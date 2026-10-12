"""Closed-loop evaluation of a policy on a LIBERO suite (paired by seed).

Thin wrapper around the same env/policy code used for mining, but it records
only per-episode outcomes (task, seed, success, length) so runs stay cheap.
Results are JSON, ready for ``report.py`` paired analysis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .envs import env_horizon, make_task_envs, step_checked, task_description
from .paths import DEFAULTS
from .policy import SmolVLABundle, load_smolvla

__all__ = ["evaluate", "EvalConfig"]


class EvalConfig:
    def __init__(self, policy_path: str, task: str = "libero_spatial", task_ids: list[int] | None = None,
                 episodes: int = 5, chunk_exec: int = 50, seed: int = 0, max_steps: int | None = None,
                 out: str = "runs/eval.json", device: str = "cuda"):
        self.policy_path = policy_path
        self.task = task
        self.task_ids = task_ids
        self.episodes = episodes
        self.chunk_exec = chunk_exec
        self.seed = seed
        self.max_steps = max_steps
        self.out = out
        self.device = device


def _rollout(bundle: SmolVLABundle, gym_env, task_desc: str, seed: int,
             chunk_exec: int, budget: int) -> tuple[bool, int]:
    obs, _ = gym_env.reset(seed=seed)
    success, step, done = False, 0, False
    while not done and step < budget:
        o = bundle.obs_to_input(obs, task_desc)
        chunk = bundle.sample_chunk(o)
        raw = bundle.unnormalize(chunk)
        for t in range(min(chunk_exec, raw.shape[1])):
            env_action = bundle.to_env_action(raw[:, t, :])
            obs, success_step, done = step_checked(gym_env, env_action)
            step += 1
            if success_step:
                success = True
            if done or obs is None:
                done = True
                break
    return bool(success), step


def evaluate(cfg: EvalConfig, log=print) -> dict:
    bundle = load_smolvla(cfg.policy_path, device=cfg.device)
    task_envs = make_task_envs(cfg.task)
    if cfg.task_ids:
        keep = set(cfg.task_ids)
        task_envs = [te for i, te in enumerate(task_envs) if i in keep]

    episodes = []
    t0 = time.time()
    for task_id, (name, gym_env) in enumerate(task_envs):
        desc = task_description(gym_env)
        for ep in range(cfg.episodes):
            seed = cfg.seed * 100000 + task_id * 1000 + ep
            budget = cfg.max_steps or env_horizon(gym_env, cfg.task)
            success, length = _rollout(bundle, gym_env, desc, seed, cfg.chunk_exec, budget)
            episodes.append({"task": name, "task_id": task_id, "episode": ep,
                             "seed": seed, "success": success, "length": length})
            log(f"    {name} ep{ep}: success={success} len={length}")
    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    out = {"policy_path": cfg.policy_path, "task": cfg.task, "episodes": episodes,
           "n": n, "successes": k, "success_rate": round(100 * k / max(n, 1), 2),
           "minutes": round((time.time() - t0) / 60, 1)}
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"[eval] {k}/{n} = {out['success_rate']}% ({out['minutes']} min) -> {cfg.out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="paired closed-loop evaluation on LIBERO")
    p.add_argument("--policy-path", default=DEFAULTS.policy_path)
    p.add_argument("--task", default="libero_spatial")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--chunk-exec", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default="runs/eval.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    evaluate(EvalConfig(policy_path=args.policy_path, task=args.task, task_ids=args.task_ids,
                        episodes=args.episodes, chunk_exec=args.chunk_exec, seed=args.seed,
                        max_steps=args.max_steps, out=args.out, device=args.device))


if __name__ == "__main__":
    main()
