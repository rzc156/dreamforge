"""Stage 1 -- mine decision points from base-policy rollouts.

Runs SmolVLA on a LIBERO suite and records, at every decision point:

  - the observation the policy saw (both cameras + robot state)
  - the action chunk the policy was about to execute (dataset raw space)
  - the exact simulator state (flattened MjSimState) for later replay
  - the final episode outcome (success / failure)

Everything is written as one .npz per episode under ``<run>/mine/<suite>/``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .envs import (env_horizon, is_success, make_task_envs, raw_env, save_sim_state,
                   step_checked, task_description)
from .policy import SmolVLABundle, load_smolvla

__all__ = ["mine", "MineConfig"]


class MineConfig:
    def __init__(self, task: str = "libero_spatial", task_ids: list[int] | None = None,
                 episodes: int = 5, chunk_exec: int = 50, seed: int = 0,
                 max_steps: int | None = None, policy_path: str = "",
                 out_dir: str = "runs/mine", save_images: bool = True,
                 wm_image_size: int = 96, device: str = "cuda"):
        self.task = task
        self.task_ids = task_ids
        self.episodes = episodes
        self.chunk_exec = chunk_exec
        self.seed = seed
        self.max_steps = max_steps
        self.policy_path = policy_path
        self.out_dir = out_dir
        self.save_images = save_images
        self.wm_image_size = wm_image_size
        self.device = device


def _agentview96(obs: dict, device: str, size: int) -> np.ndarray:
    """Downscaled agent view for the world model, (size,size,3) uint8."""
    img = obs["pixels"]["image"]  # (1,H,W,3) uint8
    if isinstance(img, np.ndarray) and img.ndim == 4:
        img = img[0]
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = torch.nn.functional.interpolate(t.to(device), size=(size, size), mode="bilinear",
                                        align_corners=False)
    arr = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return arr


def _rollout_episode(bundle: SmolVLABundle, gym_env, task_desc: str, seed: int,
                     cfg: MineConfig) -> dict:
    obs, _ = gym_env.reset(seed=seed)
    horizon = env_horizon(gym_env)
    budget = cfg.max_steps or horizon

    decision_steps: list[int] = []
    images: list[np.ndarray] = []
    images2: list[np.ndarray] = []
    thumbs: list[np.ndarray] = []
    states: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    sim_states: list[np.ndarray] = []

    success = False
    step = 0
    done = False
    while not done and step < budget:
        o = bundle.obs_to_input(obs, task_desc)
        chunk = bundle.sample_chunk(o)                 # (1, T, A) normalized
        raw_chunk = bundle.unnormalize(chunk)          # (1, T, A) dataset raw

        decision_steps.append(step)
        states.append(np.asarray(o["observation.state"][0].detach().cpu(), dtype=np.float32))
        chunks.append(np.asarray(raw_chunk[0].detach().cpu(), dtype=np.float32))
        sim_states.append(save_sim_state(gym_env))
        if cfg.save_images:
            images.append(np.asarray(obs["pixels"]["image"][0], dtype=np.uint8))
            images2.append(np.asarray(obs["pixels"]["image2"][0], dtype=np.uint8))
        thumbs.append(_agentview96(obs, cfg.device, cfg.wm_image_size))

        for t in range(min(cfg.chunk_exec, raw_chunk.shape[1])):
            env_action = bundle.to_env_action(raw_chunk[:, t, :])
            obs, success_step, done = step_checked(gym_env, env_action)
            step += 1
            if success_step:
                success = True
            if done or obs is None:
                done = True
                break

    return {
        "task_desc": task_desc,
        "seed": seed,
        "success": bool(success),
        "episode_len": step,
        "horizon": horizon,
        "decision_steps": np.asarray(decision_steps, dtype=np.int64),
        "state": np.stack(states) if states else np.zeros((0, 8), np.float32),
        "chunk_raw": np.stack(chunks) if chunks else np.zeros((0, 50, 7), np.float32),
        "sim_state": np.stack(sim_states) if sim_states else np.zeros((0, 1), np.float64),
        "img96": np.stack(thumbs) if thumbs else np.zeros((0, 96, 96, 3), np.uint8),
        "image": np.stack(images) if images else np.zeros((0, 360, 360, 3), np.uint8),
        "image2": np.stack(images2) if images2 else np.zeros((0, 360, 360, 3), np.uint8),
    }


def mine(cfg: MineConfig, log=print) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_smolvla(cfg.policy_path, device=cfg.device)
    task_envs = make_task_envs(cfg.task)
    if cfg.task_ids:
        task_envs = [te for i, te in enumerate(task_envs) if i in set(cfg.task_ids)]

    summary = {"task": cfg.task, "episodes": [], "n_decision_points": 0, "n_success": 0}
    t0 = time.time()
    for task_id, (name, gym_env) in enumerate(task_envs):
        task_desc = task_description(gym_env)
        log(f"[mine] task {name}: {task_desc[:60]}")
        for ep in range(cfg.episodes):
            seed = cfg.seed * 100000 + task_id * 1000 + ep
            try:
                rec = _rollout_episode(bundle, gym_env, task_desc, seed, cfg)
            except Exception as exc:  # one bad episode must not kill the run
                log(f"    ep{ep}: FAILED ({type(exc).__name__}: {exc})")
                summary["episodes"].append({"task": name, "episode": ep, "seed": seed,
                                            "error": f"{type(exc).__name__}: {exc}"})
                continue
            rec.update({"task": cfg.task, "task_name": name, "task_id": task_id, "episode": ep})
            fname = out_dir / f"{cfg.task}__task{task_id:02d}__ep{ep:03d}.npz"
            np.savez_compressed(
                fname,
                task_name=name, task_id=task_id, episode=ep, seed=seed,
                task_desc=task_desc, success=rec["success"], episode_len=rec["episode_len"],
                horizon=rec["horizon"], decision_steps=rec["decision_steps"],
                state=rec["state"], chunk_raw=rec["chunk_raw"], sim_state=rec["sim_state"],
                img96=rec["img96"], image=rec["image"], image2=rec["image2"],
            )
            summary["episodes"].append({
                "task": name, "episode": ep, "seed": seed, "success": rec["success"],
                "len": rec["episode_len"], "decisions": len(rec["decision_steps"]),
                "file": fname.name,
            })
            summary["n_decision_points"] += len(rec["decision_steps"])
            summary["n_success"] += int(rec["success"])
            log(f"    ep{ep}: success={rec['success']} len={rec['episode_len']} "
                f"decisions={len(rec['decision_steps'])}")

    n_ep = len(summary["episodes"])
    summary["success_rate"] = round(100 * summary["n_success"] / max(1, n_ep), 2)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"[mine] done: {n_ep} episodes, success={summary['success_rate']}%, "
        f"{summary['n_decision_points']} decision points, {summary['minutes']} min")
    return summary


def main() -> None:
    from .paths import DEFAULTS

    p = argparse.ArgumentParser(description="mine decision points from base-policy rollouts")
    p.add_argument("--task", default="libero_spatial")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--chunk-exec", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-path", default=DEFAULTS.policy_path)
    p.add_argument("--out", default=str(DEFAULTS.run_dir / "mine"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    mine(MineConfig(task=args.task, task_ids=args.task_ids, episodes=args.episodes,
                    chunk_exec=args.chunk_exec, seed=args.seed,
                    policy_path=args.policy_path, out_dir=args.out, device=args.device))


if __name__ == "__main__":
    main()
