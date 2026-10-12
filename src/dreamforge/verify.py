"""Stage 3 -- verify imagined corrections in the real simulator.

For each mined decision point we replay the exact saved simulator state and
run one of three arms, then hand control back to the unchanged base policy:

  none      : base policy continues immediately (should reproduce its failure)
  imagined  : execute the first ``exec_horizon`` steps of the CEM plan,
              then hand back to the base policy
  random    : execute a uniform random action sequence of the same length
              (same persisted sequence for every run), then hand back

Outcome per trial: final episode success after hand-off.  The comparison
``imagined vs none`` measures whether world-model optimization actually
produces corrective actions; ``imagined vs random`` measures whether any of
the benefit is specific to the model rather than to perturbation.

Results are appended to ``<run>/verify/<arm>.jsonl`` (one line per trial),
so long runs are resumable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .envs import (clear_done, env_horizon, load_sim_state, make_task_envs, obs_from_state,
                   step_checked, task_description)
from .paths import DEFAULTS
from .policy import SmolVLABundle, load_smolvla

__all__ = ["verify", "VerifyConfig"]


class VerifyConfig:
    def __init__(self, mine_dir: str, imagine_dir: str, out_dir: str, arms: list[str] | None = None,
                 exec_horizon: int = 16, chunk_exec: int = 50, max_states: int | None = None,
                 policy_path: str = "", steps_budget: int = 400, seed: int = 0,
                 device: str = "cuda", task: str = "libero_spatial"):
        self.mine_dir = mine_dir
        self.imagine_dir = imagine_dir
        self.out_dir = out_dir
        self.arms = arms or ["none", "imagined", "random"]
        self.exec_horizon = exec_horizon
        self.chunk_exec = chunk_exec
        self.max_states = max_states
        self.policy_path = policy_path
        self.steps_budget = steps_budget
        self.seed = seed
        self.device = device
        self.task = task


def _load_done(path: Path) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["file"], int(r["decision"]), r["arm"]))
    return done


def _run_trial(bundle: SmolVLABundle, gym_env, task_desc: str, sim_state: np.ndarray,
               plan: np.ndarray | None, arm: str, step_at_mine: int, horizon: int,
               cfg: VerifyConfig) -> dict:
    load_sim_state(gym_env, sim_state)
    # keep the env's internal step counter consistent with the restored state
    inner = gym_env.envs[0]._env.env
    if hasattr(inner, "timestep"):
        try:
            inner.timestep = int(step_at_mine)
        except Exception:
            pass
    clear_done(gym_env)

    steps_used = 0
    success = False
    obs = None

    # ---- correction phase ----
    if plan is not None and arm != "none":
        for t in range(min(cfg.exec_horizon, plan.shape[0])):
            env_action = bundle.to_env_action(torch.from_numpy(plan[t]).view(1, -1))
            obs, success, done = step_checked(gym_env, env_action)
            steps_used += 1
            if success or done or obs is None:
                break
    if not success:
        if obs is None:
            obs = obs_from_state(gym_env)
        # ---- hand-off phase: unchanged base policy ----
        torch.manual_seed(cfg.seed * 100003 + step_at_mine)
        budget = max(50, min(cfg.steps_budget, horizon - step_at_mine))
        while steps_used < budget and not success:
            o = bundle.obs_to_input(obs, task_desc)
            chunk = bundle.sample_chunk(o)
            raw = bundle.unnormalize(chunk)
            for t in range(min(cfg.chunk_exec, raw.shape[1])):
                env_action = bundle.to_env_action(raw[:, t, :])
                obs, success, done = step_checked(gym_env, env_action)
                steps_used += 1
                if success or done or obs is None or steps_used >= budget:
                    break
    return {"success": bool(success), "steps_used": int(steps_used),
            "completed_during_correction": bool(success and plan is not None and arm != "none"
                                                and steps_used <= min(cfg.exec_horizon, 10**9))}


def verify(cfg: VerifyConfig, log=print) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_smolvla(cfg.policy_path, device=cfg.device)
    task_envs = make_task_envs(cfg.task)
    env_by_name: dict[str, Any] = {}
    task_desc_by_name: dict[str, str] = {}
    horizon_by_name: dict[str, int] = {}
    for name, env in task_envs:
        env.reset(seed=cfg.seed)          # ensures the underlying robosuite env exists
        env_by_name[name] = env
        task_desc_by_name[name] = task_description(env)
        horizon_by_name[name] = env_horizon(env)

    done_sets = {arm: _load_done(out_dir / f"{arm}.jsonl") for arm in cfg.arms}
    handles = {arm: (out_dir / f"{arm}.jsonl").open("a") for arm in cfg.arms}

    files = sorted(Path(cfg.imagine_dir).glob("*.npz"))
    n_trials = 0
    t0 = time.time()
    for f in files:
        d = np.load(f, allow_pickle=False)
        mine_file = Path(cfg.mine_dir) / str(d["source_file"])
        m = np.load(mine_file, allow_pickle=False)
        task_name = str(d["task_name"])
        gym_env = env_by_name[task_name]
        task_desc = task_desc_by_name[task_name]
        horizon = horizon_by_name[task_name]
        S = int(d["decision_steps"].shape[0])
        n_states = S if cfg.max_states is None else min(S, cfg.max_states)
        for s in range(n_states):
            step_at = int(d["decision_steps"][s])
            for arm in cfg.arms:
                key = (f.name, s, arm)
                if key in done_sets[arm]:
                    continue
                plan = None
                if arm == "imagined":
                    plan = np.asarray(d["correction"][s], dtype=np.float32)
                elif arm == "random":
                    plan = np.asarray(d["random_actions"][s], dtype=np.float32)
                t1 = time.time()
                res = _run_trial(bundle, gym_env, task_desc,
                                 np.asarray(m["sim_state"][s], dtype=np.float64),
                                 plan, arm, step_at, horizon, cfg)
                rec = {"file": f.name, "task_name": task_name, "decision": s,
                       "decision_step": step_at, "arm": arm, "episode_success_original": bool(m["success"]),
                       "wm_best_score": float(d["wm_best_score"][s]),
                       "wm_init_score": float(d["wm_init_score"][s]),
                       **res, "trial_seconds": round(time.time() - t1, 1)}
                handles[arm].write(json.dumps(rec) + "\n")
                handles[arm].flush()
                n_trials += 1
                log(f"    {f.name} d{s} {arm:8s} success={res['success']} "
                    f"steps={res['steps_used']} ({rec['trial_seconds']}s)")

    for h in handles.values():
        h.close()
    summary = {"n_trials": n_trials, "arms": cfg.arms,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / f"verify_summary_{'_'.join(cfg.arms)}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"[verify] {n_trials} trials in {summary['minutes']} min")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="verify imagined corrections in the simulator")
    p.add_argument("--mine", default=str(DEFAULTS.run_dir / "mine"))
    p.add_argument("--imagine", default=str(DEFAULTS.run_dir / "imagine"))
    p.add_argument("--out", default=str(DEFAULTS.run_dir / "verify"))
    p.add_argument("--arms", nargs="*", default=["none", "imagined", "random"])
    p.add_argument("--exec-horizon", type=int, default=16)
    p.add_argument("--chunk-exec", type=int, default=50)
    p.add_argument("--max-states", type=int, default=None)
    p.add_argument("--policy-path", default=DEFAULTS.policy_path)
    p.add_argument("--task", default="libero_spatial")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    verify(VerifyConfig(mine_dir=args.mine, imagine_dir=args.imagine, out_dir=args.out,
                        arms=args.arms, exec_horizon=args.exec_horizon,
                        chunk_exec=args.chunk_exec, max_states=args.max_states,
                        policy_path=args.policy_path, task=args.task, device=args.device))


if __name__ == "__main__":
    main()
