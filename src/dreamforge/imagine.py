"""Stage 2 -- imagine corrective actions with the world model (CEM).

For every mined decision point (or only those from failed episodes), encode
the agent view into the world model's latent space and run CEM to search a
50-step action plan that maximizes imagined task progress.  The base policy's
own chunk seeds the search, so the result is a *correction* of the policy's
plan, not a from-scratch plan.

Outputs per mined episode file:``<run>/imagine/<same>.npz`` with
    correction      (S, H, 7)  best imagined plan (dataset raw action space)
    wm_best_score   (S,)       progress score of the best plan
    wm_init_score   (S,)       progress score of the base policy chunk
    progress_curve  (S, H)     imagined progress along the best plan
    history         (S, iters) best score per CEM iteration
    random_actions  (S, H, 7)  uniform random control plan (fixed seed)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cem import ActionCEM, CEMConfig
from .paths import DEFAULTS
from .wm import encode_image, load_world_model, make_progress_head

__all__ = ["imagine", "ImagineConfig"]


class ImagineConfig:
    def __init__(self, mine_dir: str, out_dir: str, wm: str = "base", context: str = "failure",
                 max_states: int | None = None, horizon: int = 50, population: int = 64,
                 elites: int = 8, iters: int = 6, init_std: float = 0.25, seed: int = 0,
                 action_clip: bool = True, delta: float | None = None, delta_sigma: float = 0.5,
                 device: str = "cuda"):
        self.mine_dir = mine_dir
        self.out_dir = out_dir
        self.wm = wm
        self.context = context          # failure | all
        self.max_states = max_states
        self.horizon = horizon
        self.population = population
        self.elites = elites
        self.iters = iters
        self.init_std = init_std
        self.seed = seed
        self.action_clip = action_clip
        self.delta = delta              # uniform trust region (optional)
        self.delta_sigma = delta_sigma  # per-dim trust region = delta_sigma * std
        self.device = device


def _action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Dataset action min/max per dim (for clipping / random controls)."""
    cache = Path(DEFAULTS.action_cache)
    if cache.exists():
        a = np.load(cache, mmap_mode="r")
        return np.asarray(a.min(axis=0), dtype=np.float32), np.asarray(a.max(axis=0), dtype=np.float32)
    return np.full(7, -1.0, np.float32), np.full(7, 1.0, np.float32)


def _action_scale() -> np.ndarray:
    """Per-dimension std of dataset actions (used to size the trust region)."""
    cache = Path(DEFAULTS.action_cache)
    if cache.exists():
        a = np.load(cache, mmap_mode="r")
        s = np.asarray(a).std(axis=0).astype(np.float32)
        return np.maximum(s, 1e-3)
    return np.full(7, 0.1, np.float32)


def imagine(cfg: ImagineConfig, log=print) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm, ckpt = load_world_model(DEFAULTS.wm_path(cfg.wm), device=cfg.device)
    head = make_progress_head(ckpt, device=cfg.device)
    low, high = _action_bounds()
    scale = _action_scale()
    delta_per_dim = tuple(float(x) for x in (scale * cfg.delta_sigma)) if cfg.delta_sigma else None
    cem_cfg = CEMConfig(horizon=cfg.horizon, population=cfg.population, elites=cfg.elites,
                        iters=cfg.iters, init_std=cfg.init_std, seed=cfg.seed,
                        delta=cfg.delta, delta_per_dim=delta_per_dim,
                        low=tuple(low) if cfg.action_clip else None,
                        high=tuple(high) if cfg.action_clip else None)
    cem = ActionCEM(wm, head, cem_cfg, device=cfg.device)
    rng = np.random.default_rng(cfg.seed)

    files = sorted(Path(cfg.mine_dir).glob("*.npz"))
    log(f"[imagine] wm={cfg.wm} files={len(files)} context={cfg.context}")
    summary = {"wm": cfg.wm, "context": cfg.context, "n_states": 0, "episodes": []}
    t0 = time.time()
    budget_states = cfg.max_states or 10**9

    for f in files:
        d = np.load(f, allow_pickle=False)
        success = bool(d["success"])
        if cfg.context == "failure" and success:
            continue
        S = int(d["decision_steps"].shape[0])
        if S == 0 or summary["n_states"] >= budget_states:
            continue
        take = min(S, budget_states - summary["n_states"])
        corrections, best_scores, init_scores, curves, histories, randoms = [], [], [], [], [], []
        for s in range(take):
            img = torch.from_numpy(d["img96"][s]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            z0 = encode_image(wm, img.to(cfg.device))
            init_mean = torch.from_numpy(np.asarray(d["chunk_raw"][s], dtype=np.float32))
            res = cem.optimize(z0, init_mean=init_mean)
            corrections.append(res.actions.numpy().astype(np.float32))
            best_scores.append(res.best_score)
            init_scores.append(res.init_score)
            curves.append(np.asarray(res.progress_curve, dtype=np.float32))
            histories.append(np.asarray(res.history, dtype=np.float32))
            # control: random perturbation of the *base chunk* within the same budget,
            # or unconstrained uniform sampling over the dataset range when the search
            # itself is unconstrained (delta_sigma = 0)
            base = np.asarray(d["chunk_raw"][s], dtype=np.float32)[: cfg.horizon]
            if cfg.delta_sigma:
                budget = scale * cfg.delta_sigma
                rnd = np.clip(base + rng.uniform(-1.0, 1.0, size=base.shape) * budget, low, high)
            else:
                rnd = rng.uniform(low, high, size=base.shape)
            randoms.append(rnd.astype(np.float32))
        np.savez_compressed(
            out_dir / f.name,
            correction=np.stack(corrections), wm_best_score=np.asarray(best_scores, np.float32),
            wm_init_score=np.asarray(init_scores, np.float32),
            progress_curve=np.stack(curves), history=np.stack(histories),
            random_actions=np.stack(randoms),
            decision_steps=d["decision_steps"][:take], task_name=d["task_name"],
            task_id=int(d["task_id"]), episode=int(d["episode"]), success=success,
            source_file=f.name,
        )
        summary["n_states"] += take
        summary["episodes"].append({"file": f.name, "states": take,
                                    "mean_init": float(np.mean(init_scores)),
                                    "mean_best": float(np.mean(best_scores))})
        log(f"    {f.name}: {take} states | WM score {np.mean(init_scores):.3f} -> "
            f"{np.mean(best_scores):.3f}")

    summary["minutes"] = round((time.time() - t0) / 60, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    gains = [e["mean_best"] - e["mean_init"] for e in summary["episodes"]]
    if gains:
        log(f"[imagine] done: {summary['n_states']} states | imagined gain "
            f"{np.mean(gains):+.3f} | {summary['minutes']} min")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="imagine corrective actions with the world model")
    p.add_argument("--mine", default=str(DEFAULTS.run_dir / "mine"))
    p.add_argument("--out", default=str(DEFAULTS.run_dir / "imagine"))
    p.add_argument("--wm", default="base", help="world-model name (base/aso/full/...) or path")
    p.add_argument("--context", choices=["failure", "all"], default="failure")
    p.add_argument("--max-states", type=int, default=None)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--elites", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--delta", type=float, default=None,
                   help="uniform trust-region half-width (optional; prefer --delta-sigma)")
    p.add_argument("--delta-sigma", type=float, default=0.5,
                   help="per-dimension trust region as a multiple of dataset action std (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    imagine(ImagineConfig(mine_dir=args.mine, out_dir=args.out, wm=args.wm,
                          context=args.context, max_states=args.max_states,
                          horizon=args.horizon, population=args.population,
                          elites=args.elites, iters=args.iters, seed=args.seed,
                          delta=args.delta, delta_sigma=args.delta_sigma,
                          device=args.device))


if __name__ == "__main__":
    main()
