"""World-model pre-flight diagnostic: does the model actually respond to actions?

Imagination only helps if the model's predicted future depends on the action.
A model that ignores actions ("action vacancy") will happily produce plans, but
CEM has nothing to optimize -- every plan scores the same.

This module measures, from real observation latents:

  * ``score_std``          -- spread of predicted progress across random plans
                              (0 means the model cannot tell plans apart)
  * ``action_swap_delta``  -- progress difference between a repeated +a and -a
                              plan (direct action response)
  * ``latent_swap_norm``   -- latent displacement between those two plans

Run it before trusting any world model:

    python -m dreamforge.diagnose --mine runs/main/mine --wms base aso full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .paths import DEFAULTS
from .wm import encode_image, load_world_model, make_progress_head

__all__ = ["wm_sensitivity", "diagnose"]


@torch.no_grad()
def _rollout(wm, head, z0: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z0.expand(actions.shape[0], -1).clone()
    prog = []
    for t in range(actions.shape[1]):
        z, _ = wm.predict_next(z, actions[:, t, :])
        prog.append(head(z))
    return z, torch.stack(prog, dim=1)


@torch.no_grad()
def wm_sensitivity(wm, head, img96: np.ndarray, base_chunk: np.ndarray, device: str = "cuda",
                   n_random: int = 16, seed: int = 0, horizon: int | None = None) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    img = torch.from_numpy(img96).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    z0 = encode_image(wm, img)
    H = horizon or base_chunk.shape[0]
    A = wm.cfg.action_dim

    base = torch.from_numpy(base_chunk[:H]).float().to(device).unsqueeze(0)
    rand = (torch.rand(n_random, H, A, generator=gen, device=device) * 2 - 1)

    z_base, p_base = _rollout(wm, head, z0, base)
    z_rand, p_rand = _rollout(wm, head, z0, rand)

    a_pos = torch.zeros(1, H, A, device=device); a_pos[:, :, 0] = 0.9
    a_neg = torch.zeros(1, H, A, device=device); a_neg[:, :, 0] = -0.9
    z_pos, p_pos = _rollout(wm, head, z0, a_pos)
    z_neg, p_neg = _rollout(wm, head, z0, a_neg)

    return {
        "score_base": float(p_base[0, -1]),
        "score_random_mean": float(p_rand[:, -1].mean()),
        "score_random_std": float(p_rand[:, -1].std()),
        "action_swap_delta": float(abs(p_pos[0, -1] - p_neg[0, -1])),
        "latent_swap_norm": float((z_pos - z_neg).norm(dim=-1).mean()),
        "latent_z0_norm": float(z0.norm()),
    }


def diagnose(mine_dir: str, wms: list[str], max_states: int = 8, device: str = "cuda",
             out: str | None = None, log=print) -> dict:
    files = sorted(Path(mine_dir).glob("*.npz"))[:2]
    if not files:
        raise FileNotFoundError(f"no mined .npz under {mine_dir}")
    states = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        for i in range(min(int(d["img96"].shape[0]), max_states // len(files) + 1)):
            states.append((np.asarray(d["img96"][i]), np.asarray(d["chunk_raw"][i])))
        if len(states) >= max_states:
            break
    states = states[:max_states]

    report: dict[str, dict] = {}
    for name in wms:
        try:
            wm, ckpt = load_world_model(DEFAULTS.wm_path(name), device=device)
        except Exception as exc:  # pragma: no cover
            log(f"  {name}: load failed ({exc})")
            continue
        head = make_progress_head(ckpt, device=device)
        rows = [wm_sensitivity(wm, head, img, chunk, device=device) for img, chunk in states]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        report[name] = agg
        log(f"  {name:8s} score_std={agg['score_random_std']:.4f}  "
            f"swap_delta={agg['action_swap_delta']:.4f}  "
            f"latent_swap={agg['latent_swap_norm']:.3f}  |z0|={agg['latent_z0_norm']:.1f}")
    result = {"n_states": len(states), "world_models": report}
    out_path = Path(out) if out else Path(DEFAULTS.run_dir) / "wm_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="world-model action-response diagnostic")
    p.add_argument("--mine", default=str(DEFAULTS.run_dir / "mine"))
    p.add_argument("--wms", nargs="*", default=["base", "aso"])
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    diagnose(args.mine, args.wms, max_states=args.max_states, device=args.device, out=args.out)


if __name__ == "__main__":
    main()
