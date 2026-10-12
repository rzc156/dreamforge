"""Stage 4 -- build a fine-tuning dataset from imagined corrections.

Takes the outputs of ``mine`` + ``imagine`` (+ optional ``verify`` filtering)
and writes compact sample shards:

    <run>/build/samples_XXXX.npz
        image        (N, 360, 360, 3) uint8   agent view at the mined state
        image2       (N, 360, 360, 3) uint8   wrist view
        state        (N, 8)           float32 policy state vector
        action       (N, 50, 7)       float32 imagined corrective chunk
        task         (N,)             object array of instruction strings
        meta_json    scalar           json string with provenance

Filter modes:
    all       every imagined correction from failed episodes
    lifted    corrections whose real replay succeeded while the base policy did
              not (requires verify results; this is the strong filter)
    gained    imagined progress >= base progress + delta (model-side filter,
              no simulator needed -- tests whether the WM score alone is enough
              to pick useful corrections)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import DEFAULTS

__all__ = ["build_dataset", "BuildConfig"]


class BuildConfig:
    def __init__(self, mine_dir: str, imagine_dir: str, out_dir: str, filter_mode: str = "all",
                 verify_dir: str | None = None, wm_gain: float = 0.05, shard_size: int = 200,
                 max_samples: int | None = None, max_decision_step: int | None = None):
        self.mine_dir = mine_dir
        self.imagine_dir = imagine_dir
        self.out_dir = out_dir
        self.filter_mode = filter_mode
        self.verify_dir = verify_dir
        self.wm_gain = wm_gain
        self.shard_size = shard_size
        self.max_samples = max_samples
        # only keep corrections applied early in the episode (recoverable states)
        self.max_decision_step = max_decision_step


def _load_verify(verify_dir: Path) -> dict[tuple[str, int], dict[str, bool]]:
    """file -> decision -> {arm: success}"""
    out: dict[tuple[str, int], dict[str, bool]] = {}
    for arm in ("none", "imagined", "random"):
        p = verify_dir / f"{arm}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault((r["file"], int(r["decision"])), {})[r["arm"]] = bool(r["success"])
    return out


def _state_stats(policy_path: str):
    """Mean/std used by the policy's state normalizer (for de-normalizing legacy picks)."""
    import glob

    from safetensors.torch import load_file

    cands = sorted(glob.glob(policy_path + "/**/*normalizer*safetensors", recursive=True))
    for f in cands:
        try:
            sd = load_file(f)
        except Exception:
            continue
        if "observation.state.mean" in sd and "observation.state.std" in sd:
            return (sd["observation.state.mean"].numpy().astype(np.float32),
                    sd["observation.state.std"].numpy().astype(np.float32))
    return None, None


def build_dataset(cfg: BuildConfig, log=print) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("samples_*.npz"):  # a rebuild must not mix with previous shards
        stale.unlink()
    verify = _load_verify(Path(cfg.verify_dir)) if cfg.verify_dir else {}
    mean, std = _state_stats(DEFAULTS.policy_path)

    n = 0
    shard: list[dict] = []
    shard_id = 0

    def flush_shard() -> None:
        nonlocal shard, shard_id, n
        if not shard:
            return
        path = out_dir / f"samples_{shard_id:04d}.npz"
        shard_id += 1
        np.savez_compressed(
            path,
            image=np.stack([s["image"] for s in shard]),
            image2=np.stack([s["image2"] for s in shard]),
            state=np.stack([s["state"] for s in shard]),
            action=np.stack([s["action"] for s in shard]),
            task=np.asarray([s["task"] for s in shard], dtype=object),
            meta_json=json.dumps([s["meta"] for s in shard]),
        )
        log(f"    wrote {path.name}: {len(shard)} samples (total {n})")
        shard = []

    counts = {"kept": 0, "skipped": 0}
    for f in sorted(Path(cfg.imagine_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        if bool(d["success"]):
            continue
        m = np.load(Path(cfg.mine_dir) / str(d["source_file"]), allow_pickle=False)
        S = int(d["decision_steps"].shape[0])
        for s in range(S):
            if cfg.max_samples is not None and n >= cfg.max_samples:
                break
            keep = True
            if cfg.max_decision_step is not None and int(d["decision_steps"][s]) > cfg.max_decision_step:
                keep = False
            if keep:
                if cfg.filter_mode == "lifted":
                    v = verify.get((f.name, s), {})
                    keep = v.get("imagined", False) and not v.get("none", True)
                elif cfg.filter_mode == "gained":
                    gain = float(d["wm_best_score"][s]) - float(d["wm_init_score"][s])
                    keep = gain >= cfg.wm_gain
            if not keep:
                counts["skipped"] += 1
                continue
            # state must be in dataset space (the training batch normalizes it again)
            if "state_env" in m.files:
                state = np.asarray(m["state_env"][s], dtype=np.float32)
            elif mean is not None:
                state = np.asarray(m["state"][s], dtype=np.float32) * (std + 1e-8) + mean
            else:
                state = np.asarray(m["state"][s], dtype=np.float32)
            shard.append({
                "image": np.asarray(m["image"][s], dtype=np.uint8),
                "image2": np.asarray(m["image2"][s], dtype=np.uint8),
                "state": state,
                "action": np.asarray(d["correction"][s], dtype=np.float32),
                "task": str(m["task_desc"]),
                "meta": {"file": f.name, "decision": s,
                         "step": int(d["decision_steps"][s]),
                         "wm_init": float(d["wm_init_score"][s]),
                         "wm_best": float(d["wm_best_score"][s])},
            })
            n += 1
            counts["kept"] += 1
            if len(shard) >= cfg.shard_size:
                flush_shard()
        if cfg.max_samples is not None and n >= cfg.max_samples:
            break

    flush_shard()
    summary = {"filter_mode": cfg.filter_mode, "n_samples": n, **counts,
               "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"[build] {cfg.filter_mode}: {n} samples ({counts['skipped']} skipped)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="build fine-tuning samples from imagined corrections")
    p.add_argument("--mine", default=str(DEFAULTS.run_dir / "mine"))
    p.add_argument("--imagine", default=str(DEFAULTS.run_dir / "imagine"))
    p.add_argument("--out", default=str(DEFAULTS.run_dir / "build"))
    p.add_argument("--filter", dest="filter_mode", choices=["all", "lifted", "gained"], default="all")
    p.add_argument("--verify-dir", default=None)
    p.add_argument("--wm-gain", type=float, default=0.05)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-decision-step", type=int, default=None,
                   help="keep only corrections applied at/earlier than this episode step")
    args = p.parse_args()
    build_dataset(BuildConfig(mine_dir=args.mine, imagine_dir=args.imagine, out_dir=args.out,
                              filter_mode=args.filter_mode, verify_dir=args.verify_dir,
                              wm_gain=args.wm_gain, max_samples=args.max_samples,
                              max_decision_step=args.max_decision_step))


if __name__ == "__main__":
    main()
