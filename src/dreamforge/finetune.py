"""Stage 5 -- fine-tune SmolVLA on imagined corrections + expert replay.

The default recipe mixes:
  - synthetic samples: (observed state -> imagined corrective chunk)
  - replay samples:    expert LIBERO data (prevents forgetting / collapse)

Control run: ``--synthetic none`` (replay only, same step budget) isolates the
effect of the imagined data.

Implementation notes:
  - Batch construction mirrors LeRobot's dataset item format so the policy's
    existing pre-processors (normalization, tokenization, rename) apply
    unchanged.
  - ``policy.forward`` returns the flow-matching action loss.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .paths import DEFAULTS
from .policy import _RENAME_MAP

__all__ = ["finetune", "FinetuneConfig"]


class FinetuneConfig:
    def __init__(self, samples_dir: str | None, replay_root: str, out_dir: str,
                 policy_path: str, steps: int = 3000, batch_size: int = 16,
                 replay_ratio: float = 0.5, lr: float = 5e-6, num_workers: int = 3,
                 device: str = "cuda", seed: int = 0, smoke: bool = False):
        self.samples_dir = samples_dir
        self.replay_root = replay_root
        self.out_dir = out_dir
        self.policy_path = policy_path
        self.steps = steps
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.lr = lr
        self.num_workers = num_workers
        self.device = device
        self.seed = seed
        self.smoke = smoke


class SyntheticSamples(Dataset):
    """Loads ``build`` shards lazily (one shard at a time)."""

    def __init__(self, samples_dir: str):
        self.files = sorted(Path(samples_dir).glob("samples_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no samples_*.npz under {samples_dir}")
        self.index: list[tuple[int, int]] = []
        self._cache: dict[int, dict] = {}
        for fi, f in enumerate(self.files):
            with np.load(f, allow_pickle=True) as d:
                n = int(d["state"].shape[0])
            self.index.extend([(fi, i) for i in range(n)])

    def _shard(self, fi: int) -> dict:
        if fi not in self._cache:
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[fi] = dict(np.load(self.files[fi], allow_pickle=True))
        return self._cache[fi]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fi, i = self.index[idx]
        d = self._shard(fi)
        img = torch.from_numpy(d["image"][i]).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(d["image2"][i]).permute(2, 0, 1).float() / 255.0
        return {
            "observation.images.image": img,
            "observation.images.image2": img2,
            "observation.state": torch.from_numpy(d["state"][i]).float(),
            "action": torch.from_numpy(d["action"][i]).float(),
            "task": str(d["task"][i]),
        }


def build_replay_dataset(cfg, policy_cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(repo_id="local/libero", root=cfg.replay_root)
    delta_ts = resolve_delta_timestamps(policy_cfg, meta)
    return LeRobotDataset(repo_id="local/libero", root=cfg.replay_root, delta_timestamps=delta_ts)


def _collate(samples: list[dict]) -> dict:
    """Collate for both synthetic and replay datasets (keep strings, stack tensors)."""
    keys = samples[0].keys()
    batch: dict = {}
    for k in keys:
        vals = [s[k] for s in samples]
        if isinstance(vals[0], torch.Tensor):
            batch[k] = torch.stack(vals)
        else:
            batch[k] = vals
    return batch


def finetune(cfg: FinetuneConfig, log=print) -> dict:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    import lerobot.policies  # noqa: F401

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.device = cfg.device
    policy_cfg.pretrained_path = cfg.policy_path

    replay_ds = build_replay_dataset(cfg, policy_cfg)
    policy = make_policy(policy_cfg, ds_meta=replay_ds.meta, rename_map=_RENAME_MAP).to(cfg.device)
    try:
        policy.model.vlm_with_expert.vlm.gradient_checkpointing_enable()
        log("  gradient checkpointing enabled")
    except Exception as e:  # pragma: no cover
        log(f"  gradient checkpointing unavailable: {e}")
    policy.train()
    pre, _post = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=cfg.policy_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device},
                                "rename_observations_processor": {"rename_map": _RENAME_MAP}})

    synth_loader = None
    if cfg.samples_dir and Path(cfg.samples_dir, "summary.json").exists():
        synth_ds = SyntheticSamples(cfg.samples_dir)
        synth_loader = DataLoader(synth_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, drop_last=True,
                                  collate_fn=_collate, persistent_workers=cfg.num_workers > 0)
        log(f"  synthetic samples: {len(synth_ds)}")
    replay_loader = DataLoader(replay_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, drop_last=True,
                               collate_fn=_collate, persistent_workers=cfg.num_workers > 0)
    log(f"  replay samples: {len(replay_ds)}")

    def _infinite(loader):
        while True:
            for b in loader:
                yield b

    synth_it = _infinite(synth_loader) if synth_loader is not None else None
    replay_it = _infinite(replay_loader)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-5)
    logf = (out_dir / "train.log.jsonl").open("w")
    step, t0 = 0, time.time()
    target = 5 if cfg.smoke else cfg.steps
    while step < target:
        use_synth = synth_it is not None and torch.rand(1).item() < (1.0 - cfg.replay_ratio)
        batch = next(synth_it) if use_synth else next(replay_it)
        batch = pre(batch)
        out = policy.forward(batch)
        loss = out[0] if isinstance(out, tuple) else (out["loss"] if isinstance(out, dict) else out)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss.detach()), 5),
                   "source": "synthetic" if use_synth else "replay",
                   "steps_per_sec": round(step / (time.time() - t0), 2)}
            log(json.dumps(rec))
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

    logf.close()
    summary = {"steps": step, "synthetic": cfg.samples_dir, "replay_root": cfg.replay_root,
               "replay_ratio": cfg.replay_ratio, "base_policy": cfg.policy_path,
               "minutes": round((time.time() - t0) / 60, 1)}
    if not cfg.smoke:
        policy.save_pretrained(out_dir)
        (out_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
        print("saved ->", out_dir)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="fine-tune SmolVLA on imagined corrections")
    p.add_argument("--samples", default=None, help="dir with samples_*.npz ('none' disables synthetic)")
    p.add_argument("--replay-root", default=DEFAULTS.dataset_root)
    p.add_argument("--out", default=str(DEFAULTS.run_dir / "ft_synth"))
    p.add_argument("--policy-path", default=DEFAULTS.policy_path)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--replay-ratio", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    samples = None if args.samples in (None, "none") else args.samples
    finetune(FinetuneConfig(samples_dir=samples, replay_root=args.replay_root, out_dir=args.out,
                            policy_path=args.policy_path, steps=args.steps,
                            batch_size=args.batch_size, replay_ratio=args.replay_ratio,
                            lr=args.lr, num_workers=args.num_workers, device=args.device,
                            smoke=args.smoke))


if __name__ == "__main__":
    main()
