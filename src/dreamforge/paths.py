"""Central paths / defaults for dreamforge.

All heavy assets live on the data disk; override with CLI flags or env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULTS"]


def _env_path(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Paths:
    # --- existing assets (from the author's environment) ---
    policy_path: str = _env_path(
        "DREAMFORGE_POLICY",
        "/media/rzc/PS3000/embodied-ai/models/smolvla_libero",
    )
    # world model checkpoints (relative to the research results dir)
    research_results: str = _env_path(
        "DREAMFORGE_RESEARCH_RESULTS",
        "/home/rzc/embodied-ai/research/results",
    )
    dataset_root: str = _env_path(
        "DREAMFORGE_DATASET",
        "/media/rzc/PS3000/embodied-ai/datasets/lerobot_libero",
    )
    action_cache: str = _env_path(
        "DREAMFORGE_ACTION_CACHE",
        "/media/rzc/PS3000/embodied-ai/datasets/libero_cache/actions.npy",
    )
    # --- outputs ---
    run_dir: Path = field(default_factory=lambda: Path(
        _env_path("DREAMFORGE_RUN", "/media/rzc/PS3000/embodied-projects/dreamforge/runs/default")))

    def wm_path(self, name: str = "base") -> str:
        """Named world-model checkpoints.

        base / aso / caa / cbt / full / aso_cbt / caa_aso / caa_cbt
        (from the ablation set), or a direct path when ``name`` contains '/'.
        """
        if "/" in name or name.endswith(".pt"):
            return name if name.startswith("/") else os.path.join(self.research_results, name)
        return os.path.join(self.research_results, "ablation", f"wm_{name}.pt")


DEFAULTS = Paths()
