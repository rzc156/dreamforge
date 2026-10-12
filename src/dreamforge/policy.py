"""SmolVLA policy loading and the observation -> policy-input pipeline.

The pipeline mirrors the author's evaluation code exactly, so numbers are
comparable with existing LIBERO results:

    obs (pixels/robot_state)
      -> preprocess_observation  (LeRobot format, float images)
      -> env_pre                 (state/action env processors)
      -> pre                     (device + rename + tokenizer + normalizer)
      -> policy.predict_action_chunk -> normalized chunk (1, T, 7)
      -> post                    (unnormalize -> dataset raw action space)
      -> env_post                (env action space, for stepping)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["SmolVLABundle", "load_smolvla"]

_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@dataclass
class SmolVLABundle:
    policy: Any
    pre: Any
    post: Any
    env_pre: Any
    env_post: Any
    cfg: Any
    path: str

    # ------------------------------------------------------------------ inputs
    def obs_to_input(self, obs: dict, task_desc: str) -> dict:
        """Env observation dict -> model-ready batch."""
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(obs)
        o["task"] = [task_desc]
        o = self.env_pre(o)
        o = self.pre(o)
        return o

    # ------------------------------------------------------------------ actions
    def sample_chunk(self, o: dict, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized action chunk, shape (1, chunk_size, action_dim)."""
        with torch.no_grad():
            return self.policy.predict_action_chunk(o, noise=noise)

    def unnormalize(self, chunk: torch.Tensor) -> torch.Tensor:
        """Normalized chunk (B,T,A) -> dataset raw action space (B,T,A)."""
        try:
            return self.post(chunk)
        except Exception:
            rows = [self.post(chunk[:, t, :]) for t in range(chunk.shape[1])]
            return torch.stack(rows, dim=1)

    def to_env_action(self, raw_row: torch.Tensor) -> np.ndarray:
        """Dataset raw action (1,A) -> env action (A,) as float64 numpy."""
        a = self.env_post({"action": raw_row})["action"]
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        return np.asarray(a).reshape(-1).astype(np.float64)


def load_smolvla(policy_path: str, device: str = "cuda") -> SmolVLABundle:
    import libero.libero  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    env_cfg = LiberoEnv(task="libero_spatial")  # only used for shapes/metadata
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    cfg.pretrained_path = policy_path

    policy = make_policy(cfg, env_cfg=env_cfg, rename_map=_RENAME_MAP)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": _RENAME_MAP},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return SmolVLABundle(policy=policy, pre=pre, post=post, env_pre=env_pre,
                         env_post=env_post, cfg=cfg, path=policy_path)
