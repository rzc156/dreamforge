"""Latent world model + progress head (vendored from the author's research stack).

The world model is a small CPU/GPU-friendly latent dynamics model:
    encoder: 96x96 RGB -> z (128-d)
    transition: (z, a) -> z'
    progress head: z -> progress in [0, 1]

It was trained on LIBERO expert frames (lerobot/libero) with the *raw dataset
action space* (7-dim).  Actions fed to ``predict_next`` MUST be in that space
(i.e. the output of the policy post-processor), not the policy's normalized
output.

Attribution: this file adapts code from the author's previous research project
(``near-zero-headroom``, manuscript under review).  Reused here with the
author's permission; the world-model design is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WMConfig", "LatentWorldModel", "ProgressHead", "load_world_model", "encode_image"]


@dataclass
class WMConfig:
    latent_dim: int = 128
    action_dim: int = 7
    image_size: int = 96
    hidden_dim: int = 512
    use_gru: bool = False
    decode: bool = False


class _ConvEncoder(nn.Module):
    """Image encoder.  No global average pooling (it destroys spatial layout and
    empirically kills action responsiveness).  ``legacy=True`` loads old
    checkpoints that used GAP."""

    def __init__(self, latent_dim: int, size: int = 96, legacy: bool = False):
        super().__init__()
        self.legacy = legacy
        convs = [
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        ]
        if legacy:
            self.net = nn.Sequential(*convs, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(256, latent_dim))
        else:
            self.net = nn.Sequential(*convs)
            f = size // 16
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * f * f, latent_dim * 2), nn.SiLU(),
                nn.Linear(latent_dim * 2, latent_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.legacy:
            return self.net(x)
        return self.head(self.net(x))


class LatentWorldModel(nn.Module):
    def __init__(self, cfg: WMConfig, legacy_encoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = _ConvEncoder(cfg.latent_dim, size=cfg.image_size, legacy=legacy_encoder)
        self.action_enc = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
        )
        if cfg.use_gru:
            self.rnn = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
            self.transition = nn.Sequential(
                nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            )
        else:
            self.transition = nn.Sequential(
                nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim), nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_next(self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor | None = None):
        if self.cfg.use_gru:
            ae = self.action_enc(a)
            h = self.rnn(ae, h) if h is not None else self.rnn(ae, torch.zeros_like(ae))
            return self.transition(torch.cat([z, h], dim=-1)), h
        return self.transition(torch.cat([z, a], dim=-1)), None


class ProgressHead(nn.Module):
    """z -> p in [0,1]; 0 = episode start, 1 = completed."""

    def __init__(self, latent_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def load_world_model(path: str, device: str = "cuda"):
    """Load a world-model checkpoint; returns ``(model, ckpt)``.

    The image size is recovered from the encoder weights (historical
    checkpoints stored a wrong ``cfg.image_size``).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = WMConfig(**ckpt["cfg"])
    sd = ckpt["model"]
    legacy = any("encoder.net.10.weight" in k for k in sd)
    w = sd.get("encoder.head.1.weight")
    if w is not None:
        f = int(round((w.shape[1] / 256) ** 0.5))
        cfg.image_size = f * 16
    model = LatentWorldModel(cfg, legacy_encoder=legacy)
    model.load_state_dict(sd)
    return model.to(device).eval(), ckpt


def make_progress_head(ckpt: dict, device: str = "cuda") -> ProgressHead:
    head = ProgressHead(latent_dim=ckpt["cfg"]["latent_dim"]).to(device).eval()
    head.load_state_dict(ckpt["head"])
    return head


@torch.no_grad()
def encode_image(wm: LatentWorldModel, image: torch.Tensor, size: int | None = None) -> torch.Tensor:
    """Encode a (B,3,H,W) float image in [0,1] into the latent."""
    size = size or wm.cfg.image_size
    x = image.float()
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x[:, :3].clamp(0, 1)
    return wm.encode(x)
