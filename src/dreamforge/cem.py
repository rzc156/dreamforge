"""Cross-Entropy Method (CEM) over imagined futures.

Given a start latent z0 (encoded from the current observation) and a latent
world model, search for an action sequence that maximizes the world model's
own predicted task progress.  The result is an *imagined correction*: a plan
that looks good to the model, whose real value we then measure by executing
it in the simulator (see ``verify.py``).

Space contract: actions are in the dataset raw action space, exactly the
space the world model was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .wm import LatentWorldModel, ProgressHead

__all__ = ["CEMConfig", "CEMResult", "ActionCEM"]


@dataclass
class CEMConfig:
    horizon: int = 50          # actions per plan (SmolVLA chunk_size = 50)
    population: int = 64
    elites: int = 8
    iters: int = 6
    init_std: float = 0.25
    min_std: float = 0.02
    momentum: float = 0.7
    score_tail: int = 3        # average progress over the last k imagined steps
    seed: int = 0
    # trust region: if set, each action is bounded to base +/- delta (then clipped
    # to the dataset range).  Keeps plans plausible; without it CEM drives rotation
    # dims to the extremes of the dataset range.
    delta: float | None = None
    # per-dimension trust region (preferred): base +/- delta_per_dim, e.g. 0.5 * std
    delta_per_dim: tuple[float, ...] | None = None
    # optional bounds on actions (dataset min/max); None disables clipping
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None


@dataclass
class CEMResult:
    actions: torch.Tensor                 # (H, A) best plan in raw action space
    best_score: float
    init_score: float                     # score of the initial mean (e.g. base chunk)
    progress_curve: list[float] = field(default_factory=list)  # best plan's imagined progress
    history: list[float] = field(default_factory=list)         # best score per iteration


class ActionCEM:
    def __init__(self, wm: LatentWorldModel, head: ProgressHead, cfg: CEMConfig | None = None,
                 device: str = "cuda"):
        self.wm = wm
        self.head = head
        self.cfg = cfg or CEMConfig()
        self.device = device
        self._low = None if self.cfg.low is None else torch.tensor(self.cfg.low, device=device)
        self._high = None if self.cfg.high is None else torch.tensor(self.cfg.high, device=device)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _rollout_scores(self, z0: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """actions: (P, H, A) -> scores (P,), progress (P, H)."""
        P, H, _ = actions.shape
        z = z0.expand(P, -1).contiguous()
        progress = torch.zeros(P, H, device=actions.device)
        for t in range(H):
            z, _ = self.wm.predict_next(z, actions[:, t, :])
            progress[:, t] = self.head(z)
        tail = min(self.cfg.score_tail, H)
        scores = progress[:, -tail:].mean(dim=1)
        return scores, progress

    # ------------------------------------------------------------------ optimize
    @torch.no_grad()
    def optimize(self, z0: torch.Tensor, init_mean: torch.Tensor | None = None) -> CEMResult:
        """Search a plan from latent ``z0`` (shape (1, D)).

        ``init_mean`` (H, A) seeds the search (typically the base policy chunk
        in raw action space); None starts from zeros.
        """
        cfg = self.cfg
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        H, A = cfg.horizon, self.wm.cfg.action_dim
        if init_mean is None:
            mu = torch.zeros(H, A, device=self.device)
        else:
            mu = init_mean.to(self.device).clone()
        sigma = torch.full((H, A), cfg.init_std, device=self.device)

        base_score = float(self._rollout_scores(z0, mu.unsqueeze(0))[0][0]) if init_mean is not None else float("nan")
        best_actions, best_score, best_progress = mu.clone(), -1.0, []
        history: list[float] = []

        # action bounds: global dataset bounds, optionally intersected with a
        # trust region around the seed plan
        lo_t = self._low.view(1, A).expand(H, A).clone() if self._low is not None else None
        hi_t = self._high.view(1, A).expand(H, A).clone() if self._high is not None else None
        if cfg.delta_per_dim is not None and init_mean is not None:
            d = torch.tensor(cfg.delta_per_dim, device=self.device, dtype=mu.dtype).view(1, A)
            lo_t = (mu - d) if lo_t is None else torch.maximum(mu - d, lo_t)
            hi_t = (mu + d) if hi_t is None else torch.minimum(mu + d, hi_t)
        elif cfg.delta is not None and init_mean is not None:
            d = float(cfg.delta)
            lo_t = (mu - d) if lo_t is None else torch.maximum(mu - d, lo_t)
            hi_t = (mu + d) if hi_t is None else torch.minimum(mu + d, hi_t)

        for _ in range(cfg.iters):
            eps = torch.randn(cfg.population, H, A, generator=gen, device=self.device)
            cand = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            if lo_t is not None:
                cand = torch.maximum(torch.minimum(cand, hi_t), lo_t)
            scores, progress = self._rollout_scores(z0, cand)
            k = min(cfg.elites, cfg.population)
            top = torch.topk(scores, k).indices
            elite = cand[top]
            mu = cfg.momentum * mu + (1 - cfg.momentum) * elite.mean(dim=0)
            sigma = cfg.momentum * sigma + (1 - cfg.momentum) * elite.std(dim=0)
            sigma = sigma.clamp_min(cfg.min_std)
            top1 = int(top[0])
            history.append(float(scores[top1]))
            if float(scores[top1]) > best_score:
                best_score = float(scores[top1])
                best_actions = cand[top1].clone()
                best_progress = [float(x) for x in progress[top1]]

        return CEMResult(actions=best_actions.cpu(), best_score=best_score,
                         init_score=base_score, progress_curve=best_progress, history=history)
