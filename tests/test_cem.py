"""CEM tests against a synthetic linear world model (no checkpoints needed)."""
import torch

from dreamforge.cem import ActionCEM, CEMConfig
from dreamforge.wm import LatentWorldModel, ProgressHead, WMConfig


class _FakeWM(LatentWorldModel):
    """z' = z + A a, progress = sigmoid(sum(z)); action dim 2 for the test."""

    def __init__(self):
        cfg = WMConfig(latent_dim=4, action_dim=2, image_size=32)
        super().__init__(cfg)
        self.A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]])

    def predict_next(self, z, a, h=None):
        return z + a @ self.A.T, None


def test_cem_improves_over_init():
    torch.manual_seed(0)
    wm = _FakeWM()
    head = ProgressHead(latent_dim=4)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-2].bias.fill_(1.0)  # monotone in z sum
    cem = ActionCEM(wm, head, CEMConfig(horizon=8, population=64, elites=8, iters=6,
                                        init_std=0.5, seed=0), device="cpu")
    z0 = torch.zeros(1, 4)
    init = torch.zeros(8, 2)
    res = cem.optimize(z0, init_mean=init)
    assert res.best_score >= res.init_score - 1e-6
    assert len(res.progress_curve) == 8
    assert len(res.history) == 6


def test_cem_respects_bounds():
    torch.manual_seed(0)
    wm = _FakeWM()
    head = ProgressHead(latent_dim=4)
    cem = ActionCEM(wm, head, CEMConfig(horizon=4, population=32, elites=4, iters=2,
                                        low=(-1.0, -1.0), high=(1.0, 1.0)), device="cpu")
    res = cem.optimize(torch.zeros(1, 4), init_mean=torch.zeros(4, 2))
    assert res.actions.min() >= -1.0 - 1e-6 and res.actions.max() <= 1.0 + 1e-6
