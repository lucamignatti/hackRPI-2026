"""Successor-feature fallback (PLAN.md 7.6).

B is replaced by a frozen hand-built feature map phi(g); F becomes psi trained
with the projected SF loss
    L_SF = mean( psi(s,a,z)^T z - phi(g')^T z - gamma * psibar(s',a',z)^T z )^2
(twin psi, min over targets). z for a reward is the least-squares regression of
the reward on phi(g). Same learner interface as FBCPR.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .. import spec
from .config import Config
from .fb_cpr import FBCPR

K = spec.N_ANCHORS
PHI_DIM = spec.G_DIM + (2 * K + 1) * 2 + K + 1     # 59 + 18 + 4 + 1 = 82


class Phi(nn.Module):
    """Frozen features of g: normalised g, anchor / payload distances, exp(-distance)
    shaping, holding x closeness of payload to each anchor, and a bias."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", spec.g_scale())
        self.d = PHI_DIM

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        gn = (g / self.scale).clamp(-5.0, 5.0)
        ad = spec.g_field(g, "anchor_rel_drone").unflatten(-1, (K, 3)).norm(dim=-1)       # [..., K]
        ap = spec.g_field(g, "anchor_rel_payload").unflatten(-1, (K, 3)).norm(dim=-1)     # [..., K]
        pd = spec.g_field(g, "payload_rel").norm(dim=-1, keepdim=True)                    # [..., 1]
        dist = torch.cat([ad, ap, pd], -1) / 15.0
        close = torch.exp(-torch.cat([ad, ap, pd], -1))
        hold = spec.g_field(g, "holding") * torch.exp(-ap)
        one = torch.ones_like(pd)
        return torch.cat([gn, dist, close, hold, one], -1)


def regress_z(phi: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
    """Least-squares w with phi @ w ~= reward (min-norm for rank-deficient phi)."""
    X = phi.detach().double().cpu()
    r = reward.detach().double().cpu().reshape(-1, 1)
    w = torch.linalg.lstsq(X, r, driver="gelsd").solution.squeeze(-1)
    return w.float().to(phi.device)


class SFLearner(FBCPR):
    method = "sf"

    def _z_dim(self) -> int:
        return PHI_DIM

    def _build_B(self) -> nn.Module:
        return Phi()

    def _repr_loss(self, batch, z, a_next) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        obs, nobs, ng, done = batch["obs"], batch["next_obs"], batch["next_g"], batch["done"]
        with torch.no_grad():
            tq = (self.F_t(nobs, z, a_next) * z.unsqueeze(0)).sum(-1).min(0).values      # [b]
            y = (self.B(ng) * z).sum(-1) + cfg.gamma * (1.0 - done) * tq
        q = (self.F(obs, z, batch["action"]) * z.unsqueeze(0)).sum(-1)                   # [n, b]
        loss = (q - y.unsqueeze(0)).pow(2).mean(-1).sum()
        return loss, {"fb_loss": loss.item(), "fb_offdiag": 0.0, "fb_diag": 0.0, "ortho_loss": 0.0}

    @torch.no_grad()
    def infer_z(self, g: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return regress_z(self.B(g), reward)
