"""Networks for SkyHand (PLAN.md section 8).

All networks except B and D share one design: z-conditioned query tokens that
cross-attend to state tokens. Inputs are obs dicts with a flat leading batch
dimension [M, ...] (agents already flattened into M), using the keys and
per-row shapes of spec.OBS_SPEC.

Torch >= 2.0 and the standard library only; no genesis.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import spec


def LN(d: int) -> nn.LayerNorm:
    return nn.LayerNorm(d, elementwise_affine=False)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def sphere(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project z onto the sphere of radius sqrt(d) along its last dim."""
    d = z.shape[-1]
    return math.sqrt(d) * z / z.norm(dim=-1, keepdim=True).clamp_min(eps)


# ============================================================ blocks (PLAN 8.2)
class Gate(nn.Module):
    """Per-feature gate in (0, 2), exactly 1 at init: y * 2 sigmoid(W x + b)."""

    def __init__(self, w: int):
        super().__init__()
        self.lin = nn.Linear(w, w)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.lin(x))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y * self.gate(x)


class SwiGLU(nn.Module):
    def __init__(self, w: int, mult: int = 4):
        super().__init__()
        self.w1, self.w3 = nn.Linear(w, mult * w), nn.Linear(w, mult * w)
        self.w2 = nn.Linear(mult * w, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm cross-attention block with K-norm only, gated residuals, SwiGLU MLP."""

    def __init__(self, w: int, heads: int):
        super().__init__()
        assert w % heads == 0, (w, heads)
        self.h, self.dh = heads, w // heads
        self.nq, self.nc, self.nm, self.nk = LN(w), LN(w), LN(w), LN(w // heads)
        self.q, self.k, self.v, self.o = (nn.Linear(w, w) for _ in range(4))
        self.mlp = SwiGLU(w)
        self.ga, self.gm = Gate(w), Gate(w)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q: [B, Nq, w], ctx: [B, Nc, w], mask: [B, Nc] bool, True = attend."""
        x, c = self.nq(q), self.nc(ctx)
        Q = self.q(x).unflatten(-1, (self.h, self.dh)).transpose(1, 2)
        K = self.nk(self.k(c).unflatten(-1, (self.h, self.dh))).transpose(1, 2)   # K-norm only
        V = self.v(c).unflatten(-1, (self.h, self.dh)).transpose(1, 2)
        am = None
        if mask is not None:
            mask = mask | ~mask.any(-1, keepdim=True)          # guard all-masked rows
            am = mask[:, None, None, :]                         # [B, 1, 1, Nc]
        a = F.scaled_dot_product_attention(Q, K, V, attn_mask=am).transpose(1, 2).flatten(-2)
        q = q + self.ga(x, self.o(a))
        x = self.nm(q)
        return q + self.gm(x, self.mlp(x))


class MLPBlock(nn.Module):
    """Pre-norm gated SwiGLU residual block (used by B and D)."""

    def __init__(self, w: int):
        super().__init__()
        self.n = LN(w)
        self.mlp = SwiGLU(w)
        self.g = Gate(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n(x)
        return x + self.g(h, self.mlp(h))


# ============================================================ tokens (PLAN 8.1)
LIDAR_PR, LIDAR_PC = 4, 8                    # 4 x 8 grid of patches
DOWN_PR, DOWN_PC = 2, 2                      # 4 downward patches


def decode_lidar(img: torch.Tensor) -> torch.Tensor:
    """uint8 LiDAR image -> float in [0, 1] (range / 255 normalisation)."""
    return img.float() * (1.0 / 255.0)


def _patchify(img: torch.Tensor, pr: int, pc: int) -> torch.Tensor:
    """[M, C, H, W] -> [M, pr*pc, C*(H/pr)*(W/pc)]."""
    m, c, h, w = img.shape
    ph, pw = h // pr, w // pc
    x = img.reshape(m, c, pr, ph, pc, pw).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(m, pr * pc, c * ph * pw)


class TokenEmbed(nn.Module):
    """Turns an obs dict into context tokens [M, Nc, w] and a mask [M, Nc]."""

    def __init__(self, w: int, priv: bool, action: bool):
        super().__init__()
        self.use_priv, self.use_action = priv, action
        self.proprio = nn.Linear(spec.PROPRIO_DIM, w)
        self.anchor = nn.Linear(spec.ANCHOR_DIM, w)
        self.anchor_slot = nn.Parameter(torch.randn(spec.N_ANCHORS, w) * 0.02)
        self.mate = nn.Linear(spec.MATE_DIM, w)
        self.mate_slot = nn.Parameter(torch.randn(spec.MAX_MATES, w) * 0.02)
        lid_dim = spec.LIDAR_CH * (spec.LIDAR_V // LIDAR_PR) * (spec.LIDAR_H // LIDAR_PC)
        self.lidar = nn.Linear(lid_dim, w)
        self.lidar_row = nn.Parameter(torch.randn(LIDAR_PR, 1, w) * 0.02)   # elevation
        self.lidar_col = nn.Parameter(torch.randn(1, LIDAR_PC, w) * 0.02)   # azimuth (learned, wraps by construction)
        down_dim = spec.LIDAR_CH * (spec.DOWN_V // DOWN_PR) * (spec.DOWN_H // DOWN_PC)
        self.down = nn.Linear(down_dim, w)
        self.down_pos = nn.Parameter(torch.randn(DOWN_PR * DOWN_PC, w) * 0.02)
        if priv:
            self.priv = nn.Linear(spec.PRIV_DIM, w)
        if action:
            self.action = nn.Linear(spec.ACTION_DIM, w)

    def forward(self, obs: Dict[str, torch.Tensor],
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = obs["proprio"]
        m, dev = p.shape[0], p.device
        toks, masks = [], []
        ones = lambda k: torch.ones(m, k, dtype=torch.bool, device=dev)
        toks.append(self.proprio(symlog(p)).unsqueeze(1)); masks.append(ones(1))
        anc = obs["anchors"]
        toks.append(self.anchor(symlog(anc)) + self.anchor_slot)
        masks.append(anc[..., 3] < 0.5)                         # type one-hot slot 0 = "none"
        toks.append(self.mate(symlog(obs["mates"])) + self.mate_slot)
        masks.append(obs["mates_mask"].bool())
        lid = _patchify(decode_lidar(obs["lidar"]), LIDAR_PR, LIDAR_PC)
        pos = (self.lidar_row + self.lidar_col).reshape(LIDAR_PR * LIDAR_PC, -1)
        toks.append(self.lidar(lid) + pos); masks.append(ones(LIDAR_PR * LIDAR_PC))
        dn = _patchify(decode_lidar(obs["lidar_down"]), DOWN_PR, DOWN_PC)
        toks.append(self.down(dn) + self.down_pos); masks.append(ones(DOWN_PR * DOWN_PC))
        if self.use_priv:
            toks.append(self.priv(symlog(obs["priv"]))); masks.append(obs["priv_mask"].bool())
        if self.use_action:
            assert action is not None, "this network needs an action token"
            toks.append(self.action(action).unsqueeze(1)); masks.append(ones(1))
        return torch.cat(toks, 1), torch.cat(masks, 1)


# ============================================================ query networks (PLAN 8.3)
class QueryNet(nn.Module):
    """z-conditioned queries cross-attending to state tokens; mean -> LN -> linear head."""

    def __init__(self, d: int, out_dim: int, width: int, heads: int, blocks: int,
                 n_queries: int = 8, priv: bool = True, action: bool = True, tanh: bool = False):
        super().__init__()
        self.d, self.nq, self.w, self.tanh = d, n_queries, width, tanh
        self.embed = TokenEmbed(width, priv=priv, action=action)
        self.base = nn.Parameter(torch.randn(n_queries, width) * 0.02)
        self.zproj = nn.Linear(d, n_queries * width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(blocks)])
        self.norm = LN(width)
        self.head = nn.Linear(width, out_dim)

    def forward(self, obs: Dict[str, torch.Tensor], z: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx, mask = self.embed(obs, action)
        q = self.base + self.zproj(sphere(z)).unflatten(-1, (self.nq, self.w))
        for b in self.blocks:
            q = b(q, ctx, mask)
        y = self.head(self.norm(q.mean(1)))
        return torch.tanh(y) if self.tanh else y


class Actor(QueryNet):
    def __init__(self, d: int, width: int = 256, heads: int = 4, blocks: int = 3,
                 n_queries: int = 8, priv: bool = False):
        super().__init__(d, spec.ACTION_DIM, width, heads, blocks, n_queries,
                         priv=priv, action=False, tanh=True)

    def forward(self, obs, z, action=None):            # noqa: D401
        return super().forward(obs, z, None)


class Critic(QueryNet):
    """Privileged critic with an action token. out_dim = d for F, 1 for Q_D / Q_R."""

    def __init__(self, d: int, out_dim: int, width: int = 256, heads: int = 4,
                 blocks: int = 4, n_queries: int = 8):
        super().__init__(d, out_dim, width, heads, blocks, n_queries, priv=True, action=True)
        self.scalar = out_dim == 1

    def forward(self, obs, z, action=None):
        y = super().forward(obs, z, action)
        return y.squeeze(-1) if self.scalar else y


class FEnsemble(nn.Module):
    """Twin forward networks; returns [n, M, d]."""

    def __init__(self, d: int, width: int = 256, heads: int = 4, blocks: int = 4,
                 n_queries: int = 8, n: int = 2):
        super().__init__()
        self.nets = nn.ModuleList([Critic(d, d, width, heads, blocks, n_queries) for _ in range(n)])

    def forward(self, obs, z, action):
        return torch.stack([f(obs, z, action) for f in self.nets], 0)


class BNet(nn.Module):
    """Backward embedding B(g): MLP blocks over g / g_scale, L2-normalised to sqrt(d)."""

    def __init__(self, d: int, width: int = 256, blocks: int = 3):
        super().__init__()
        self.d = d
        self.register_buffer("scale", spec.g_scale())
        self.inp = nn.Linear(spec.G_DIM, width)
        self.blocks = nn.ModuleList([MLPBlock(width) for _ in range(blocks)])
        self.norm = LN(width)
        self.out = nn.Linear(width, d)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        x = self.inp(g / self.scale)
        for b in self.blocks:
            x = b(x)
        return sphere(self.out(self.norm(x)))


class Discriminator(nn.Module):
    """D(g, z): probability that (g, z) came from a demo. forward returns (prob, logits)."""

    def __init__(self, d: int, width: int = 256, blocks: int = 3):
        super().__init__()
        self.d = d
        self.register_buffer("scale", spec.g_scale())
        self.inp = nn.Linear(spec.G_DIM + d, width)
        self.blocks = nn.ModuleList([MLPBlock(width) for _ in range(blocks)])
        self.norm = LN(width)
        self.out = nn.Linear(width, 1)

    def features(self, g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenated, normalised input (used for the gradient penalty)."""
        return torch.cat([g / self.scale, sphere(z) / math.sqrt(self.d)], -1)

    def logits_from_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x)).squeeze(-1)

    def forward(self, g: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits_from_features(self.features(g, z))
        return torch.sigmoid(logits), logits


def build_networks(cfg, d: int) -> Dict[str, nn.Module]:
    """All learner networks from a Config. B is omitted here (method specific)."""
    return {
        "F": FEnsemble(d, cfg.f_width, cfg.f_heads, cfg.f_blocks, cfg.n_queries),
        "actor": Actor(d, cfg.actor_width, cfg.actor_heads, cfg.actor_blocks, cfg.n_queries,
                       priv=cfg.actor_priv),
        "QD": Critic(d, 1, cfg.critic_width, cfg.critic_heads, cfg.critic_blocks, cfg.n_queries),
        "QR": Critic(d, 1, cfg.critic_width, cfg.critic_heads, cfg.critic_blocks, cfg.n_queries),
        "D": Discriminator(d, cfg.disc_width, cfg.disc_blocks),
    }
