"""FB-CPR learner (PLAN.md 7.2): forward-backward representation, discriminator
regulariser toward expert demos, style critic Q_D, penalty critic Q_R, actor.

Notation: obs = critic obs dict (actor keys + privileged), g = goal-space
vector, z on the sqrt(d) sphere. Targets are Polyak averages (bars in PLAN).
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import spec
from .config import Config
from .nets import BNet, build_networks, sphere
from .zsample import ZSampler


# ============================================================ penalties for Q_R
def penalties(g: torch.Tensor, cfg: Optional[Config] = None) -> torch.Tensor:
    """Non-negative physical-limit cost per state from raw g, shape g.shape[:-1].
    collision + shield intervention + tilt above limit + jerk + thrust excess."""
    cfg = cfg or Config()
    f = lambda n: spec.g_field(g, n).squeeze(-1)
    cost = cfg.pen_collision * f("collision")
    cost = cost + cfg.pen_shield * f("shield")
    cost = cost + cfg.pen_tilt * (f("tilt") - cfg.tilt_limit).clamp_min(0.0)
    cost = cost + cfg.pen_jerk * (f("jerk") / 50.0).clamp(0.0, 10.0)
    cost = cost + cfg.pen_thrust * (f("thrust") - cfg.thrust_limit).clamp_min(0.0)
    return cost


# ============================================================ losses
def fb_measure_loss(Fk: torch.Tensor, Bn: torch.Tensor, target_M: torch.Tensor,
                    done: torch.Tensor, gamma: float):
    """Fk [n, b, d] online forward, Bn [b, d] = B(g'), target_M [b, b] = min_l Fbar_l^T Bbar(g').
    Returns (loss, offdiag, diag) summed over the ensemble."""
    n, b, _ = Fk.shape
    M = torch.einsum("nid,jd->nij", Fk, Bn)                                  # [n, b, b]
    diff = M - gamma * (1.0 - done).view(1, b, 1) * target_M.unsqueeze(0)
    off = 1.0 - torch.eye(b, device=Fk.device)
    offdiag = (diff.pow(2) * off).sum((1, 2)) / off.sum()
    diag = torch.diagonal(M, dim1=1, dim2=2).mean(-1)
    loss = (offdiag - 2.0 * diag).sum()
    return loss, offdiag.mean(), diag.mean()


def ortho_loss(Bm: torch.Tensor) -> torch.Tensor:
    """Frobenius orthonormality ||E[B B^T] - I||^2 over a batch Bm [b, d].
    Equals mean_ij (B_i.B_j)^2 - 2 mean_i |B_i|^2 + d (PLAN eq. 25-26 plus a constant)."""
    b, d = Bm.shape
    cov = Bm.t() @ Bm / b
    return (cov - torch.eye(d, device=Bm.device)).pow(2).sum()


def polyak(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for pt, p in zip(target.parameters(), online.parameters()):
            pt.lerp_(p, tau)
        for bt, b in zip(target.buffers(), online.buffers()):
            bt.copy_(b)


def _frozen_copy(m: nn.Module) -> nn.Module:
    t = copy.deepcopy(m)
    for p in t.parameters():
        p.requires_grad_(False)
    return t


# ============================================================ learner
class FBCPR(nn.Module):
    method = "fb"

    def __init__(self, cfg: Config, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.d = self._z_dim()
        nets = build_networks(cfg, self.d)
        self.F, self.actor, self.QD, self.QR, self.D = nets["F"], nets["actor"], nets["QD"], nets["QR"], nets["D"]
        self.B = self._build_B()
        self.F_t, self.B_t = _frozen_copy(self.F), _frozen_copy(self.B)
        self.QD_t, self.QR_t = _frozen_copy(self.QD), _frozen_copy(self.QR)
        self.to(device)
        self.device = torch.device(device)
        groups = [{"params": self.F.parameters(), "lr": cfg.lr_f}]
        bp = [p for p in self.B.parameters() if p.requires_grad]
        if bp:
            groups.append({"params": bp, "lr": cfg.lr_b})
        self.opt_fb = torch.optim.Adam(groups)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_qd = torch.optim.Adam(self.QD.parameters(), lr=cfg.lr_critic)
        self.opt_qr = torch.optim.Adam(self.QR.parameters(), lr=cfg.lr_critic)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.lr_disc)
        self.zs = ZSampler(self.d, cfg.z_demo_frac, cfg.z_buffer_frac, cfg.z_uniform_frac, device)
        self.num_updates = 0

    # -------------------------------------------------- method hooks
    def _z_dim(self) -> int:
        return self.cfg.d

    def _build_B(self) -> nn.Module:
        return BNet(self.cfg.d, self.cfg.b_width, self.cfg.b_blocks)

    def encode_g(self, g: torch.Tensor) -> torch.Tensor:
        return self.B(g)

    def _repr_loss(self, batch, z, a_next) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        obs, nobs, ng, done = batch["obs"], batch["next_obs"], batch["next_g"], batch["done"]
        with torch.no_grad():
            tF = self.F_t(nobs, z, a_next)                                    # [n, b, d]
            tB = self.B_t(ng)
            tM = torch.einsum("nid,jd->nij", tF, tB).min(0).values            # [b, b]
        Fk = self.F(obs, z, batch["action"])
        Bn = self.B(ng)
        l_fb, off, diag = fb_measure_loss(Fk, Bn, tM, done, cfg.gamma)
        l_or = ortho_loss(self.B(batch["g"]))
        loss = l_fb + cfg.ortho_coef * l_or
        return loss, {"fb_loss": l_fb.item(), "fb_offdiag": off.item(), "fb_diag": diag.item(),
                      "ortho_loss": l_or.item()}

    # -------------------------------------------------- helpers
    def _step(self, opt, loss, params=None):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            ps = [p for grp in opt.param_groups for p in grp["params"]]
            torch.nn.utils.clip_grad_norm_(ps, self.cfg.grad_clip)
        opt.step()

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], z: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        a = self.actor(obs, z)
        if noise > 0:
            a = (a + noise * torch.randn_like(a)).clamp(-1.0, 1.0)
        return a

    @torch.no_grad()
    def infer_z(self, g: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """z_r = sqrt(d) normalize(mean_i r_i B(g_i)) (PLAN 6.3)."""
        e = self.encode_g(g)
        return sphere((reward.reshape(-1, 1).float() * e).mean(0))

    # -------------------------------------------------- update
    def update(self, batch: Dict[str, object], demo=None) -> Dict[str, float]:
        """One FB-CPR gradient step on a replay batch; demo is a DemoStore or None."""
        cfg = self.cfg
        obs, nobs = batch["obs"], batch["next_obs"]
        g, ng, done = batch["g"], batch["next_g"], batch["done"]
        b = g.shape[0]
        have_demo = demo is not None and len(demo) > 0
        m: Dict[str, float] = {}
        # z for this batch from the mixture (buffer states = this batch's next g)
        z = self.zs.sample(b, self.encode_g, demo=demo, g_pool=ng)
        with torch.no_grad():
            noise = (cfg.target_noise * torch.randn(b, spec.ACTION_DIM, device=g.device)
                     ).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
            a_next = (self.actor(nobs, z) + noise).clamp(-1.0, 1.0)

        # ---- representation (FB or SF)
        loss, mm = self._repr_loss(batch, z, a_next)
        self._step(self.opt_fb, loss)
        m.update(mm)

        # ---- discriminator
        if have_demo:
            L = demo.seg_len
            k = max(1, b // L)
            segs = demo.sample_segments(k, g.device)
            with torch.no_grad():
                z_tau = self.zs.encode_segments(self.encode_g, segs)              # [k, d]
            pos_g = segs.reshape(k * L, -1)
            pos_z = z_tau.unsqueeze(1).expand(k, L, self.d).reshape(k * L, self.d)
            neg_g, neg_z = g, batch["z"]
            _, lp = self.D(pos_g, pos_z)
            _, ln_ = self.D(neg_g, neg_z)
            bce = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                F.binary_cross_entropy_with_logits(ln_, torch.zeros_like(ln_))
            nn_ = min(pos_g.shape[0], b)
            xp = self.D.features(pos_g[:nn_], pos_z[:nn_])
            xn = self.D.features(neg_g[:nn_], neg_z[:nn_])
            alpha = torch.rand(nn_, 1, device=g.device)
            xi = (alpha * xp + (1 - alpha) * xn).detach().requires_grad_(True)
            li = self.D.logits_from_features(xi)
            grad = torch.autograd.grad(li.sum(), xi, create_graph=True)[0]
            gp = grad.pow(2).sum(-1).mean()
            self._step(self.opt_d, bce + cfg.gp_coef * gp)
            m.update({"d_loss": bce.item(), "d_gp": gp.item(),
                      "d_acc_pos": (lp > 0).float().mean().item(), "d_acc_neg": (ln_ < 0).float().mean().item()})
            with torch.no_grad():
                _, lr_ = self.D(ng, z)
                r_d = lr_.clamp(-cfg.disc_reward_clip, cfg.disc_reward_clip)
        else:
            m.update({"d_loss": 0.0, "d_gp": 0.0, "d_acc_pos": 0.0, "d_acc_neg": 0.0})
            r_d = torch.zeros(b, device=g.device)

        # ---- style critic Q_D and penalty critic Q_R (TD)
        r_r = -penalties(ng, cfg)
        with torch.no_grad():
            nb = cfg.gamma * (1.0 - done)
            y_d = r_d + nb * self.QD_t(nobs, z, a_next)
            y_r = r_r + nb * self.QR_t(nobs, z, a_next)
        l_qd = F.mse_loss(self.QD(obs, z, batch["action"]), y_d)
        self._step(self.opt_qd, l_qd)
        l_qr = F.mse_loss(self.QR(obs, z, batch["action"]), y_r)
        self._step(self.opt_qr, l_qr)

        # ---- actor
        frozen = [p for net in (self.F, self.QD, self.QR) for p in net.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        a = self.actor(obs, z)
        Fa = self.F(obs, z, a)                                                    # [n, b, d]
        q_fb = (Fa * z.unsqueeze(0)).sum(-1).min(0).values
        q_d, q_r = self.QD(obs, z, a), self.QR(obs, z, a)
        w = q_fb.abs().mean().detach() if cfg.scale_reg else 1.0
        l_pi = -(q_fb + w * (cfg.lambda_d * q_d + cfg.lambda_r * q_r)).mean()
        self._step(self.opt_actor, l_pi)
        for p in frozen:
            p.requires_grad_(True)

        # ---- targets
        polyak(self.F_t, self.F, cfg.tau)
        polyak(self.B_t, self.B, cfg.tau)
        polyak(self.QD_t, self.QD, cfg.tau)
        polyak(self.QR_t, self.QR, cfg.tau)
        self.num_updates += 1
        m.update({"qd_loss": l_qd.item(), "qr_loss": l_qr.item(), "actor_loss": l_pi.item(),
                  "q_fb": q_fb.mean().item(), "q_d": q_d.mean().item(), "q_r": q_r.mean().item(),
                  "r_d": r_d.mean().item(), "r_r": r_r.mean().item()})
        return m
