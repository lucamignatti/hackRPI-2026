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
from . import dist as D
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


def fb_q_loss(Fz: torch.Tensor, Bz: torch.Tensor, target_Fz: torch.Tensor,
              done: torch.Tensor, gamma: float) -> torch.Tensor:
    """Auxiliary diagonal loss (Touati, Rapin, Ollivier 2023, eq. 9 / 42), summed over the ensemble:
    mean_i ( F_k(s_i,a_i,z_i)^T z_i - B(s'_i)^T Cov^+ z_i - gamma min_l Fbar_l(s'_i,a'_i,z_i)^T z_i )^2.
    Fz [n, b]; Bz [b] = B(s')^T Cov^+ z (no grad); target_Fz [b]; done [b]."""
    y = Bz + gamma * (1.0 - done) * target_Fz
    return (Fz - y.unsqueeze(0)).pow(2).mean(-1).sum()


def _pinv(m: torch.Tensor) -> torch.Tensor:
    """Pseudo-inverse; computed on CPU where the device lacks a kernel (e.g. MPS)."""
    if m.device.type == "mps":
        return torch.linalg.pinv(m.cpu()).to(m.device)
    return torch.linalg.pinv(m)


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
        # auxiliary diagonal loss: trains F^T z directly on the z it was sampled with
        coef = cfg.q_loss_coef if cfg.q_loss_coef >= 0 else 1.0 / self.d
        l_q = torch.zeros((), device=Fk.device)
        if coef > 0:
            with torch.no_grad():
                Bd = Bn.detach()
                cov = Bd.t() @ Bd / Bd.shape[0]
                Bz = ((Bd @ _pinv(cov)) * z).sum(-1)                              # [b]
                tFz = (tF * z.unsqueeze(0)).sum(-1).min(0).values                 # [b]
            l_q = fb_q_loss((Fk * z.unsqueeze(0)).sum(-1), Bz, tFz, done, cfg.gamma)
            loss = loss + coef * l_q
        return loss, {"fb_loss": l_fb.item(), "fb_offdiag": off.item(), "fb_diag": diag.item(),
                      "ortho_loss": l_or.item(), "fb_q_loss": l_q.item()}

    # -------------------------------------------------- helpers
    def _step(self, opt, loss, params=None):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ps = [p for grp in opt.param_groups for p in grp["params"]]
        D.allreduce_grads(ps)                          # data-parallel: every rank steps on the mean gradient
        if self.cfg.grad_clip > 0:
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
    # One update = _prepare (eager: everything random or data-dependent, plus the pseudo-inverse) and
    # _core (pure tensor math with static shapes and no host syncs). With cfg.cuda_graph the core is
    # captured once into a CUDA graph and replayed, which removes the per-kernel launch overhead.
    def _prepare(self, batch: Dict[str, object], demo) -> Tuple[Dict[str, object], bool, bool]:
        cfg = self.cfg
        g, ng = batch["g"], batch["next_g"]
        b = g.shape[0]
        have_demo = demo is not None and len(demo) > 0
        if D.active():   # the discriminator steps on every rank or on none (its all-reduce must match up)
            t = torch.tensor([1 if have_demo else 0], device=g.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            have_demo = bool(t.item())
        # discriminator negatives are policy states only: expert-flown transitions are excluded
        expert = batch.get("expert")
        neg = torch.ones(b, dtype=torch.bool, device=g.device) if expert is None else ~expert.to(g.device).bool()
        step_d = have_demo and bool(neg.any())
        if D.active():   # every rank steps D or none does: a rank skipping it would hang the all-reduce
            t = torch.tensor([1 if step_d else 0], device=g.device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            step_d = bool(t.item())
        inp: Dict[str, object] = {k: batch[k] for k in ("obs", "next_obs", "action", "g", "next_g", "done")}
        inp["z_beh"] = batch["z"]
        inp["neg"] = neg.float()
        with torch.no_grad():
            # z for this batch from the mixture (buffer states = this batch's next g)
            z = self.zs.sample(b, self.encode_g, demo=demo, g_pool=ng)
            inp["z"] = z
            inp["noise"] = (cfg.target_noise * torch.randn(b, spec.ACTION_DIM, device=g.device)
                            ).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)
            if self.method == "fb":
                coef = cfg.q_loss_coef if cfg.q_loss_coef >= 0 else 1.0 / self.d
                if coef > 0:   # Bz = B(s')^T Cov^+ z for the auxiliary diagonal loss
                    Bd = self.B(ng)
                    cov = Bd.t() @ Bd / Bd.shape[0]
                    inp["Bz"] = ((Bd @ _pinv(cov)) * z).sum(-1)
            if step_d:
                L = demo.seg_len
                k = max(1, b // L)
                segs = demo.sample_segments(k, g.device)
                z_tau = self.zs.encode_segments(self.encode_g, segs)              # [k, d]
                inp["pos_g"] = segs.reshape(k * L, -1)
                inp["pos_z"] = z_tau.unsqueeze(1).expand(k, L, self.d).reshape(k * L, self.d).contiguous()
                inp["alpha"] = torch.rand(min(k * L, b), 1, device=g.device)
        return inp, have_demo, step_d

    def _repr_core(self, inp, a_next) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """FB measure loss + orthonormality + auxiliary diagonal loss, from prepared inputs."""
        cfg = self.cfg
        obs, nobs, ng, done, z = inp["obs"], inp["next_obs"], inp["next_g"], inp["done"], inp["z"]
        with torch.no_grad():
            tF = self.F_t(nobs, z, a_next)                                    # [n, b, d]
            tB = self.B_t(ng)
            tM = torch.einsum("nid,jd->nij", tF, tB).min(0).values            # [b, b]
        Fk = self.F(obs, z, inp["action"])
        Bn = self.B(ng)
        l_fb, off, diag = fb_measure_loss(Fk, Bn, tM, done, cfg.gamma)
        l_or = ortho_loss(self.B(inp["g"]))
        loss = l_fb + cfg.ortho_coef * l_or
        l_q = torch.zeros((), device=Fk.device)
        if "Bz" in inp:
            coef = cfg.q_loss_coef if cfg.q_loss_coef >= 0 else 1.0 / self.d
            with torch.no_grad():
                tFz = (tF * z.unsqueeze(0)).sum(-1).min(0).values                 # [b]
            l_q = fb_q_loss((Fk * z.unsqueeze(0)).sum(-1), inp["Bz"], tFz, done, cfg.gamma)
            loss = loss + coef * l_q
        return loss, {"fb_loss": l_fb.detach(), "fb_offdiag": off.detach(), "fb_diag": diag.detach(),
                      "ortho_loss": l_or.detach(), "fb_q_loss": l_q.detach()}

    def _core(self, inp, have_demo: bool, step_d: bool) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        obs, nobs, g, ng, done, z = (inp["obs"], inp["next_obs"], inp["g"], inp["next_g"], inp["done"], inp["z"])
        b = g.shape[0]
        neg = inp["neg"]
        m: Dict[str, torch.Tensor] = {}
        zero = torch.zeros((), device=g.device)
        with torch.no_grad():
            a_next = (self.actor(nobs, z) + inp["noise"]).clamp(-1.0, 1.0)

        # ---- representation (FB or SF)
        if self.method == "fb":
            loss, mm = self._repr_core(inp, a_next)
        else:
            loss, mf = self._repr_loss({"obs": obs, "next_obs": nobs, "next_g": ng, "done": done,
                                        "g": g, "action": inp["action"]}, z, a_next)
            mm = {k: torch.as_tensor(v, device=g.device, dtype=torch.float32) for k, v in mf.items()}
        self._step(self.opt_fb, loss)
        m.update(mm)

        # ---- discriminator (negatives = policy-flown rows, by mask: static shapes)
        if step_d:
            pos_g, pos_z, alpha = inp["pos_g"], inp["pos_z"], inp["alpha"]
            z_beh = inp["z_beh"]
            _, lp = self.D(pos_g, pos_z)
            _, ln_ = self.D(g, z_beh)
            nsum = neg.sum().clamp_min(1.0)
            bce = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                (F.binary_cross_entropy_with_logits(ln_, torch.zeros_like(ln_), reduction="none") * neg).sum() / nsum
            nn_ = alpha.shape[0]
            xp = self.D.features(pos_g[:nn_], pos_z[:nn_])
            xn = self.D.features(g[:nn_], z_beh[:nn_])
            xi = (alpha * xp + (1 - alpha) * xn).detach().requires_grad_(True)
            li = self.D.logits_from_features(xi)
            grad = torch.autograd.grad(li.sum(), xi, create_graph=True)[0]
            wgp = neg[:nn_]
            gp = (grad.pow(2).sum(-1) * wgp).sum() / wgp.sum().clamp_min(1.0)
            self._step(self.opt_d, bce + cfg.gp_coef * gp)
            m.update({"d_loss": bce.detach(), "d_gp": gp.detach(),
                      "d_acc_pos": (lp > 0).float().mean().detach(),
                      "d_acc_neg": (((ln_ < 0).float() * neg).sum() / nsum).detach()})
            with torch.no_grad():
                _, lr_ = self.D(ng, z)
                r_d = lr_.clamp(-cfg.disc_reward_clip, cfg.disc_reward_clip)
        elif have_demo:   # no policy negatives in this batch: skip the D step, still use D's reward
            m.update({"d_loss": zero, "d_gp": zero, "d_acc_pos": zero, "d_acc_neg": zero})
            with torch.no_grad():
                _, lr_ = self.D(ng, z)
                r_d = lr_.clamp(-cfg.disc_reward_clip, cfg.disc_reward_clip)
        else:
            m.update({"d_loss": zero, "d_gp": zero, "d_acc_pos": zero, "d_acc_neg": zero})
            r_d = torch.zeros(b, device=g.device)

        # ---- style critic Q_D and penalty critic Q_R (TD)
        r_r = -penalties(ng, cfg)
        with torch.no_grad():
            nb = cfg.gamma * (1.0 - done)
            y_d = r_d + nb * self.QD_t(nobs, z, a_next)
            y_r = r_r + nb * self.QR_t(nobs, z, a_next)
        l_qd = F.mse_loss(self.QD(obs, z, inp["action"]), y_d)
        self._step(self.opt_qd, l_qd)
        l_qr = F.mse_loss(self.QR(obs, z, inp["action"]), y_r)
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
        m.update({"qd_loss": l_qd.detach(), "qr_loss": l_qr.detach(), "actor_loss": l_pi.detach(),
                  "q_fb": q_fb.mean().detach(), "q_d": q_d.mean().detach(), "q_r": q_r.mean().detach(),
                  "r_d": r_d.mean().detach(), "r_r": r_r.mean().detach()})
        return m

    # -------------------------------------------------- CUDA graph
    GRAPH_WARMUP = 3

    def _graph_supported(self, step_d: bool) -> bool:
        return bool(getattr(self.cfg, "cuda_graph", False)) and self.method == "fb" and step_d \
            and self.device.type == "cuda"

    def _make_capturable(self) -> None:
        """Adam must keep its step counters on the GPU to be captured."""
        for opt in (self.opt_fb, self.opt_actor, self.opt_qd, self.opt_qr, self.opt_d):
            for grp in opt.param_groups:
                grp["capturable"] = True
            for st in opt.state.values():
                if "step" in st and torch.is_tensor(st["step"]) and st["step"].device.type != "cuda":
                    st["step"] = st["step"].to(self.device, dtype=torch.float32)

    def _graph_update(self, inp) -> Dict[str, torch.Tensor]:
        g = getattr(self, "_graph", None)
        if g is None:
            self._make_capturable()
            self._static = _clone_tree(inp)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                 # warm up allocator / autograd on a side stream
                for _ in range(self.GRAPH_WARMUP):
                    _copy_tree(self._static, inp)
                    self._core(self._static, True, True)
            torch.cuda.current_stream().wait_stream(side)
            for opt in (self.opt_fb, self.opt_actor, self.opt_qd, self.opt_qr, self.opt_d):
                opt.zero_grad(set_to_none=True)
            g = torch.cuda.CUDAGraph()
            _copy_tree(self._static, inp)
            with torch.cuda.graph(g):
                self._static_out = self._core(self._static, True, True)
            self._graph = g
            self.num_updates += self.GRAPH_WARMUP          # the warm-up iterations were real updates
        _copy_tree(self._static, inp)
        g.replay()
        return self._static_out

    def _apply_lr(self) -> None:
        """Learning rates = config rates x lr_scale, ramped linearly from 1x over lr_warmup updates
        counted from the first update of this process (so a resumed run ramps again)."""
        cfg = self.cfg
        scale = float(getattr(cfg, "lr_scale", 1.0))
        warm = int(getattr(cfg, "lr_warmup", 0))
        if scale == 1.0 and not hasattr(self, "_lr_t0"):
            return
        if not hasattr(self, "_lr_t0"):
            self._lr_t0 = self.num_updates
        f = 1.0 if warm <= 0 else min(1.0, (self.num_updates - self._lr_t0 + 1) / warm)
        mult = 1.0 + (scale - 1.0) * f
        base = {id(self.opt_actor): [cfg.lr_actor], id(self.opt_qd): [cfg.lr_critic], id(self.opt_qr): [cfg.lr_critic],
                id(self.opt_d): [cfg.lr_disc], id(self.opt_fb): [cfg.lr_f, cfg.lr_b]}
        for opt in (self.opt_fb, self.opt_actor, self.opt_qd, self.opt_qr, self.opt_d):
            for grp, lr0 in zip(opt.param_groups, base[id(opt)]):
                grp["lr"] = lr0 * mult
        self._lr_mult = mult

    def update(self, batch: Dict[str, object], demo=None) -> Dict[str, float]:
        """One FB-CPR gradient step on a replay batch; demo is a DemoStore or None."""
        self._apply_lr()
        inp, have_demo, step_d = self._prepare(batch, demo)
        warming = getattr(self, "_lr_mult", float(getattr(self.cfg, "lr_scale", 1.0))) != float(
            getattr(self.cfg, "lr_scale", 1.0))
        if self._graph_supported(step_d) and not warming:   # graphs bake the learning rate in: capture after warm-up
            m = self._graph_update(inp)
        else:
            m = self._core(inp, have_demo, step_d)
        self.num_updates += 1
        keys = list(m.keys())
        vals = torch.stack([m[k].float().reshape(()) for k in keys]).tolist()   # one host sync per update
        return dict(zip(keys, vals))


def _clone_tree(x):
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    return x.clone() if torch.is_tensor(x) else x


def _copy_tree(dst, src) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            _copy_tree(dst[k], v)
        elif torch.is_tensor(v):
            dst[k].copy_(v)
