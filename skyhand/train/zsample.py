"""z sampling (PLAN.md 7.3).

Mixture: demo segment encodings z_tau = mean_t B(g_t) over a segment (default
50%), B(g) of random buffer states (25%), uniform on the sqrt(d) sphere (25%).
All samples are projected onto the sqrt(d) sphere. Missing sources (empty
demo store / buffer) give their share to the remaining ones.

WorldZ keeps one team z per world for `resample_steps` policy steps, with
staggered timers, shared by every agent in the world.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .nets import sphere

Encode = Callable[[torch.Tensor], torch.Tensor]


class ZSampler:
    def __init__(self, d: int, demo_frac: float = 0.5, buffer_frac: float = 0.25,
                 uniform_frac: float = 0.25, device="cpu"):
        self.d = d
        self.probs = torch.tensor([demo_frac, buffer_frac, uniform_frac], dtype=torch.float32)
        assert (self.probs >= 0).all() and self.probs.sum() > 0
        self.device = torch.device(device)

    def uniform(self, n: int) -> torch.Tensor:
        return sphere(torch.randn(n, self.d, device=self.device))

    @torch.no_grad()
    def encode_segments(self, encode: Encode, segs: torch.Tensor) -> torch.Tensor:
        """segs [k, L, G] -> z_tau [k, d] = sphere(mean_t B(g_t))."""
        k, L, G = segs.shape
        e = encode(segs.reshape(k * L, G).to(self.device)).reshape(k, L, -1)
        return sphere(e.mean(1))

    def choose(self, n: int, have_demo: bool, have_buffer: bool) -> torch.Tensor:
        """Category per row: 0 demo, 1 buffer, 2 uniform."""
        p = self.probs.clone()
        if not have_demo:
            p[0] = 0.0
        if not have_buffer:
            p[1] = 0.0
        if p.sum() <= 0:
            p = torch.tensor([0.0, 0.0, 1.0])
        return torch.multinomial(p / p.sum(), n, replacement=True).to(self.device)

    @torch.no_grad()
    def sample(self, n: int, encode: Optional[Encode] = None, buffer=None, demo=None,
               g_pool: Optional[torch.Tensor] = None) -> torch.Tensor:
        """n z vectors from the mixture. Buffer states come from g_pool [k, G] if
        given (e.g. the training batch's next g), else from buffer.sample_g."""
        have_demo = encode is not None and demo is not None and len(demo) > 0
        have_buf = encode is not None and ((g_pool is not None and len(g_pool) > 0)
                                           or (buffer is not None and buffer.obs_size > 0))
        cat = self.choose(n, have_demo, have_buf)
        z = self.uniform(n)
        i0 = torch.nonzero(cat == 0).squeeze(-1)
        if i0.numel():
            z[i0] = self.encode_segments(encode, demo.sample_segments(i0.numel(), self.device))
        i1 = torch.nonzero(cat == 1).squeeze(-1)
        if i1.numel():
            if g_pool is not None and len(g_pool) > 0:
                g = g_pool[torch.randint(0, len(g_pool), (i1.numel(),), device=g_pool.device)]
            else:
                g = buffer.sample_g(i1.numel())
            z[i1] = sphere(encode(g.to(self.device)))
        return z


class WorldZ:
    """Per-world team z with staggered resampling every `resample_steps` steps."""

    def __init__(self, num_envs: int, sampler: ZSampler, resample_steps: int, device="cpu"):
        self.n, self.sampler, self.period = num_envs, sampler, int(resample_steps)
        self.device = torch.device(device)
        self.z = sampler.uniform(num_envs).to(self.device)
        # staggered: world i first resamples after a random 1..period steps
        self.timer = torch.randint(1, self.period + 1, (num_envs,), device=self.device)

    def step(self, encode: Optional[Encode] = None, buffer=None, demo=None) -> torch.Tensor:
        """Advance one policy step; resample worlds whose window ended. Returns their ids."""
        self.timer -= 1
        ids = torch.nonzero(self.timer <= 0).squeeze(-1)
        if ids.numel():
            self.z[ids] = self.sampler.sample(ids.numel(), encode, buffer, demo).to(self.device)
            self.timer[ids] = self.period
        return ids

    def resample(self, ids: torch.Tensor, encode=None, buffer=None, demo=None) -> None:
        ids = torch.as_tensor(ids, device=self.device).reshape(-1)
        if ids.numel():
            self.z[ids] = self.sampler.sample(ids.numel(), encode, buffer, demo).to(self.device)
            self.timer[ids] = self.period

    def agent_z(self, num_agents: int) -> torch.Tensor:
        """[N, A, d]: every agent in a world shares the world's z."""
        return self.z.unsqueeze(1).expand(self.n, num_agents, self.z.shape[-1])
