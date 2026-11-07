"""Agent: the one object the mission and demo layers use.

    agent = Agent.load("runs/x/agent.pt", device="cpu")
    z = agent.infer_z(g, reward)                 # [d] from buffer states and reward labels
    z = agent.encode_g(g_goal)                   # reach: z = B(g_goal)
    actions = agent.act(obs, z.expand(N, -1))    # obs from env.reset()/step(), [N, A, 5]
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from .. import spec
from .config import Config
from .fb_cpr import FBCPR
from .sf_fallback import SFLearner


def make_learner(cfg: Config, device="cpu") -> FBCPR:
    if cfg.method == "fb":
        return FBCPR(cfg, device)
    if cfg.method == "sf":
        return SFLearner(cfg, device)
    raise ValueError(f"unknown method {cfg.method!r}")


class Agent:
    def __init__(self, cfg: Optional[Config] = None, device="cpu", learner: Optional[FBCPR] = None):
        self.cfg = cfg or Config()
        self.device = torch.device(device)
        self.learner = learner if learner is not None else make_learner(self.cfg, self.device)
        # Sample of replay-buffer (and demo) states g [n, G_DIM] saved with the checkpoint,
        # so z can be inferred from a reward program without the training buffer.
        self.g_pool: Optional[torch.Tensor] = None

    # -------------------------------------------------- properties
    @property
    def method(self) -> str:
        return self.learner.method

    @property
    def d(self) -> int:
        return self.learner.d

    # -------------------------------------------------- inference
    def _actor_obs(self, obs: Dict[str, torch.Tensor], n: int, a: int) -> Dict[str, torch.Tensor]:
        keys = spec.CRITIC_KEYS if self.cfg.actor_priv else spec.ACTOR_KEYS
        return {k: obs[k].to(self.device).reshape((n * a,) + tuple(obs[k].shape[2:])) for k in keys}

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], z: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """obs: env obs dict [N, A, ...]; z: [N, d] (team z, broadcast over agents),
        [N, A, d], or [d]. Returns actions [N, A, ACTION_DIM] in [-1, 1]."""
        n, a = obs["proprio"].shape[:2]
        z = z.to(self.device).float()
        if z.dim() == 1:
            z = z.expand(n, -1)
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(n, a, z.shape[-1])
        self.learner.actor.eval()
        out = self.learner.act(self._actor_obs(obs, n, a), z.reshape(n * a, -1), noise)
        return out.reshape(n, a, spec.ACTION_DIM)

    @torch.no_grad()
    def encode_g(self, g: torch.Tensor) -> torch.Tensor:
        """B(g) (FB) or phi(g) (SF), [..., d]."""
        return self.learner.encode_g(g.to(self.device).float())

    @torch.no_grad()
    def infer_z(self, g: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """z for a reward labelled on states g [n, G] with reward [n]. FB: sqrt(d) normalize(mean r B(g));
        SF: least squares of r on phi(g)."""
        return self.learner.infer_z(g.to(self.device).float(), reward.to(self.device).float())

    def update(self, batch, demo=None) -> Dict[str, float]:
        return self.learner.update(batch, demo)

    # -------------------------------------------------- persistence
    def state_dict(self, with_optim: bool = False) -> dict:
        L = self.learner
        st = {"config": self.cfg.to_dict(), "method": self.method, "d": self.d,
              "nets": {k: v.state_dict() for k, v in L.named_children()},
              "num_updates": L.num_updates,
              "g_pool": None if self.g_pool is None else self.g_pool.detach().cpu()}
        if with_optim:
            st["optim"] = {k: getattr(L, k).state_dict()
                           for k in ("opt_fb", "opt_actor", "opt_qd", "opt_qr", "opt_d")}
        return st

    def save(self, path: str, with_optim: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(with_optim), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, device="cpu") -> "Agent":
        st = torch.load(path, map_location=device, weights_only=False)
        cfg = Config.from_dict(st["config"])
        cfg.method = st["method"]
        ag = cls(cfg, device)
        L = ag.learner
        for k, sd in st["nets"].items():
            getattr(L, k).load_state_dict(sd)
        L.num_updates = st.get("num_updates", 0)
        gp = st.get("g_pool")
        ag.g_pool = None if gp is None else gp.to(device)
        if "optim" in st:
            for k, sd in st["optim"].items():
                getattr(L, k).load_state_dict(sd)
        return ag
