"""Hyperparameters for the SkyHand learning stack (PLAN.md 7.4 and 8.3/8.5).

Python >= 3.9, torch >= 2.0, no genesis. Every field has a default so
`Config()` is the full-size run; tests override sizes with `tiny_config()`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from .. import spec


@dataclass
class Config:
    # ------------------------------------------------------------ method
    method: str = "fb"                 # "fb" (FB-CPR) or "sf" (successor-feature fallback, PLAN 7.6)
    d: int = 256                       # z dimension (FB). SF uses sf_fallback.PHI_DIM instead.

    # ------------------------------------------------------------ architecture (PLAN 8.3)
    n_queries: int = 8
    f_width: int = 256
    f_heads: int = 4
    f_blocks: int = 4
    actor_width: int = 256
    actor_heads: int = 4
    actor_blocks: int = 3
    critic_width: int = 256            # Q_D, Q_R
    critic_heads: int = 4
    critic_blocks: int = 3
    b_width: int = 256
    b_blocks: int = 3
    disc_width: int = 256
    disc_blocks: int = 3
    actor_priv: bool = False           # cut line (PLAN 7.1): give the actor privileged tokens too
    action_queries: bool = True        # critics: add an action embedding to every query, not only an action token

    # ------------------------------------------------------------ optimisation (PLAN 7.4)
    gamma: float = 0.99
    lr_f: float = 1e-4
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4            # Q_D, Q_R
    lr_b: float = 1e-5
    lr_disc: float = 1e-5
    tau: float = 0.005                 # Polyak
    target_noise: float = 0.2          # TD3 smoothing sigma
    target_noise_clip: float = 0.3
    ortho_coef: float = 100.0
    gp_coef: float = 10.0
    lambda_d: float = 0.05             # style critic weight in the actor loss
    lambda_r: float = 0.02             # penalty critic weight in the actor loss
    scale_reg: bool = True             # Motivo: scale lambda terms by mean |F^T z| (detached)
    disc_reward_clip: float = 10.0     # clamp on log(D / (1 - D))
    grad_clip: float = 0.0             # 0 = off
    q_loss_coef: float = -1.0          # auxiliary diagonal F loss (Touati 2023 eq. 9); < 0 means 1/d

    # ------------------------------------------------------------ penalties for Q_R (computed from g)
    pen_collision: float = 1.0
    pen_shield: float = 0.5
    pen_tilt: float = 1.0
    tilt_limit: float = 0.5            # rad, tilt above this is penalised
    pen_jerk: float = 0.1              # times jerk / 50
    pen_thrust: float = 0.5
    thrust_limit: float = 1.5          # collective / hover above this is penalised

    # ------------------------------------------------------------ domain randomisation (training)
    domain_rand: bool = False          # master switch for all domain randomisation (off: no sim2real needed)
    rand_drone_dynamics: bool = True   # drone mass and drag randomisation
    enable_wind: bool = True           # ambient/mean wind in arena
    dynamic_wind: bool = True          # dynamic stochastic wind turbulence
    enable_gusts: bool = True          # dynamic gusts

    # ------------------------------------------------------------ evaluation
    eval_every: int = 0                # 0 = disabled, >0 = evaluate every N iterations
    eval_envs: int = 16                # parallel evaluation worlds
    eval_steps: int = 100              # steps per evaluation rollout
    eval_domain_rand: bool = False     # master switch for domain randomisation during eval (clean by default)
    eval_rand_drone_dynamics: bool = False # drone mass/drag randomisation during eval
    eval_enable_wind: bool = False     # wind during eval
    eval_dynamic_wind: bool = False    # dynamic wind turbulence during eval
    eval_enable_gusts: bool = False    # gusts during eval

    # ------------------------------------------------------------ z sampling (PLAN 7.3)
    z_demo_frac: float = 0.5
    z_buffer_frac: float = 0.25
    z_uniform_frac: float = 0.25
    z_resample_steps: int = spec.Z_RESAMPLE_STEPS
    demo_seg_len: int = 8

    # ------------------------------------------------------------ data
    buffer_capacity: int = 2_000_000
    buffer_device: str = ""            # "" = same as training device; "cpu" keeps it in host memory
    demo_capacity: int = 50_000        # demo segments
    z_infer_samples: int = 10_000      # reward labels for z inference

    # ------------------------------------------------------------ loop (PLAN 7.3)
    batch_size: int = 4096
    updates_per_step: int = 2          # G
    expl_noise: float = 0.2
    warm_steps: int = 200              # env steps of expert + random actions before learning
    warm_expert_frac: float = 0.5      # fraction of worlds driven by the expert during warm start
    expert_world_frac: float = 0.1     # fraction of training worlds flown by the expert after warm start
    learning_starts: int = 0           # min transitions before updates (0 = batch_size)

    # ------------------------------------------------------------ run
    scene: str = "trash"               # trash | delivery | ocean | all ("both" = all)
    backend: str = "torch"
    num_envs: int = 1024
    device: str = "cuda"
    steps: int = 100_000               # training iterations (env steps per world)
    out: str = "runs/default"
    seed: int = 0
    lidar: bool = True
    log_every: int = 50
    save_every: int = 2000
    demo_every: int = 500              # iterations between expert demo collections
    demo_envs: int = 32                # worlds in the separate expert demo env
    demo_steps: int = 64               # steps per demo collection (demo env keeps running between calls)
    demo_initial_steps: int = 600      # first collection: a full episode, so demos include grasp and carry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def tiny_config(**kw) -> Config:
    """Small sizes for tests and laptops."""
    base = dict(d=16, n_queries=4,
                f_width=32, f_heads=2, f_blocks=1,
                actor_width=32, actor_heads=2, actor_blocks=1,
                critic_width=32, critic_heads=2, critic_blocks=1,
                b_width=32, b_blocks=1, disc_width=32, disc_blocks=1,
                buffer_capacity=4096, demo_capacity=256, batch_size=32,
                num_envs=4, device="cpu", lidar=False, warm_steps=4,
                log_every=2, save_every=4, demo_every=4, demo_envs=2, demo_steps=16, demo_initial_steps=16)
    base.update(kw)
    return Config(**base)
