"""Velocity-command cascade controller shared by both physics backends.

action [-1, 1]^5 -> desired world velocity + yaw rate -> collective thrust (N, along
body z) and body torque (N m). Runs every physics step.
"""
from __future__ import annotations

import torch

from .. import spec
from .math3d import quat_rotate_inv, yaw_of

GRAVITY = 9.81
INERTIA = (0.03, 0.03, 0.05)   # kg m^2, diagonal


class VelocityController:
    def __init__(self, mass: float = spec.DRONE_MASS, kp_vel: float = 3.0,
                 kp_att: float = 150.0, kd_att: float = 20.0, kp_yaw: float = 2.0,
                 max_tilt: float = spec.MAX_TILT, max_thrust_ratio: float = 2.5):
        self.mass = mass
        self.kp_vel = kp_vel
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_yaw = kp_yaw
        self.max_tilt = max_tilt
        self.max_thrust = max_thrust_ratio * mass * GRAVITY
        self.inertia = torch.tensor(INERTIA)

    @staticmethod
    def command(action: torch.Tensor):
        """action [..., 5] in [-1, 1] -> (v_des [..., 3], yaw_rate [...])."""
        a = action.clamp(-1.0, 1.0)
        scale = a.new_tensor([spec.MAX_SPEED, spec.MAX_SPEED, spec.MAX_VZ])
        return a[..., :3] * scale, a[..., 3] * spec.MAX_YAW_RATE

    def __call__(self, action, quat, vel, ang_vel_body, mass=None, v_des=None):
        """Return (thrust [...], torque_body [..., 3], hover_ratio [...]).

        v_des overrides the velocity from the action (used by the safety shield).
        mass: optional per-agent mass [...] (domain randomisation + payload).
        """
        cmd_v, yaw_rate_des = self.command(action)
        if v_des is None:
            v_des = cmd_v
        m = self.mass if mass is None else mass
        if not torch.is_tensor(m):
            m = torch.full(vel.shape[:-1], float(m), device=vel.device)
        # desired acceleration, then tilt-limited thrust vector
        acc = self.kp_vel * (v_des - vel)
        acc_z = acc[..., 2] + GRAVITY
        acc_xy = acc[..., :2]
        max_xy = acc_z.clamp_min(1.0) * torch.tan(torch.tensor(self.max_tilt))
        n_xy = acc_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        acc_xy = acc_xy * torch.minimum(torch.ones_like(n_xy), max_xy.unsqueeze(-1) / n_xy)
        f_des = torch.cat([acc_xy, acc_z.clamp_min(0.5).unsqueeze(-1)], dim=-1) * m.unsqueeze(-1)
        # desired body z axis in world, expressed in the current body frame
        z_des_body = quat_rotate_inv(quat, f_des / f_des.norm(dim=-1, keepdim=True))
        # attitude error: rotation taking body z (0,0,1) to z_des_body
        err = torch.stack([-z_des_body[..., 1], z_des_body[..., 0]], dim=-1)  # small-angle roll, pitch
        # thrust = projection of desired force onto current body z
        bz_world_dot = (z_des_body[..., 2]).clamp_min(0.2)
        thrust = (f_des.norm(dim=-1) * bz_world_dot).clamp(0.0, self.max_thrust)
        inertia = self.inertia.to(vel.device)
        rp_torque = inertia[:2] * (self.kp_att * err - self.kd_att * ang_vel_body[..., :2])
        yaw_torque = inertia[2] * (self.kp_yaw * 4.0 * (yaw_rate_des - ang_vel_body[..., 2]))
        torque = torch.cat([rp_torque, yaw_torque.unsqueeze(-1)], dim=-1)
        hover = thrust / (m * GRAVITY)
        return thrust, torque, hover
