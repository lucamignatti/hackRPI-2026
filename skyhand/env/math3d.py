"""Batched quaternion and vector helpers. Quaternions are (w, x, y, z)."""
from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v (body) into world by q."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the body frame of q."""
    return quat_rotate(quat_conj(q), v)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(0.5 * yaw), z, z, torch.sin(0.5 * yaw)], dim=-1)


def quat_integrate(q: torch.Tensor, ang_vel_body: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate orientation with body-frame angular velocity."""
    omega = torch.cat([torch.zeros_like(ang_vel_body[..., :1]), ang_vel_body], dim=-1)
    return quat_normalize(q + 0.5 * dt * quat_mul(q, omega))


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q: torch.Tensor) -> torch.Tensor:
    """Angle between body z axis and world z axis."""
    bz = quat_rotate(q, q.new_tensor([0.0, 0.0, 1.0]).expand(q.shape[:-1] + (3,)))
    return torch.acos(bz[..., 2].clamp(-1.0, 1.0))


def rotmat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [..., 3, 3] to quaternion (w, x, y, z), numerically safe."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    qw = torch.sqrt((1.0 + tr).clamp_min(1e-9)) * 0.5
    qx = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-9)) * 0.5
    qy = torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-9)) * 0.5
    qz = torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-9)) * 0.5
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    return quat_normalize(torch.stack([qw, qx, qy, qz], dim=-1))
