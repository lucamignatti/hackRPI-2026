"""Genesis backend: same API and scenes as SkyEnv, with Genesis integrating the
drones' rigid-body dynamics.

Every world in a batch has its own terrain, buildings, trees, items, and animals,
but Genesis fixes geometry when the scene is built. So world geometry, contacts
with it, the LiDAR, and all task logic stay in the shared torch code, identical
in both backends. Genesis receives the shared force model (thrust, gravity, drag,
wind, payload and net loads) as an external wrench each physics step, scaled to
its body mass, with its own gravity off.

Desktop and laptops only; where Genesis is not installed, use backend="torch".
cfg.genesis_lidar=True adds Genesis' raycaster for the spherical scan; in
genesis-world 1.4.5 its rays start inside the drone body and return it at ~7 cm,
and it sees only the drones (no world geometry is in the Genesis scene), so it is
off by default.
"""
from __future__ import annotations

import torch

from ... import spec
from ..core import SkyEnv
from ..math3d import quat_rotate, quat_rotate_inv

_GS_READY = False
DRONE_BOX = (0.4, 0.4, 0.1)
DRONE_RHO = spec.DRONE_MASS / (DRONE_BOX[0] * DRONE_BOX[1] * DRONE_BOX[2])
FAR = torch.tensor([0.0, 0.0, -100.0])


def _init_genesis(device: torch.device):
    global _GS_READY
    import genesis as gs
    if not _GS_READY:
        backend = gs.gpu if device.type in ("cuda", "mps") else gs.cpu
        gs.init(backend=backend, precision="32", logging_level="warning")
        _GS_READY = True
    return gs


class GenesisSkyEnv(SkyEnv):
    def _post_init(self) -> None:
        gs = _init_genesis(self.device)
        self._gs_mod = gs
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=spec.PHYS_DT, substeps=1, gravity=(0.0, 0.0, 0.0)),
                         show_viewer=False)
        self._drones = [scene.add_entity(gs.morphs.Box(size=DRONE_BOX, pos=(3.0 * i, 0.0, 2.0)),
                                         material=gs.materials.Rigid(rho=DRONE_RHO)) for i in range(self.num_agents)]
        self._lidars = []
        if self.cfg.genesis_lidar:
            el = torch.linspace(spec.LIDAR_V_FOV[0], spec.LIDAR_V_FOV[1], spec.LIDAR_V).tolist()
            az = (torch.arange(spec.LIDAR_H) * (360.0 / spec.LIDAR_H)).tolist()
            self._lidars = [scene.add_sensor(gs.sensors.Lidar(
                pattern=gs.sensors.SphericalPattern(angles=(az, el)), entity_idx=d.idx,
                min_range=0.35, max_range=spec.LIDAR_MAX_RANGE)) for d in self._drones]
        scene.build(n_envs=self.num_envs)
        self._scene = scene
        self._solver = scene.sim.rigid_solver
        self._gmass = torch.tensor([float(d.get_mass()) for d in self._drones], device=self.device)

    def _to_gs(self, x: torch.Tensor) -> torch.Tensor:
        import genesis as gs
        return x.to(gs.device)

    def _reset_backend(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "_scene"):
            return
        ids = env_ids.to(torch.long)
        if len(ids) == 0:
            return
        gid = self._to_gs(ids)
        for j, d in enumerate(self._drones):
            pos = torch.where(self.alive[ids, j].unsqueeze(-1), self.pos[ids, j], FAR.to(self.device))
            d.set_pos(self._to_gs(pos), envs_idx=gid)
            d.set_quat(self._to_gs(self.quat[ids, j]), envs_idx=gid)
            ang_w = quat_rotate(self.quat[ids, j], self.ang[ids, j])
            d.set_dofs_velocity(self._to_gs(torch.cat([self.vel[ids, j], ang_w], -1)), envs_idx=gid)

    def _integrate_drones(self, force: torch.Tensor, torque: torch.Tensor, dt: float) -> None:
        scale = (self._gmass.unsqueeze(0) / self.mass).unsqueeze(-1)
        f = force * scale
        tq = quat_rotate(self.quat, torque)
        for j, d in enumerate(self._drones):
            self._solver.apply_links_external_wrench(
                force=self._to_gs(f[:, j:j + 1]), torque=self._to_gs(tq[:, j:j + 1]),
                links_idx=[d.base_link_idx], envs_idx=None)
        self._scene.step()
        for j, d in enumerate(self._drones):
            alive = self.alive[:, j].unsqueeze(-1)
            quat = d.get_quat().to(self.device)
            self.pos[:, j] = torch.where(alive, d.get_pos().to(self.device), self.pos[:, j])
            self.quat[:, j] = torch.where(alive, quat, self.quat[:, j])
            self.vel[:, j] = torch.where(alive, d.get_vel().to(self.device), self.vel[:, j])
            self.ang[:, j] = torch.where(alive, quat_rotate_inv(quat, d.get_ang().to(self.device)), self.ang[:, j])

    def _collisions(self) -> torch.Tensor:
        before = self.pos.clone()
        hit = super()._collisions()
        # the shared code may move drones (resting on the ground); push those worlds back to Genesis
        moved = (self.pos != before).any(-1).any(-1)
        if moved.any():
            self._reset_backend(torch.nonzero(moved).squeeze(-1))
        return hit

    def _solid_ranges(self):
        if not self._lidars:
            return None, None
        rs = []
        for lid in self._lidars:
            d = lid.read().distances.to(self.device)              # [N, H, V]
            d = d.transpose(1, 2).reshape(self.num_envs, -1)      # [N, V*H]
            rs.append(torch.where((d > 0) & (d < spec.LIDAR_MAX_RANGE - 1e-3), d, torch.full_like(d, float("inf"))))
        return torch.stack(rs, 1), None
