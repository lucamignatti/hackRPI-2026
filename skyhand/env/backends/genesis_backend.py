"""Genesis backend: same API and scenes as SkyEnv, with Genesis doing drone rigid-body
dynamics and contacts, and Genesis' raycaster providing the spherical LiDAR.

Everything Genesis lacks stays in shared torch code: controller, waves, net, fish,
payload pendulum, shield, anchors, goal space. Desktop and laptops only; where Genesis is not
installed, train with backend="torch".

Differences from the torch backend, by design:
- Gravity is off in Genesis; the shared force model (thrust, gravity, drag, wind,
  payload and net loads) is applied as an external wrench, scaled to the Genesis
  body mass so domain-randomised masses behave the same.
- Geometry is fixed at scene build, so obstacle sizes are drawn once per env
  instance and shared by every world (positions still vary per world). The
  payload uses one fixed box size.
- Obstacles and the payload are kinematic: their poses come from the shared torch
  logic and are written to Genesis every physics step.
- Net, fish, and the water surface are not Genesis bodies; the green channel and
  the downward grid are still ray-cast in torch against them.
- LiDAR: by default the shared torch ray caster runs against the same boxes,
  spheres, and planes Genesis simulates, so readings match the torch backend.
  cfg.genesis_lidar=True switches the spherical scan to Genesis' raycaster, but in
  genesis-world 1.4.5 rays start inside the drone body and return it at ~7 cm
  (min_range does not filter these), so it is off until that is solved.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ... import spec
from ..core import BOAT_HALF, EXTRA_OBSTACLE_SLOTS, SkyEnv
from ..math3d import quat_rotate, quat_rotate_inv

_GS_READY = False
DRONE_BOX = (0.4, 0.4, 0.1)
DRONE_RHO = spec.DRONE_MASS / (DRONE_BOX[0] * DRONE_BOX[1] * DRONE_BOX[2])
PAYLOAD_BOX = (0.4, 0.35, 0.4)
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
        n, a = self.num_envs, self.num_agents
        g = torch.Generator().manual_seed(self.cfg.seed + 1234)
        # obstacle slot sizes, fixed for this instance
        half = torch.zeros(self.nb, 3)
        if self.scene == "trash":
            nob = spec.TRASH_N_OBSTACLES
            r = torch.rand(nob, 3, generator=g)
            half[:nob] = torch.stack([0.4 + 1.1 * r[:, 0], 0.4 + 1.1 * r[:, 1], 0.75 + 1.25 * r[:, 2]], -1)
            half[nob:] = 0.6
        else:
            half[0] = torch.tensor(BOAT_HALF)
            half[1:] = 0.6
        self._slot_half = half.to(self.device)
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=spec.PHYS_DT, substeps=1, gravity=(0.0, 0.0, 0.0)),
                         show_viewer=False)
        if self.scene == "trash":
            scene.add_entity(gs.morphs.Plane())
        self._drones = [scene.add_entity(gs.morphs.Box(size=DRONE_BOX, pos=(3.0 * i, 0.0, 2.0)),
                                         material=gs.materials.Rigid(rho=DRONE_RHO)) for i in range(a)]
        self._boxes = [scene.add_entity(gs.morphs.Box(size=tuple(float(2 * x) for x in half[b]),
                                                      pos=(0.0, 0.0, -100.0 - 5 * b)))
                       for b in range(self.nb)]
        self._payload = None
        if self.scene == "trash":
            self._payload = scene.add_entity(gs.morphs.Box(size=PAYLOAD_BOX, pos=(0.0, 0.0, -150.0)))
        el = torch.linspace(spec.LIDAR_V_FOV[0], spec.LIDAR_V_FOV[1], spec.LIDAR_V).tolist()
        az = (torch.arange(spec.LIDAR_H) * (360.0 / spec.LIDAR_H)).tolist()
        self._lidars = []
        if self.cfg.genesis_lidar:
            self._lidars = [scene.add_sensor(gs.sensors.Lidar(
                pattern=gs.sensors.SphericalPattern(angles=(az, el)), entity_idx=d.idx,
                min_range=0.35, max_range=spec.LIDAR_MAX_RANGE)) for d in self._drones]
        scene.build(n_envs=n)
        self._scene = scene
        self._solver = scene.sim.rigid_solver
        self._gmass = torch.tensor([float(d.get_mass()) for d in self._drones], device=self.device)

    # --------------------------------------------------------------- resets
    def _reset_worlds(self, ids, m, a) -> None:
        super()._reset_worlds(ids, m, a)
        half = self._slot_half
        self.box_h[ids] = half
        if self.scene == "trash":
            nob = spec.TRASH_N_OBSTACLES
            self.box_c[ids, :nob, 2] = half[:nob, 2]
            self.pl_size[ids] = torch.tensor(PAYLOAD_BOX, device=self.device)
            self.pl_pos[ids, 2] = PAYLOAD_BOX[2] / 2

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
        self._sync_kinematics(ids)

    # --------------------------------------------------------------- dynamics hooks
    def _integrate_drones(self, force: torch.Tensor, torque: torch.Tensor, dt: float) -> None:
        n, a = self.num_envs, self.num_agents
        scale = (self._gmass.unsqueeze(0) / self.mass).unsqueeze(-1)
        f = force * scale
        tq = quat_rotate(self.quat, torque)
        for j, d in enumerate(self._drones):
            self._solver.apply_links_external_wrench(
                force=self._to_gs(f[:, j:j + 1]), torque=self._to_gs(tq[:, j:j + 1]),
                links_idx=[d.base_link_idx], envs_idx=None)
        self._scene.step()
        for j, d in enumerate(self._drones):
            alive = self.alive[:, j]
            pos = d.get_pos().to(self.device)
            quat = d.get_quat().to(self.device)
            vel = d.get_vel().to(self.device)
            ang = quat_rotate_inv(quat, d.get_ang().to(self.device))
            self.pos[:, j] = torch.where(alive.unsqueeze(-1), pos, self.pos[:, j])
            self.quat[:, j] = torch.where(alive.unsqueeze(-1), quat, self.quat[:, j])
            self.vel[:, j] = torch.where(alive.unsqueeze(-1), vel, self.vel[:, j])
            self.ang[:, j] = torch.where(alive.unsqueeze(-1), ang, self.ang[:, j])

    def _sync_kinematics(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if not hasattr(self, "_scene"):
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        gid = None if env_ids is None else self._to_gs(ids)
        far = FAR.to(self.device)
        for b, ent in enumerate(self._boxes):
            p = torch.where(self.box_active[ids, b].unsqueeze(-1), self.box_c[ids, b], far + torch.tensor([0.0, 0.0, -5.0 * b], device=self.device))
            ent.set_pos(self._to_gs(p), envs_idx=gid)
        if self._payload is not None:
            self._payload.set_pos(self._to_gs(self.pl_pos[ids]), envs_idx=gid)

    def _solid_ranges(self):
        if not self._lidars:
            return None, None
        rs = []
        for lid in self._lidars:
            d = lid.read().distances.to(self.device)              # [N, H, V]
            d = d.transpose(1, 2).reshape(self.num_envs, -1)      # [N, V*H]
            d = torch.where((d > 0) & (d < spec.LIDAR_MAX_RANGE - 1e-3), d, torch.full_like(d, float("inf")))
            rs.append(d)
        return torch.stack(rs, 1), None

    def _geometry(self):
        """With Genesis' raycaster, drones (and the trash payload) are Genesis boxes it
        already sees, so drop their analytic spheres; fish and net nodes stay."""
        geom = super()._geometry()
        if not self._lidars:
            return geom
        skip = self.num_agents + (1 if self.scene == "trash" else 0)
        geom["sph_c"] = geom["sph_c"][:, skip:]
        geom["sph_r"] = geom["sph_r"][:, skip:]
        self._drone_spheres_first = False
        return geom
