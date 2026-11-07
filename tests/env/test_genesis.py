"""Genesis backend tests. Skipped automatically where genesis-world is not installed."""
import pytest
import torch

from skyhand import spec
from skyhand.env.core import make_env
from skyhand.env.lidar import decode_range
from skyhand.spec import AnchorBinding

pytestmark = pytest.mark.genesis


@pytest.fixture(scope="module")
def trash():
    return make_env("trash", 2, backend="genesis", seed=0)


def test_obs_contract_and_hover(trash):
    env = trash
    obs = env.reset()
    spec.check_obs(obs, 2, 1)
    z0 = env.pos[:, 0, 2].clone()
    for _ in range(40):
        out = env.step(torch.zeros(2, 1, spec.ACTION_DIM))
    spec.check_obs(out.obs, 2, 1)
    assert (env.pos[:, 0, 2] - z0).abs().max() < 0.5


def test_velocity_tracking(trash):
    env = trash
    env.reset()
    env.box_active[:] = False
    env.zone_active[:] = False
    env._sync_kinematics()
    act = torch.zeros(2, 1, spec.ACTION_DIM)
    act[..., 1] = 0.4
    for _ in range(30):
        env.step(act)
    assert abs(float(env.vel[:, 0, 1].mean()) - 0.4 * spec.MAX_SPEED) < 0.6


def test_lidar_sees_ground_and_obstacles(trash):
    obs = trash.reset()
    nir = decode_range(obs["lidar"][:, 0, 0])
    assert (nir < spec.LIDAR_MAX_RANGE - 0.5).any()


@pytest.mark.xfail(reason="genesis-world 1.4.5 raycaster returns the drone's own body (~7 cm)", strict=False)
def test_genesis_raycaster_matches_analytic_ground():
    env = make_env("trash", 2, backend="genesis", seed=0, genesis_lidar=True)
    env.reset()
    env.box_active[:] = False
    env._sync_kinematics()
    env._scene.step()
    ext, _ = env._solid_ranges()
    from skyhand.env.lidar import spherical_dirs
    from skyhand.env.math3d import quat_rotate
    dirs = spherical_dirs().reshape(-1, 3)
    dw = quat_rotate(env.quat[:, 0:1].expand(2, dirs.shape[0], 4), dirs.unsqueeze(0).expand(2, -1, 3))
    expect = env.pos[:, 0, 2:3] / (-dw[..., 2]).clamp_min(1e-6)
    down = (dw[..., 2] < -0.3) & (expect < spec.LIDAR_MAX_RANGE - 1)
    ok = torch.isfinite(ext[:, 0]) & down
    assert ok.sum() > 10
    assert (ext[:, 0][ok] - expect[ok]).abs().median() < 0.15


def test_expert_grasps_in_genesis():
    env = make_env("trash", 4, backend="genesis", seed=1, lidar=False)
    env.reset()
    env.bind_anchors(torch.arange(4), [AnchorBinding("handle", entity="payload.handle"), AnchorBinding("zone", entity="bin")])
    grasp = torch.zeros(4, dtype=torch.bool)
    for _ in range(300):
        out = env.step(env.expert_action())
        grasp |= out.events["grasp"][:, 0]
    assert grasp.any()


def test_ocean_runs_in_genesis():
    env = make_env("ocean", 2, backend="genesis", seed=2)
    obs = env.reset()
    for _ in range(10):
        out = env.step(env.expert_action())
    spec.check_obs(out.obs, 2, env.num_agents)
    st = env.render_state(0)
    assert st["net"] is not None and len(st["fish"]) > 0
