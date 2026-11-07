"""Genesis backend tests. Skipped automatically where genesis-world is not installed."""
import pytest
import torch

from skyhand import spec
from skyhand.env.core import make_env
from skyhand.spec import AnchorBinding

pytestmark = pytest.mark.genesis


@pytest.mark.parametrize("scene", ["trash", "delivery", "ocean"])
def test_obs_contract_all_scenes(scene):
    env = make_env(scene, 2, backend="genesis", seed=0)
    obs = env.reset()
    spec.check_obs(obs, 2, env.num_agents)
    for _ in range(5):
        out = env.step(torch.zeros(2, env.num_agents, spec.ACTION_DIM))
    spec.check_obs(out.obs, 2, env.num_agents)


def test_hover_and_velocity_tracking():
    env = make_env("trash", 2, backend="genesis", seed=0, lidar=False, city_prob=1.0, shield=False)
    env.reset()
    env.box_active[:] = False
    env.sph_active[:] = False
    env.pos[:, 0, 2] = 5.0
    env._reset_backend(torch.arange(2))
    z0 = env.pos[:, 0, 2].clone()
    for _ in range(40):
        env.step(torch.zeros(2, 1, spec.ACTION_DIM))
    assert (env.pos[:, 0, 2] - z0).abs().max() < 0.5
    act = torch.zeros(2, 1, spec.ACTION_DIM)
    act[..., 1] = 0.4
    for _ in range(30):
        env.step(act)
    assert abs(float(env.vel[:, 0, 1].mean()) - 0.4 * env.max_speed) < 0.6


def test_expert_grasps_in_genesis():
    env = make_env("trash", 4, backend="genesis", seed=1, lidar=False, city_prob=0.0)
    env.reset()
    for i in range(4):
        k = next(k for k in range(spec.MAX_ITEMS) if bool(env.it_active[i, k]) and bool(env.catalog.is_trash[env.it_type[i, k]]))
        env.bind_anchors([i], [AnchorBinding("handle", entity=f"item_{k}.handle"), AnchorBinding("zone", entity="bin")])
    grasp = torch.zeros(4, dtype=torch.bool)
    for _ in range(400):
        out = env.step(env.expert_action())
        grasp |= out.events["grasp"][:, 0]
    assert grasp.any()
