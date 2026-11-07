import json

import pytest
import torch

from skyhand import spec
from skyhand.spec import AnchorBinding
from skyhand.env.core import BIN_RADIUS, SkyEnv, make_env


def zeros(env):
    return torch.zeros(env.num_envs, env.num_agents, spec.ACTION_DIM)


@pytest.mark.parametrize("scene", ["trash", "ocean"])
def test_obs_contract(scene):
    env = make_env(scene, 4)
    obs = env.reset()
    spec.check_obs(obs, 4, env.num_agents)
    out = env.step(zeros(env))
    spec.check_obs(out.obs, 4, env.num_agents)
    assert out.done.shape == (4,) and out.truncated.shape == (4,)
    assert set(out.events) == set(spec.EVENT_KEYS)
    for v in out.events.values():
        assert v.shape == (4, env.num_agents) and v.dtype == torch.bool


@pytest.mark.parametrize("scene", ["trash", "ocean"])
def test_seed_determinism(scene):
    a = make_env(scene, 3, seed=7, lidar=False)
    b = make_env(scene, 3, seed=7, lidar=False)
    for _ in range(10):
        oa, ob = a.step(zeros(a)), b.step(zeros(b))
    assert torch.allclose(oa.obs["g"], ob.obs["g"])


def test_hover_and_velocity_tracking():
    env = make_env("trash", 2, lidar=False, domain_rand=False, shield=False)
    env.reset()
    z0 = env.pos[:, 0, 2].clone()
    for _ in range(60):
        env.step(zeros(env))
    assert (env.pos[:, 0, 2] - z0).abs().max() < 0.3
    act = zeros(env)
    act[..., 0] = 0.5                                   # 3 m/s along x
    env.reset()
    for _ in range(40):
        env.step(act)
    assert abs(float(env.vel[:, 0, 0].mean()) - 0.5 * spec.MAX_SPEED) < 0.4


def test_partial_reset_leaves_other_worlds():
    env = make_env("trash", 4, lidar=False)
    env.reset()
    for _ in range(5):
        env.step(zeros(env))
    before = env.pos[1:].clone()
    env.reset(torch.tensor([0]))
    assert torch.allclose(env.pos[1:], before)
    assert int(env.steps[0]) == 0 and int(env.steps[1]) == 5


def test_bind_anchors_entity_and_point():
    env = make_env("trash", 2, lidar=False)
    env.reset()
    env.bind_anchors(torch.arange(2), [AnchorBinding("handle", entity="payload.handle"),
                                       AnchorBinding("point", point=(1.0, 2.0, 3.0))])
    apos, arad, akind = env.anchor_positions()
    assert torch.allclose(apos[:, 0, 0], env.payload_handle())
    assert torch.allclose(apos[:, 0, 1], torch.tensor([1.0, 2.0, 3.0]).expand(2, 3))
    assert int(akind[0, 0, 2]) == 0                      # unset -> none
    with pytest.raises(KeyError):
        env.bind_anchors(torch.arange(2), [AnchorBinding("point", entity="no_such_thing")])
    g = env._obs()["g"]
    rel = spec.anchor_rel(g, "anchor_rel_drone", 1)
    assert torch.allclose(rel, torch.tensor([1.0, 2.0, 3.0]) - env.pos, atol=1e-5)


def test_entity_descriptions_cover_entities():
    for scene in ("trash", "ocean"):
        env = make_env(scene, 1, lidar=False)
        d = env.entity_descriptions()
        assert set(d) == set(env.entity_names())


def test_grasp_and_release_events():
    env = make_env("trash", 1, lidar=False, domain_rand=False, shield=False)
    env.reset()
    off = torch.tensor(spec.GRIPPER_OFFSET)
    env.pos[0, 0] = env.payload_handle()[0] - off
    env.vel.zero_()
    act = zeros(env)
    act[..., 4] = 1.0
    out = env.step(act)
    assert out.events["grasp"][0, 0]
    assert int(env.pl_held_by[0]) == 0
    act[..., 4] = -1.0
    out = env.step(act)
    assert out.events["release"][0, 0] and int(env.pl_held_by[0]) == -1


def test_carried_payload_swings_and_loads_drone():
    env = make_env("trash", 1, lidar=False, domain_rand=False, shield=False)
    env.reset()
    env.pos[0, 0] = env.payload_handle()[0] - torch.tensor(spec.GRIPPER_OFFSET)
    act = zeros(env)
    act[..., 4] = 1.0
    env.step(act)
    act[..., 2] = 0.6
    for _ in range(20):
        env.step(act)
    assert float(env.pl_pos[0, 2]) > float(env.pl_size[0, 2]) / 2 + 0.3   # lifted off the ground
    act[..., 2] = 0.0
    act[..., 0] = 1.0
    swing = 0.0
    for _ in range(20):
        env.step(act)
        swing = max(swing, float(env.swing[0]))
    assert swing > 0.05


def test_delivered_event_in_bin():
    env = make_env("trash", 1, lidar=False)
    env.reset()
    env.pl_pos[0, :2] = env.zone_c[0, 0, :2] + 0.2
    env.pl_pos[0, 2] = env.pl_size[0, 2] / 2
    out = env.step(zeros(env))
    assert out.events["delivered"][0, 0]


def test_obstacle_collision_ends_episode():
    env = make_env("trash", 1, lidar=False, shield=False)
    env.reset()
    env.pos[0, 0] = env.box_c[0, 2] + torch.tensor([0.0, 0.0, 0.0])
    out = env.step(zeros(env))
    assert out.done[0] and out.events["collision"][0, 0]


def test_shield_keeps_drone_out_of_nofly_zone():
    env = make_env("trash", 1, lidar=False, domain_rand=False)
    env.reset()
    env.box_active[:] = False
    env.zone_active[0, 1] = True
    env.zone_c[0, 1] = torch.tensor([5.0, 0.0, 0.0])
    env.zone_r[0, 1] = 1.5
    env.pos[0, 0] = torch.tensor([0.0, 0.0, 2.0])
    act = zeros(env)
    act[..., 0] = 0.6
    hits = 0
    for _ in range(80):
        out = env.step(act)
        hits += int(out.events["shield"].sum())
        d = (env.pos[0, 0, :2] - env.zone_c[0, 1, :2]).norm()
        assert d > 1.5
    assert hits > 0


@pytest.mark.parametrize("scene", ["trash", "ocean"])
def test_copy_state_makes_identical_futures(scene):
    env = make_env(scene, 3, lidar=False, domain_rand=False)
    env.reset()
    for _ in range(5):
        env.step(zeros(env))
    env.copy_state(0, torch.tensor([1, 2]))
    act = torch.rand(env.num_envs, env.num_agents, spec.ACTION_DIM) * 2 - 1
    act[1:] = act[0]
    for _ in range(10):
        out = env.step(act)
    assert torch.allclose(out.obs["g"][0], out.obs["g"][1], atol=1e-4)
    assert torch.allclose(out.obs["g"][0], out.obs["g"][2], atol=1e-4)


def test_get_set_state_roundtrip():
    env = make_env("ocean", 2, lidar=False)
    env.reset()
    for _ in range(3):
        env.step(zeros(env))
    st = env.get_state()
    g0 = env._obs()["g"].clone()
    for _ in range(5):
        env.step(zeros(env))
    env.set_state(st)
    assert torch.allclose(env._obs()["g"], g0, atol=1e-5)


@pytest.mark.parametrize("scene", ["trash", "ocean"])
def test_render_state_schema(scene):
    env = make_env(scene, 2)
    env.reset()
    env.step(zeros(env))
    st = env.render_state(0)
    json.dumps(st)
    for k in ("t", "scene", "drones", "payloads", "obstacles", "zones", "anchors", "waves", "boat", "net", "fish",
              "lidar", "shield_hits", "wind"):
        assert k in st
    assert len(st["drones"]) == env.num_agents
    if scene == "ocean":
        assert len(st["net"]["nodes"]) == spec.NET_ROWS * spec.NET_COLS
        assert len(st["fish"]) == spec.N_FISH_SCHOOLS * spec.FISH_PER_SCHOOL
        assert set(st["waves"]) == {"amp", "k", "dir", "omega", "phase"}


def test_perturbations():
    env = make_env("trash", 1, lidar=False)
    env.reset()
    n_obs = len(env.render_state(0)["obstacles"])
    env.perturb(0, "drop_obstacle", pos=[1.0, 1.0])
    assert len(env.render_state(0)["obstacles"]) == n_obs + 1
    env.perturb(0, "gust", vec=[5.0, 0.0, 0.0], duration=1.0)
    assert float(env.gust[0, 0]) == 5.0
    env.perturb(0, "move_payload", pos=[2.0, -3.0])
    assert torch.allclose(env.pl_pos[0, :2], torch.tensor([2.0, -3.0]))
    oc = make_env("ocean", 1, lidar=False)
    oc.reset()
    a0 = oc.waves.params(0)["amp"][0]
    oc.perturb(0, "waves", scale=2.0)
    assert abs(oc.waves.params(0)["amp"][0] - 2 * a0) < 1e-6
    oc.perturb(0, "disable_drone", agent=1)
    assert not bool(oc.alive[0, 1])
    with pytest.raises(ValueError):
        oc.perturb(0, "explode")


def test_lidar_sees_obstacles_in_trash():
    env = make_env("trash", 2)
    obs = env.reset()
    nir = obs["lidar"][:, :, 0]
    assert (nir < 255).any(), "no LiDAR returns at all"


@pytest.mark.slow
def test_trash_expert_delivers():
    env = make_env("trash", 32, lidar=False, seed=1)
    env.reset()
    env.bind_anchors(torch.arange(32), [AnchorBinding("handle", entity="payload.handle"), AnchorBinding("zone", entity="bin")])
    delivered = torch.zeros(32, dtype=torch.bool)
    for _ in range(600):
        out = env.step(env.expert_action())
        delivered |= out.events["delivered"][:, 0]
    assert delivered.float().mean() >= 0.6


@pytest.mark.slow
def test_ocean_expert_grabs_corners():
    env = make_env("ocean", 16, lidar=False, seed=3)
    env.reset()
    grasp = torch.zeros(16, env.num_agents, dtype=torch.bool)
    for _ in range(300):
        out = env.step(env.expert_action())
        grasp |= out.events["grasp"]
    assert grasp.float().mean() >= 0.6
