"""Environment tests (torch backend): scenes, physics, items, terrain, ocean, rendering."""
import json
import math

import pytest
import torch

from skyhand import spec
from skyhand.spec import AnchorBinding
from skyhand.env.core import BK, SK, ZK, SkyEnv, make_env

SCENES = list(spec.SCENES)


def zeros(env):
    return torch.zeros(env.num_envs, env.num_agents, spec.ACTION_DIM)


def first_item(env, i, trash=True):
    for k in range(spec.MAX_ITEMS):
        if bool(env.it_active[i, k]) and bool(env.catalog.is_trash[env.it_type[i, k]]) == trash:
            return k
    return None


def clear_world(env, i):
    env.box_active[i] = False
    env.sph_active[i] = False
    env.zone_active[i, 1:3] = False


# ---------------------------------------------------------------- contract
@pytest.mark.parametrize("scene", SCENES)
def test_obs_contract(scene):
    env = make_env(scene, 3)
    obs = env.reset()
    spec.check_obs(obs, 3, env.num_agents)
    out = env.step(zeros(env))
    spec.check_obs(out.obs, 3, env.num_agents)
    assert out.done.shape == (3,) and out.truncated.shape == (3,)
    assert set(out.events) == set(spec.EVENT_KEYS)
    for v in out.events.values():
        assert v.shape == (3, env.num_agents) and v.dtype == torch.bool


@pytest.mark.parametrize("scene", SCENES)
def test_seed_determinism(scene):
    a = make_env(scene, 3, seed=7, lidar=False)
    b = make_env(scene, 3, seed=7, lidar=False)
    for _ in range(8):
        oa, ob = a.step(zeros(a)), b.step(zeros(b))
    assert torch.allclose(oa.obs["g"], ob.obs["g"])


def test_domain_randomisation_off_by_default():
    env = make_env("trash", 4, lidar=False)
    assert not env.cfg.domain_rand
    for _ in range(20):
        env.step(zeros(env))
    assert float(env.wind.abs().max()) == 0.0 and float(env.gust.abs().max()) == 0.0
    assert torch.allclose(env.mass, torch.full_like(env.mass, spec.DRONE_MASS))


def test_per_scene_speed_scale():
    for scene in SCENES:
        env = make_env(scene, 2, lidar=False, shield=False)
        assert env.max_speed == spec.SCENE_MAX_SPEED[scene]
        assert env.arena_half == spec.SCENE_ARENA_HALF[scene]
    env = make_env("delivery", 2, lidar=False, shield=False)
    env.reset()
    clear_world(env, slice(None))
    env.pos[..., 2] = env.surface_height(env.pos[..., :2]) + 60.0
    act = zeros(env)
    act[..., 0] = 0.8
    for _ in range(80):
        env.step(act)
    assert float(env.vel[:, 0, 0].mean()) > 12.0          # fast long-range flight


# ---------------------------------------------------------------- flight
def test_hover_and_velocity_tracking():
    env = make_env("trash", 2, lidar=False, shield=False, city_prob=1.0)
    env.reset()
    clear_world(env, slice(None))
    env.pos[..., 2] = 5.0
    z0 = env.pos[:, 0, 2].clone()
    for _ in range(60):
        env.step(zeros(env))
    assert (env.pos[:, 0, 2] - z0).abs().max() < 0.3
    act = zeros(env)
    act[..., 0] = 0.5
    for _ in range(40):
        env.step(act)
    assert abs(float(env.vel[:, 0, 0].mean()) - 0.5 * env.max_speed) < 0.4


def test_partial_reset_leaves_other_worlds():
    env = make_env("trash", 4, lidar=False)
    for _ in range(5):
        env.step(zeros(env))
    before = env.pos[1:].clone()
    env.reset(torch.tensor([0]))
    assert torch.allclose(env.pos[1:], before)
    assert int(env.steps[0]) == 0 and int(env.steps[1]) == 5


# ---------------------------------------------------------------- anchors and entities
def test_bind_anchors_entity_and_point():
    env = make_env("trash", 2, lidar=False)
    k = first_item(env, 0)
    env.bind_anchors(torch.tensor([0]), [AnchorBinding("handle", entity=f"item_{k}.handle"),
                                         AnchorBinding("point", point=(1.0, 2.0, 3.0))])
    apos, arad, akind = env.anchor_positions()
    assert torch.allclose(apos[0, 0, 0], env.item_handles()[0, k])
    assert torch.allclose(apos[0, 0, 1], torch.tensor([1.0, 2.0, 3.0]))
    assert int(akind[0, 0, 2]) == 0
    with pytest.raises(KeyError):
        env.bind_anchors(torch.tensor([0]), [AnchorBinding("point", entity="payload.handle")])
    g = env.observe()["g"]
    assert torch.allclose(spec.anchor_rel(g, "anchor_rel_drone", 1)[0], torch.tensor([1.0, 2.0, 3.0]) - env.pos[0], atol=1e-5)


@pytest.mark.parametrize("scene", SCENES)
def test_entity_descriptions(scene):
    env = make_env(scene, 1, lidar=False)
    d = env.entity_descriptions()
    assert set(d) == set(env.entity_names())
    if scene == "trash":
        k = first_item(env, 0)
        assert spec.ITEM_TYPES[int(env.it_type[0, k])] in d[f"item_{k}"] and "(trash)" in d[f"item_{k}"]
        assert "handle (grab here)" in d[f"item_{k}.handle"]
    if scene == "delivery":
        assert d["item_0"].startswith("cooler") and "hospital" in d["pad_dropoff"]
    if scene == "ocean":
        assert "whale_0" in d and "net corner" in d["net.corner_0"]


def test_goal_space_payload_fields():
    env = make_env("trash", 1, lidar=False, city_prob=0.0)
    k = first_item(env, 0)
    env.bind_anchors([0], [AnchorBinding("handle", entity=f"item_{k}.handle")])
    g = env.observe()["g"][0, 0]
    t = int(env.it_type[0, k])
    onehot = g[spec.G_SLICES["payload_type"]]
    assert int(onehot.argmax()) == t and float(onehot.sum()) == 1.0
    assert float(g[spec.G_SLICES["payload_is_trash"]]) == 1.0
    assert torch.allclose(g[spec.G_SLICES["payload_rel"]], env.it_pos[0, k] - env.pos[0, 0], atol=1e-5)
    assert float(g[spec.G_SLICES["trash_left"]]) == 1.0


def test_priv_tokens_are_nearest_entities():
    env = make_env("trash", 2, lidar=False, city_prob=1.0)
    obs = env.observe()
    assert obs["priv_mask"].sum(-1).max() <= spec.MAX_PRIV
    rel = obs["priv"][..., :3].norm(dim=-1)
    m = obs["priv_mask"]
    d = torch.where(m, rel, torch.full_like(rel, float("inf")))
    assert (d[..., 1:] >= d[..., :-1] - 1e-4).all()       # sorted nearest first


# ---------------------------------------------------------------- items
def test_grasp_release_and_swing():
    env = make_env("trash", 1, lidar=False, shield=False, city_prob=0.0)
    clear_world(env, 0)
    k = first_item(env, 0)
    env.pos[0, 0] = env.item_handles()[0, k] - torch.tensor(spec.GRIPPER_OFFSET)
    env.vel.zero_()
    act = zeros(env)
    act[..., 4] = 1.0
    out = env.step(act)
    assert out.events["grasp"][0, 0] and int(env.it_held_by[0, k]) == 0 and int(env.holding[0, 0]) == k
    act[..., 2] = 0.8
    for _ in range(20):
        env.step(act)
    assert float(env.it_pos[0, k, 2]) > float(env.terrain.height(env.it_pos[:, k:k + 1, :2])[0, 0]) + 0.5
    act[..., 2] = 0.0
    act[..., 0] = 1.0
    swing = 0.0
    for _ in range(20):
        env.step(act)
        swing = max(swing, float(env.swing[0, 0]))
    assert swing > 0.05
    act[..., 4] = -1.0
    out = env.step(act)
    assert out.events["release"][0, 0] and int(env.it_held_by[0, k]) == -1


def test_trash_delivered_into_bin_is_removed():
    env = make_env("trash", 1, lidar=False, city_prob=0.0)
    k = first_item(env, 0)
    before = float(env.observe()["g"][0, 0, spec.G_SLICES["trash_left"]])
    env.it_pos[0, k, :2] = env.zone_c[0, 0, :2]
    env.it_pos[0, k, 2] = env.zone_c[0, 0, 2] + 2.0
    out = env.step(zeros(env))
    assert out.events["delivered"][0, 0]
    assert not bool(env.it_active[0, k])
    after = float(out.obs["g"][0, 0, spec.G_SLICES["trash_left"]])
    assert after < before


def test_cooler_must_be_set_down_gently():
    env = make_env("delivery", 1, lidar=False)
    pad = env.zone_c[0, spec.ZONE_SLOT["pad_dropoff"]]
    env.it_pos[0, 0] = pad + torch.tensor([0.0, 0.0, 3.0])        # dropped from 3 m: not a delivery yet
    assert not bool(env._check_delivered()[0])
    env.it_pos[0, 0] = pad + torch.tensor([0.0, 0.0, float(env.it_half[0, 0, 2]) + 0.05])
    assert bool(env._check_delivered()[0]) and bool(env.it_active[0, 0])   # the cooler stays on the pad


def test_item_sinks_in_lake():
    env = make_env("trash", 16, lidar=False, city_prob=0.0, seed=2)
    lvl = env.terrain.water_level(env.t)
    wet = env.terrain.h < (lvl[:, None, None] - 0.8)
    i = int(torch.nonzero(wet.flatten(1).any(-1))[0])
    r, c = [int(x) for x in torch.nonzero(wet[i])[0]]
    k = first_item(env, i)
    env.it_pos[i, k, :2] = torch.stack([env.terrain.gx[r, c], env.terrain.gy[r, c]])
    env.it_pos[i, k, 2] = float(lvl[i]) + 1.0
    lost = False
    for _ in range(30):
        out = env.step(zeros(env))
        lost |= bool(out.events["lost"][i, 0])
    assert lost and not bool(env.it_active[i, k])


# ---------------------------------------------------------------- collisions, terrain, shield
def test_box_and_sphere_collisions_end_episode():
    env = make_env("trash", 2, lidar=False, shield=False, city_prob=1.0)
    b = int(torch.nonzero(env.box_kind[0] == BK["building"])[0])
    env.pos[0, 0] = env.box_c[0, b]
    env.sph_active[1, 0] = True
    env.sph_r[1, 0] = 1.0
    env.sph_moving[1, 0] = False
    env.sph_c[1, 0] = env.pos[1, 0] + torch.tensor([0.5, 0.0, 0.0])
    out = env.step(zeros(env))
    assert out.done.all() and out.events["collision"].all()


def test_terrain_soft_landing_hard_crash_and_water():
    env = make_env("trash", 3, lidar=False, shield=False, city_prob=0.0, seed=4)
    for i in range(3):
        clear_world(env, i)
    g = env.ground_height(env.pos[..., :2])
    env.pos[0, 0, 2] = g[0, 0] + spec.DRONE_RADIUS + 0.02         # gentle touchdown
    env.pos[1, 0, 2] = g[1, 0] + spec.DRONE_RADIUS + 0.02
    env.vel[1, 0, 2] = -4.0                                        # hard impact
    out = env.step(zeros(env))
    assert not bool(out.done[0]) and bool(out.done[1])
    lvl = env.terrain.water_level(env.t)
    wet = env.terrain.h < (lvl[:, None, None] - 0.5)
    i = int(torch.nonzero(wet.flatten(1).any(-1))[0])
    r, c = [int(x) for x in torch.nonzero(wet[i])[0]]
    env.pos[i, 0] = torch.stack([env.terrain.gx[r, c], env.terrain.gy[r, c], lvl[i] - 0.2])
    env.vel[i] = 0.0
    out = env.step(zeros(env))
    assert not bool(out.done[i]) and bool(out.events["shield"][i, 0])   # skims the lake, penalised, not terminal
    assert float(env.pos[i, 0, 2]) >= float(lvl[i]) - 0.2


def test_shield_keeps_drone_out_of_nofly_and_off_the_ground():
    env = make_env("trash", 1, lidar=False, city_prob=1.0)
    clear_world(env, 0)
    env.zone_active[0, 1] = True
    env.zone_kind[0, 1] = ZK["nofly"]
    env.zone_c[0, 1] = env.pos[0, 0] + torch.tensor([5.0, 0.0, 0.0])
    env.zone_r[0, 1] = 1.5
    act = zeros(env)
    act[..., 0] = 0.6
    hits = 0
    for _ in range(80):
        out = env.step(act)
        hits += int(out.events["shield"].sum())
        assert (env.pos[0, 0, :2] - env.zone_c[0, 1, :2]).norm() > 1.5
    assert hits > 0 and not out.done.any()
    act = zeros(env)
    act[..., 2] = -1.0                                             # dive at the street
    for _ in range(80):
        out = env.step(act)
        assert not out.done.any()
    assert float(env.pos[0, 0, 2]) > spec.DRONE_RADIUS - 0.05


# ---------------------------------------------------------------- layouts
def test_city_layout():
    env = make_env("trash", 8, lidar=False, city_prob=1.0, seed=3)
    assert (env.variant == 0).all()
    for i in range(8):
        kinds = env.box_kind[i][env.box_active[i]].tolist()
        assert kinds.count(BK["building"]) >= 4 and kinds.count(BK["car"]) >= 1 and kinds.count(BK["truck"]) == 1
        assert int(env.zone_kind[i, 0]) == ZK["truck"]
        trash = env.catalog.is_trash[env.it_type[i]] & env.it_active[i]
        assert int(trash.sum()) >= 5                              # piles of trash
        assert bool((env.it_active[i] & ~env.catalog.is_trash[env.it_type[i]]).any())   # and things that are not
        assert float(env.terrain.h[i].abs().max()) == 0.0
    cars = env.box_kind == BK["car"]
    c0 = env.box_c[cars].clone()
    for _ in range(10):
        env.step(zeros(env))
    assert (env.box_c[cars] - c0).norm(dim=-1).min() > 0.5           # cars drive


def test_terrain_litter_layout():
    env = make_env("trash", 8, lidar=False, city_prob=0.0, seed=3)
    assert (env.variant == 1).all()
    lvl = env.terrain.water_level(env.t)
    gz = env.terrain.height(env.it_pos[..., :2])
    dry = gz > lvl[:, None]
    assert dry[env.it_active].all()                                  # litter lies on dry land
    assert ((env.terrain.h < lvl[:, None, None]).flatten(1).float().mean(-1) > 0).any()   # lakes exist


def test_birds_stay_removed_from_every_scene():
    """Bird simulation and rendering are intentionally absent from the shared scene catalog."""
    assert "bird" not in SK


def test_tide_changes_water_level():
    env = make_env("trash", 1, lidar=False, city_prob=0.0)
    a = float(env.terrain.water_level(torch.tensor([0.0]))[0])
    b = float(env.terrain.water_level(torch.tensor([15.0]))[0])
    assert abs(a - b) > 0.01


def test_delivery_layout_is_long_range_and_remote():
    env = make_env("delivery", 4, lidar=False, seed=2)
    d = (env.zone_c[:, spec.ZONE_SLOT["pad_dropoff"], :2] - env.zone_c[:, spec.ZONE_SLOT["pad_pickup"], :2]).norm(dim=-1)
    assert (d > 250).all()                                           # clinics far apart
    relief = env.terrain.h.flatten(1).amax(-1) - env.terrain.h.flatten(1).amin(-1)
    assert (relief > 30).all()                                       # hills, mesas, valleys
    assert ((env.box_kind == BK["building"]) & env.box_active).sum(-1).min() >= 2   # a clinic at each pad
    assert torch.allclose(env.it_pos[:, 0, :2], env.zone_c[:, 3, :2])  # cooler waits at the pickup pad


# ---------------------------------------------------------------- ocean
def test_fish_swim_in_schools():
    env = make_env("ocean", 4, lidar=False)
    for _ in range(100):
        env.step(zeros(env))
    c = env.fish.school_centers()[:, env.fish.school]
    assert float((env.fish.pos - c).norm(dim=-1).mean()) < 1.5


def test_animals_move_and_bycatch_when_tangled():
    env = make_env("ocean", 2, lidar=False)
    p0 = env.animals.pos.clone()
    for _ in range(20):
        env.step(zeros(env))
    assert (env.animals.pos - p0).norm(dim=-1).mean() > 0.3
    env.pos[0, 0] = env.net_handles()[0, 0] - torch.tensor(spec.GRIPPER_OFFSET)   # drone 0 over its corner
    env.vel[0, 0] = 0.0
    env.corner_held_by[0, 0] = 0                                     # someone holds the net
    env.holding[0, 0] = 0
    env.animals.pos[0, 1] = env.net.pos[0, 20]                       # a dolphin swims into it
    out = env.step(zeros(env))
    assert bool(out.events["bycatch"][0, 0]) and bool(env.bycatch[0])
    assert float(out.obs["g"][0, 0, spec.G_SLICES["bycatch"]]) == 1.0


# ---------------------------------------------------------------- state, rendering, perturbations
@pytest.mark.parametrize("scene", SCENES)
def test_copy_state_makes_identical_futures(scene):
    env = make_env(scene, 3, lidar=False)
    for _ in range(5):
        env.step(zeros(env))
    env.copy_state(0, torch.tensor([1, 2]))
    act = torch.rand(env.num_envs, env.num_agents, spec.ACTION_DIM) * 2 - 1
    act[1:] = act[0]
    for _ in range(10):
        out = env.step(act)
    assert torch.allclose(out.obs["g"][0], out.obs["g"][1], atol=1e-4)
    assert torch.allclose(out.obs["g"][0], out.obs["g"][2], atol=1e-4)


@pytest.mark.parametrize("scene", SCENES)
def test_render_state_schema(scene):
    env = make_env(scene, 2)
    env.step(zeros(env))
    st = env.render_state(0)
    json.dumps(st)
    for k in ("t", "scene", "variant", "arena_half", "terrain", "water", "drones", "items", "boxes", "spheres",
              "zones", "anchors", "boat", "net", "fish", "animals", "lidar", "shield_hits", "wind"):
        assert k in st, k
    assert len(st["drones"]) == env.num_agents
    if scene == "ocean":
        assert st["terrain"] is None and st["water"]["waves"] is not None and len(st["animals"]) == spec.N_ANIMALS
    else:
        h = st["terrain"]["heights"]
        res = spec.SCENE_TERRAIN_RES[scene]
        assert len(h) == res and len(h[0]) == res and st["terrain"]["res"] == res
        assert env.render_state(0, include_terrain=False)["terrain"]["heights"] is None
        assert all(it["type"] in spec.ITEM_TYPES for it in st["items"])


def test_perturbations():
    env = make_env("trash", 1, lidar=False, city_prob=0.0)
    n0 = len(env.render_state(0)["boxes"])
    env.perturb(0, "drop_obstacle", pos=[1.0, 1.0])
    assert len(env.render_state(0)["boxes"]) == n0 + 1
    env.perturb(0, "gust", vec=[5.0, 0.0, 0.0])                   # ignored: gusts are off by default
    assert float(env.gust.abs().max()) == 0.0
    k = first_item(env, 0)
    env.perturb(0, "move_payload", item=k, pos=[2.0, -3.0])
    assert torch.allclose(env.it_pos[0, k, :2], torch.tensor([2.0, -3.0]))
    env.perturb(0, "tide", offset=0.5)
    assert float(env.terrain.tide_offset[0]) == 0.5
    oc = make_env("ocean", 1, lidar=False)
    a0 = oc.waves.params(0)["amp"][0]
    oc.perturb(0, "waves", scale=2.0)
    assert abs(oc.waves.params(0)["amp"][0] - 2 * a0) < 1e-6
    oc.perturb(0, "disable_drone", agent=1)
    assert not bool(oc.alive[0, 1])
    with pytest.raises(ValueError):
        oc.perturb(0, "explode")


@pytest.mark.parametrize("scene", SCENES)
def test_lidar_sees_the_world(scene):
    env = make_env(scene, 2)
    obs = env.reset()
    assert (obs["lidar"][:, :, 0] < 255).any() and (obs["lidar_down"][:, :, 0] < 255).any()


def test_world_lidar_staggers_one_agent_per_tick():
    env = make_env("world", 1, lidar=False, seed=3)
    env.reset()
    calls = []
    env._scan = lambda ids, agent_ids=None: calls.append((ids.tolist(), tuple(agent_ids)))
    env.cfg.lidar = True
    for _ in range(env.num_agents):
        env.step(zeros(env))
    assert calls == [([0], (2,)), ([0], (1,)), ([0], (0,))]


# ---------------------------------------------------------------- expert (slow)
def _expert_success(scene, seed, **kw):
    env = make_env(scene, 16, seed=seed, lidar=False, **kw)
    for i in range(16):
        if scene == "trash":
            env.bind_anchors([i], [AnchorBinding("handle", entity=f"item_{first_item(env, i)}.handle"),
                                   AnchorBinding("zone", entity="bin")])
        else:
            env.bind_anchors([i], [AnchorBinding("handle", entity="item_0.handle"),
                                   AnchorBinding("zone", entity="pad_dropoff")])
    deliv = torch.zeros(16, dtype=torch.bool)
    done = torch.zeros(16, dtype=torch.bool)
    for _ in range(env.episode_steps):
        out = env.step(env.expert_action())
        deliv |= out.events["delivered"][:, 0] & ~done
        done |= out.done
    return float(deliv.float().mean())


@pytest.mark.slow
@pytest.mark.parametrize("scene,kw", [("trash", {"city_prob": 1.0}), ("trash", {"city_prob": 0.0}), ("delivery", {})])
def test_expert_delivers(scene, kw):
    assert _expert_success(scene, 5, **kw) >= 0.7


@pytest.mark.slow
def test_ocean_expert_hauls_net():
    env = make_env("ocean", 16, seed=3, lidar=False)
    grasp = torch.zeros(16, env.num_agents, dtype=torch.bool)
    for _ in range(300):
        out = env.step(env.expert_action())
        grasp |= out.events["grasp"]
    assert grasp.float().mean() >= 0.6


def test_payload_stays_on_released_item_and_delivered_items_stay_listed():
    env = make_env("trash", 1, lidar=False, shield=False, city_prob=0.0)
    clear_world(env, 0)
    k = first_item(env, 0)
    env.bind_anchors([0], [AnchorBinding("handle", entity=f"item_{k}.handle"), AnchorBinding("zone", entity="bin")])
    env.pos[0, 0] = env.item_handles()[0, k] - torch.tensor(spec.GRIPPER_OFFSET)
    act = zeros(env)
    act[..., 4] = 1.0
    env.step(act)
    assert int(env.holding[0, 0]) == k
    # carry it over the bin and let go
    env.pos[0, 0, :2] = env.zone_c[0, 0, :2]
    env.pos[0, 0, 2] = env.zone_c[0, 0, 2] + 3.0
    env.it_pos[0, k] = env.pos[0, 0] - torch.tensor([0.0, 0.0, 0.8])
    env.bind_anchors([0], [AnchorBinding("point", point=(30.0, 30.0, 3.0))])     # a0 no longer names an item
    act[..., 4] = -1.0
    out = env.step(act)
    assert out.events["release"][0, 0]
    assert int(env.payload_index()[0, 0]) == k                       # still the released item
    rel = out.obs["g"][0, 0, spec.G_SLICES["payload_rel"]]
    assert torch.allclose(rel, env.it_pos[0, k] - env.pos[0, 0], atol=1e-4)
    names = [it["name"] for it in env.render_state(0)["items"] if it["delivered"]]
    assert f"item_{k}" in names                                      # delivered trash stays listed


# ---------------------------------------------------------------- generic world
ZSL = spec.ZONE_SLOT


def test_world_layout_contains_everything():
    env = make_env("world", 4, lidar=False, seed=1)
    assert env.num_agents == spec.WORLD_AGENTS and env.arena_half == 200.0 and env.terrain.res == 128
    for i in range(4):
        kinds = env.box_kind[i][env.box_active[i]].tolist()
        assert kinds.count(BK["building"]) >= 30 and kinds.count(BK["truck"]) == 1
        assert int(env.box_kind[i, env.boat_slot if hasattr(env, "boat_slot") else spec.MAX_BOXES - 2]) == BK["boat"]
        for z in ("bin", "pad_pickup", "pad_dropoff", "boat"):
            assert bool(env.zone_active[i, ZSL[z]]), z
        types = env.it_type[i][env.it_active[i]]
        assert bool((types == env.catalog.id("cooler")).any())
        assert bool(env.catalog.is_trash[types].any()) and bool((~env.catalog.is_trash[types] & (types != env.catalog.id("cooler"))).any())
    pads = (env.zone_c[:, ZSL["pad_pickup"], :2] - env.zone_c[:, ZSL["pad_dropoff"], :2]).norm(dim=-1)
    assert (pads > 100).all()
    lvl = env.terrain.water_level(env.t)
    dry = lambda xy: env.terrain.height(xy) > lvl[:, None]
    assert dry(env.pos[..., :2]).all()                                        # drones start on land
    assert dry(env.it_pos[..., :2])[env.it_active].all()                      # items lie on land
    assert not dry(env.zone_c[:, ZSL["boat"]:ZSL["boat"] + 1, :2]).any()      # the boat floats
    assert (env.terrain.height(env.fish.pos[..., :2]) < -3).all()            # fish live offshore
    assert (env.terrain.height(env.net.pos[..., :2]) < -3).all()
    wet = (env.terrain.h < lvl[:, None, None]).flatten(1).float().mean(-1)
    assert ((wet > 0.1) & (wet < 0.45)).all()                                 # coast and lakes, mostly land


def test_world_water_is_tide_plus_waves():
    env = make_env("world", 2, lidar=False)
    xy = env.zone_c[:, ZSL["boat"]:ZSL["boat"] + 1, :2]
    h = env.water_height(xy)[:, 0]
    lvl = env.terrain.water_level(env.t)
    w = env.waves.height(xy, env.t)[:, 0]
    assert torch.allclose(h, lvl + w, atol=1e-5)
    st = env.render_state(0)
    assert st["water"]["waves"] is not None and st["terrain"] is not None and st["net"] is not None


def test_world_grasps_items_and_net_corners():
    env = make_env("world", 1, lidar=False, shield=False, random_anchors=False)
    clear_world(env, 0)
    k = first_item(env, 0)
    env.pos[0, 0] = env.item_handles()[0, k] - torch.tensor(spec.GRIPPER_OFFSET)
    env.pos[0, 1] = env.net_handles()[0, 2] - torch.tensor(spec.GRIPPER_OFFSET)
    env.vel.zero_()
    act = zeros(env)
    act[:, :2, 4] = 1.0
    out = env.step(act)
    assert out.events["grasp"][0, 0] and out.events["grasp"][0, 1]
    assert int(env.holding[0, 0]) == k and int(env.it_held_by[0, k]) == 0
    assert int(env.holding[0, 1]) == env.corner_offset + 2 and int(env.corner_held_by[0, 2]) == 1
    g = out.obs["g"][0]
    assert float(g[0, spec.G_SLICES["payload_type"]].sum()) == 1.0           # drone 0 carries an item
    assert float(g[1, spec.G_SLICES["payload_type"]].sum()) == 0.0           # drone 1 holds a net corner


@pytest.mark.slow
def test_world_expert_independent_tasks():
    N = 8
    env = make_env("world", N, seed=3, lidar=False, random_anchors=False)
    cooler = env.catalog.id("cooler")
    for i in range(N):
        kc = int(torch.nonzero(env.it_type[i] == cooler)[0])
        env.bind_anchors([i], [AnchorBinding("handle", entity=f"item_{first_item(env, i)}.handle"),
                               AnchorBinding("zone", entity="bin")], agent=0)
        env.bind_anchors([i], [AnchorBinding("handle", entity=f"item_{kc}.handle"),
                               AnchorBinding("zone", entity="pad_dropoff")], agent=1)
        env.bind_anchors([i], [AnchorBinding("point", point=(0.0, 0.0, 30.0))], agent=2)
    trash_in = torch.zeros(N, dtype=torch.bool)
    cooler_in = torch.zeros(N, dtype=torch.bool)
    done = torch.zeros(N, dtype=torch.bool)
    for _ in range(env.episode_steps):
        out = env.step(env.expert_action())
        trash_in |= (env.it_delivered & env.catalog.is_trash[env.it_type]).any(-1) & ~done
        cooler_in |= (env.it_delivered & (env.it_type == cooler)).any(-1) & ~done
        done |= out.done
    assert trash_in.float().mean() >= 0.7 and cooler_in.float().mean() >= 0.7


@pytest.mark.slow
def test_world_expert_team_grabs_net():
    N = 8
    env = make_env("world", N, seed=4, lidar=False, random_anchors=False)
    for ag in range(3):
        env.bind_anchors(torch.arange(N), [AnchorBinding("handle", entity=f"net.corner_{ag}"),
                                           AnchorBinding("zone", entity="boat")], agent=ag)
    grasp = torch.zeros(N, 3, dtype=torch.bool)
    for _ in range(900):
        out = env.step(env.expert_action())
        grasp |= out.events["grasp"]
    assert grasp.float().mean() >= 0.8


# ---------------------------------------------------------------- anchor selectors and active flags
def _active(env):
    return env.observe()["g"][..., spec.G_SLICES["anchor_active"]]


def test_anchor_active_flags_mark_unused_slots():
    env = make_env("trash", 1, lidar=False, random_anchors=False)
    k = first_item(env, 0)
    env.bind_anchors([0], [AnchorBinding("handle", entity=f"item_{k}.handle"), AnchorBinding("zone", entity="bin")])
    assert _active(env)[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0]
    env.bind_anchors([0], [AnchorBinding("point", point=tuple(env.pos[0, 0].tolist()))])
    act = _active(env)[0, 0]
    assert act.tolist() == [1.0, 0.0, 0.0, 0.0]                         # an anchor at the drone is still "in use"


def test_nearest_trash_selector_targets_closest_trash_and_sticks_when_held():
    env = make_env("trash", 1, lidar=False, shield=False, city_prob=0.0, random_anchors=False)
    clear_world(env, 0)
    env.bind_anchors([0], [AnchorBinding("handle", entity="nearest:trash"), AnchorBinding("zone", entity="bin")])
    trash = env.it_active[0] & env.catalog.is_trash[env.it_type[0]]
    d = (env.item_handles()[0] - env.pos[0, 0]).norm(dim=-1)
    want = int(torch.where(trash, d, torch.full_like(d, 1e9)).argmin())
    assert int(env.anchor_item()[0, 0]) == want
    apos, _, akind = env.anchor_positions()
    assert torch.allclose(apos[0, 0, 0], env.item_handles()[0, want]) and int(akind[0, 0, 0]) == 2
    # grab it and fly next to a different trash item: the selector stays on the carried one
    env.pos[0, 0] = env.item_handles()[0, want] - torch.tensor(spec.GRIPPER_OFFSET)
    act = zeros(env)
    act[..., 4] = 1.0
    env.step(act)
    other = next(j for j in range(spec.MAX_ITEMS) if j != want and bool(trash[j]))
    env.pos[0, 0, :2] = env.it_pos[0, other, :2]
    assert int(env.anchor_item()[0, 0]) == want


def test_selector_with_nothing_left_reads_as_unused():
    env = make_env("trash", 1, lidar=False, random_anchors=False)
    env.bind_anchors([0], [AnchorBinding("handle", entity="nearest:trash")])
    assert float(_active(env)[0, 0, 0]) == 1.0
    env.it_active[0] &= ~env.catalog.is_trash[env.it_type[0]]          # all trash gone
    assert float(_active(env)[0, 0, 0]) == 0.0
    _, _, akind = env.anchor_positions()
    assert int(akind[0, 0, 0]) == 0


def test_type_selector_only_matches_that_type():
    env = make_env("trash", 6, lidar=False, city_prob=0.0, random_anchors=False, seed=2)
    env.bind_anchors(torch.arange(6), [AnchorBinding("handle", entity="nearest:pizza_box")])
    it = env.anchor_item()[:, 0]
    has = (env.it_active & (env.it_type == env.catalog.id("pizza_box"))).any(-1)
    for i in range(6):
        if bool(has[i]):
            assert int(env.it_type[i, int(it[i])]) == env.catalog.id("pizza_box")
        else:
            assert int(it[i]) == -1


def test_net_corner_selector_skips_corners_held_by_teammates():
    env = make_env("ocean", 1, lidar=False, random_anchors=False)
    env.bind_anchors([0], [AnchorBinding("handle", entity="nearest:net_corner")], agent=1)
    env.pos[0, 1, :2] = env.net_handles()[0, 0, :2]                     # drone 1 sits over corner 0
    assert int(env.anchor_handle()[0, 1]) == env.corner_offset + 0
    env.corner_held_by[0, 0] = 0                                         # but drone 0 holds it
    env.holding[0, 0] = env.corner_offset + 0
    assert int(env.anchor_handle()[0, 1]) != env.corner_offset + 0
    env.bind_anchors([0], [AnchorBinding("point", entity="nearest:animal")], agent=2)
    apos, _, _ = env.anchor_positions()
    d = (env.animals.pos[0] - env.pos[0, 2]).norm(dim=-1)
    assert torch.allclose(apos[0, 2, 0], env.animals.pos[0, int(d.argmin())])


def test_pretraining_uses_selectors():
    env = make_env("world", 64, lidar=False, seed=0)
    assert bool((env.anc_sel > 0).any())
    spec.check_obs(env.observe(), 64, env.num_agents)




# ---------------------------------------------------------------- dense city, 4-drone ocean, net collisions, fish
def test_city_is_densely_packed():
    import numpy as np
    env = make_env("trash", 4, lidar=False, city_prob=1.0, seed=3)
    H = env.arena_half
    xs = torch.linspace(-H, H, 200)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    for i in range(4):
        m = env.box_active[i] & (env.box_kind[i] == BK["building"])
        assert int(m.sum()) >= 30
        c, h = env.box_c[i][m], env.box_h[i][m]
        occ = ((X[..., None] - c[:, 0]).abs() < h[:, 0]) & ((Y[..., None] - c[:, 1]).abs() < h[:, 1])
        assert occ.any(-1).float().mean() > 0.25                         # blocks are built up, streets narrow


def test_ocean_has_four_drones_one_per_corner():
    env = make_env("ocean", 2, lidar=False)
    assert env.num_agents == 4 == spec.OCEAN_AGENTS
    spec.check_obs(env.observe(), 2, 4)


def test_drone_cannot_fly_through_the_net_and_contact_is_not_terminal():
    env = make_env("ocean", 1, lidar=False, shield=False)
    edge = 2                                                             # a floating edge node (row 0)
    env.pos[0, 0] = env.net.pos[0, edge] + torch.tensor([0.0, 0.0, 0.3])  # just above the rope
    env.vel.zero_()
    env.vel[0, 0, 2] = -3.0                                              # diving into it
    out = env.step(zeros(env))
    assert not bool(out.done[0]) and not bool(out.events["collision"][0, 0])
    assert bool(out.events["shield"][0, 0])                              # penalised like a shield save
    d = env.net.segment_distance(env.pos)[0, 0]
    assert float(d) >= spec.DRONE_RADIUS + 0.05 - 1e-3                   # pushed back out of the ropes


def test_net_cannot_pass_through_the_boat_or_seabed():
    from skyhand.env.core import BOAT_SLOT
    env = make_env("ocean", 1, lidar=False)
    c, h = env.box_c[0, BOAT_SLOT], env.box_h[0, BOAT_SLOT]
    env.net.pos[0, 30] = c                                               # a node inside the hull
    env.net.pos[0, 31, 2] = -40.0                                        # a node under the seabed
    env.step(zeros(env))
    rel = (env.net.pos[0, 30] - env.box_c[0, BOAT_SLOT]).abs()
    assert not bool((rel < env.box_h[0, BOAT_SLOT]).all())
    assert float(env.net.pos[0, 31, 2]) >= float(env.ground_height(env.net.pos[:, 31:32, :2])[0, 0])


def test_fish_motion_is_stochastic_but_seeded():
    a = make_env("ocean", 2, lidar=False, seed=5)
    b = make_env("ocean", 2, lidar=False, seed=5)
    for _ in range(20):
        a.step(zeros(a))
        b.step(zeros(b))
    assert torch.allclose(a.fish.pos, b.fish.pos)                        # same seed, same fish
    v = a.fish.vel
    turn = []
    for _ in range(10):
        a.step(zeros(a))
        turn.append((a.fish.vel - v).norm(dim=-1).mean())
        v = a.fish.vel.clone()
    assert float(torch.stack(turn).mean()) > 0.02                       # fish keep changing course


def test_net_never_tears_corner_slips_instead():
    env = make_env("ocean", 1, lidar=False, shield=False)
    env.pos[0, 0] = env.net_handles()[0, 0] - torch.tensor(spec.GRIPPER_OFFSET)
    env.vel.zero_()
    env.corner_held_by[0, 0] = 0
    env.holding[0, 0] = env.corner_offset                               # handle index of corner 0
    env.pos[0, 0, :2] += 12.0                                            # yank the corner far away
    out = env.step(zeros(env))
    assert not bool(out.done[0])                                         # no tear, the episode goes on
    assert bool(out.events["slip"][0, 0]) and bool(out.events["release"][0, 0])
    assert int(env.corner_held_by[0, 0]) == -1 and int(env.holding[0, 0]) == -1


@pytest.mark.parametrize("scene", SCENES)
def test_distance_fields_use_one_cap_in_every_scene(scene):
    env = make_env(scene, 2, lidar=False)
    g = env.step(zeros(env)).obs["g"]
    for name in ("mate_dist", "animal_dist", "obstacle_dist"):
        v = spec.g_field(g, name)
        assert float(v.max()) <= spec.FAR_DIST + 1e-4, (scene, name)
    if env.num_agents == 1:
        assert torch.all(spec.g_field(g, "mate_dist") == spec.FAR_DIST)
    if scene in ("trash", "delivery"):
        assert torch.all(spec.g_field(g, "animal_dist") == spec.FAR_DIST)


def test_only_the_crashed_drone_is_terminal():
    env = make_env("ocean", 2, lidar=False, shield=False)
    from skyhand.env.core import BOAT_SLOT
    env.pos[0, 1] = env.box_c[0, BOAT_SLOT]                              # drone 1 flies into the boat
    env.vel.zero_()
    out = env.step(zeros(env))
    assert bool(out.done[0]) and not bool(out.done[1])
    ad = out.info["agent_done"]
    assert ad[0].tolist() == [False, True, False, False] and not ad[1].any()


@pytest.mark.parametrize("scene", ["trash", "delivery"])
def test_arena_edge_is_a_soft_wall_not_a_terminal(scene):
    """Flying off is not an escape: the drone is held at the wall, the episode goes on, and the
    touch counts as a shield intervention."""
    env = make_env(scene, 1, lidar=False, shield=False)
    act = zeros(env)
    act[..., 0] = 1.0                                                    # full speed east
    act[..., 2] = 1.0                                                    # and up
    lim = env.arena_half + env.WALL_MARGIN
    touched = False
    for _ in range(env.episode_steps - 1):                               # long enough to cross the arena
        out = env.step(act)
        assert not bool(out.done[0])
        touched |= bool(out.events["shield"][0, 0])
        assert float(env.pos[0, 0, 0]) <= lim + 1e-4
    assert touched and float(env.pos[0, 0, 0]) > lim - 1.0               # pressed against the wall
def test_selector_single_item_scene_no_out_of_bounds():
    env = make_env("delivery", 1, lidar=False, seed=0)
    env.bind_anchors([0], [AnchorBinding(kind="handle", entity="nearest:trash")])
    env.holding[0, 0] = 2
    src = env.anchor_src()
    assert src.shape == (1, 1, spec.N_ANCHORS)


def test_fake_env_selector_and_active_flags():
    from demo.fake_env import FakeEnv
    from demo.pipeline import Pipeline
    pipe = Pipeline()
    env = FakeEnv("trash")
    m = pipe.compile("pick up all the trash", "trash", env)
    pipe.bind(env, m, torch.tensor([0]))
    obs = env.observe()
    pos, kind, rad = env._anchor_state()
    assert kind[0, 0] == spec.ANCHOR_TYPES.index("handle")
    assert bool(obs["g"][0, 0, spec.G_SLICES["anchor_active"].start] == 1.0)
    assert not torch.allclose(pos[0, 0, 0], torch.zeros(3))
