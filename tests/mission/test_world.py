"""World scene: aliases, the cooler, single missions with world names, per-drone splits, the team net haul."""
from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from skyhand import spec
from skyhand.mission import AgentMission, MissionSpec, compile_mission, compile_reward, offline_compile, to_bindings
from skyhand.mission.compile import _EXAMPLES, build_prompt, parse_and_validate
from skyhand.mission.infer_z import mission_z
from skyhand.mission.schema import MissionError, bind_mission, find_cooler, resolve_entity, scene_from_entities

from mission_fakes import FakeEnv, world

WORLD = world()
WORLD_ENTS = list(WORLD)
HOVER = "-abs(height - 3) - 0.3*speed - 0.2*tilt"
CARRY_DONE = "distxy(payload, a1) < 1.0 and not holding"


def _check(text):
    env = FakeEnv("world", WORLD)
    assert env.num_agents == 3
    ms = offline_compile(text, "world", env.entity_descriptions())
    assert ms.offline and ms.scene == "world" and ms.explanation
    g = torch.randn(8, spec.G_DIM) * 3
    for _, e in ms.expressions():
        if e is not None:
            out = compile_reward(e)(g)
            assert out.shape == (8,) and torch.isfinite(out).all()
    n_steps = max(ms.n_steps(ag) for ag in range(3))
    for i in range(n_steps):
        bind_mission(env, ms, step=i)
    bind_mission(env, ms)
    return ms, env


# ------------------------------------------------------------------ schema / aliases
def test_scene_detected_and_accepted():
    assert scene_from_entities(WORLD_ENTS) == "world"
    assert MissionSpec(scene="world", reward="speed").scene == "world"


@pytest.mark.parametrize("alias,target", [
    ("truck", "bin"), ("garbage truck", "bin"), ("dumpster", "bin"), ("hospital", "pad_dropoff"),
    ("helipad", "pad_dropoff"), ("pickup", "pad_pickup"), ("ship", "boat"), ("vessel", "boat"),
    ("net", "net.center"), ("whale", "whale_0"), ("dolphin", "dolphin_0"), ("fish", "fish_school_0"),
    ("nofly", "nofly_0"), ("no-fly zone", "nofly_0"), ("item 3 handle", "item_3.handle"), ("drone", "drone_0"),
    ("cooler", "item_5"), ("organ", "item_5"), ("organ cooler", "item_5"), ("package", "item_5"),
    ("cooler.handle", "item_5.handle"), ("organ.handle", "item_5.handle"), ("cooler handle", "item_5.handle"),
])
def test_world_aliases(alias, target):
    ms = MissionSpec.model_validate({"scene": "world", "anchors": {"a0": alias}, "reward": "1"},
                                    context={"entities": WORLD})
    assert ms.anchors["a0"] == target
    assert resolve_entity(alias, WORLD, scene="world") == target


def test_world_cooler_needs_descriptions_and_ambiguous_aliases():
    # with names only the cooler cannot be found (item_0 is a trash bag in the world)
    with pytest.raises(MissionError, match="descriptions"):
        resolve_entity("cooler", WORLD_ENTS, scene="world")
    with pytest.raises((ValidationError, MissionError)):
        MissionSpec.model_validate({"scene": "world", "anchors": {"a0": "cooler"}, "reward": "1"},
                                   context={"entities": WORLD_ENTS})
    # "dropoff" means the truck in trash and the hospital in delivery: not guessed in the world
    for name in ("dropoff", "payload", "handle"):
        with pytest.raises(MissionError):
            resolve_entity(name, WORLD, scene="world")
    # real names never need descriptions
    assert resolve_entity("item_5.handle", WORLD_ENTS, scene="world") == "item_5.handle"


def test_find_cooler_prefers_active():
    ents = {"item_0": "cooler (organ transport) (inactive)", "item_3": "bottle (trash) at [0, 0, 0]",
            "item_7": "cooler (organ transport) at [1.0, 2.0, 3.0]"}
    assert find_cooler(ents) == "item_7"
    assert find_cooler(WORLD) == "item_5"
    assert find_cooler(WORLD_ENTS) is None


def test_single_scene_aliases_unchanged():
    from mission_fakes import delivery
    assert resolve_entity("cooler", list(delivery())) == "item_0"   # delivery: still item_0, names suffice


# ------------------------------------------------------------------ agents field
def _split_spec(**kw):
    data = {"scene": "world", "reward": HOVER,
            "agents": {"0": {"anchors": {"a0": "item_0.handle", "a1": "truck"}, "reward": "-dist(drone, a0)",
                             "done_when": CARRY_DONE, "explanation": "trash"},
                       1: {"anchors": {"a0": "cooler.handle", "a1": "hospital", "a2": "nofly_0"},
                           "reward": "-dist(payload, a1)", "explanation": "organ",
                           "steps": [{"reward": "holding", "done_when": "holding"},
                                     {"anchors": {"a0": None, "a1": "pad_dropoff"}, "reward": "-dist(payload, a1)"}]}}}
    data.update(kw)
    return MissionSpec.model_validate(data, context={"entities": WORLD})


def test_agents_shape_and_bindings():
    ms = _split_spec()
    assert set(ms.agents) == {0, 1} and isinstance(ms.agents[1], AgentMission)
    assert ms.agents[0].anchors["a1"] == "bin" and ms.agents[1].anchors["a0"] == "item_5.handle"
    assert ms.agents[1].anchors["a1"] == "pad_dropoff"
    b0, b1, b2 = (to_bindings(ms, agent=ag, entities=WORLD) for ag in range(3))
    assert [(b.entity, b.kind) for b in b0[:2]] == [("item_0.handle", "handle"), ("bin", "zone")]
    assert [(b.entity, b.kind) for b in b1[:3]] == [("item_5.handle", "handle"), ("pad_dropoff", "zone"),
                                                   ("nofly_0", "zone")]
    assert all(b.kind == "none" for b in b2)           # drone 2 is not listed: top-level (no anchors)
    assert ms.stage(None, 0).reward == "-dist(drone, a0)" and ms.stage(None, 2).reward == HOVER
    # steps: drone 1 has two, drone 0 none (keeps its top level), past the end -> last step
    assert ms.n_steps(1) == 2 and ms.n_steps(0) == 0
    assert ms.stage(0, 1).anchors["a0"] == "item_5.handle"         # step without anchors inherits drone 1's
    assert ms.stage(1, 1).anchors["a0"] is None and ms.stage(5, 1).reward == "-dist(payload, a1)"
    assert ms.stage(1, 0).reward == "-dist(drone, a0)"
    assert mission_z is not None and "drone 1: organ" in ms.summary()
    labels = [lab for lab, _ in ms.expressions()]
    assert "agents[1].steps[1].reward" in labels and "agents[0].done_when" in labels


def test_agents_bind_mission_uses_descriptions():
    env = FakeEnv("world", WORLD)
    ms = MissionSpec(scene="world", reward=HOVER,
                     agents={1: {"anchors": {"a0": "cooler.handle", "a1": "hospital"}, "reward": "1"}})
    bind_mission(env, ms)   # aliases unresolved so far: bind_mission reads env.entity_descriptions()
    assert env.bound[1][0].entity == "item_5.handle" and env.bound[1][1].entity == "pad_dropoff"
    assert all(b.kind == "none" for b in env.bound[0])


def test_agents_inherit_top_level_and_templates():
    ms = MissionSpec.model_validate({"scene": "world", "anchors": {"a0": "net.corner_{agent}", "a1": "boat"},
                                     "reward": "1", "agents": {2: {"reward": "-dist(drone, a1)"}}},
                                    context={"entities": WORLD})
    assert [to_bindings(ms, agent=ag)[0].entity for ag in range(3)] == ["net.corner_0", "net.corner_1",
                                                                         "net.corner_2"]
    assert ms.stage(None, 2).reward == "-dist(drone, a1)" and ms.stage(None, 1).reward == "1"


def test_agents_validation():
    with pytest.raises(ValidationError):
        MissionSpec(scene="world", reward="1", agents={5: {"reward": "1"}})
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="1", agents={1: {"reward": "1"}})          # trash has one drone
    with pytest.raises(ValidationError):
        MissionSpec(scene="world", reward="1", agents={"x": {"reward": "1"}})
    with pytest.raises(ValidationError):
        MissionSpec(scene="world", reward="1", agents={0: {"reward": "__import__('os')"}})
    with pytest.raises((ValidationError, MissionError)):
        MissionSpec.model_validate({"scene": "world", "reward": "1",
                                    "agents": {0: {"anchors": {"a0": "rocket"}, "reward": "1"}}},
                                   context={"entities": WORLD})
    assert MissionSpec(scene="world", reward="1", agents={"drone_1": {"reward": "1"}}).agents[1].reward == "1"


# ------------------------------------------------------------------ prompt / Gemini (mocked)
def test_world_prompt_and_example():
    p = build_prompt("drone 0 cleans up and drone 1 delivers the organ", "world", WORLD)
    for s in ('"agents"', "Scene: world", "item_5", "cooler (organ transport)", "400 x 400 m", "NOT item_0"):
        assert s in p, s
    ex = [e for e in _EXAMPLES if e[0] == "world"]
    assert ex and "agents" in ex[0][3]
    ms = parse_and_validate(json.dumps(ex[0][3]), ex[0][2], "world", list(ex[0][1]))
    assert ms.agents[0].anchors["a0"] == "item_2.handle" and ms.agents[1].anchors["a0"] == "item_5.handle"


class _Client:
    def __init__(self, out):
        class M:
            def generate_content(self, model, contents, config=None):
                class R:
                    text = out
                return R()
        self.models = M()


def test_compile_world_with_cooler_alias():
    out = {"anchors": {"a0": None}, "reward": HOVER, "explanation": "split",
           "agents": {"1": {"anchors": {"a0": "cooler.handle", "a1": "hospital"}, "reward": "-dist(payload, a1)",
                            "explanation": "organ"}}}
    ms = compile_mission("drone 1 delivers the organ", "world", WORLD, client=_Client(json.dumps(out)))
    assert not ms.offline and ms.agents[1].anchors == {"a0": "item_5.handle", "a1": "pad_dropoff", "a2": None,
                                                       "a3": None}


# ------------------------------------------------------------------ offline: single missions in the world
def test_world_offline_trash_single():
    ms, env = _check("pick up the pizza box and put it in the garbage truck")
    assert ms.agents is None and ms.anchors["a0"] == "item_1.handle" and ms.anchors["a1"] == "bin"
    assert "payload_is('pizza_box')" in ms.reward and ms.done_when == CARRY_DONE
    assert all(env.bound[ag][0].entity == "item_1.handle" for ag in range(3))


def test_world_offline_trash_pile_skips_cooler_and_inactive():
    ms, _ = _check("pick up all the trash")
    assert ms.anchors["a0"] == "nearest:trash" and ms.anchors["a1"] == "bin"
    assert ms.done_when == "trash_left < 0.01 and not holding" and "cooler" in ms.explanation


@pytest.mark.parametrize("text", ["deliver the organ cooler gently to the hospital",
                                  "rush the donor heart to the hospital", "take the cooler to the drop-off pad"])
def test_world_offline_delivery(text):
    ms, env = _check(text)
    assert ms.agents is None
    assert ms.anchors == {"a0": "item_5.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"}
    for term in ("swing", "jerk", "tilt", "5*inside(a2)", "5*inside(a3)", "payload_is('cooler')"):
        assert term in ms.reward, term
    assert ms.done_when == CARRY_DONE and len(ms.steps) == 2
    assert [(b.entity, b.kind) for b in env.bound[0]] == [("item_5.handle", "handle"), ("pad_dropoff", "zone"),
                                                          ("nofly_0", "zone"), ("nofly_1", "zone")]


def test_world_offline_net_haul_all_three_drones():
    ms, env = _check("net the fish and haul them to the boat, keep the dolphins out of the net")
    assert ms.agents is None
    assert ms.anchors["a0"] == "net.corner_{agent}" and ms.anchors["a1"] == "boat"
    assert ms.anchors["a2"] == "fish_school_1"          # nearest school to drone_0
    assert "10*bycatch" in ms.reward and "not bycatch" in ms.done_when and len(ms.steps) == 3
    bind_mission(env, ms)
    assert [env.bound[ag][0].entity for ag in range(3)] == ["net.corner_0", "net.corner_1", "net.corner_2"]
    assert all(env.bound[ag][0].kind == "handle" and env.bound[ag][1].entity == "boat" for ag in range(3))


def test_world_offline_goto():
    ms, _ = _check("fly to the hospital")
    assert ms.anchors["a0"] == "pad_dropoff" and ms.agents is None
    ms, _ = _check("go to the boat")
    assert ms.anchors["a0"] == "boat"


# ------------------------------------------------------------------ offline: per-drone splits
def test_world_offline_two_drone_split():
    ms, env = _check("drone 0 picks up the trash bag and drone 1 delivers the cooler to the hospital")
    assert set(ms.agents) == {0, 1}
    a0, a1 = ms.agents[0], ms.agents[1]
    assert a0.anchors["a0"] == "item_0.handle" and a0.anchors["a1"] == "bin" and a0.done_when == CARRY_DONE
    assert "payload_is('trash_bag')" in a0.reward
    assert a1.anchors == {"a0": "item_5.handle", "a1": "pad_dropoff", "a2": "nofly_0", "a3": "nofly_1"}
    assert "swing" in a1.reward and len(a1.steps) == 2
    assert ms.reward == HOVER and all(v is None for v in ms.anchors.values())
    assert "Drone 0" in ms.explanation and "Drone 1" in ms.explanation and "Drone 2 hovers" in ms.explanation
    bind_mission(env, ms)
    assert env.bound[0][0].entity == "item_0.handle" and env.bound[1][0].entity == "item_5.handle"
    assert all(b.kind == "none" for b in env.bound[2])
    bind_mission(env, ms, step=1)
    assert env.bound[0][1].entity == "bin" and env.bound[1][1].entity == "pad_dropoff"


def test_world_offline_split_uses_each_drones_position():
    ms, _ = _check("Drone 0 picks up some trash, drone 2 picks up some trash")
    assert set(ms.agents) == {0, 2}
    assert ms.agents[0].anchors["a0"] == "item_1.handle"    # nearest trash to drone_0 (the backpack is skipped)
    assert ms.agents[2].anchors["a0"] == "item_2.handle"    # nearest trash to drone_2 (out east)
    assert "Drone 1 hovers" in ms.explanation


def test_world_offline_split_with_net_takes_whole_team():
    ms, env = _check("drone 0 nets the fish and drone 1 delivers the cooler")
    assert ms.agents is None and ms.anchors["a0"] == "net.corner_{agent}" and ms.warnings
    assert [env.bound[ag][0].entity for ag in range(3)] == ["net.corner_0", "net.corner_1", "net.corner_2"]


def test_world_offline_no_split_for_drone_as_target():
    ms, _ = _check("fly to drone 1")
    assert ms.agents is None and ms.anchors["a0"] == "drone_1"
