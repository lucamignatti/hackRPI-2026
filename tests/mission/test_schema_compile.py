from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from skyhand import spec
from skyhand.env.core import make_env
from skyhand.mission import MissionSpec, compile_mission, offline_compile, to_bindings
from skyhand.mission.schema import MissionError, bind_mission

TRASH_ENTS = ["payload.handle", "payload", "bin", "nofly_0"] + [f"obstacle_{i}" for i in range(6)] + ["drone_0"]
OCEAN_ENTS = [f"net.corner_{i}" for i in range(4)] + ["net.center", "boat", "fish_school_0", "fish_school_1",
                                                      "nofly_0", "drone_0", "drone_1", "drone_2"]


@pytest.fixture(scope="module")
def trash_env():
    return make_env("trash", 2, lidar=False)


@pytest.fixture(scope="module")
def ocean_env():
    return make_env("ocean", 2, lidar=False)


def test_schema_unknown_entity():
    data = {"scene": "trash", "anchors": {"a0": "rocket"}, "reward": "-dist(drone, a0)"}
    with pytest.raises((ValidationError, MissionError)):
        MissionSpec.model_validate(data, context={"entities": TRASH_ENTS})
    ms = MissionSpec.model_validate(data)   # no entity list: syntax only
    with pytest.raises(MissionError):
        ms.check_entities(TRASH_ENTS)


def test_schema_bad_fields():
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="speed.real")
    with pytest.raises(ValidationError):
        MissionSpec(scene="mars", reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", anchors={"a7": "bin"}, reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", anchors={"a0": [1, 2]}, reward="speed")
    with pytest.raises(ValidationError):
        MissionSpec(scene="trash", reward="speed", steps=[{"reward": "__import__('os')"}])


def test_aliases_and_templates():
    ms = MissionSpec.model_validate({"scene": "trash", "anchors": {"a0": "bag.handle", "a1": "nofly"},
                                     "reward": "1"}, context={"entities": TRASH_ENTS})
    assert ms.anchors["a0"] == "payload.handle" and ms.anchors["a1"] == "nofly_0"
    ms = MissionSpec.model_validate({"scene": "ocean", "anchors": {"a0": "net.corner_{agent}"}, "reward": "1"},
                                    context={"entities": OCEAN_ENTS})
    assert [to_bindings(ms, agent=i)[0].entity for i in range(3)] == ["net.corner_0", "net.corner_1", "net.corner_2"]


def test_to_bindings_kinds():
    ms = MissionSpec(scene="ocean", anchors={"a0": "net.corner_2", "a1": "boat", "a2": "fish_school_0",
                                             "a3": [1, 2, 3]}, reward="1")
    b = to_bindings(ms)
    assert [x.kind for x in b] == ["handle", "zone", "point", "point"]
    assert b[3].point == (1.0, 2.0, 3.0) and b[3].entity is None
    ms = MissionSpec(scene="trash", anchors={"a0": "payload.handle", "a1": "bin", "a2": "nofly_0", "a3": None},
                     reward="1", anchor_radius={"a3": 2.0})
    assert [x.kind for x in to_bindings(ms)] == ["handle", "zone", "zone", "none"]
    ms2 = MissionSpec(scene="trash", anchors={"a3": [0, 0, 0]}, anchor_radius={"a3": 2.0}, reward="1")
    assert to_bindings(ms2)[3].kind == "zone" and to_bindings(ms2)[3].radius == 2.0
    ms3 = MissionSpec(scene="trash", anchors={"a0": "payload"}, anchor_kinds={"a0": "zone"}, reward="1")
    assert to_bindings(ms3)[0].kind == "zone"


class FakeModels:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents)
        assert config["response_mime_type"] == "application/json"
        txt = self.outputs.pop(0)

        class R:
            text = txt
        return R()


class FakeClient:
    def __init__(self, outputs):
        self.models = FakeModels(outputs)


GOOD = {"anchors": {"a0": "payload.handle", "a1": "bin", "a2": None, "a3": None},
        "reward": "where(holding, -dist(payload, a1), -dist(drone, a0))",
        "done_when": "distxy(payload, a1) < 0.6 and not holding",
        "explanation": "Pick up the bag and drop it in the bin."}


def ents_dict(names):
    return {n: f"{n} at [0.0, 0.0, 0.0]" for n in names}


def test_compile_valid():
    c = FakeClient([json.dumps(GOOD)])
    ms = compile_mission("pick up the bag", "trash", ents_dict(TRASH_ENTS), client=c)
    assert not ms.offline and ms.anchors["a0"] == "payload.handle" and ms.text == "pick up the bag"
    p = c.models.prompts[0]
    assert "payload.handle" in p and "obstacle_dist" in p and "pick up the bag" in p


def test_compile_retry_on_bad_dsl():
    bad = dict(GOOD, reward="__import__('os').system('ls')")
    c = FakeClient(["```json\n" + json.dumps(bad) + "\n```", json.dumps(GOOD)])
    ms = compile_mission("pick up the bag", "trash", ents_dict(TRASH_ENTS), client=c)
    assert not ms.offline and ms.reward == GOOD["reward"]
    assert len(c.models.prompts) == 2 and "rejected" in c.models.prompts[1]


def test_compile_retry_on_unknown_entity():
    bad = dict(GOOD, anchors={"a0": "rocket_launcher"})
    c = FakeClient([json.dumps(bad), "not json at all", json.dumps(GOOD)])
    ms = compile_mission("pick up the bag", "trash", ents_dict(TRASH_ENTS), client=c)
    assert not ms.offline
    assert "rocket_launcher" in c.models.prompts[1] and "JSON" in c.models.prompts[2]


def test_compile_falls_back_offline():
    c = FakeClient([json.dumps(dict(GOOD, reward="nope("))] * 3)
    ms = compile_mission("pick up the trash bag", "trash", ents_dict(TRASH_ENTS), client=c, max_retries=3)
    assert ms.offline and ms.warnings
    with pytest.raises(ValueError):
        compile_mission("x", "trash", ents_dict(TRASH_ENTS), client=FakeClient(["{}"] * 2), max_retries=2,
                        fallback=False)


def test_compile_no_key_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ms = compile_mission("pick up the bag", "trash", ents_dict(TRASH_ENTS))
    assert ms.offline


def _check_offline(env, text):
    ents = env.entity_descriptions()
    ms = offline_compile(text, env.scene, ents)
    assert ms.offline and ms.explanation
    g = torch.randn(8, spec.G_DIM)
    from skyhand.mission import compile_reward
    assert compile_reward(ms.reward)(g).shape == (8,)
    bind_mission(env, ms)
    for i in range(len(ms.steps or [])):
        bind_mission(env, ms, step=i)
    for ag in range(env.num_agents):
        env.bind_anchors(torch.arange(env.num_envs), to_bindings(ms, agent=ag, entities=env.entity_names()), agent=ag)
    obs = env.step(torch.zeros(env.num_envs, env.num_agents, spec.ACTION_DIM)).obs
    assert torch.isfinite(compile_reward(ms.reward)(obs["g"])).all()
    return ms


def test_offline_trash_pickup(trash_env):
    ms = _check_offline(trash_env, "grab the bag and drop it in the bin without getting near the moving carts")
    assert ms.anchors["a0"] == "payload.handle" and ms.anchors["a1"] == "bin"
    assert "obstacle_dist" in ms.reward and ms.steps


def test_offline_organ(trash_env):
    ms = _check_offline(trash_env, "deliver the organ cooler gently and avoid the no-fly zone")
    assert "swing" in ms.reward and "inside(a2)" in ms.reward and ms.anchors["a2"] == "nofly_0"
    assert to_bindings(ms)[2].kind == "zone"


def test_offline_net(ocean_env):
    ms = _check_offline(ocean_env, "net the school closest to the boat")
    assert ms.anchors["a0"] == "net.corner_{agent}" and ms.anchors["a1"] == "boat"
    assert ms.anchors["a2"].startswith("fish_school")
    assert int(ocean_env.anc_src[0, 1, 0]) == ocean_env.entity_names().index("net.corner_1")


def test_offline_goto_avoid(trash_env, ocean_env):
    ms = _check_offline(trash_env, "fly to obstacle 3 and avoid the no-fly zone")
    assert ms.anchors["a0"] == "obstacle_3" and "nofly_0" in ms.anchors.values()
    ms = _check_offline(ocean_env, "go to the boat")
    assert ms.anchors["a0"] == "boat"
    ms = _check_offline(trash_env, "go to (2, -3) at 4 m")
    assert ms.anchors["a0"] == [2.0, -3.0, 4.0]
    ms = _check_offline(trash_env, "do a backflip")
    assert "not understood" in ms.explanation
