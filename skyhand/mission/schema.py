"""MissionSpec: what the language layer produces and the operator approves.

Anchors map a0..a3 to an entity name from env.entity_names(), a fixed world
point [x, y, z], or None. An entity name may contain the placeholder
"{agent}", replaced by the drone index when binding, so a team mission can give
every drone its own net corner ("net.corner_{agent}").

Validation against a scene's entity list is done by passing the list as
pydantic validation context (``MissionSpec.model_validate(data,
context={"entities": names})``) or by calling ``spec.check_entities(names)``.
Common aliases ("truck" -> "bin", "cooler" -> "item_0" in delivery, "hospital"
-> "pad_dropoff", "nofly" -> "nofly_0", "item 3 handle" -> "item_3.handle") are
rewritten to the real entity names when an entity list is available.

Wherever an entity list is accepted, a mapping name -> description (as from
env.entity_descriptions()) works too. The world scene needs it to resolve
"cooler" / "organ": there the cooler is whichever item_i is described as a cooler.

Multi-drone scenes (ocean, world) can give each drone its own job with the
optional ``agents`` field: {drone index: AgentMission}. A drone listed there
uses its own anchors / reward / done_when / steps; drones not listed use the
top-level mission. Without ``agents`` every drone shares the top-level anchors
("{agent}" templates give each its own net corner).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .. import spec as S
from .dsl import DSLError, compile_reward

ANCHOR_KEYS = tuple(f"a{k}" for k in range(S.N_ANCHORS))
AnchorValue = Union[str, List[float], None]

# Aliases are applied only when the target exists in the scene's entity list. Keys are lower-case with
# spaces / dashes turned into underscores. Per-scene tables win over the common one.
_COMMON_ALIASES = {
    "nofly": "nofly_0", "no_fly": "nofly_0", "nofly_zone": "nofly_0", "no_fly_zone": "nofly_0",
    "restricted_zone": "nofly_0", "restricted_area": "nofly_0", "drone": "drone_0",
}
_SCENE_ALIASES: Dict[str, Dict[str, str]] = {
    "trash": {
        "truck": "bin", "garbage_truck": "bin", "trash_truck": "bin", "dump_truck": "bin", "trash_bin": "bin",
        "garbage_bin": "bin", "trash_can": "bin", "garbage_can": "bin", "dumpster": "bin", "can": "bin",
        "dropoff": "bin", "drop_off": "bin", "drop_zone": "bin",
    },
    "delivery": {
        "cooler": "item_0", "organ": "item_0", "organ_cooler": "item_0", "package": "item_0", "payload": "item_0",
        "cooler.handle": "item_0.handle", "organ.handle": "item_0.handle", "payload.handle": "item_0.handle",
        "handle": "item_0.handle", "cooler_handle": "item_0.handle",
        "hospital": "pad_dropoff", "helipad": "pad_dropoff", "dropoff": "pad_dropoff", "drop_off": "pad_dropoff",
        "dropoff_pad": "pad_dropoff", "destination": "pad_dropoff", "landing_pad": "pad_dropoff",
        "drop_zone": "pad_dropoff", "bin": "pad_dropoff",
        "pickup": "pad_pickup", "pick_up": "pad_pickup", "pickup_pad": "pad_pickup", "origin": "pad_pickup",
        "start": "pad_pickup", "donor_hospital": "pad_pickup",
    },
    "ocean": {
        "net": "net.center", "net_center": "net.center", "net.centroid": "net.center", "ship": "boat",
        "vessel": "boat", "deck": "boat", "fish": "fish_school_0", "school": "fish_school_0",
        "fish_school": "fish_school_0", "whale": "whale_0", "dolphin": "dolphin_0", "turtle": "turtle_0",
    },
}



def _world_aliases() -> Dict[str, str]:
    """Union of the single-scene tables. Delivery's item_0 aliases are left out (the world's cooler is found by
    type, see _COOLER_KEYS), and keys the scenes disagree on ("dropoff" = truck or hospital pad) are dropped."""
    out: Dict[str, str] = {}
    clash = set()
    for sc in ("trash", "delivery", "ocean"):
        for k, v in _SCENE_ALIASES[sc].items():
            if v.startswith("item_0") or (sc == "delivery" and k == "bin"):
                continue
            if k in out and out[k] != v:
                clash.add(k)
            out[k] = v
    for k in clash:
        out.pop(k)
    return out


_SCENE_ALIASES["world"] = _world_aliases()
# Aliases naming the organ cooler. In the world they resolve to the item described as a cooler (or its handle);
# the generic payload words ("payload", "handle") are not used there since the world has many items.
_COOLER_KEYS = {k: v.endswith(".handle") for k, v in _SCENE_ALIASES["delivery"].items()
                if v.startswith("item_0") and k not in ("payload", "payload.handle", "handle")}
MAX_AGENTS = max(S.SCENE_AGENTS.values())


def find_cooler(entities) -> Optional[str]:
    """The item_i described as a cooler (active ones first), from a name -> description mapping; else None."""
    if not isinstance(entities, Mapping):
        return None
    found = []
    for name, desc in entities.items():
        m = re.fullmatch(r"item_(\d+)", name)
        if m and re.search(r"\bcooler\b", str(desc).lower()):
            found.append((re.search(r"\binactive\b", str(desc).lower()) is not None, int(m.group(1)), name))
    return sorted(found)[0][2] if found else None


class MissionError(ValueError):
    """Mission spec does not fit the scene (unknown entity, bad anchor, ...)."""


def infer_kind(entity: str) -> str:
    """Anchor kind from an entity name: handle, zone or point.

    ".handle" names and net.corner_i -> handle; bin, boat, pad_*, nofly_* -> zone; everything else -> point.
    """
    if entity.endswith(".handle") or re.fullmatch(r"net\.corner_(\d+|\{agent\})", entity):
        return "handle"
    if entity in ("bin", "boat") or entity.startswith("nofly") or entity.startswith("pad_"):
        return "zone"
    return "point"


def n_agents_from_entities(entities: Iterable[str]) -> int:
    return max(1, sum(1 for e in entities if re.fullmatch(r"drone_\d+", e)))


def scene_from_entities(entities: Iterable[str]) -> str:
    """Guess the scene from its entity names (used for scene-specific aliases)."""
    ents = set(entities)
    pads = "pad_pickup" in ents or "pad_dropoff" in ents
    sea = "boat" in ents or "net.center" in ents or any(e.startswith("net.corner") for e in ents)
    if pads and sea:
        return "world"
    if pads:
        return "delivery"
    if "boat" in ents or "net.center" in ents or any(e.startswith("net.corner") for e in ents):
        return "ocean"
    return "trash"


def _alias(name: str, entities: Sequence[str], scene: Optional[str] = None) -> Optional[str]:
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    m = re.fullmatch(r"item_?#?(\d+)(?:[._]?(handle))?", key)
    if m:
        key = f"item_{m.group(1)}" + (".handle" if m.group(2) else "")
        if key in entities:
            return key
    scenes = [scene] if scene else []
    scenes.append(scene_from_entities(entities))   # the entity list wins when the scene field is off
    if "world" in scenes and key in _COOLER_KEYS:
        cooler = find_cooler(entities)
        if cooler is None:
            return None
        tgt = cooler + ".handle" if _COOLER_KEYS[key] else cooler
        return tgt if tgt in entities else (cooler if cooler in entities else None)
    for table in [_SCENE_ALIASES.get(sc, {}) for sc in scenes] + [_COMMON_ALIASES]:
        tgt = table.get(key)
        if tgt is not None and tgt in entities:
            return tgt
    return None


def resolve_entity(name: str, entities: Sequence[str], agent: int = 0, scene: Optional[str] = None) -> str:
    """Map an anchor entity (alias / {agent} template) to a concrete entity name or raise MissionError."""
    concrete = name.replace("{agent}", str(agent))
    if concrete in entities:
        return concrete
    alias = _alias(concrete, entities, scene)
    if alias is not None:
        return alias
    hint = ""
    if re.sub(r"[\s\-]+", "_", concrete.strip().lower()) in _COOLER_KEYS and not isinstance(entities, Mapping):
        hint = " (to resolve the cooler in the world scene pass entity descriptions, not just names)"
    raise MissionError(f"unknown entity {concrete!r}{hint}; known entities: {list(entities)}")


def _normalise_template(name: str, entities: Sequence[str], scene: Optional[str] = None) -> str:
    """Apply aliases to a (possibly templated) entity name without substituting {agent}."""
    if "{agent}" in name or name in entities:
        return name
    alias = _alias(name, entities, scene)
    return alias if alias is not None else name


def _check_dsl(v: Optional[str], what: str) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    try:
        compile_reward(v)
    except DSLError as e:
        raise ValueError(f"{what}: {e}") from None
    return v


def _check_anchor_dict(v):
    if v is None:
        return {k: None for k in ANCHOR_KEYS}
    if not isinstance(v, dict):
        raise ValueError("anchors must be an object mapping a0..a3 to an entity name, [x, y, z] or null")
    out = {}
    for k, val in v.items():
        if k not in ANCHOR_KEYS:
            raise ValueError(f"anchor key {k!r} not in {list(ANCHOR_KEYS)}")
        if isinstance(val, str):
            val = val.strip()
            if val.lower() in ("", "none", "null"):
                val = None
        elif isinstance(val, (list, tuple)):
            if len(val) != 3:
                raise ValueError(f"anchor {k}: a point needs exactly 3 numbers, got {val!r}")
            try:
                val = [float(x) for x in val]
            except (TypeError, ValueError):
                raise ValueError(f"anchor {k}: point coordinates must be numbers, got {val!r}") from None
        elif val is not None:
            raise ValueError(f"anchor {k}: expected entity name, [x, y, z] or null, got {val!r}")
        out[k] = val
    for k in ANCHOR_KEYS:
        out.setdefault(k, None)
    return out


def _check_kinds(v):
    if v is None:
        return None
    for k, kind in v.items():
        if k not in ANCHOR_KEYS:
            raise ValueError(f"anchor_kinds key {k!r} not in {list(ANCHOR_KEYS)}")
        if kind not in S.ANCHOR_TYPES:
            raise ValueError(f"anchor_kinds[{k}]={kind!r} not in {list(S.ANCHOR_TYPES)}")
    return v


class MissionStep(BaseModel):
    """One stage of a chained mission. anchors=None inherits the parent's anchors."""
    anchors: Optional[Dict[str, AnchorValue]] = None
    anchor_kinds: Optional[Dict[str, str]] = None
    anchor_radius: Optional[Dict[str, float]] = None
    reward: str
    done_when: Optional[str] = None
    explanation: str = ""

    @field_validator("anchors", mode="before")
    @classmethod
    def _anchors(cls, v):
        return None if v is None else _check_anchor_dict(v)

    @field_validator("anchor_kinds")
    @classmethod
    def _kinds(cls, v):
        return _check_kinds(v)

    @field_validator("reward")
    @classmethod
    def _reward(cls, v):
        return _check_dsl(v, "reward")

    @field_validator("done_when")
    @classmethod
    def _done(cls, v):
        return _check_dsl(v, "done_when")


class AgentMission(MissionStep):
    """One drone's own mission inside a multi-drone MissionSpec (see MissionSpec.agents).

    anchors=None inherits the top-level anchors. steps: optional chain, as for the top-level mission.
    """
    steps: Optional[List[MissionStep]] = None


class MissionSpec(BaseModel):
    text: str = ""
    scene: str = "trash"
    anchors: Dict[str, AnchorValue] = Field(default_factory=lambda: {k: None for k in ANCHOR_KEYS})
    anchor_kinds: Optional[Dict[str, str]] = None
    anchor_radius: Optional[Dict[str, float]] = None   # zone radius for fixed points / overrides
    reward: str
    done_when: Optional[str] = None
    explanation: str = ""
    steps: Optional[List[MissionStep]] = None
    agents: Optional[Dict[int, AgentMission]] = None    # per-drone jobs {drone index: sub-mission}
    offline: bool = False                               # True: produced by offline_compile
    warnings: List[str] = Field(default_factory=list)

    @field_validator("scene")
    @classmethod
    def _scene(cls, v):
        if v not in S.SCENES:
            raise ValueError(f"scene {v!r} not in {list(S.SCENES)}")
        return v

    @field_validator("anchors", mode="before")
    @classmethod
    def _anchors(cls, v):
        return _check_anchor_dict(v)

    @field_validator("anchor_kinds")
    @classmethod
    def _kinds(cls, v):
        return _check_kinds(v)

    @field_validator("reward")
    @classmethod
    def _reward(cls, v):
        return _check_dsl(v, "reward")

    @field_validator("done_when")
    @classmethod
    def _done(cls, v):
        return _check_dsl(v, "done_when")

    @field_validator("agents", mode="before")
    @classmethod
    def _agents(cls, v):
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("agents must be an object mapping a drone index (0, 1, ...) to a sub-mission")
        out = {}
        for k, sub in v.items():
            try:
                ag = int(str(k).strip().replace("drone_", ""))
            except ValueError:
                raise ValueError(f"agents key {k!r} is not a drone index") from None
            if not 0 <= ag < MAX_AGENTS:
                raise ValueError(f"agents key {k!r} not in 0..{MAX_AGENTS - 1}")
            out[ag] = sub
        return out or None

    @model_validator(mode="after")
    def _entities(self, info: ValidationInfo):
        if self.agents:
            n = S.SCENE_AGENTS.get(self.scene, MAX_AGENTS)
            bad = [ag for ag in self.agents if ag >= n]
            if bad:
                raise ValueError(f"agents {bad}: the {self.scene} scene has {n} drone(s) (indices 0..{n - 1})")
        ctx = info.context if info is not None else None
        if ctx and ctx.get("entities") is not None:
            self.check_entities(ctx["entities"])
        return self

    # -------------------------------------------------------------- helpers
    def _anchor_sets_for(self, agent: Optional[int]) -> List[Dict[str, AnchorValue]]:
        if agent is not None and self.agents and agent in self.agents:
            sub = self.agents[agent]
            out = [sub.anchors] if sub.anchors is not None else []
            steps = sub.steps
        else:
            out, steps = [self.anchors], self.steps
        out += [st.anchors for st in steps or [] if st.anchors is not None]
        return out

    def anchor_sets(self) -> List[Dict[str, AnchorValue]]:
        """Every anchor dict in the mission: top-level, its steps, then each drone's sub-mission and steps."""
        out = self._anchor_sets_for(None)
        for ag in sorted(self.agents or {}):
            out += self._anchor_sets_for(ag)
        return out

    def check_entities(self, entities) -> "MissionSpec":
        """Raise MissionError if an anchor names an unknown entity; rewrites aliases in place.

        entities: entity names, or a name -> description mapping (needed for the world's cooler)."""
        if not isinstance(entities, Mapping):
            entities = list(entities)
        n_ag = n_agents_from_entities(entities)
        if self.agents:
            bad = [ag for ag in self.agents if ag >= n_ag]
            if bad:
                raise MissionError(f"agents {bad}: this world has {n_ag} drone(s)")
        groups = [(self._anchor_sets_for(None), range(n_ag))]
        groups += [(self._anchor_sets_for(ag), [ag]) for ag in sorted(self.agents or {})]
        for sets, ags in groups:
            for anchors in sets:
                for k, val in list(anchors.items()):
                    if isinstance(val, str):
                        val = _normalise_template(val, entities, self.scene)
                        for ag in ags:
                            try:
                                resolve_entity(val, entities, ag, self.scene)
                            except MissionError as e:
                                who = f"agents[{ag}] " if self.agents and ag in self.agents else ""
                                raise MissionError(f"{who}anchor {k}: {e}") from None
                        anchors[k] = val
        return self

    def stage(self, i: Optional[int] = None, agent: Optional[int] = None) -> "MissionStep":
        """Step i of drone `agent`'s mission as a MissionStep with anchors resolved.

        i=None -> the (sub-)mission's top level. agent=None, or a drone not in `agents`, -> the shared top-level
        mission. A drone with its own sub-mission uses that; since drones in a split can have different numbers of
        steps, an i past the end of a sub-mission's steps gives its last step (that drone has finished).
        """
        if agent is not None and self.agents and agent in self.agents:
            sub = self.agents[agent]
            inherit = sub.anchors is None
            top = MissionStep(anchors=self.anchors if inherit else sub.anchors,
                              anchor_kinds=sub.anchor_kinds if sub.anchor_kinds is not None or not inherit
                              else self.anchor_kinds,
                              anchor_radius=sub.anchor_radius if sub.anchor_radius is not None or not inherit
                              else self.anchor_radius,
                              reward=sub.reward, done_when=sub.done_when, explanation=sub.explanation)
            steps = sub.steps
            if steps and i is not None:
                i = min(i, len(steps) - 1)
        else:
            top = MissionStep(anchors=self.anchors, anchor_kinds=self.anchor_kinds,
                              anchor_radius=self.anchor_radius, reward=self.reward,
                              done_when=self.done_when, explanation=self.explanation)
            steps = self.steps
            if steps and i is not None and self.agents:
                i = min(i, len(steps) - 1)
        if i is None or not steps:
            return top
        st = steps[i]
        return MissionStep(anchors=st.anchors if st.anchors is not None else top.anchors,
                           anchor_kinds=st.anchor_kinds if st.anchor_kinds is not None else top.anchor_kinds,
                           anchor_radius=st.anchor_radius if st.anchor_radius is not None else top.anchor_radius,
                           reward=st.reward, done_when=st.done_when, explanation=st.explanation)

    def n_steps(self, agent: Optional[int] = None) -> int:
        """Number of steps in drone `agent`'s mission (0 = no chain)."""
        if agent is not None and self.agents and agent in self.agents:
            return len(self.agents[agent].steps or [])
        return len(self.steps or [])

    def expressions(self) -> List[tuple]:
        """(label, expression) for every reward / done_when in the mission, None ones included."""
        out = [("reward", self.reward), ("done_when", self.done_when)]
        for i, st in enumerate(self.steps or []):
            out += [(f"steps[{i}].reward", st.reward), (f"steps[{i}].done_when", st.done_when)]
        for ag in sorted(self.agents or {}):
            sub = self.agents[ag]
            out += [(f"agents[{ag}].reward", sub.reward), (f"agents[{ag}].done_when", sub.done_when)]
            for i, st in enumerate(sub.steps or []):
                out += [(f"agents[{ag}].steps[{i}].reward", st.reward),
                        (f"agents[{ag}].steps[{i}].done_when", st.done_when)]
        return out

    def summary(self) -> str:
        """Plain-language text for the approval screen."""
        lines = [self.explanation or "(no explanation)"]
        for k in ANCHOR_KEYS:
            v = self.anchors.get(k)
            if v is not None:
                lines.append(f"  {k} = {v}")
        lines.append(f"  reward: {self.reward}")
        if self.done_when:
            lines.append(f"  done when: {self.done_when}")
        for i, st in enumerate(self.steps or []):
            lines.append(f"  step {i + 1}: {st.explanation or st.reward}")
        for ag in sorted(self.agents or {}):
            sub = self.agents[ag]
            lines.append(f"  drone {ag}: {sub.explanation or sub.reward}")
            for k in ANCHOR_KEYS:
                v = (sub.anchors or {}).get(k)
                if v is not None:
                    lines.append(f"    {k} = {v}")
            for i, st in enumerate(sub.steps or []):
                lines.append(f"    step {i + 1}: {st.explanation or st.reward}")
        if self.offline:
            lines.append("  (compiled offline, no Gemini)")
        return "\n".join(lines)


def to_bindings(mission: MissionSpec, agent: int = 0, step: Optional[int] = None,
                entities=None) -> List[S.AnchorBinding]:
    """AnchorBindings for anchors a0..a3 (length N_ANCHORS) of drone `agent`.

    agent: the drone; picks its sub-mission when mission.agents lists it, and is substituted into "{agent}"
    templates. step: use that step's anchors (see MissionSpec.stage).
    entities: names or a name -> description mapping; if given, aliases are resolved and unknown names raise
    MissionError.
    Kinds come from anchor_kinds when set, else are inferred from the entity name
    (".handle" / "net.corner_i" -> handle; bin / boat / pad_* / nofly_* -> zone; else point).
    A fixed point is a "point", or a "zone" if anchor_radius gives it a radius > 0.
    """
    st = mission.stage(step, agent)
    kinds = st.anchor_kinds or {}
    radii = st.anchor_radius or {}
    out: List[S.AnchorBinding] = []
    for k in ANCHOR_KEYS:
        val = (st.anchors or {}).get(k)
        r = float(radii.get(k, 0.0))
        if val is None:
            out.append(S.AnchorBinding(kind="none"))
        elif isinstance(val, str):
            name = resolve_entity(val, entities, agent, mission.scene) if entities is not None else val.replace("{agent}", str(agent))
            out.append(S.AnchorBinding(kind=kinds.get(k, infer_kind(name)), entity=name, radius=r))
        else:
            kind = kinds.get(k, "zone" if r > 0 else "point")
            out.append(S.AnchorBinding(kind=kind, point=tuple(float(x) for x in val), radius=r))
    return out


def bind_mission(env, mission: MissionSpec, env_ids=None, step: Optional[int] = None) -> None:
    """Bind a mission's anchors in env worlds env_ids (default all), per agent so {agent} templates and per-drone
    sub-missions (mission.agents) work. Uses env.entity_descriptions() when the env has it (world cooler)."""
    import torch
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ents = env.entity_names()
    describe = getattr(env, "entity_descriptions", None)
    if callable(describe):
        try:
            descs = describe()
            if isinstance(descs, Mapping) and set(descs) >= set(ents):
                ents = descs
        except Exception:   # descriptions are a convenience; names are enough for most missions
            pass
    for ag in range(env.num_agents):
        env.bind_anchors(env_ids, to_bindings(mission, agent=ag, step=step, entities=ents), agent=ag)
