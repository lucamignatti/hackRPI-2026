"""MissionSpec: what the language layer produces and the operator approves.

Anchors map a0..a3 to an entity name from env.entity_names(), a fixed world
point [x, y, z], or None. An entity name may contain the placeholder
"{agent}", replaced by the drone index when binding, so a team mission can give
every drone its own net corner ("net.corner_{agent}").

Validation against a scene's entity list is done by passing the list as
pydantic validation context (``MissionSpec.model_validate(data,
context={"entities": names})``) or by calling ``spec.check_entities(names)``.
Common aliases ("bag.handle", "cooler", "nofly") are rewritten to the real
entity names when an entity list is available.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .. import spec as S
from .dsl import DSLError, compile_reward

ANCHOR_KEYS = tuple(f"a{k}" for k in range(S.N_ANCHORS))
AnchorValue = Union[str, List[float], None]

_ALIASES = {
    "bag": "payload", "trash": "payload", "trash_bag": "payload", "cooler": "payload", "organ": "payload",
    "box": "payload", "package": "payload",
    "bag.handle": "payload.handle", "trash.handle": "payload.handle", "cooler.handle": "payload.handle",
    "handle": "payload.handle", "payload_handle": "payload.handle",
    "nofly": "nofly_0", "no_fly": "nofly_0", "no-fly": "nofly_0", "nofly_zone": "nofly_0",
    "net": "net.center", "net_center": "net.center", "net.centroid": "net.center",
    "trash_bin": "bin", "drone": "drone_0",
}


class MissionError(ValueError):
    """Mission spec does not fit the scene (unknown entity, bad anchor, ...)."""


def infer_kind(entity: str) -> str:
    """Anchor kind from an entity name: handle, zone or point."""
    if entity.endswith(".handle") or re.fullmatch(r"net\.corner_(\d+|\{agent\})", entity):
        return "handle"
    if entity in ("bin", "boat") or entity.startswith("nofly"):
        return "zone"
    return "point"


def n_agents_from_entities(entities: Iterable[str]) -> int:
    return max(1, sum(1 for e in entities if re.fullmatch(r"drone_\d+", e)))


def resolve_entity(name: str, entities: Sequence[str], agent: int = 0) -> str:
    """Map an anchor entity (alias / {agent} template) to a concrete entity name or raise MissionError."""
    concrete = name.replace("{agent}", str(agent))
    if concrete in entities:
        return concrete
    alias = _ALIASES.get(concrete.lower().replace(" ", "_"))
    if alias is not None and alias in entities:
        return alias
    raise MissionError(f"unknown entity {concrete!r}; known entities: {list(entities)}")


def _normalise_template(name: str, entities: Sequence[str]) -> str:
    """Apply aliases to a (possibly templated) entity name without substituting {agent}."""
    if "{agent}" in name:
        return name
    if name in entities:
        return name
    alias = _ALIASES.get(name.lower().replace(" ", "_"))
    if alias is not None and alias in entities:
        return alias
    return name


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

    @model_validator(mode="after")
    def _entities(self, info: ValidationInfo):
        ctx = info.context if info is not None else None
        if ctx and ctx.get("entities") is not None:
            self.check_entities(ctx["entities"])
        return self

    # -------------------------------------------------------------- helpers
    def anchor_sets(self) -> List[Dict[str, AnchorValue]]:
        out = [self.anchors]
        for st in self.steps or []:
            if st.anchors is not None:
                out.append(st.anchors)
        return out

    def check_entities(self, entities: Sequence[str]) -> "MissionSpec":
        """Raise MissionError if an anchor names an unknown entity; rewrites aliases in place."""
        entities = list(entities)
        n_ag = n_agents_from_entities(entities)
        for anchors in self.anchor_sets():
            for k, val in list(anchors.items()):
                if isinstance(val, str):
                    val = _normalise_template(val, entities)
                    for ag in range(n_ag):
                        try:
                            resolve_entity(val, entities, ag)
                        except MissionError as e:
                            raise MissionError(f"anchor {k}: {e}") from None
                    anchors[k] = val
        return self

    def stage(self, i: Optional[int]) -> "MissionStep":
        """Step i as a MissionStep with anchors resolved (None -> the top-level mission)."""
        if i is None or not self.steps:
            return MissionStep(anchors=self.anchors, anchor_kinds=self.anchor_kinds,
                               anchor_radius=self.anchor_radius, reward=self.reward,
                               done_when=self.done_when, explanation=self.explanation)
        st = self.steps[i]
        return MissionStep(anchors=st.anchors if st.anchors is not None else self.anchors,
                           anchor_kinds=st.anchor_kinds if st.anchor_kinds is not None else self.anchor_kinds,
                           anchor_radius=st.anchor_radius if st.anchor_radius is not None else self.anchor_radius,
                           reward=st.reward, done_when=st.done_when, explanation=st.explanation)

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
        if self.offline:
            lines.append("  (compiled offline, no Gemini)")
        return "\n".join(lines)


def to_bindings(mission: MissionSpec, agent: int = 0, step: Optional[int] = None,
                entities: Optional[Sequence[str]] = None) -> List[S.AnchorBinding]:
    """AnchorBindings for anchors a0..a3 (length N_ANCHORS).

    agent: substituted into "{agent}" templates. step: use steps[step]'s anchors.
    entities: if given, aliases are resolved and unknown names raise MissionError.
    Kinds come from anchor_kinds when set, else are inferred from the entity name
    (".handle" / "net.corner_i" -> handle; bin / boat / nofly_* -> zone; else point).
    A fixed point is a "point", or a "zone" if anchor_radius gives it a radius > 0.
    """
    st = mission.stage(step)
    kinds = st.anchor_kinds or {}
    radii = st.anchor_radius or {}
    out: List[S.AnchorBinding] = []
    for k in ANCHOR_KEYS:
        val = (st.anchors or {}).get(k)
        r = float(radii.get(k, 0.0))
        if val is None:
            out.append(S.AnchorBinding(kind="none"))
        elif isinstance(val, str):
            name = resolve_entity(val, entities, agent) if entities is not None else val.replace("{agent}", str(agent))
            out.append(S.AnchorBinding(kind=kinds.get(k, infer_kind(name)), entity=name, radius=r))
        else:
            kind = kinds.get(k, "zone" if r > 0 else "point")
            out.append(S.AnchorBinding(kind=kind, point=tuple(float(x) for x in val), radius=r))
    return out


def bind_mission(env, mission: MissionSpec, env_ids=None, step: Optional[int] = None) -> None:
    """Bind a mission's anchors in env worlds env_ids (default all), per agent so {agent} templates work."""
    import torch
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ents = env.entity_names()
    for ag in range(env.num_agents):
        env.bind_anchors(env_ids, to_bindings(mission, agent=ag, step=step, entities=ents), agent=ag)
