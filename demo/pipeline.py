"""Adapter between the demo server and the (possibly missing) mission / training packages.

Everything external is imported lazily and every piece has a labelled fallback,
so the demo always runs and never misrepresents what is flying:

  policy:  "trained policy"            skyhand.train load(path, device) -> agent.act(obs, z)
           "scripted fallback (env)"   env.expert_action()
           "scripted fallback"         ScriptedController below (fly to a0, grip, fly to a1, drop)
  mission: "Gemini" / "offline parser"  skyhand.mission compile_mission / offline_compile
           "keyword parser"            keyword_compile below
  z:       reward_to_z from skyhand.mission, else the same formula implemented here.

Tests inject fakes through Pipeline(agent=..., mission_api=...).
"""
from __future__ import annotations

import importlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from skyhand import spec
from skyhand.spec import AnchorBinding

from . import rewardlang

log = logging.getLogger("skyhand.demo")

POLICY_TRAINED = "trained policy"
POLICY_ENV_EXPERT = "scripted fallback (env expert)"
POLICY_SCRIPTED = "scripted fallback"
MISSION_GEMINI = "Gemini"
MISSION_OFFLINE = "offline parser"
MISSION_KEYWORD = "keyword parser"


# ------------------------------------------------------------------ mission
@dataclass
class CompiledMission:
    text: str
    scene: str
    anchors: Dict[str, Any]                  # "a0" -> entity name | [x, y, z] | None
    reward: str
    done_when: Optional[str]
    explanation: List[str]
    source: str                              # MISSION_* label
    reward_fn: Callable[[torch.Tensor], torch.Tensor] = field(repr=False, default=None)
    done_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = field(repr=False, default=None)
    bindings: List[AnchorBinding] = field(repr=False, default_factory=list)
    raw: Any = field(repr=False, default=None)
    per_agent_bindings: Optional[Dict[int, List[AnchorBinding]]] = field(repr=False, default=None)
    warnings: List[str] = field(default_factory=list)

    def public(self, descriptions: Optional[Dict[str, str]] = None) -> dict:
        descriptions = descriptions or {}
        anchors = {}
        for k, v in self.anchors.items():
            if v is None:
                continue
            if isinstance(v, str) and "{agent}" in v:
                d0 = descriptions.get(v.replace("{agent}", "0"), "")
                desc = f"each drone its own ({v.replace('{agent}', 'i')} for drone i); drone 0: {d0}"
            elif isinstance(v, str):
                desc = descriptions.get(v, "")
            else:
                desc = f"fixed point {[round(float(x), 2) for x in v]}"
            anchors[k] = {"target": v if isinstance(v, str) else [float(x) for x in v], "description": desc}
        return {"text": self.text, "scene": self.scene, "anchors": anchors, "reward": self.reward,
                "done_when": self.done_when, "explanation": self.explanation, "source": self.source,
                "warnings": self.warnings}


def _binding_for(target, entity_kinds: Dict[str, str]) -> AnchorBinding:
    if target is None:
        return AnchorBinding(kind="none")
    if isinstance(target, str):
        return AnchorBinding(kind=entity_kinds.get(target, "point"), entity=target)
    return AnchorBinding(kind="point", point=[float(x) for x in target])


def bindings_from_anchors(anchors: Dict[str, Any], entity_kinds: Dict[str, str]) -> List[AnchorBinding]:
    return [_binding_for(anchors.get(f"a{k}"), entity_kinds) for k in range(spec.N_ANCHORS)]


def entity_kinds(env) -> Dict[str, str]:
    return {n: spec.ANCHOR_TYPES[int(k)] for n, k in zip(env.entity_names(), env._ent_kind.tolist())}


def _has(text: str, *words) -> bool:
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


def _closest(env, names: Sequence[str], to: str) -> str:
    try:
        pos = env._entity_positions()[0]
        all_names = env.entity_names()
        p0 = pos[all_names.index(to)]
        return min(names, key=lambda n: float((pos[all_names.index(n)] - p0)[:2].norm()))
    except Exception:
        return names[0]


def keyword_compile(text: str, scene: str, env) -> CompiledMission:
    """Tiny keyword parser: picks one of a few templates and adds modifiers."""
    t = text.lower()
    names = env.entity_names()
    anchors: Dict[str, Any] = {f"a{k}": None for k in range(spec.N_ANCHORS)}
    terms: List[str] = []
    done = None
    nofly_active = bool(env.zone_active[0, 1])
    if scene == "trash":
        reach_only = _has(t, "go to", "fly to", "hover", "visit", "touch") and not _has(t, "grab", "pick", "carry", "drop", "deliver", "bring")
        if reach_only:
            target = "bin" if _has(t, "bin", "can") else "payload.handle" if _has(t, "bag", "trash", "cooler", "payload") else \
                "nofly_0" if _has(t, "no-fly", "nofly") else "bin"
            anchors["a0"] = target
            terms.append("-dist(drone, a0)")
            done = "dist(drone, a0) < 1.0"
        else:
            anchors["a0"], anchors["a1"] = "payload.handle", "bin"
            terms += ["-dist(drone, a0)*(1 - holding)", "-dist(payload, a1)", "2*holding"]
            done = "not holding and dist(payload, a1) < 0.7"
        if _has(t, "cooler", "organ", "medical", "heart", "smooth", "gentle", "careful", "fragile", "steady"):
            terms += ["-0.5*swing", "-0.01*jerk"]
        if _has(t, "no-fly", "nofly", "restricted", "hospital", "airspace", "organ") and nofly_active:
            anchors["a2"] = "nofly_0"
            terms.append("-5*inside(a2)")
        if _has(t, "avoid", "obstacle", "cart", "moving", "careful", "safe", "without hitting", "near"):
            terms.append("-2*(obstacle_dist < 1.0)")
    else:
        if _has(t, "go to", "fly to", "hover", "visit") and not _has(t, "net", "fish", "haul"):
            anchors["a0"] = "boat" if _has(t, "boat") else _closest(env, [n for n in names if n.startswith("fish_school")], "boat")
            terms.append("-dist(drone, a0)")
            done = "dist(drone, a0) < 1.5"
        else:
            schools = [n for n in names if n.startswith("fish_school")]
            m = re.search(r"school\s*(\d)", t)
            school = f"fish_school_{m.group(1)}" if m and f"fish_school_{m.group(1)}" in names else \
                _closest(env, schools, "boat") if _has(t, "closest", "nearest") else schools[0]
            anchors["a0"], anchors["a1"], anchors["a2"] = "net.corner_0", "boat", school
            terms += ["-dist(drone, a0)*(1 - holding)", "2*holding", "-0.5*dist(payload, a2)*holding",
                      "3*fish_in_net", "-dist(payload, a1)*fish_in_net"]
            done = "fish_in_net > 0.2 and dist(payload, a1) < 3.0"
        if _has(t, "gentle", "careful", "tear", "smooth"):
            terms.append("-2*net_stretch")
        if _has(t, "no-fly", "nofly", "restricted") and nofly_active:
            anchors["a3"] = "nofly_0"
            terms.append("-5*inside(a3)")
        if _has(t, "avoid", "collide", "safe", "teammate"):
            terms.append("-2*(mate_dist < 1.0)")
    reward = _join_terms(terms)
    return _finish(text, scene, anchors, reward, done, None, MISSION_KEYWORD, env)


def _join_terms(terms: Sequence[str]) -> str:
    out = ""
    for t in terms:
        t = t.strip()
        if not out:
            out = t
        elif t.startswith("-"):
            out += " - " + t[1:].strip()
        else:
            out += " + " + t
    return out or "0"


def _finish(text, scene, anchors, reward, done, explanation, source, env, reward_fn=None, done_fn=None,
            bindings=None, raw=None) -> CompiledMission:
    shown = {k: (v if isinstance(v, str) else "fixed point") for k, v in anchors.items() if v is not None}
    expl = rewardlang.explain(reward, done, shown, strict=bool(explanation))
    if explanation:
        expl = ([explanation] if isinstance(explanation, str) else list(explanation)) + expl
    return CompiledMission(
        text=text, scene=scene, anchors=anchors, reward=reward, done_when=done, explanation=expl, source=source,
        reward_fn=reward_fn or rewardlang.compile_expr(reward),
        done_fn=done_fn if done_fn is not None else (rewardlang.compile_expr(done) if done else None),
        bindings=bindings or bindings_from_anchors(anchors, entity_kinds(env)), raw=raw)


# ------------------------------------------------------------------ external APIs
def _import_attr(names: Sequence[str], modules: Sequence[str]) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for mod in modules:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for n in names:
            if n not in found and callable(getattr(m, n, None)):
                found[n] = getattr(m, n)
    return found


MISSION_NAMES = ("compile_mission", "offline_compile", "compile_reward", "to_bindings", "reward_to_z", "cem_refine")


def load_mission_api() -> Dict[str, Any]:
    return _import_attr(MISSION_NAMES, ["skyhand.mission", "skyhand.mission.compile", "skyhand.mission.compiler",
                                        "skyhand.mission.gemini", "skyhand.mission.reward", "skyhand.mission.language",
                                        "skyhand.mission.ztask", "skyhand.mission.zinfer", "skyhand.mission.cem"])


def load_agent(path: Optional[str], device: str = "cpu"):
    """Return the trained agent or None (with a logged reason)."""
    if not path:
        return None
    if not os.path.exists(path):
        log.warning("checkpoint %s not found; using scripted fallback", path)
        return None
    api = _import_attr(["load"], ["skyhand.train", "skyhand.train.agent", "skyhand.train.fb", "skyhand.train.fbcpr"])
    if "load" not in api:
        try:
            from skyhand.train.agent import Agent  # type: ignore
            api["load"] = Agent.load
        except Exception:
            pass
    if "load" not in api:
        log.warning("skyhand.train has no load(); using scripted fallback")
        return None
    try:
        return api["load"](path, device)
    except Exception as e:  # pragma: no cover - depends on other package
        log.warning("loading %s failed (%s); using scripted fallback", path, e)
        return None


def _accepts(fn, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ------------------------------------------------------------------ fallback reward -> z
def reward_to_z_fallback(reward_fn, B, g_samples: torch.Tensor, d: int) -> torch.Tensor:
    """z = sqrt(d) * normalize(mean_i r(g_i) B(g_i))  (PLAN.md 6.3)."""
    with torch.no_grad():
        r = reward_fn(g_samples).float().reshape(-1)
        b = B(g_samples).float().reshape(r.shape[0], -1)
        z = (r.unsqueeze(-1) * b).mean(0)
        return math.sqrt(d) * z / z.norm().clamp_min(1e-8)


def pca_2d(zs: np.ndarray) -> np.ndarray:
    """Project rows of zs [M, d] to 2D with PCA via SVD. Returns [M, 2]."""
    zs = np.asarray(zs, dtype=np.float64)
    if zs.ndim != 2 or zs.shape[0] == 0:
        return np.zeros((0, 2))
    x = zs - zs.mean(0, keepdims=True)
    if zs.shape[0] == 1:
        return np.zeros((1, 2))
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2]
    out = x @ comps.T
    if out.shape[1] < 2:
        out = np.concatenate([out, np.zeros((out.shape[0], 2 - out.shape[1]))], 1)
    return out


# Named reference missions for the z map (the three highlighted missions + simple ones).
REFERENCE_MISSIONS = [
    {"name": "Trash pickup", "scene": "trash", "anchors": {"a0": "payload.handle", "a1": "bin"},
     "reward": "-dist(drone, a0)*(1 - holding) - dist(payload, a1) + 2*holding - 2*(obstacle_dist < 1.0)"},
    {"name": "Organ delivery", "scene": "trash", "anchors": {"a0": "payload.handle", "a1": "bin", "a2": "nofly_0"},
     "reward": "-dist(drone, a0)*(1 - holding) - dist(payload, a1) + 2*holding - 0.5*swing - 0.01*jerk - 5*inside(a2)"},
    {"name": "Net fishing", "scene": "ocean", "anchors": {"a0": "net.corner_0", "a1": "boat", "a2": "fish_school_0"},
     "reward": "-dist(drone, a0)*(1 - holding) + 2*holding + 3*fish_in_net - dist(payload, a1)*fish_in_net"},
    {"name": "Go to bin", "scene": "trash", "anchors": {"a0": "bin"}, "reward": "-dist(drone, a0)"},
    {"name": "Hover at 3 m", "scene": "trash", "anchors": {}, "reward": "-abs(height - 3) - 0.1*speed"},
    {"name": "Fly fast", "scene": "trash", "anchors": {}, "reward": "speed - 2*collision"},
    {"name": "Go to boat", "scene": "ocean", "anchors": {"a0": "boat"}, "reward": "-dist(drone, a0)"},
]


# ------------------------------------------------------------------ scripted controller
class ScriptedController:
    """Fly above a0, descend and grip, carry to above a1, lower and release.

    Uses only actor observations (obs["anchors"], obs["proprio"]). Phases per
    [world, agent]: 0 approach a0 at cruise height, 1 descend + grip, 2 carry to
    a1 at cruise height, 3 lower over a1, 4 released / climb and hover.
    """

    def __init__(self, scene: str, num_envs: int, num_agents: int, device="cpu"):
        self.scene = scene
        self.cruise = 4.5 if scene == "trash" else 2.5
        self.drop_h = 1.2 if scene == "trash" else 1.5
        self.min_h = 0.15 if scene == "trash" else 0.45
        self.grab_r = 0.08 if scene == "trash" else 0.2
        self.phase = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
        self.miss = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)
        self.stage = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.phase.zero_()
            self.miss.zero_()
            self.stage.zero_()
        else:
            self.phase[env_ids] = 0
            self.miss[env_ids] = 0
            self.stage[env_ids] = 0

    def __call__(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        anc, pro = obs["anchors"], obs["proprio"]
        dev = anc.device
        self.phase, self.miss, self.stage = self.phase.to(dev), self.miss.to(dev), self.stage.to(dev)
        rel0, rel1 = anc[..., 0, 0:3], anc[..., 1, 0:3]
        has0, has1 = anc[..., 0, 3] < 0.5, anc[..., 1, 3] < 0.5       # one-hot index 0 is "none"
        vel, grip_closed, holding, height = pro[..., 0:3], pro[..., 9] > 0.5, pro[..., 10] > 0.5, pro[..., 11]
        speed = vel.norm(dim=-1)
        ph = self.phase
        off = torch.tensor([0.0, 0.0, -spec.GRIPPER_OFFSET[2]], device=dev)
        grip_rel = rel0 + off                                          # where the drone must go to put gripper on a0
        xy0, xy1 = rel0[..., :2].norm(dim=-1), rel1[..., :2].norm(dim=-1)
        # transitions
        ph = torch.where((ph == 0) & has0 & (xy0 < 0.4), torch.ones_like(ph), ph)
        ph = torch.where((ph == 1) & (xy0 > 1.2), torch.zeros_like(ph), ph)
        ph = torch.where((ph <= 1) & holding, torch.full_like(ph, 2), ph)
        ph = torch.where((ph >= 2) & (ph <= 3) & ~holding, torch.where(has0, torch.zeros_like(ph), torch.full_like(ph, 4)), ph)
        if self.scene == "trash":
            ph = torch.where((ph == 2) & has1 & (xy1 < 0.35) & (speed < 0.6), torch.full_like(ph, 3), ph)
        # targets (relative vectors the drone should move along); in transit each agent
        # flies at its own altitude so paths never cross at the same height
        lane = torch.arange(anc.shape[1], device=dev, dtype=height.dtype).unsqueeze(0) * 1.2
        cruise_dz = (self.cruise + lane - height)
        zeros = torch.zeros_like(height)
        t_hover = torch.stack([zeros, zeros, cruise_dz], -1)
        t0 = torch.cat([rel0[..., :2], cruise_dz.unsqueeze(-1)], -1)
        dz1 = torch.maximum(grip_rel[..., 2], self.min_h - height)
        t1 = torch.cat([grip_rel[..., :2], dz1.unsqueeze(-1)], -1)
        if self.scene == "trash":
            t2 = torch.cat([rel1[..., :2], cruise_dz.unsqueeze(-1)], -1)
            t3 = torch.cat([rel1[..., :2], (self.drop_h - height).unsqueeze(-1)], -1)
        else:
            # ocean haul: move the whole net (its centroid) over a2 (fish), then over a1 (boat),
            # all drones with the same horizontal velocity, once every live drone holds a corner.
            cen = spec.g_field(obs["g"], "net_centroid_rel") if "g" in obs else torch.zeros_like(rel0)
            rel2 = anc[..., 2, 0:3]
            has2 = anc[..., 2, 3] < 0.5
            team_ok = holding.all(-1, keepdim=True) if "mates" not in obs else \
                (holding & ((obs["mates"][..., 6] > 0.5) | ~obs["mates_mask"]).all(-1)).all(-1, keepdim=True).expand_as(holding)
            to_fish = (rel2 - cen)[..., :2]
            to_boat = (rel1 - cen)[..., :2]
            fish_done = (~has2) | (to_fish.norm(dim=-1) < 1.0) | (self.stage == 1)
            self.stage = torch.where(fish_done & team_ok & (ph == 2), torch.ones_like(self.stage), self.stage)
            dxy = torch.where((self.stage == 0).unsqueeze(-1), to_fish, to_boat)
            dxy = torch.where(team_ok.unsqueeze(-1), dxy, torch.zeros_like(dxy))
            haul_dz = (self.cruise - height)
            t2 = torch.cat([dxy, haul_dz.unsqueeze(-1)], -1)
            t3 = t2
            if "mates" in obs:   # keep apart from teammates
                m_rel = obs["mates"][..., 0:3]
                dm = m_rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)
                push = (-(m_rel / dm) * (2.5 - dm).clamp_min(0.0) * 3.0 * obs["mates_mask"].unsqueeze(-1).float()).sum(-2)
                t0 = t0 + push
                t1 = t1 + push * (height > self.min_h + 0.3).unsqueeze(-1).float()
                t2 = t2 + push
        tgt = torch.where((ph == 0).unsqueeze(-1), t0, t_hover)
        tgt = torch.where((ph == 1).unsqueeze(-1), t1, tgt)
        tgt = torch.where((ph == 2).unsqueeze(-1), t2, tgt)
        tgt = torch.where((ph == 3).unsqueeze(-1), t3, tgt)
        tgt = torch.where(((ph == 0) & ~has0).unsqueeze(-1), t_hover, tgt)
        if self.scene == "trash":
            tgt = torch.where(((ph == 2) & ~has1).unsqueeze(-1), t_hover, tgt)
        # gripper
        near = (ph == 1) & (grip_rel.norm(dim=-1) < self.grab_r) & (speed < 0.4)
        if self.scene != "trash":   # corners may sit at/under the surface: grab from the lowest safe height
            near = (ph == 1) & (grip_rel[..., :2].norm(dim=-1) < 0.15) & (height < self.min_h + 0.1) & (speed < 0.5)
        self.miss = torch.where(grip_closed & ~holding & (ph == 1), self.miss + 1, torch.zeros_like(self.miss))
        retry = self.miss >= 4
        close = (near & ~retry) | (holding & (ph >= 2) & (ph <= 3))
        release = (ph == 3) & (height < self.drop_h + 0.15) & (xy1 < 0.35) & (speed < 0.35)
        if self.scene != "trash":
            release = torch.zeros_like(release)     # ocean: keep hauling the corner
        close = close & ~release
        ph = torch.where(release, torch.full_like(ph, 4), ph)
        self.miss = torch.where(retry, torch.zeros_like(self.miss), self.miss)
        self.phase = ph
        # velocity command
        vmax_xy = torch.where(ph == 1, torch.full_like(height, 1.0), torch.full_like(height, 3.0))
        vmax_xy = torch.where(ph == 3, torch.full_like(height, 1.0), vmax_xy)
        if self.scene != "trash":
            vmax_xy = torch.where(ph >= 2, torch.full_like(height, 1.2), vmax_xy)
        v = 1.2 * tgt
        vxy = v[..., :2]
        n = vxy.norm(dim=-1, keepdim=True)
        vxy = vxy * torch.clamp(vmax_xy.unsqueeze(-1) / n.clamp_min(1e-6), max=1.0)
        vz_max = torch.where(ph == 1, torch.full_like(height, 0.6), torch.full_like(height, 2.0))
        vz = torch.maximum(torch.minimum(v[..., 2], vz_max), -vz_max)
        act = torch.zeros(anc.shape[:2] + (spec.ACTION_DIM,), device=dev)
        act[..., 0] = vxy[..., 0] / spec.MAX_SPEED
        act[..., 1] = vxy[..., 1] / spec.MAX_SPEED
        act[..., 2] = vz / spec.MAX_VZ
        act[..., 4] = torch.where(close, torch.ones_like(height), -torch.ones_like(height))
        return act.clamp(-1.0, 1.0)


# ------------------------------------------------------------------ pipeline
class Pipeline:
    """One object the server talks to. Inject `agent` / `mission_api` in tests."""

    def __init__(self, agent=None, mission_api: Optional[Dict[str, Any]] = None, device: str = "cpu",
                 use_env_expert: bool = False, z_samples_envs: int = 32, z_samples_steps: int = 60,
                 cem: bool = False):
        self.agent = agent
        self.mission_api = load_mission_api() if mission_api is None else mission_api
        self.device = device
        self.use_env_expert = use_env_expert
        self.z_samples_envs = z_samples_envs
        self.z_samples_steps = z_samples_steps
        self.cem = cem
        self._sample_cache: Dict[Any, torch.Tensor] = {}
        self._zmap_cache: Optional[Dict[str, np.ndarray]] = None
        self._env_expert_ok: Optional[bool] = None

    # -------------------------------------------------- labels
    @property
    def policy_mode(self) -> str:
        if self.agent is not None:
            return POLICY_TRAINED
        if self._env_expert_ok:
            return POLICY_ENV_EXPERT
        return POLICY_SCRIPTED

    @property
    def mission_mode(self) -> str:
        if "compile_mission" in self.mission_api:
            return MISSION_GEMINI if _gemini_key() else MISSION_OFFLINE   # per-mission label comes from spec.offline
        if "offline_compile" in self.mission_api:
            return MISSION_OFFLINE
        return MISSION_KEYWORD

    # -------------------------------------------------- mission
    def compile(self, text: str, scene: str, env) -> CompiledMission:
        api = self.mission_api
        desc = env.entity_descriptions()
        for name in ("compile_mission", "offline_compile"):
            if name not in api:
                continue
            try:
                ms = api[name](text, scene, desc)
                label = MISSION_OFFLINE if (name == "offline_compile" or _get(ms, "offline", False)) else MISSION_GEMINI
                return self._from_spec(ms, text, scene, env, label)
            except Exception as e:
                log.warning("%s failed (%s: %s)", name, type(e).__name__, e)
        return keyword_compile(text, scene, env)

    def _from_spec(self, ms, text, scene, env, label) -> CompiledMission:
        src = _get(ms, "source")
        if isinstance(src, str) and src:
            label = MISSION_GEMINI if "gemini" in src.lower() else MISSION_OFFLINE if "offline" in src.lower() else src
        anchors = dict(_get(ms, "anchors", {}) or {})
        anchors = {k: (v if (v is None or isinstance(v, str)) else list(v)) for k, v in anchors.items()}
        for k in range(spec.N_ANCHORS):
            anchors.setdefault(f"a{k}", None)
        reward = str(_get(ms, "reward"))
        done = _get(ms, "done_when") or None
        api = self.mission_api
        reward_fn = done_fn = bindings = None
        if "compile_reward" in api:
            reward_fn = api["compile_reward"](reward)
            done_fn = api["compile_reward"](done) if done else None
        binder = None
        if "to_bindings" in api:
            ents = env.entity_names()
            per_agent = {ag: list(api["to_bindings"](ms, agent=ag, entities=ents)) for ag in range(env.num_agents)} \
                if _accepts(api["to_bindings"], "agent") else None
            bindings = per_agent[0] if per_agent else list(api["to_bindings"](ms))
            if per_agent:
                binder = per_agent
        warnings = list(_get(ms, "warnings", []) or [])
        out = _finish(text, scene, anchors, reward, done, _get(ms, "explanation"), label, env,
                      reward_fn=reward_fn, done_fn=done_fn, bindings=bindings, raw=ms)
        out.per_agent_bindings = binder
        out.warnings = warnings
        return out

    def bind(self, env, mission: CompiledMission, env_ids) -> None:
        """Bind anchors. Mission-package specs bind per agent (their {agent} templates);
        keyword-parser specs in the ocean give each drone its own net corner (corner k -> k + agent)."""
        if mission.per_agent_bindings:
            for ag, b in mission.per_agent_bindings.items():
                env.bind_anchors(env_ids, b, agent=ag)
            return
        b = mission.bindings
        if env.num_agents > 1 and any(x.entity and x.entity.startswith("net.corner_") for x in b):
            for ag in range(env.num_agents):
                per = []
                for x in b:
                    if x.entity and x.entity.startswith("net.corner_"):
                        c = (int(x.entity.rsplit("_", 1)[1]) + ag) % 4
                        x = AnchorBinding(kind=x.kind, entity=f"net.corner_{c}", radius=x.radius)
                    per.append(x)
                env.bind_anchors(env_ids, per, agent=ag)
        else:
            env.bind_anchors(env_ids, b)

    # -------------------------------------------------- policy
    def make_controller(self, env) -> Callable[[Dict[str, torch.Tensor], Optional[torch.Tensor]], torch.Tensor]:
        """Return act(obs, z) for this env."""
        if self.agent is not None:
            agent = self.agent

            def act_trained(obs, z):
                with torch.no_grad():
                    if z is None:
                        z = torch.zeros(int(agent.d), device=env.device)
                    return agent.act(obs, z, noise=0.0)
            return act_trained
        scripted = ScriptedController(env.scene, env.num_envs, env.num_agents, env.device)
        if self.use_env_expert and self._env_expert_ok is not False:
            try:
                env.expert_action()
                self._env_expert_ok = True
            except Exception as e:
                log.info("env.expert_action unavailable (%s: %s); using built-in scripted controller", type(e).__name__, e)
                self._env_expert_ok = False

        def act_scripted(obs, z):
            if self._env_expert_ok:
                try:
                    return env.expert_action()
                except Exception:
                    self._env_expert_ok = False
            return scripted(obs)
        act_scripted.reset = scripted.reset  # type: ignore[attr-defined]
        return act_scripted

    # -------------------------------------------------- z
    def _g_samples(self, scene: str, anchors: Dict[str, Any], make_env) -> torch.Tensor:
        key = (scene, tuple(sorted((k, str(v)) for k, v in anchors.items() if v is not None)))
        if key in self._sample_cache:
            return self._sample_cache[key]
        env = make_env(scene, self.z_samples_envs)
        mission = _finish("", scene, {f"a{k}": anchors.get(f"a{k}") for k in range(spec.N_ANCHORS)},
                          "0", None, None, "", env)
        ids = torch.arange(env.num_envs, device=env.device)
        self.bind(env, mission, ids)
        ctrl = ScriptedController(scene, env.num_envs, env.num_agents, env.device)
        obs = env._obs()
        gs = []
        gen = torch.Generator().manual_seed(1)
        for _ in range(self.z_samples_steps):
            act = ctrl(obs) + 0.3 * torch.randn(env.num_envs, env.num_agents, spec.ACTION_DIM, generator=gen).to(env.device)
            out = env.step(act)
            obs = out.obs
            gs.append(obs["g"].reshape(-1, spec.G_DIM).clone())
            dead = out.done | out.truncated
            if dead.any():
                rid = torch.nonzero(dead).squeeze(-1)
                obs = env.reset(rid)
                self.bind(env, mission, rid)
                ctrl.reset(rid)
                obs = env._obs()
        g = torch.cat(gs)
        self._sample_cache[key] = g
        return g

    def compute_z(self, reward_fn, scene: str, anchors: Dict[str, Any], make_env) -> Optional[torch.Tensor]:
        if self.agent is None:
            return None
        g = self._g_samples(scene, anchors, make_env).to(self.device)
        B = self.agent.encode_g
        if "reward_to_z" in self.mission_api:
            try:
                return torch.as_tensor(self.mission_api["reward_to_z"](reward_fn, B, g, int(self.agent.d))).float().reshape(-1)
            except Exception as e:
                log.warning("reward_to_z failed (%s); using built-in formula", e)
        return reward_to_z_fallback(reward_fn, B, g, int(self.agent.d)).reshape(-1)

    def refine_z(self, env, z0, reward_fn, done_fn=None, n_candidates: Optional[int] = None):
        """CEM refinement in clones of world 0 (needs env.num_envs > 1 and --cem)."""
        if not (self.cem and self.agent is not None and "cem_refine" in self.mission_api and env.num_envs > 1):
            return z0, False
        try:
            agent = self.agent
            nc = n_candidates or env.num_envs - 1
            out = self.mission_api["cem_refine"](env, lambda obs, z: agent.act(obs, z, noise=0.0), z0, reward_fn,
                                                 src_env=0, n_candidates=nc, done_fn=done_fn,
                                                 dst_ids=list(range(1, nc + 1)))
            z = out[0] if isinstance(out, tuple) else out
            return torch.as_tensor(z).float().reshape(-1), True
        except Exception as e:
            log.warning("cem_refine failed (%s); keeping z_r", e)
            return z0, False

    def reference_z(self, make_env) -> Dict[str, np.ndarray]:
        if self.agent is None:
            return {}
        if self._zmap_cache is None:
            out = {}
            for m in REFERENCE_MISSIONS:
                try:
                    comp = self.mission_api.get("compile_reward", rewardlang.compile_expr)
                    z = self.compute_z(comp(m["reward"]), m["scene"], m["anchors"], make_env)
                    out[m["name"]] = z.detach().cpu().numpy()
                except Exception as e:
                    log.warning("reference z for %s failed: %s", m["name"], e)
            self._zmap_cache = out
        return self._zmap_cache

    def zmap(self, make_env, current: Optional[torch.Tensor], current_name: str = "your mission") -> dict:
        if self.agent is None:
            return {"available": False, "points": [], "note": "z map available after training"}
        ref = self.reference_z(make_env)
        names = list(ref)
        zs = [ref[n] for n in names]
        if current is not None:
            names.append(current_name)
            zs.append(current.detach().cpu().numpy().reshape(-1))
        if not zs:
            return {"available": False, "points": [], "note": "no reference missions"}
        p = pca_2d(np.stack(zs))
        scale = max(float(np.abs(p).max()), 1e-6)
        pts = [{"name": n, "x": float(x / scale), "y": float(y / scale), "current": (current is not None and i == len(names) - 1)}
               for i, (n, (x, y)) in enumerate(zip(names, p))]
        return {"available": True, "points": pts, "note": ""}
