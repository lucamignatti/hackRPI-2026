"""SkyHand judge demo server.

    python -m demo.server --port 8000 --host 0.0.0.0 --backend torch --device cpu \
        --checkpoint runs/x/agent.pt

The server owns one env (display world 0, plus clone worlds when --cem is on), steps
it at 20 Hz in real time in a worker thread, and streams env.render_state(0) plus
demo extras over /ws. Judges compile a mission, approve it, and perturb the world.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import torch
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from skyhand import spec

from .pipeline import CompiledMission, Pipeline, load_agent, observe

log = logging.getLogger("skyhand.demo")
WEB_DIR = Path(__file__).resolve().parent / "web"
PERTURB_KINDS = ("drop_obstacle", "gust", "waves", "move_payload", "disable_drone", "tide")
SIM_REAL = "simulator"
SIM_FAKE = "kinematic preview"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
MAX_VOICE_BYTES = 10 * 1024 * 1024
VOICE_TYPES = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/x-m4a": "m4a",
}


async def transcribe_elevenlabs(audio: bytes, content_type: str) -> dict:
    """Send one push-to-talk recording to Scribe v2; the API key never reaches the browser."""
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    mime = content_type.split(";", 1)[0].strip().lower()
    extension = VOICE_TYPES.get(mime)
    if extension is None:
        raise ValueError(f"unsupported audio type: {mime or 'missing'}")
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                data={"model_id": "scribe_v2", "tag_audio_events": "false"},
                files={"file": (f"mission.{extension}", audio, mime)},
            )
    except httpx.RequestError as exc:
        raise RuntimeError("could not reach ElevenLabs") from exc
    if response.is_error:
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("message")
        except (ValueError, AttributeError):
            detail = response.text
        detail = str(detail or response.reason_phrase)[:300]
        raise RuntimeError(f"ElevenLabs transcription failed ({response.status_code}): {detail}")
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("ElevenLabs returned an invalid response") from exc
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("ElevenLabs returned an empty transcript")
    # A stable response shape keeps provider details out of the mission compiler.
    return {"text": text, "language": result.get("language_code"),
            "provider": "ElevenLabs", "model": "scribe_v2"}


def default_make_env(backend: str = "torch", device: str = "cpu", lidar: bool = True, fake: bool = False, **env_kw):
    """make_env(scene, num_envs, **kw) factory. fake=True uses demo.fake_env (no physics)."""
    if fake:
        from .fake_env import make_fake_env as make_env
    else:
        from skyhand.env.core import make_env

    def mk(scene: str, num_envs: int, **kw):
        opts = dict(env_kw)
        opts.update(kw)
        opts.setdefault("lidar", lidar)
        opts.setdefault("random_anchors", False)
        if fake:
            return make_env(scene, num_envs=num_envs, device=device, **opts)
        return make_env(scene, num_envs=num_envs, device=device, backend=backend, **opts)
    return mk


def gusts_enabled(env) -> bool:
    """Whether env.perturb(..., "gust") has any effect (the env ignores gusts unless enabled)."""
    if hasattr(env, "gusts_enabled"):
        return bool(env.gusts_enabled)
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return False
    if not getattr(cfg, "domain_rand", True):
        return False
    return bool(getattr(cfg, "enable_gusts", False))


def _item_index(item) -> Optional[int]:
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return int(item)
    m = str(item).split(".")[0]
    if m.startswith("item_") and m[5:].isdigit():
        return int(m[5:])
    raise ValueError(f"item must be an index or item_<i>, got {item!r}")


class DemoSession:
    """All demo state. Thread-safe: every env access holds self.lock."""

    def __init__(self, scene: str = "trash", num_envs: int = 1, pipeline: Optional[Pipeline] = None,
                 make_env: Optional[Callable] = None, sample_make_env: Optional[Callable] = None,
                 seed: Optional[int] = None, max_steps: Optional[int] = None):
        self.lock = threading.RLock()
        self.pipeline = pipeline or Pipeline()
        self.make_env = make_env or default_make_env()
        # z sampling env: no LiDAR (B only reads g), anchors bound by the pipeline
        self.sample_make_env = sample_make_env or (lambda sc, n: self.make_env(sc, n, lidar=False))
        self.num_envs = max(1, int(num_envs))
        self.seed = seed
        self._max_steps_arg = max_steps          # None: the env's episode_steps (per scene)
        self._episode = 0
        self.zmap: Dict[str, Any] = {"available": False, "points": [], "note": "z map available after training"}
        self._build_id = 0
        self._build(scene)

    # ------------------------------------------------------------ setup
    def _build(self, scene: str) -> None:
        if scene not in spec.SCENES:
            raise ValueError(f"scene must be one of {spec.SCENES}")
        kw = {} if self.seed is None else {"seed": self.seed}
        if self.seed is None:
            kw["seed"] = int(time.time()) % 100000
        self.scene = scene
        self.env = self.make_env(scene, self.num_envs, **kw)
        self.obs = observe(self.env)
        self._build_id += 1
        self._terrain: Optional[dict] = None       # last full terrain (with heights), "key" = build:version
        self._sent_key: Optional[str] = None       # terrain key of the last broadcast frame
        self._render_kw = 0                         # which render_state keyword this env accepts
        self.gusts = gusts_enabled(self.env)
        self.arena_half = float(getattr(self.env, "arena_half", spec.SCENE_ARENA_HALF.get(scene, spec.ARENA_HALF)))
        self.max_steps = int(self._max_steps_arg or getattr(self.env, "episode_steps",
                                                           spec.SCENE_EPISODE_STEPS.get(scene, spec.EPISODE_STEPS)))
        self.sim = SIM_FAKE if getattr(self.env, "is_fake", False) else SIM_REAL
        self.controller = self.pipeline.make_controller(self.env)
        self.pending: Optional[CompiledMission] = None
        self.active: Optional[CompiledMission] = None
        self.z: Optional[torch.Tensor] = None
        self.flying = False
        self.status = "idle"
        self.message = "Type a mission and press Compile."
        self.step_count = 0
        self.current_segment_idx = 0
        self.segment_step_count = 0
        self.reward: Optional[float] = None
        self.reward_sum = 0.0
        self.cem_used = False
        self.last_result: Optional[str] = None
        self.done_agents: set = set()              # drones whose own done_when held (per-drone missions)
        self.zmap = self.pipeline.zmap(self.sample_make_env, None) if self.pipeline.agent is not None else \
            {"available": False, "points": [], "note": "z map available after training"}

    def set_scene(self, scene: str) -> dict:
        with self.lock:
            self._build(scene)
        return self.info()

    # ------------------------------------------------------------ mission
    def compile(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty mission")
        with self.lock:
            env, scene = self.env, self.scene
            desc = env.entity_descriptions()
        m = self.pipeline.compile(text, scene, env)
        with self.lock:
            self.pending = m
            self.status = "awaiting approval" if not self.flying else self.status
            self.message = "Review the compiled reward, then Approve or Cancel."
        return m.public(desc)

    def cancel(self) -> dict:
        with self.lock:
            self.pending = None
            if not self.flying:
                self.status, self.message = "idle", "Mission cancelled."
        return self.info()

    def approve(self) -> dict:
        with self.lock:
            m = self.pending
            if m is None:
                raise ValueError("no compiled mission to approve")
            self.current_segment_idx = 0
            self.segment_step_count = 0
            ids = torch.arange(self.env.num_envs, device=self.env.device)
            if m.agents or not (m.segments and len(m.segments) > 0):
                self.pipeline.bind(self.env, m, ids)
            else:
                first_seg = m.segments[0]
                if first_seg and first_seg.bindings:
                    self.pipeline.bind(self.env, first_seg, ids)
                else:
                    self.pipeline.bind(self.env, m, ids)
            self.flying = False
            self.status, self.message = "computing z", "Computing the task vector z from the reward..."
            scene = self.scene
            episode = self._episode = self._episode + 1
        z = None
        if self.pipeline.agent is not None:
            if m.agents:
                z = self.pipeline.compute_z(m.reward_fn, scene, m.anchors, self.sample_make_env, mission=m, method="argmax")
            elif m.segments and len(m.segments) > 1:
                self.pipeline.compute_segment_zs(m, scene, self.sample_make_env, method="argmax")
                z = m.segments[0].z
            else:
                z = self.pipeline.compute_z(m.reward_fn, scene, m.anchors, self.sample_make_env, mission=m, method="argmax")
                if m.segments:
                    m.segments[0].z = z
            with self.lock:
                if self.pipeline.cem and self.env.num_envs > 1:
                    self.status, self.message = "refining z", "Refining z with CEM in cloned worlds..."
                    z, self.cem_used = self.pipeline.refine_z(self.env, z, m.reward_fn, m.done_fn)
                    self.pipeline.bind(self.env, m, ids)
            zmap = self.pipeline.zmap(self.sample_make_env, z)
        else:
            zmap = {"available": False, "points": [], "note": "z map available after training"}
        with self.lock:
            if episode != self._episode or self.scene != scene:
                return self.info()      # superseded by a reset / scene switch
            self.active, self.pending, self.z, self.zmap = m, None, z, zmap
            self.obs = observe(self.env)
            if hasattr(self.controller, "reset"):
                self.controller.reset()
            self.step_count, self.segment_step_count, self.reward_sum, self.reward = 0, 0, 0.0, None
            self.done_agents = set()
            self.flying, self.status = True, "flying"
            self.last_result = None
            seg_prefix = f"[Segment 1/{len(m.segments)}: {m.segments[0].text}] " if (m.segments and len(m.segments) > 1) else ""
            self.message = f"{seg_prefix}Flying: {m.text}"
        return self.info()

    def stop(self) -> dict:
        with self.lock:
            self._episode += 1
            if self.flying or self.status in ("computing z", "refining z"):
                self.flying, self.status, self.message = False, "stopped", "Stopped by operator."
        return self.info()

    def reset(self) -> dict:
        with self.lock:
            self._episode += 1
            self.obs = self.env.reset()
            if self.active is not None:
                self.pipeline.bind(self.env, self.active, torch.arange(self.env.num_envs, device=self.env.device))
                self.obs = observe(self.env)
            if hasattr(self.controller, "reset"):
                self.controller.reset()
            self.flying = False
            self.step_count, self.current_segment_idx, self.segment_step_count = 0, 0, 0
            self.reward, self.reward_sum = 0.0, 0.0
            self.done_agents = set()
            self.status, self.message = "idle", "World reset. Compile a mission (or re-approve) to fly."
            if self.active is not None and self.pending is None:
                self.pending = self.active
                self.status = "awaiting approval"
        return self.info()

    # ------------------------------------------------------------ perturbations
    def perturb(self, kind: str, params: Dict[str, Any]) -> dict:
        if kind not in PERTURB_KINDS:
            raise ValueError(f"unknown perturbation {kind!r}; one of {PERTURB_KINDS}")
        kw: Dict[str, Any] = {}
        if kind in ("drop_obstacle", "move_payload"):
            pos = params.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                raise ValueError("pos [x, y] required")
            lim = self.arena_half
            kw["pos"] = [max(-lim, min(lim, float(pos[0]))), max(-lim, min(lim, float(pos[1])))]
            if kind == "drop_obstacle" and "half" in params:
                kw["half"] = max(0.2, min(2.0, float(params["half"])))
            if kind == "move_payload":
                idx = _item_index(params.get("item"))
                kw["item"] = self._default_item() if idx is None else max(0, min(spec.MAX_ITEMS - 1, idx))
        elif kind == "gust":
            vec = params.get("vec")
            if vec is None:
                ang = float(params.get("angle", torch.rand(1).item() * 2 * math.pi))
                mag = float(params.get("strength", 5.0))
                vec = [mag * math.cos(ang), mag * math.sin(ang), 0.0]
            kw["vec"] = [max(-12.0, min(12.0, float(v))) for v in list(vec)[:3]] + [0.0] * (3 - len(list(vec)[:3]))
            kw["duration"] = max(0.1, min(10.0, float(params.get("duration", 2.0))))
        elif kind == "waves":
            kw["scale"] = max(0.0, min(4.0, float(params.get("scale", 1.5))))
        elif kind == "tide":
            kw["offset"] = max(-2.0, min(2.0, float(params.get("offset", 0.0))))
        elif kind == "disable_drone":
            ag = int(params.get("agent", 0))
            if not 0 <= ag < self.env.num_agents:
                raise ValueError(f"agent must be in [0, {self.env.num_agents})")
            kw["agent"] = ag
        with self.lock:
            note = ""
            if kind == "waves" and self.scene not in ("ocean", "world"):
                note = "waves only exist in the ocean and world scenes"
            elif kind == "tide" and self.scene == "ocean":
                note = "the tide only moves lakes in terrain worlds (use the wave slider in the ocean)"
            elif kind == "move_payload" and self.scene == "ocean":
                note = "no items in the ocean scene"
            elif kind == "gust" and not self.gusts:
                note = "gusts are off in this env (start the server with --gusts)"
            self.env.perturb(0, kind, **kw)
            self.obs = observe(self.env)
        return {"ok": True, "kind": kind, "params": kw, "note": note}

    def _default_item(self) -> int:
        """Item moved by move_payload when none is given: the one a0 is bound to, else item 0."""
        m = self.active or self.pending
        a0 = m.anchors.get("a0") if m is not None else None
        try:
            idx = _item_index(a0) if isinstance(a0, str) and a0.startswith("item_") else None
        except ValueError:
            idx = None
        return 0 if idx is None else idx

    # ------------------------------------------------------------ simulation
    def _idle_action(self) -> torch.Tensor:
        a = torch.zeros(self.env.num_envs, self.env.num_agents, spec.ACTION_DIM, device=self.env.device)
        a[..., 4] = -1.0
        return a

    def tick(self) -> None:
        """Advance the sim one policy step (20 Hz). Called from the sim thread."""
        with self.lock:
            env = self.env
            if self.flying:
                act = self.controller(self.obs, self.z)
                if self.done_agents and self.active is not None and self.active.agents:
                    act = self._hold_done(act)
            else:
                act = self._idle_action()
            out = env.step(act)
            self.obs = out.obs
            if not self.flying:
                return
            self.step_count += 1
            self.segment_step_count += 1
            m = self.active
            g0 = out.obs["g"][0]
            seg = m.segments[self.current_segment_idx] if (m.segments and self.current_segment_idx < len(m.segments)) else None
            try:
                if m.agent_reward_fns:   # per-drone jobs: mean of each drone's own reward
                    rs = [float(fn(g0[ag:ag + 1]).reshape(-1)[0]) for ag, fn in m.agent_reward_fns.items()
                          if ag < g0.shape[0]]
                    self.reward = sum(rs) / max(len(rs), 1)
                elif seg is not None and seg.reward_fn is not None:
                    self.reward = float(seg.reward_fn(g0[0:1]).reshape(-1)[0])
                else:
                    self.reward = float(m.reward_fn(g0[0:1]).reshape(-1)[0])
                self.reward_sum += self.reward

                # Check segment detector
                ok = False
                if m.agent_done_fns:
                    ok = self._done(m, g0)
                elif seg is not None and seg.done_fn is not None:
                    ok = bool((seg.done_fn(g0).reshape(-1) > 0.5).any())
                elif m.done_fn is not None:
                    ok = bool((m.done_fn(g0).reshape(-1) > 0.5).any())

                if not ok and not m.agents and seg is not None and seg.max_steps is not None and self.segment_step_count >= seg.max_steps:
                    ok = True
            except Exception as e:  # never kill the loop on a bad program
                log.warning("reward evaluation failed: %s", e)
                ok = False
            delivered = self._is_carry(m) and "delivered" in out.events and bool(out.events["delivered"][0].any())

            # Auto-switch to next segment if detector fired and more segments remain
            if not m.agents and (ok or (delivered and self.current_segment_idx < len(m.segments) - 1)) and m.segments and self.current_segment_idx < len(m.segments) - 1:
                self.current_segment_idx += 1
                self.segment_step_count = 0
                self.done_agents.clear()
                nxt = m.segments[self.current_segment_idx]
                ids = torch.arange(env.num_envs, device=env.device)
                if nxt.bindings:
                    self.pipeline.bind(env, nxt, ids)
                if nxt.z is not None:
                    self.z = nxt.z
                if hasattr(self.controller, "reset"):
                    self.controller.reset()
                self.message = f"Segment {self.current_segment_idx + 1}/{len(m.segments)}: {nxt.text}"
                log.info("Transitioned to segment %d/%d: %s", self.current_segment_idx + 1, len(m.segments), nxt.text)
            elif ok:
                who = f" (all {len(m.segments)} segments complete)" if (m.segments and len(m.segments) > 1 and not m.agents) else (" (every drone reached its goal)" if m.agent_done_fns else "")
                self._finish("success", f"Mission complete in {self.step_count * spec.POLICY_DT:.1f} s{who}.")
            elif delivered:
                self._finish("success", f"Item delivered in {self.step_count * spec.POLICY_DT:.1f} s.")
            elif bool(out.done[0]):
                why = "collision" if bool(out.events["collision"][0].any()) else "crash / out of bounds"
                self._finish("failure", f"Episode ended: {why}.")
            elif self.step_count >= self.max_steps:
                self._finish("timeout", f"Time limit ({self.max_steps * spec.POLICY_DT:.0f} s) reached without completing.")

    def _hold_done(self, act: torch.Tensor) -> torch.Tensor:
        """A drone that met its own goal (per-drone jobs) holds position while the others work:
        zero velocity / yaw command, gripper unchanged."""
        act = act.clone()
        for ag in self.done_agents:
            if ag < act.shape[1]:
                act[:, ag, 0:4] = 0.0
        return act

    def _done(self, m: CompiledMission, g0: torch.Tensor) -> bool:
        """Shared mission: any drone meets done_when. Per-drone jobs: every drone with a done_when
        has met its own at some point (latched in self.done_agents); drones that only stand by do not count."""
        if not m.agent_done_fns:
            return m.done_fn is not None and bool((m.done_fn(g0).reshape(-1) > 0.5).any())
        need = {ag for ag, fn in m.agent_done_fns.items() if fn is not None and ag < g0.shape[0]}
        for ag in need - self.done_agents:
            if bool(m.agent_done_fns[ag](g0[ag:ag + 1]).reshape(-1)[0] > 0.5):
                self.done_agents.add(ag)
        return bool(need) and need <= self.done_agents

    @staticmethod
    def _is_carry(m: CompiledMission) -> bool:
        """Carry missions (a0 = item handle, a1 = drop-off) also succeed on the env's "delivered" event."""
        if m.agents:
            return False     # per-drone jobs: one drone's delivery does not end the mission
        a0, a1 = m.anchors.get("a0"), m.anchors.get("a1")
        return isinstance(a0, str) and a0.startswith("item_") and a1 in ("bin", "pad_dropoff")

    def _finish(self, result: str, msg: str) -> None:
        self.flying = False
        self.status, self.message, self.last_result = result, msg, result

    # ------------------------------------------------------------ views
    def info(self) -> dict:
        with self.lock:
            return {"scene": self.scene, "status": self.status, "message": self.message,
                    "policy_mode": self.pipeline.policy_mode, "mission_mode": self.pipeline.mission_mode,
                    "num_agents": self.env.num_agents, "num_envs": self.env.num_envs,
                    "mission": self.active.public() if self.active else None,
                    "pending": self.pending.public() if self.pending else None,
                    "flying": self.flying, "cem": self.pipeline.cem, "gusts": self.gusts, "sim": self.sim,
                    "scenes": list(spec.SCENES)}

    def extras(self) -> dict:
        g = self.obs["g"][0, 0]
        z = self.z
        return {
            "status": self.status, "message": self.message, "flying": self.flying, "scene": self.scene,
            "policy_mode": self.pipeline.policy_mode,
            "mission_mode": self.active.source if self.active else (self.pending.source if self.pending else self.pipeline.mission_mode),
            "mission_text": self.active.text if self.active else None,
            "current_segment": self.current_segment_idx if (self.active and self.active.segments) else 0,
            "total_segments": len(self.active.segments) if (self.active and self.active.segments) else 1,
            "segment_text": self.active.segments[self.current_segment_idx].text if (self.active and self.active.segments and self.current_segment_idx < len(self.active.segments)) else "",
            "reward": self.reward, "reward_sum": self.reward_sum if self.flying or self.last_result else None,
            "shield_hits": int(self.env.shield_count[0]) if hasattr(self.env, "shield_count") else 0,
            "swing": float(spec.g_field(g, "swing")[0]), "speed": float(spec.g_field(g, "speed")[0]),
            "step": self.step_count, "max_steps": self.max_steps, "time": self.step_count * spec.POLICY_DT,
            "result": self.last_result,
            "z_norm": (float(z.norm(dim=-1).mean()) if z.dim() == 2 else float(z.norm())) if z is not None else None,
            "z_per_drone": bool(z is not None and z.dim() == 2),
            "cem_used": self.cem_used, "zmap": self.zmap, "num_agents": self.env.num_agents,
            "gusts": self.gusts, "sim": self.sim,
            # drone picker: what each drone is doing, and which per-drone goals are met
            "agent_tasks": (self.active or self.pending).tasks(self.env.num_agents) if (self.active or self.pending)
            else [""] * self.env.num_agents,
            "agents_done": sorted(self.done_agents),
        }

    def _render(self) -> dict:
        """env.render_state(0) with the terrain heights cached here and stripped from the frame.

        state["terrain"] keeps version/res/half and gains "key" (build id + version, unique across
        scene switches). The full terrain is in self._terrain."""
        st = self._call_render(False)
        ter = st.get("terrain")
        if not ter:
            self._terrain = None
            return st
        key = f"{self._build_id}:{ter.get('version', 0)}"
        if ter.get("heights") is None and (self._terrain is None or self._terrain["key"] != key):
            full = self._call_render(True).get("terrain") or {}     # the heights changed: fetch them once
            if full.get("heights") is not None:
                ter = full
                key = f"{self._build_id}:{ter.get('version', 0)}"
        if ter.get("heights") is not None:
            self._terrain = dict(ter, key=key)
        st["terrain"] = {k: v for k, v in ter.items() if k != "heights"}
        st["terrain"]["key"] = key
        return st

    def _call_render(self, heights: bool) -> dict:
        """render_state(0) asking for the heights or not, whichever keyword the env takes
        (include_terrain=..., spec's send_terrain=..., or none: then the env decides)."""
        kws = [{"include_terrain": heights}, {"send_terrain": heights}, {}]
        start = getattr(self, "_render_kw", 0)
        for i in range(start, len(kws)):
            try:
                st = self.env.render_state(0, **kws[i])
                self._render_kw = i
                return st
            except TypeError:
                if i == len(kws) - 1:
                    raise
        raise RuntimeError("unreachable")

    def terrain(self) -> Optional[dict]:
        """Full terrain (with heights) of the current world, or None (flat ground / ocean)."""
        with self.lock:
            if self._terrain is None:
                self._render()
            return self._terrain

    def frame(self, include_terrain: bool = True) -> dict:
        """One frame. include_terrain=True always attaches the heights; False attaches them
        only when the terrain changed since the last frame built with False (the broadcast)."""
        with self.lock:
            st = self._render()
            ter = st.get("terrain")
            if ter and self._terrain is not None and self._terrain["key"] == ter["key"]:
                if include_terrain or self._sent_key != ter["key"]:
                    st["terrain"] = self._terrain
                if not include_terrain:
                    self._sent_key = ter["key"]
            return {"type": "frame", "state": st, "demo": self.extras()}


class Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def broadcast(self, text: str) -> None:
        async def send(ws):
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=1.0)
            except Exception:
                if ws in self.clients:
                    self.clients.remove(ws)
        await asyncio.gather(*(send(ws) for ws in list(self.clients)))


def create_app(session: Optional[DemoSession] = None, speed: float = 1.0, max_fps: float = 30.0,
               session_factory: Optional[Callable[[], DemoSession]] = None) -> FastAPI:
    hub = Hub()
    sim_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
    work_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="work")
    holder: Dict[str, DemoSession] = {}

    def S() -> DemoSession:
        return holder["s"]

    async def sim_loop():
        loop = asyncio.get_running_loop()
        period = spec.POLICY_DT / max(speed, 1e-3)
        min_send = 1.0 / max_fps
        last_send = 0.0
        while True:
            t0 = loop.time()
            try:
                await loop.run_in_executor(sim_pool, S().tick)
                if hub.clients and loop.time() - last_send >= min_send * 0.95:
                    msg = await loop.run_in_executor(sim_pool, lambda: json.dumps(S().frame(include_terrain=False)))
                    last_send = loop.time()
                    await hub.broadcast(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the demo alive
                log.exception("sim tick failed: %s", e)
                await asyncio.sleep(0.5)
            await asyncio.sleep(max(0.0, period - (loop.time() - t0)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if "s" not in holder:
            holder["s"] = session if session is not None else (session_factory or DemoSession)()
        task = asyncio.create_task(sim_loop())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except BaseException:
                pass

    app = FastAPI(title="SkyHand demo", lifespan=lifespan)
    if session is not None:
        holder["s"] = session
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    async def run(fn, *a):
        try:
            return await asyncio.get_running_loop().run_in_executor(work_pool, fn, *a)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/")
    async def index():
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/api/health")
    async def health():
        s = S()
        return {"ok": True, "scene": s.scene, "status": s.status, "policy_mode": s.pipeline.policy_mode,
                "mission_mode": s.pipeline.mission_mode, "clients": len(hub.clients)}

    @app.get("/api/state")
    async def state():
        return await run(S().frame)

    @app.get("/api/info")
    async def info():
        return S().info()

    @app.post("/api/scene")
    async def scene(body: Dict[str, Any] = Body(...)):
        return await run(S().set_scene, str(body.get("scene", "")))

    @app.post("/api/key")
    async def post_api_key(body: Dict[str, Any] = Body(...)):
        key = str(body.get("key", "")).strip()
        if not key:
            raise HTTPException(400, "empty key")
        env_file = Path(__file__).resolve().parents[1] / ".env"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={key}\n")
        import os
        os.environ["GEMINI_API_KEY"] = key
        S().pipeline.mission_api = load_mission_api()
        return {"status": "ok", "mission_mode": S().pipeline.mission_mode}

    @app.post("/api/mission")
    async def mission(body: Dict[str, Any] = Body(...)):
        return await run(S().compile, str(body.get("text", "")))

    @app.post("/api/transcribe")
    async def transcribe(request: Request):
        """Accept a raw browser recording and return reviewable text; never auto-compile it."""
        if not os.environ.get("ELEVENLABS_API_KEY", "").strip():
            raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured")
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        if declared > MAX_VOICE_BYTES:
            raise HTTPException(status_code=413, detail="voice recording is too large")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in VOICE_TYPES:
            raise HTTPException(status_code=415, detail=f"unsupported audio type: {content_type or 'missing'}")
        audio = await request.body()
        if not audio:
            raise HTTPException(status_code=400, detail="empty voice recording")
        if len(audio) > MAX_VOICE_BYTES:
            raise HTTPException(status_code=413, detail="voice recording is too large")
        try:
            return await transcribe_elevenlabs(audio, content_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/api/cancel")
    async def cancel():
        return await run(S().cancel)

    @app.post("/api/approve")
    async def approve():
        return await run(S().approve)

    @app.post("/api/stop")
    async def stop():
        return await run(S().stop)

    @app.post("/api/reset")
    async def reset():
        return await run(S().reset)

    @app.post("/api/perturb")
    async def perturb(body: Dict[str, Any] = Body(...)):
        kind = str(body.get("kind", ""))
        params = {k: v for k, v in body.items() if k != "kind"}
        return await run(S().perturb, kind, params)

    @app.get("/api/terrain")
    async def terrain():
        return {"terrain": await run(S().terrain)}

    @app.get("/api/zmap")
    async def zmap():
        return S().zmap

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.append(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "hello", **S().info()}))
            ter = await run(S().terrain)
            await websocket.send_text(json.dumps({"type": "terrain", "terrain": ter}))
            while True:
                txt = await websocket.receive_text()
                try:
                    msg = json.loads(txt)
                except ValueError:
                    continue
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "perturb":
                    try:
                        res = await run(S().perturb, str(msg.get("kind")), {k: v for k, v in msg.items() if k not in ("type", "kind")})
                        await websocket.send_text(json.dumps({"type": "perturbed", **res}))
                    except HTTPException as e:
                        await websocket.send_text(json.dumps({"type": "error", "detail": e.detail}))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if websocket in hub.clients:
                hub.clients.remove(websocket)

    app.state.hub = hub
    app.state.session = lambda: holder.get("s")
    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SkyHand judge demo server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", default="torch", choices=["torch", "genesis"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None, help="trained agent .pt (omit: scripted fallback, labelled in the UI)")
    ap.add_argument("--scene", default="trash", choices=list(spec.SCENES))
    ap.add_argument("--num-envs", type=int, default=1, help="worlds in the display env (world 0 is shown)")
    ap.add_argument("--cem", action="store_true", help="refine z with CEM in clone worlds (needs a checkpoint)")
    ap.add_argument("--cem-candidates", type=int, default=64)
    ap.add_argument("--env-expert", action="store_true", help="fallback flies env.expert_action() instead of the built-in controller")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--fake-env", action="store_true",
                    help="kinematic preview env (demo/fake_env.py, no physics; labelled in the UI)")
    # env options: left unset, the env's own defaults apply (domain randomisation and gusts are off by default)
    opt = dict(action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--domain-rand", help="domain randomisation on/off", **opt)
    ap.add_argument("--rand-dynamics", help="drone dynamics randomisation on/off", **opt)
    ap.add_argument("--wind", help="ambient wind on/off", **opt)
    ap.add_argument("--dynamic-wind", help="stochastic wind turbulence on/off", **opt)
    ap.add_argument("--gusts", help="dynamic gusts on/off (the Gust button needs them on)", **opt)
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed multiplier (1 = real time)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = load_agent(args.checkpoint, args.device)
    pipe = Pipeline(agent=agent, device=args.device, use_env_expert=args.env_expert, cem=args.cem)
    num_envs = max(args.num_envs, args.cem_candidates + 1) if (args.cem and agent is not None) else args.num_envs
    env_kw = {name: val for name, val in (("domain_rand", args.domain_rand),
                                          ("rand_drone_dynamics", args.rand_dynamics),
                                          ("enable_wind", args.wind), ("dynamic_wind", args.dynamic_wind),
                                          ("enable_gusts", args.gusts)) if val is not None}
    mk = default_make_env(args.backend, args.device, lidar=not args.no_lidar, fake=args.fake_env, **env_kw)
    session = DemoSession(args.scene, num_envs, pipeline=pipe, make_env=mk, seed=args.seed)
    log.info("policy: %s | mission compiler: %s | backend %s on %s | %d worlds",
             pipe.policy_mode, pipe.mission_mode, args.backend, args.device, num_envs)
    import uvicorn
    uvicorn.run(create_app(session, speed=args.speed), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
