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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from skyhand import spec

from .pipeline import CompiledMission, Pipeline, load_agent

log = logging.getLogger("skyhand.demo")
WEB_DIR = Path(__file__).resolve().parent / "web"
PERTURB_KINDS = ("drop_obstacle", "gust", "waves", "move_payload", "disable_drone")
MAX_EPISODE_STEPS = 2 * spec.EPISODE_STEPS          # 60 s of flight per approved mission


def default_make_env(backend: str = "torch", device: str = "cpu", lidar: bool = True):
    from skyhand.env.core import make_env

    def mk(scene: str, num_envs: int, **kw):
        kw.setdefault("lidar", lidar)
        kw.setdefault("random_anchors", False)
        return make_env(scene, num_envs=num_envs, device=device, backend=backend, **kw)
    return mk


class DemoSession:
    """All demo state. Thread-safe: every env access holds self.lock."""

    def __init__(self, scene: str = "trash", num_envs: int = 1, pipeline: Optional[Pipeline] = None,
                 make_env: Optional[Callable] = None, sample_make_env: Optional[Callable] = None,
                 seed: Optional[int] = None, max_steps: int = MAX_EPISODE_STEPS):
        self.lock = threading.RLock()
        self.pipeline = pipeline or Pipeline()
        self.make_env = make_env or default_make_env()
        # z sampling env: no LiDAR (B only reads g), anchors bound by the pipeline
        self.sample_make_env = sample_make_env or (lambda sc, n: self.make_env(sc, n, lidar=False))
        self.num_envs = max(1, int(num_envs))
        self.seed = seed
        self.max_steps = max_steps
        self._episode = 0
        self.zmap: Dict[str, Any] = {"available": False, "points": [], "note": "z map available after training"}
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
        self.obs = self.env._obs() if hasattr(self.env, "_obs") else self.env.reset()
        self.controller = self.pipeline.make_controller(self.env)
        self.pending: Optional[CompiledMission] = None
        self.active: Optional[CompiledMission] = None
        self.z: Optional[torch.Tensor] = None
        self.flying = False
        self.status = "idle"
        self.message = "Type a mission and press Compile."
        self.step_count = 0
        self.reward: Optional[float] = None
        self.reward_sum = 0.0
        self.cem_used = False
        self.last_result: Optional[str] = None
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
            ids = torch.arange(self.env.num_envs, device=self.env.device)
            self.pipeline.bind(self.env, m, ids)
            self.flying = False
            self.status, self.message = "computing z", "Computing the task vector z from the reward..."
            scene = self.scene
            episode = self._episode = self._episode + 1
        z = None
        if self.pipeline.agent is not None:
            z = self.pipeline.compute_z(m.reward_fn, scene, m.anchors, self.sample_make_env)
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
            self.obs = self.env._obs()
            if hasattr(self.controller, "reset"):
                self.controller.reset()
            self.step_count, self.reward_sum, self.reward = 0, 0.0, None
            self.flying, self.status = True, "flying"
            self.last_result = None
            self.message = f"Flying: {m.text}"
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
                self.obs = self.env._obs()
            if hasattr(self.controller, "reset"):
                self.controller.reset()
            self.flying = False
            self.step_count, self.reward, self.reward_sum = 0, None, 0.0
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
            lim = spec.ARENA_HALF
            kw["pos"] = [max(-lim, min(lim, float(pos[0]))), max(-lim, min(lim, float(pos[1])))]
            if kind == "drop_obstacle" and "half" in params:
                kw["half"] = max(0.2, min(2.0, float(params["half"])))
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
        elif kind == "disable_drone":
            ag = int(params.get("agent", 0))
            if not 0 <= ag < self.env.num_agents:
                raise ValueError(f"agent must be in [0, {self.env.num_agents})")
            kw["agent"] = ag
        with self.lock:
            note = ""
            if kind == "waves" and self.scene != "ocean":
                note = "waves only exist in the ocean scene"
            elif kind == "move_payload" and self.scene != "trash":
                note = "no payload in the ocean scene"
            self.env.perturb(0, kind, **kw)
            self.obs = self.env._obs()
        return {"ok": True, "kind": kind, "params": kw, "note": note}

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
            else:
                act = self._idle_action()
            out = env.step(act)
            self.obs = out.obs
            if not self.flying:
                return
            self.step_count += 1
            m = self.active
            g0 = out.obs["g"][0]
            try:
                self.reward = float(m.reward_fn(g0[0:1]).reshape(-1)[0])
                self.reward_sum += self.reward
                ok = m.done_fn is not None and bool((m.done_fn(g0).reshape(-1) > 0.5).any())
            except Exception as e:  # never kill the loop on a bad program
                log.warning("reward evaluation failed: %s", e)
                ok = False
            if ok:
                self._finish("success", f"Mission complete in {self.step_count * spec.POLICY_DT:.1f} s.")
            elif bool(out.done[0]):
                why = "collision" if bool(out.events["collision"][0].any()) else \
                    "net torn" if bool(out.events["tear"][0].any()) else "crash / out of bounds"
                self._finish("failure", f"Episode ended: {why}.")
            elif self.step_count >= self.max_steps:
                self._finish("timeout", f"Time limit ({self.max_steps * spec.POLICY_DT:.0f} s) reached without completing.")

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
                    "flying": self.flying, "cem": self.pipeline.cem}

    def extras(self) -> dict:
        g = self.obs["g"][0, 0]
        z = self.z
        return {
            "status": self.status, "message": self.message, "flying": self.flying, "scene": self.scene,
            "policy_mode": self.pipeline.policy_mode,
            "mission_mode": self.active.source if self.active else (self.pending.source if self.pending else self.pipeline.mission_mode),
            "mission_text": self.active.text if self.active else None,
            "reward": self.reward, "reward_sum": self.reward_sum if self.flying or self.last_result else None,
            "shield_hits": int(self.env.shield_count[0]) if hasattr(self.env, "shield_count") else 0,
            "swing": float(spec.g_field(g, "swing")[0]), "speed": float(spec.g_field(g, "speed")[0]),
            "step": self.step_count, "max_steps": self.max_steps, "time": self.step_count * spec.POLICY_DT,
            "result": self.last_result, "z_norm": float(z.norm()) if z is not None else None,
            "cem_used": self.cem_used, "zmap": self.zmap, "num_agents": self.env.num_agents,
        }

    def frame(self) -> dict:
        with self.lock:
            return {"type": "frame", "state": self.env.render_state(0), "demo": self.extras()}


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
                    msg = await loop.run_in_executor(sim_pool, lambda: json.dumps(S().frame()))
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

    @app.post("/api/mission")
    async def mission(body: Dict[str, Any] = Body(...)):
        return await run(S().compile, str(body.get("text", "")))

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

    @app.get("/api/zmap")
    async def zmap():
        return S().zmap

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.append(websocket)
        try:
            await websocket.send_text(json.dumps({"type": "hello", **S().info()}))
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
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed multiplier (1 = real time)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = load_agent(args.checkpoint, args.device)
    pipe = Pipeline(agent=agent, device=args.device, use_env_expert=args.env_expert, cem=args.cem)
    num_envs = max(args.num_envs, args.cem_candidates + 1) if (args.cem and agent is not None) else args.num_envs
    mk = default_make_env(args.backend, args.device, lidar=not args.no_lidar)
    session = DemoSession(args.scene, num_envs, pipeline=pipe, make_env=mk, seed=args.seed)
    log.info("policy: %s | mission compiler: %s | backend %s on %s | %d worlds",
             pipe.policy_mode, pipe.mission_mode, args.backend, args.device, num_envs)
    import uvicorn
    uvicorn.run(create_app(session, speed=args.speed), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
