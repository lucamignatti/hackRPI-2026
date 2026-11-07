"""Tests for skyhand/train (tiny configs, CPU)."""
from __future__ import annotations

import ast
import copy
import csv
import math
import os
import pathlib

import pytest
import torch

from skyhand import spec
from skyhand.env.core import make_env
from skyhand.train import nets
from skyhand.train.agent import Agent
from skyhand.train.buffer import DemoStore, ReplayBuffer, flatten_obs
from skyhand.train.config import tiny_config
from skyhand.train.fb_cpr import FBCPR, ortho_loss, penalties
from skyhand.train.sf_fallback import PHI_DIM, Phi, SFLearner, regress_z
from skyhand.train.zsample import WorldZ, ZSampler

TRAIN_DIR = pathlib.Path(__file__).resolve().parents[2] / "skyhand" / "train"


@pytest.fixture(scope="module")
def trash_obs():
    torch.manual_seed(0)
    env = make_env("trash", 4, lidar=False)
    return env, env.step(torch.rand(4, 1, 5) * 2 - 1).obs


@pytest.fixture(scope="module")
def ocean_obs():
    torch.manual_seed(0)
    env = make_env("ocean", 4, lidar=True)
    out = env.step(torch.rand(4, 3, 5) * 2 - 1)
    return env, out.obs


def _flat(obs, keys=spec.CRITIC_KEYS + ("g",)):
    return flatten_obs(obs, keys)


def _fill(buf, env, steps, z, rand=True, stream=0):
    obs = env.reset()
    for _ in range(steps):
        a = torch.rand(env.num_envs, env.num_agents, 5) * 2 - 1
        out = env.step(a)
        reset = out.done | out.truncated
        buf.add(obs, a, out.obs, out.done, reset, z, stream=stream)
        obs = env.reset(torch.nonzero(reset).squeeze(-1)) if reset.any() else out.obs
    return obs


# ============================================================ blocks
def test_gate_identity_at_init_and_range():
    g = nets.Gate(16)
    x = torch.randn(5, 16)
    assert torch.equal(g.gate(x), torch.ones(5, 16))
    y = torch.randn(5, 16)
    assert torch.equal(g(x, y), y)
    torch.nn.init.normal_(g.lin.weight, std=1.0)
    torch.nn.init.normal_(g.lin.bias, std=1.0)
    v = g.gate(x)
    assert (v > 0).all() and (v < 2).all()


def test_block_shapes_and_gates_start_at_one():
    b = nets.Block(32, 4)
    q, c = torch.randn(3, 5, 32), torch.randn(3, 7, 32)
    assert b(q, c).shape == (3, 5, 32)
    assert b(q, c, torch.rand(3, 7) > 0.5).shape == (3, 5, 32)
    for gate in (b.ga, b.gm):
        assert torch.equal(gate.gate(torch.randn(4, 32)), torch.ones(4, 32))
    mb = nets.MLPBlock(32)
    assert mb(torch.randn(6, 32)).shape == (6, 32)
    assert b.mlp.w1.out_features == 128 and b.mlp.w3.out_features == 128 and b.mlp.w2.in_features == 128


def test_k_norm_only():
    torch.manual_seed(0)
    b = nets.Block(32, 4)
    q, c = torch.randn(2, 3, 32), torch.randn(2, 6, 32)
    y0 = b(q, c)
    bk = copy.deepcopy(b)
    with torch.no_grad():                 # scaling K is a no-op: K is LayerNormed per head
        bk.k.weight.mul_(7.0); bk.k.bias.mul_(7.0)
    assert torch.allclose(bk(q, c), y0, atol=1e-5)
    bq = copy.deepcopy(b)
    with torch.no_grad():                 # scaling Q changes the output: Q is not normalised
        bq.q.weight.mul_(7.0); bq.q.bias.mul_(7.0)
    assert not torch.allclose(bq(q, c), y0, atol=1e-3)
    assert b.nk.normalized_shape == (8,) and not b.nk.elementwise_affine


def test_masked_tokens_do_not_affect_output(ocean_obs):
    torch.manual_seed(0)
    b = nets.Block(32, 4)
    q, c = torch.randn(2, 3, 32), torch.randn(2, 6, 32)
    mask = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    y0 = b(q, c, mask)
    c2 = c.clone(); c2[~mask] = torch.randn(int((~mask).sum()), 32) * 100
    assert torch.allclose(b(q, c2, mask), y0, atol=1e-5)
    # all-masked rows are guarded (finite)
    assert torch.isfinite(b(q, c, torch.zeros(2, 6, dtype=torch.bool))).all()
    # network level: masked privileged and mate tokens
    _, obs = ocean_obs
    o = _flat(obs)
    critic = nets.Critic(16, 1, 32, 2, 1, 4)
    z, act = torch.randn(o["proprio"].shape[0], 16), torch.rand(o["proprio"].shape[0], 5)
    y0 = critic(o, z, act)
    op = {k: v.clone() for k, v in o.items()}
    op["priv"][~op["priv_mask"]] = 1e3
    assert torch.allclose(critic(op, z, act), y0, atol=1e-5)
    o2 = {k: v.clone() for k, v in o.items()}
    o2["priv"][~o2["priv_mask"]] = 1e3
    o2["mates_mask"][:, 1] = False
    o3 = {k: v.clone() for k, v in o2.items()}
    o3["mates"][:, 1] = -1e3
    assert torch.allclose(critic(o2, z, act), critic(o3, z, act), atol=1e-5)
    assert (~o["priv_mask"]).any()


@pytest.mark.parametrize("which", ["trash", "ocean"])
def test_network_forward_shapes(which, trash_obs, ocean_obs):
    env, obs = trash_obs if which == "trash" else ocean_obs
    cfg = tiny_config()
    L = FBCPR(cfg)
    o = _flat(obs)
    m = env.num_envs * env.num_agents
    z = nets.sphere(torch.randn(m, cfg.d))
    a = L.actor(o, z)
    assert a.shape == (m, 5) and (a.abs() <= 1).all()
    assert L.F(o, z, a).shape == (2, m, cfg.d)
    assert L.QD(o, z, a).shape == (m,) and L.QR(o, z, a).shape == (m,)
    bg = L.B(o["g"])
    assert bg.shape == (m, cfg.d)
    assert torch.allclose(bg.norm(dim=-1), torch.full((m,), math.sqrt(cfg.d)), atol=1e-4)
    p, lg = L.D(o["g"], z)
    assert p.shape == (m,) and lg.shape == (m,) and ((p > 0) & (p < 1)).all()
    # actor ignores privileged keys entirely
    ao = {k: o[k] for k in spec.ACTOR_KEYS}
    assert torch.allclose(L.actor(ao, z), a)
    if which == "ocean":
        assert obs["lidar"].dtype == torch.uint8 and (obs["lidar"] != 255).any()


# ============================================================ losses
def test_ortho_loss():
    b, d = 64, 16
    Q, _ = torch.linalg.qr(torch.randn(b, d))
    Bm = math.sqrt(b) * Q                      # E[B B^T] = I
    assert ortho_loss(Bm).item() < 1e-6
    assert ortho_loss(nets.sphere(torch.randn(b, d) + 2.0)).item() > 1.0


def _batch_from_env(env, n=64, d=16):
    buf = ReplayBuffer(4096, d)
    _fill(buf, env, 20, nets.sphere(torch.randn(env.num_envs, d)))
    torch.manual_seed(1)
    return buf, buf.sample(n)


def test_fb_loss_decreases_on_fixed_batch(trash_obs):
    torch.manual_seed(0)
    env = make_env("trash", 8, lidar=False)
    _, batch = _batch_from_env(env, 64)
    L = FBCPR(tiny_config(lr_f=1e-3, lr_b=1e-3))
    z = nets.sphere(torch.randn(64, 16))
    with torch.no_grad():
        a_next = L.actor(batch["next_obs"], z)
    losses = []
    for _ in range(50):
        loss, m = L._repr_loss(batch, z, a_next)
        L._step(L.opt_fb, loss)
        losses.append(m["fb_loss"] + 100 * m["ortho_loss"])
    assert losses[-1] < losses[0]
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_full_update_finite_and_polyak():
    torch.manual_seed(0)
    env = make_env("ocean", 4, lidar=True)
    buf, batch = _batch_from_env(env, 32)
    demo = DemoStore(64, 8)
    demo.add_rollout(torch.randn(16, 6, spec.G_DIM))
    L = FBCPR(tiny_config())
    before = {k: [p.clone() for p in getattr(L, k).parameters()] for k in ("F_t", "B_t", "QD_t", "QR_t")}
    m = L.update(batch, demo)
    assert all(math.isfinite(v) for v in m.values()), m
    for k in ("fb_loss", "ortho_loss", "d_loss", "d_gp", "qd_loss", "qr_loss", "actor_loss"):
        assert k in m
    for k, old in before.items():
        online = getattr(L, k[:-2])
        moved = False
        for pt, p0, p in zip(getattr(L, k).parameters(), old, online.parameters()):
            assert torch.allclose(pt, 0.995 * p0 + 0.005 * p, atol=1e-6)
            moved |= not torch.equal(pt, p0)
        assert moved, k
    # without demos too
    assert all(math.isfinite(v) for v in L.update(buf.sample(32)).values())


def test_penalties():
    g = torch.zeros(3, spec.G_DIM)
    assert torch.equal(penalties(g), torch.zeros(3))
    g[0, spec.G_SLICES["collision"]] = 1
    g[1, spec.G_SLICES["tilt"]] = 0.8
    g[2, spec.G_SLICES["thrust"]] = 2.0
    p = penalties(g)
    assert (p > 0).all() and p.shape == (3,)


# ============================================================ z sampling
def test_zsampler_proportions_and_norm():
    torch.manual_seed(0)
    zs = ZSampler(16)
    cat = zs.choose(20000, True, True)
    frac = torch.bincount(cat, minlength=3).float() / 20000
    assert torch.allclose(frac, torch.tensor([0.5, 0.25, 0.25]), atol=0.02)
    cat = zs.choose(1000, False, True)
    assert (cat != 0).all()
    demo = DemoStore(32, 8); demo.add_rollout(torch.randn(16, 4, spec.G_DIM))
    B = nets.BNet(16, 32, 1)
    z = zs.sample(500, B, demo=demo, g_pool=torch.randn(50, spec.G_DIM))
    assert z.shape == (500, 16)
    assert torch.allclose(z.norm(dim=-1), torch.full((500,), 4.0), atol=1e-4)
    assert torch.allclose(zs.uniform(10).norm(dim=-1), torch.full((10,), 4.0), atol=1e-4)


def test_world_z_window_and_team_share():
    torch.manual_seed(0)
    period = 5
    wz = WorldZ(6, ZSampler(8), period)
    assert len(set(wz.timer.tolist())) > 1           # staggered
    last_change = torch.zeros(6, dtype=torch.long)
    prev = wz.z.clone()
    for t in range(1, 31):
        ids = wz.step()
        changed = (wz.z != prev).any(-1)
        assert set(torch.nonzero(changed).squeeze(-1).tolist()) <= set(ids.tolist())
        for i in ids.tolist():
            if t > period:
                assert t - last_change[i] == period
            last_change[i] = t
        prev = wz.z.clone()
        az = wz.agent_z(3)
        assert az.shape == (6, 3, 8)
        assert torch.equal(az[:, 0], az[:, 1]) and torch.equal(az[:, 0], az[:, 2])


# ============================================================ buffer
def test_buffer_round_trip(ocean_obs):
    torch.manual_seed(0)
    env = make_env("ocean", 4, lidar=True)
    buf = ReplayBuffer(512, 16)
    obs = env.reset()
    z = nets.sphere(torch.randn(4, 16))
    a = torch.rand(4, 3, 5) * 2 - 1
    out = env.step(a)
    buf.add(obs, a, out.obs, out.done, torch.zeros(4, dtype=torch.bool), z)
    a2 = torch.rand(4, 3, 5) * 2 - 1
    out2 = env.step(a2)
    reset = torch.tensor([True, False, False, False])
    buf.add(out.obs, a2, out2.obs, out2.done, reset, z)
    assert len(buf) == 24
    assert buf.obs_size == 36                         # second step reused 12 rows
    s = buf.sample(64)
    assert s["obs"]["lidar"].dtype == torch.uint8 and s["obs"]["lidar"].shape == (64, 3, 16, 64)
    assert s["action"].shape == (64, 5) and s["g"].shape == (64, spec.G_DIM)
    assert s["next_g"].shape == (64, spec.G_DIM) and s["done"].shape == (64,) and s["z"].shape == (64, 16)
    # exact round trip of every transition
    f1, f1n = _flat(obs), _flat(out.obs)
    f2n = _flat(out2.obs)
    for t in range(24):
        oi, ni = buf.obs_idx[t], buf.next_idx[t]
        src_o, src_n = (f1, f1n) if t < 12 else (f1n, f2n)
        j = t % 12
        for k in buf.obs:
            assert torch.equal(buf.obs[k][oi], src_o[k][j]), k
            assert torch.equal(buf.obs[k][ni], src_n[k][j]), k
        assert torch.equal(buf.action[t], (a if t < 12 else a2).reshape(12, 5)[j])
    # after the reset of world 0 its rows are written fresh, others reused
    env.reset(torch.tensor([0]))
    g = buf.sample_g(10)
    assert g.shape == (10, spec.G_DIM)


def test_buffer_ring_overwrite_stays_valid():
    torch.manual_seed(0)
    env = make_env("trash", 8, lidar=False)
    buf = ReplayBuffer(64, 16, obs_capacity=80)
    _fill(buf, env, 40, nets.sphere(torch.randn(8, 16)))
    assert len(buf) == 64
    s = buf.sample(256)
    t = torch.arange(64)
    assert buf._valid(t).float().mean() > 0.5
    assert torch.isfinite(s["g"]).all()


def test_demo_store_segments():
    ds = DemoStore(100, 8)
    g = torch.randn(20, 3, spec.G_DIM)
    cut = torch.zeros(20, 3, dtype=torch.bool)
    cut[3, 0] = True                                   # world 0 episode ends inside window 0
    n = ds.add_rollout(g, cut)
    assert n == 2 * 3 - 1
    assert ds.sample_segments(7).shape == (7, 8, spec.G_DIM)
    assert ds.sample_g(5).shape == (5, spec.G_DIM)
    assert torch.equal(ds.segs[0], g[0:8, 1])


# ============================================================ SF fallback
def test_sf_regression_recovers_linear_reward():
    torch.manual_seed(0)
    phi = Phi()
    g = torch.randn(3000, spec.G_DIM) * spec.g_scale()
    f = phi(g)
    assert f.shape == (3000, PHI_DIM)
    w = torch.randn(PHI_DIM)
    r = f @ w
    w_hat = regress_z(f, r)
    assert torch.allclose(f @ w_hat, r, atol=1e-3)
    assert torch.allclose(w_hat, w, atol=1e-3)


def test_sf_learner_update_runs():
    torch.manual_seed(0)
    env = make_env("trash", 4, lidar=False)
    buf, batch = _batch_from_env(env, 32, d=PHI_DIM)
    L = SFLearner(tiny_config(method="sf"))
    assert L.d == PHI_DIM
    m = L.update(batch)
    assert all(math.isfinite(v) for v in m.values())


# ============================================================ agent
@pytest.mark.parametrize("method", ["fb", "sf"])
def test_agent_save_load_identical_actions(tmp_path, ocean_obs, method):
    torch.manual_seed(0)
    _, obs = ocean_obs
    ag = Agent(tiny_config(method=method))
    assert ag.method == method
    z = nets.sphere(torch.randn(4, ag.d))
    a0 = ag.act(obs, z)
    assert a0.shape == (4, 3, 5)
    p = str(tmp_path / "a.pt")
    ag.save(p, with_optim=True)
    ag2 = Agent.load(p, device="cpu")
    assert ag2.d == ag.d and ag2.method == method
    assert torch.equal(ag2.act(obs, z), a0)
    assert torch.equal(ag2.encode_g(obs["g"]), ag.encode_g(obs["g"]))
    zr = ag.infer_z(obs["g"].reshape(-1, spec.G_DIM), torch.randn(12))
    assert zr.shape == (ag.d,)
    assert ag.act(obs, zr).shape == (4, 3, 5)


# ============================================================ end to end
def test_train_end_to_end(tmp_path):
    from skyhand.train.train import main
    out = str(tmp_path / "run")
    path = main(["--tiny", "--scene", "both", "--device", "cpu", "--steps", "6", "--out", out,
                 "--set", "batch_size=16", "--set", "log_every=2"])
    assert os.path.exists(path)
    rows = list(csv.DictReader(open(os.path.join(out, "metrics.csv"))))
    assert len(rows) == 3 and "fb_loss" in rows[-1]
    ag = Agent.load(path)
    assert ag.g_pool is not None and ag.g_pool.shape[1] == spec.G_DIM
    assert ag.infer_z(ag.g_pool, ag.g_pool[:, 0]).shape == (ag.d,)
    env = make_env("trash", 2, lidar=False)
    assert ag.act(env.reset(), torch.randn(2, ag.d)).shape == (2, 1, 5)


def test_train_without_expert(tmp_path, monkeypatch):
    from skyhand.env.core import SkyEnv
    from skyhand.train.train import main

    def no_expert(self):
        raise NotImplementedError
    monkeypatch.setattr(SkyEnv, "expert_action", no_expert)
    out = str(tmp_path / "run_sf")
    path = main(["--tiny", "--scene", "trash", "--device", "cpu", "--steps", "3", "--out", out,
                 "--method", "sf", "--set", "batch_size=8"])
    assert os.path.exists(path)


def test_cli_flags_disable_dynamics_and_wind():
    from skyhand.train.train import parse_args
    cfg = parse_args(["--tiny", "--no-domain-rand", "--no-rand-dynamics", "--no-wind",
                      "--no-dynamic-wind", "--no-gusts", "--eval-every", "10",
                      "--eval-envs", "4", "--eval-steps", "20", "--eval-domain-rand",
                      "--eval-rand-dynamics", "--eval-wind", "--eval-dynamic-wind", "--eval-gusts"])
    assert cfg.domain_rand is False
    assert cfg.rand_drone_dynamics is False
    assert cfg.enable_wind is False
    assert cfg.dynamic_wind is False
    assert cfg.enable_gusts is False
    assert cfg.eval_every == 10
    assert cfg.eval_envs == 4
    assert cfg.eval_steps == 20
    assert cfg.eval_domain_rand is True
    assert cfg.eval_rand_drone_dynamics is True
    assert cfg.eval_enable_wind is True
    assert cfg.eval_dynamic_wind is True
    assert cfg.eval_enable_gusts is True


def test_trainer_evaluation_and_cli_eval(tmp_path):
    from skyhand.train.train import Trainer, main
    cfg = tiny_config(scene="trash", num_envs=2, steps=2, out=str(tmp_path / "eval_run"),
                      eval_every=1, eval_envs=2, eval_steps=2)
    trainer = Trainer(cfg)
    m = trainer.evaluate(steps=2)
    assert "eval_collision_rate" in m
    assert "eval_shield_rate" in m
    assert "eval_done_rate" in m
    p = trainer.save()
    res = main(["--tiny", "--eval", p, "--device", "cpu"])
    assert res == p


# ============================================================ py39 / minimal-deps compatibility
def test_py39_compat_and_no_genesis():
    files = sorted(TRAIN_DIR.glob("*.py"))
    assert len(files) >= 8
    for f in files:
        src = f.read_text()
        tree = ast.parse(src)
        if f.name != "__init__.py" or src.strip():
            assert "from __future__ import annotations" in src, f.name
        for node in ast.walk(tree):
            assert type(node).__name__ not in ("Match", "TypeAlias"), f.name
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert not any("genesis" in n for n in names), f.name


# ============================================================ review fixes
def test_demos_include_grasp_and_carry(tmp_path):
    """Demo env keeps running across collections, so demos reach the grasp and carry phases."""
    from skyhand.train.train import Trainer
    cfg = tiny_config(scene="trash", out=str(tmp_path), demo_envs=16, demo_initial_steps=400,
                      demo_steps=64, demo_capacity=8192)
    tr = Trainer(cfg)
    tr.collect_demos()
    tr.collect_demos()
    segs = tr.demo.segs[: len(tr.demo)]
    holding = spec.g_field(segs, "holding").squeeze(-1)
    assert holding.mean() > 0.05, f"only {holding.mean():.1%} of demo states hold the payload"
    # the second call continued the same episodes instead of resetting them
    assert int(tr.demo_envs["trash"].steps.max()) > 64


def test_fb_q_loss_matches_formula():
    from skyhand.train.fb_cpr import fb_q_loss
    torch.manual_seed(0)
    Fz = torch.randn(2, 5)
    Bz, tFz = torch.randn(5), torch.randn(5)
    done = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    y = Bz + 0.9 * (1 - done) * tFz
    ref = sum(((Fz[k] - y) ** 2).mean() for k in range(2))
    assert torch.allclose(fb_q_loss(Fz, Bz, tFz, done, 0.9), ref)


def test_aux_loss_reported_and_switchable():
    torch.manual_seed(0)
    env = make_env("trash", 4, lidar=False)
    _, batch = _batch_from_env(env, 32)
    m = FBCPR(tiny_config()).update(batch)
    assert math.isfinite(m["fb_q_loss"]) and m["fb_q_loss"] > 0
    m0 = FBCPR(tiny_config(q_loss_coef=0.0)).update(batch)
    assert m0["fb_q_loss"] == 0.0


def test_critic_queries_depend_on_action(trash_obs):
    _, obs = trash_obs
    o = _flat(obs)
    z = nets.sphere(torch.randn(4, 16))
    crit = nets.Critic(16, 16, width=32, heads=2, blocks=1, n_queries=4)
    assert crit.aproj is not None
    assert nets.Actor(16, width=32, heads=2, blocks=1, n_queries=4).aproj is None
    a = torch.zeros(4, spec.ACTION_DIM, requires_grad=True)
    (crit(o, z, a) * z).sum().backward()
    assert a.grad.abs().sum() > 0
    old = nets.Critic(16, 16, width=32, heads=2, blocks=1, n_queries=4, action_queries=False)
    assert old.aproj is None


def test_expert_worlds_flagged_and_excluded_from_disc_negatives(tmp_path):
    from skyhand.train.train import Trainer
    cfg = tiny_config(scene="trash", out=str(tmp_path), num_envs=4, expert_world_frac=0.5,
                      warm_expert_frac=0.5, batch_size=16)
    tr = Trainer(cfg)
    for _ in range(6):
        tr.env_step(tr.streams[0], warm=False)
    flags = tr.buffer.expert[: len(tr.buffer)]
    assert flags.any() and (~flags).any()
    assert flags.float().mean() == pytest.approx(0.5)
    # a batch made only of expert transitions: no discriminator step, but the update still runs
    tr.collect_demos()
    batch = tr.buffer.sample(16)
    batch["expert"] = torch.ones_like(batch["expert"])
    m = tr.agent.update(batch, tr.demo)
    assert m["d_loss"] == 0.0 and all(math.isfinite(v) for v in m.values())
    batch["expert"] = torch.zeros_like(batch["expert"])
    assert tr.agent.update(batch, tr.demo)["d_loss"] > 0.0
