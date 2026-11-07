from __future__ import annotations

import math

import torch

from skyhand import spec
from skyhand.env.core import make_env
from skyhand.mission import TextToZ, avoid_z, build_dataset, cem_refine, goal_z, reward_to_z

D = 8


def orthonormal_B(seed=0):
    torch.manual_seed(seed)
    W = torch.linalg.qr(torch.randn(spec.G_DIM, D))[0]   # [G, D], orthonormal columns
    return W, (lambda g: g @ W)


def test_reward_to_z_axis():
    W, B = orthonormal_B()
    g = torch.randn(20000, spec.G_DIM)
    for k in (0, 3, 7):
        z = reward_to_z(lambda gg, k=k: gg @ W[:, k], B, g)
        assert z.shape == (D,)
        assert abs(z.norm().item() - math.sqrt(D)) < 1e-4
        e = torch.zeros(D)
        e[k] = 1
        assert torch.nn.functional.cosine_similarity(z, e, dim=0) > 0.98
    z = reward_to_z("speed", B, g, center=True)
    assert abs(z.norm().item() - math.sqrt(D)) < 1e-4


def test_goal_and_avoid_z():
    W, B = orthonormal_B()
    g_goal = W[:, 0] * 5
    z = goal_z(B, g_goal)
    assert torch.allclose(z, torch.tensor([math.sqrt(D)] + [0.0] * (D - 1)), atol=1e-4)
    za = avoid_z(B, g_goal, torch.stack([W[:, 1], W[:, 1]]), lam=1.0)
    assert za[0] > 0 and za[1] < 0 and abs(za.norm().item() - math.sqrt(D)) < 1e-4


def toy_policy(obs, z):
    """Fly along the unit direction in z[:3]; gripper open."""
    n, a = obs["g"].shape[:2]
    v = z[:, :3] / z[:, :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    act = torch.zeros(n, a, spec.ACTION_DIM)
    act[..., :3] = v.unsqueeze(1)
    act[..., 4] = -1.0
    return act


def test_cem_refine_improves():
    env = make_env("trash", 16, lidar=False, shield=False)
    target = env.pos[0, 0].clone() + torch.tensor([5.0, 3.0, 1.0])
    target[2] = target[2].clamp(1.5, 6.0)
    target[:2] = target[:2].clamp(-12, 12)
    b = [spec.AnchorBinding(kind="point", point=target.tolist())] + [spec.AnchorBinding(kind="none")] * 3
    toward = target - env.pos[0, 0]
    side = torch.cross(toward, torch.tensor([0.0, 0.0, 1.0]), dim=0)   # perpendicular start direction
    z0 = torch.zeros(D)
    z0[:3] = side / side.norm()
    z0[3:] = 0.5
    src_before = env.pos[0].clone()
    gen = torch.Generator().manual_seed(0)
    z, info = cem_refine(env, toy_policy, z0, "-dist(drone, a0)", src_env=0, n_candidates=12, horizon=30,
                         iters=3, sigma=0.5, bindings=b, generator=gen)
    assert z.shape == (D,) and abs(z.norm().item() - math.sqrt(D)) < 1e-4
    assert info["best_score"] > info["z0_score"] + 3.0
    assert len(info["history"]) == 3 and 0 not in info["dst_ids"]
    assert torch.dot(z[:3], toward) > 0
    assert torch.allclose(env.pos[0], src_before)   # source world restored


def test_text2z_fit_predict(tmp_path):
    torch.manual_seed(0)
    zs = torch.nn.functional.normalize(torch.randn(4, D), dim=-1) * math.sqrt(D)
    texts = ["pick up the trash bag", "deliver the organ cooler gently", "net the fish and haul to the boat",
             "fly to obstacle 3"]
    pairs = build_dataset(texts, compile_fn=lambda t: t, reward_to_z_fn=lambda t: zs[texts.index(t)])
    m = TextToZ(D)
    m.fit(pairs, epochs=150)
    for i, t in enumerate(texts):
        sims = torch.nn.functional.cosine_similarity(m.predict(t).unsqueeze(0), zs, dim=-1)
        assert int(sims.argmax()) == i and sims[i] > 0.9
    assert int(torch.nn.functional.cosine_similarity(m.predict("grab the trash bag").unsqueeze(0), zs).argmax()) == 0
    p = tmp_path / "t2z.pt"
    m.save(str(p))
    m2 = TextToZ.load(str(p))
    assert torch.allclose(m2.predict(texts[1]), m.predict(texts[1]))
