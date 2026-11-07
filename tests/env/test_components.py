import math

import torch

from skyhand import spec
from skyhand.env.fish import FishSchools
from skyhand.env.lidar import (LidarModel, decode_range, encode_range, ray_aabb, ray_plane_z,
                               ray_spheres, spherical_dirs, down_dirs)
from skyhand.env.net import NetSim
from skyhand.env.shield import shield_velocity
from skyhand.env.waves import WaveField


def test_wave_height_matches_formula():
    w = WaveField(2)
    xy = torch.tensor([[[1.0, 2.0]], [[-3.0, 0.5]]])
    t = torch.tensor([0.7, 3.1])
    h = w.height(xy, t)
    for i in range(2):
        ref = 0.0
        for j in range(spec.N_WAVES):
            th = w.k[i, j] * (math.cos(w.dir[i, j]) * xy[i, 0, 0] + math.sin(w.dir[i, j]) * xy[i, 0, 1]) \
                - w.omega[i, j] * t[i] + w.phase[i, j]
            ref += w.amp[i, j] * math.cos(th)
        assert abs(float(h[i, 0]) - float(ref)) < 1e-5
    p = w.params(0)
    assert set(p) == {"amp", "k", "dir", "omega", "phase"}


def test_wave_ray_hits_surface_and_velocity_decays():
    w = WaveField(1)
    t = torch.tensor([1.0])
    o = torch.tensor([[[0.0, 0.0, 5.0]]])
    d = torch.tensor([[[0.3, 0.0, -1.0]]])
    d = d / d.norm(dim=-1, keepdim=True)
    s = w.ray_hit(o, d, t)
    p = o + d * s.unsqueeze(-1)
    assert abs(float(p[0, 0, 2] - w.height(p[..., :2], t)[0, 0])) < 1e-3
    shallow = w.water_velocity(torch.tensor([[[0.0, 0.0, -0.1]]]), t).norm()
    deep = w.water_velocity(torch.tensor([[[0.0, 0.0, -8.0]]]), t).norm()
    assert deep < shallow


def test_shield_removes_inward_velocity():
    pos = torch.tensor([[[0.0, 0.0, 2.0]]])
    v = torch.tensor([[[3.0, 0.0, 0.0]]])
    c = torch.tensor([[[2.5, 0.0, 0.0]]])
    r = torch.tensor([[1.0]])
    act = torch.tensor([[True]])
    vs, hit = shield_velocity(pos, v, c, r, act)
    assert hit.all() and float(vs[0, 0, 0]) <= 1e-6
    v_out = torch.tensor([[[-3.0, 0.0, 0.0]]])
    vs2, hit2 = shield_velocity(pos, v_out, c, r, act)
    assert not hit2.any() and torch.allclose(vs2, v_out)
    # boxes
    bc = torch.tensor([[[2.0, 0.0, 2.0]]])
    bh = torch.tensor([[[0.5, 0.5, 0.5]]])
    vs3, hit3 = shield_velocity(pos, v, None, None, None, bc, bh, torch.tensor([[True]]))
    assert hit3.all() and float(vs3[0, 0, 0]) <= 1e-6


def test_net_rest_and_tension():
    net = NetSim(2)
    net.reset(torch.arange(2), torch.zeros(2, 3), torch.zeros(2))
    pinned = torch.zeros(2, net.p, dtype=torch.bool)
    _, stretch = net.step(0.05, pinned, net.pos.clone())
    assert float(stretch.max()) < 1e-3
    # lift all four corners 3 m: tension pulls the pins down, net hangs below them
    pinned[:, net.corners] = True
    pin = net.pos.clone()
    pin[:, net.corners, 2] = 3.0
    for _ in range(40):
        f, stretch = net.step(0.05, pinned, pin)
    assert (f[:, net.corners, 2] < 0).all()
    assert (net.pos[..., 2] <= 3.0 + 1e-4).all()
    assert not net.torn.any()


def test_net_tears_when_overstretched():
    net = NetSim(1)
    net.reset(torch.arange(1), torch.zeros(1, 3), torch.zeros(1))
    pinned = torch.zeros(1, net.p, dtype=torch.bool)
    pinned[:, net.corners[:2]] = True
    pin = net.pos.clone()
    pin[:, net.corners[0], 0] -= 20.0
    pin[:, net.corners[1], 0] += 20.0
    net.step(0.05, pinned, pin)
    assert net.torn.all()


def test_net_contains():
    net = NetSim(1)
    net.reset(torch.arange(1), torch.zeros(1, 3), torch.zeros(1))
    net.pos[0, :, 2] = -1.0
    net.pos[0, net.corners, 2] = 1.0
    pts = torch.tensor([[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 0.0, -2.0]]])
    assert net.contains(pts)[0].tolist() == [True, False, False]


def test_fish_stay_under_surface():
    f = FishSchools(4)
    f.reset(torch.arange(4))
    surf = torch.zeros(4, f.f)
    for _ in range(100):
        f.step(0.05, torch.full((4, 1, 3), 100.0), surf)
    assert (f.pos[..., 2] < 0.0).all()
    assert (f.pos[..., 2] > -3.0).all()


def test_ray_primitives():
    o = torch.tensor([[[0.0, 0.0, 0.0]]])
    d = torch.tensor([[[1.0, 0.0, 0.0]]])
    t = ray_aabb(o, d, torch.tensor([[[5.0, 0.0, 0.0]]]), torch.tensor([[[1.0, 1.0, 1.0]]]))
    assert abs(float(t) - 4.0) < 1e-5
    t = ray_spheres(o, d, torch.tensor([[[5.0, 0.0, 0.0]]]), torch.tensor([[0.5]]))
    assert abs(float(t) - 4.5) < 1e-5
    t = ray_spheres(o, -d, torch.tensor([[[5.0, 0.0, 0.0]]]), torch.tensor([[0.5]]))
    assert math.isinf(float(t))
    oz = torch.tensor([[[0.0, 0.0, 2.0]]])
    dz = torch.tensor([[[0.0, 0.0, -1.0]]])
    assert abs(float(ray_plane_z(oz, dz, torch.tensor([0.0]))) - 2.0) < 1e-5


def test_range_encoding_roundtrip():
    r = torch.tensor([0.0, 5.0, 29.0, float("inf"), 100.0])
    q = encode_range(r)
    assert q.dtype == torch.uint8
    assert q[3] == 255 and q[4] == 255
    assert torch.allclose(decode_range(q[:3]), r[:3], atol=spec.LIDAR_MAX_RANGE / 255)


def test_ray_patterns():
    s = spherical_dirs()
    assert s.shape == (spec.LIDAR_V, spec.LIDAR_H, 3)
    assert torch.allclose(s.norm(dim=-1), torch.ones(1), atol=1e-5)
    dd = down_dirs()
    assert (dd[..., 2] < 0).all()


def test_green_penetrates_water_and_nir_does_not():
    w = WaveField(1)
    w.amp.zero_()
    lm = LidarModel(noise_std=0.0, dropout=0.0)
    origin = torch.tensor([[[0.0, 0.0, 3.0]]])
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    fish = {"sph_c": torch.tensor([[[0.0, 0.0, -1.5]]]), "sph_r": torch.tensor([[0.8]]),
            "ground_z": torch.tensor([-12.0])}
    img, down, pts = lm.scan(origin, quat, fish, waves=w, t=torch.zeros(1), turbidity=torch.tensor([0.2]), noise=False)
    nir = decode_range(down[0, 0, 0])
    green = decode_range(down[0, 0, 1])
    center = (spec.DOWN_V // 2, spec.DOWN_H // 2)
    assert abs(float(nir[center]) - 3.0) < 0.3            # surface
    assert abs(float(green[center]) - 3.7) < 0.3          # fish top under water
    assert down[0, 0, 2][center] > 0
    # murky water hides the same fish
    img2, down2, _ = lm.scan(origin, quat, fish, waves=w, t=torch.zeros(1), turbidity=torch.tensor([5.0]), noise=False)
    assert down2[0, 0, 1][center] == 255
