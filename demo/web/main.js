// SkyHand judge demo viewer. Renders env.render_state frames streamed over /ws.
// World frame is z-up, metres (same as the simulator), so we set three.js up = +z.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const ARENA_HALF = 15;
const AGENT_COLORS = [0x4ea8ff, 0xffc94e, 0xc77dff, 0x5ce1e6];
const ANCHOR_COLORS = [0xff6b6b, 0x2fbf71, 0xf2b84b, 0xb388ff];
const EXAMPLES = {
  trash: [
    "grab the bag and drop it in the bin without getting near the moving carts",
    "carry the cooler as smoothly as possible and stay out of the no-fly zone",
    "fly to the bin and hover",
  ],
  ocean: [
    "net the school closest to the boat",
    "gently haul the net over the fish and bring it to the boat",
    "fly to the boat",
  ],
};

// ------------------------------------------------------------------ wave surface
// Must match skyhand/env/waves.py: h = sum amp*cos(k*(cos(dir)*x + sin(dir)*y) - omega*t + phase)
// (render_state already multiplies amp by the live wave scale).
export function waveHeight(w, x, y, t) {
  if (!w) return 0;
  let h = 0;
  for (let i = 0; i < w.amp.length; i++) {
    h += w.amp[i] * Math.cos(w.k[i] * (Math.cos(w.dir[i]) * x + Math.sin(w.dir[i]) * y) - w.omega[i] * t + w.phase[i]);
  }
  return h;
}

// ------------------------------------------------------------------ renderer / scene
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0b1016);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0b1016, 45, 110);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 400);
camera.position.set(-15, -18, 13);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 1);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.update();

scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x223322, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(-10, -6, 25);
scene.add(sun);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// static: trash ground
const trashGroup = new THREE.Group();
{
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA_HALF, 2 * ARENA_HALF),
    new THREE.MeshStandardMaterial({ color: 0x24302a, roughness: 0.95 }));
  trashGroup.add(ground);
  const grid = new THREE.GridHelper(2 * ARENA_HALF, 30, 0x3b4a40, 0x2d3a32);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.01;
  trashGroup.add(grid);
  const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * ARENA_HALF, 2 * ARENA_HALF, 0.01)),
    new THREE.LineBasicMaterial({ color: 0x5b6b60 }));
  trashGroup.add(border);
}
scene.add(trashGroup);

// static: ocean (animated surface + dim seabed)
const oceanGroup = new THREE.Group();
const WATER_SEG = 96;
const waterGeo = new THREE.PlaneGeometry(2 * ARENA_HALF + 10, 2 * ARENA_HALF + 10, WATER_SEG, WATER_SEG);
const water = new THREE.Mesh(waterGeo, new THREE.MeshStandardMaterial({
  color: 0x1f5f8b, roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false,
}));
water.renderOrder = 2;
oceanGroup.add(water);
{
  const bed = new THREE.Mesh(new THREE.PlaneGeometry(2 * ARENA_HALF + 10, 2 * ARENA_HALF + 10),
    new THREE.MeshStandardMaterial({ color: 0x0c2233, roughness: 1 }));
  bed.position.z = -8;
  oceanGroup.add(bed);
}
scene.add(oceanGroup);

function updateWater(w, t) {
  const pos = waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setZ(i, waveHeight(w, pos.getX(i), pos.getY(i), t));
  pos.needsUpdate = true;
  waterGeo.computeVertexNormals();
}

// ------------------------------------------------------------------ dynamic objects
const dyn = new THREE.Group();
scene.add(dyn);

function makeDrone(color) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.3, emissive: 0x000000 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.12), mat);
  g.add(body);
  const armMat = new THREE.MeshStandardMaterial({ color: 0x2b313a });
  for (const a of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.04, 0.03), armMat);
    arm.rotation.z = a;
    g.add(arm);
  }
  const rotorMat = new THREE.MeshStandardMaterial({ color: 0xdfe8f2, transparent: true, opacity: 0.55 });
  for (const [x, y] of [[0.28, 0.28], [-0.28, 0.28], [0.28, -0.28], [-0.28, -0.28]]) {
    const r = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.015, 20), rotorMat);
    r.rotation.x = Math.PI / 2;
    r.position.set(x, y, 0.07);
    g.add(r);
  }
  const gripMat = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, emissive: 0x000000 });
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.14, 8), gripMat);
  stem.rotation.x = Math.PI / 2;
  stem.position.z = -0.1;
  g.add(stem);
  const claws = [];
  for (const s of [-1, 1]) {
    const c = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.05, 0.08), gripMat);
    c.position.set(0.05 * s, 0, -0.19);
    g.add(c);
    claws.push([c, s]);
  }
  const front = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  front.rotation.z = -Math.PI / 2;
  front.position.set(0.22, 0, 0.0);
  g.add(front);
  g.userData = { mat, gripMat, claws, target: new THREE.Vector3(), qtarget: new THREE.Quaternion(), fresh: true };
  return g;
}

function labelSprite(text, color) {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 32;
  const x = c.getContext("2d");
  x.fillStyle = "rgba(10,14,20,0.8)";
  x.fillRect(0, 0, 64, 32);
  x.fillStyle = "#" + color.toString(16).padStart(6, "0");
  x.font = "bold 22px sans-serif";
  x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, 32, 17);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(0.9, 0.45, 1);
  s.renderOrder = 10;
  return s;
}

const drones = [];
const obstacles = new Map();
const zones = new Map();
const payloads = new Map();
const anchors = [];
for (let k = 0; k < 4; k++) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.18), new THREE.MeshBasicMaterial({ color: ANCHOR_COLORS[k] }));
  g.add(m);
  const l = labelSprite("a" + k, ANCHOR_COLORS[k]);
  l.position.z = 0.55;
  g.add(l);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1, 48), new THREE.MeshBasicMaterial({
    color: ANCHOR_COLORS[k], transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
  g.add(ring);
  g.userData = { ring };
  g.visible = false;
  dyn.add(g);
  anchors.push(g);
}

const boat = new THREE.Group();
{
  const hull = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xd9dee5, roughness: 0.6 }));
  boat.add(hull);
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.6, 0.6), new THREE.MeshStandardMaterial({ color: 0x8a5a3c }));
  cabin.position.set(-0.2, 0, 0.6);
  boat.add(cabin);
  boat.userData = { hull, cabin };
  boat.visible = false;
  dyn.add(boat);
}

const netMat = new THREE.LineBasicMaterial({ color: 0xffe08a, transparent: true, opacity: 0.9 });
let net = null;
function ensureNet(rows, cols) {
  if (net && net.userData.rows === rows && net.userData.cols === cols) return net;
  if (net) dyn.remove(net);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(rows * cols * 3), 3));
  const idx = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const i = r * cols + c;
    if (c + 1 < cols) idx.push(i, i + 1);
    if (r + 1 < rows) idx.push(i, i + cols);
  }
  geo.setIndex(idx);
  net = new THREE.LineSegments(geo, netMat);
  net.frustumCulled = false;
  net.userData = { rows, cols };
  dyn.add(net);
  return net;
}

const MAX_FISH = 256;
const fishMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 8, 6), new THREE.MeshStandardMaterial({ color: 0xff9f43, emissive: 0x3a1a00 }), MAX_FISH);
fishMesh.count = 0;
fishMesh.frustumCulled = false;
dyn.add(fishMesh);
const fishDummy = new THREE.Object3D();
fishDummy.scale.set(1.8, 0.8, 0.8);

const MAX_PTS = 8192;
function makePoints(color, size) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAX_PTS * 3), 3));
  geo.setDrawRange(0, 0);
  const p = new THREE.Points(geo, new THREE.PointsMaterial({ color, size, sizeAttenuation: true, transparent: true, opacity: 0.85, depthWrite: false }));
  p.frustumCulled = false;
  p.renderOrder = 5;
  dyn.add(p);
  return p;
}
const nirPts = makePoints(0x9aa3ad, 0.07);
const greenPts = makePoints(0x39ff6a, 0.13);
function setPoints(p, arr) {
  const a = p.geometry.attributes.position.array;
  const n = Math.min(arr.length, MAX_PTS);
  for (let i = 0; i < n; i++) { a[3 * i] = arr[i][0]; a[3 * i + 1] = arr[i][1]; a[3 * i + 2] = arr[i][2]; }
  p.geometry.attributes.position.needsUpdate = true;
  p.geometry.setDrawRange(0, n);
}

const dropMarker = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.6, 32), new THREE.MeshBasicMaterial({ color: 0xf2b84b, side: THREE.DoubleSide }));
dropMarker.visible = false;
scene.add(dropMarker);

function syncPool(map, items, create, update) {
  const seen = new Set();
  for (const it of items) {
    let o = map.get(it.name);
    if (!o) { o = create(it); map.set(it.name, o); dyn.add(o); }
    update(o, it);
    seen.add(it.name);
  }
  for (const [name, o] of map) if (!seen.has(name)) { dyn.remove(o); map.delete(name); }
}

const unitBox = new THREE.BoxGeometry(1, 1, 1);
function createObstacle(it) {
  const m = new THREE.Mesh(unitBox, new THREE.MeshStandardMaterial({ color: 0x6b7a8f, roughness: 0.8 }));
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(unitBox), new THREE.LineBasicMaterial({ color: 0x0b1016 }));
  m.add(e);
  return m;
}
function updateObstacle(m, it) {
  m.position.set(...it.pos);
  m.scale.set(...it.size);
  m.material.color.setHex(it.moving ? 0xff8a3d : (/obstacle_(6|1)$/.test(it.name) && dropped.has(it.name) ? 0xe05d9a : 0x6b7a8f));
}
const dropped = new Set();

function createZone(it) {
  const g = new THREE.Group();
  if (it.kind === "nofly") {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 40, 1, true), new THREE.MeshBasicMaterial({
      color: 0xff4d6d, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
    c.rotation.x = Math.PI / 2;
    g.add(c);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 48), new THREE.MeshBasicMaterial({ color: 0xff4d6d, side: THREE.DoubleSide }));
    ring.position.z = 0.03;
    g.add(ring);
    g.userData = { body: c, ring };
  } else if (it.kind === "bin") {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(1, 0.85, 1, 32, 1, true), new THREE.MeshStandardMaterial({
      color: 0x3d8bfd, side: THREE.DoubleSide, roughness: 0.6 }));
    c.rotation.x = Math.PI / 2;
    g.add(c);
    const bottom = new THREE.Mesh(new THREE.CircleGeometry(0.85, 32), new THREE.MeshStandardMaterial({ color: 0x1d3f73 }));
    bottom.position.z = 0.02;
    g.add(bottom);
    g.userData = { body: c, bottom };
  } else {
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1, 64), new THREE.MeshBasicMaterial({
      color: 0x2fbf71, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
    g.add(ring);
    g.userData = { ring };
  }
  return g;
}
function updateZone(g, it) {
  const r = it.radius, h = it.height;
  if (it.kind === "nofly") {
    g.position.set(it.pos[0], it.pos[1], 0);
    g.userData.body.scale.set(r, h, r);
    g.userData.body.position.z = h / 2;
    g.userData.ring.scale.set(r, r, 1);
  } else if (it.kind === "bin") {
    g.position.set(it.pos[0], it.pos[1], 0);
    g.userData.body.scale.set(r, h, r);
    g.userData.body.position.z = h / 2;
    g.userData.bottom.scale.set(r, r, 1);
  } else {
    g.position.set(it.pos[0], it.pos[1], (it.pos[2] || 0) + 0.05);
    g.userData.ring.scale.set(r, r, 1);
  }
}

function createPayload(it) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(unitBox, new THREE.MeshStandardMaterial({ color: 0x2f6b3a, roughness: 0.9 }));
  g.add(body);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(0.07, 0.015, 8, 16), new THREE.MeshStandardMaterial({ color: 0xffffff }));
  handle.rotation.x = Math.PI / 2;
  g.add(handle);
  const cross = new THREE.Group();
  const cm = new THREE.MeshBasicMaterial({ color: 0xe0303f });
  const a = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.06, 0.01), cm), b = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.2, 0.01), cm);
  cross.add(a, b);
  g.add(cross);
  g.userData = { body, handle, cross };
  return g;
}
function updatePayload(g, it) {
  g.position.set(...it.pos);
  const [sx, sy, sz] = it.size;
  g.userData.body.scale.set(sx, sy, sz);
  g.userData.body.material.color.setHex(it.kind === "cooler" ? 0xe8eef5 : 0x2f6b3a);
  g.userData.handle.position.z = sz / 2 + 0.07;
  g.userData.cross.visible = it.kind === "cooler";
  g.userData.cross.position.z = sz / 2 + 0.006;
}

// ------------------------------------------------------------------ frame handling
let latest = null, latestAt = 0, curScene = null;
const $ = (id) => document.getElementById(id);

function applyFrame(st) {
  if (st.scene !== curScene) {
    curScene = st.scene;
    trashGroup.visible = st.scene === "trash";
    oceanGroup.visible = st.scene === "ocean";
    document.body.classList.toggle("scene-trash", st.scene === "trash");
    document.body.classList.toggle("scene-ocean", st.scene === "ocean");
    for (const b of document.querySelectorAll("#scene-toggle button")) b.classList.toggle("active", b.dataset.scene === st.scene);
    renderExamples(st.scene);
    dropped.clear();
  }
  // drones
  while (drones.length < st.drones.length) {
    const d = makeDrone(AGENT_COLORS[drones.length % AGENT_COLORS.length]);
    dyn.add(d);
    drones.push(d);
  }
  drones.forEach((d, i) => {
    const s = st.drones[i];
    if (!s) { d.visible = false; return; }
    d.visible = s.alive !== false;
    d.userData.target.set(...s.pos);
    d.userData.qtarget.set(s.quat[1], s.quat[2], s.quat[3], s.quat[0]);   // (w,x,y,z) -> three (x,y,z,w)
    if (d.userData.fresh) { d.position.copy(d.userData.target); d.quaternion.copy(d.userData.qtarget); d.userData.fresh = false; }
    d.userData.mat.emissive.setHex(s.holding ? 0x2fbf71 : 0x000000);
    d.userData.gripMat.color.setHex(s.holding ? 0x2fbf71 : s.grip ? 0xf2b84b : 0x9aa3ad);
    for (const [c, sgn] of d.userData.claws) c.position.x = (s.grip ? 0.025 : 0.06) * sgn;
  });
  syncPool(obstacles, st.obstacles, createObstacle, updateObstacle);
  syncPool(zones, st.zones, createZone, updateZone);
  syncPool(payloads, st.payloads || [], createPayload, updatePayload);
  st.anchors.forEach((a, k) => {
    const g = anchors[k];
    g.visible = a.kind !== "none";
    g.position.set(...a.pos);
    g.userData.ring.visible = a.radius > 0;
    if (a.radius > 0) g.userData.ring.scale.set(a.radius, a.radius, 1);
  });
  if (st.boat) {
    boat.visible = true;
    const [bx, by, bz] = st.boat.size;
    boat.userData.hull.scale.set(bx, by, bz);
    boat.userData.cabin.position.set(-bx * 0.15, 0, bz / 2 + 0.3);
    boat.userData.cabin.scale.set(bx / 1.2, by / 1.6, 1);
    boat.userData.base = st.boat.pos;
  } else boat.visible = false;
  if (st.net) {
    const n = ensureNet(st.net.rows, st.net.cols);
    n.visible = true;
    const arr = n.geometry.attributes.position.array;
    st.net.nodes.forEach((p, i) => { arr[3 * i] = p[0]; arr[3 * i + 1] = p[1]; arr[3 * i + 2] = p[2]; });
    n.geometry.attributes.position.needsUpdate = true;
  } else if (net) net.visible = false;
  const fish = st.fish || [];
  fishMesh.count = Math.min(fish.length, MAX_FISH);
  for (let i = 0; i < fishMesh.count; i++) {
    fishDummy.position.set(...fish[i]);
    fishDummy.rotation.z = (i * 0.7) % (2 * Math.PI);
    fishDummy.updateMatrix();
    fishMesh.setMatrixAt(i, fishDummy.matrix);
  }
  fishMesh.instanceMatrix.needsUpdate = true;
  setPoints(nirPts, (st.lidar && st.lidar.nir) || []);
  setPoints(greenPts, (st.lidar && st.lidar.green) || []);
  const w = st.wind || [0, 0, 0];
  $("g-wind").textContent = Math.hypot(w[0], w[1], w[2]).toFixed(1) + " m/s";
}

function fmt(v, d = 2) { return v === null || v === undefined ? "-" : Number(v).toFixed(d); }

let lastDemo = null;
function applyDemo(d) {
  lastDemo = d;
  const fallback = d.policy_mode !== "trained policy";
  const pm = $("policy-mode");
  pm.textContent = "policy: " + d.policy_mode;
  pm.className = "pill " + (fallback ? "fallback" : "good");
  const mm = $("mission-mode");
  mm.textContent = "compiler: " + d.mission_mode;
  mm.className = "pill " + (d.mission_mode === "Gemini" ? "good" : "fallback");
  const s = $("status");
  s.textContent = d.status;
  s.className = "pill status " + d.status.replace(/\s+/g, "-");
  $("message").textContent = d.message || "";
  $("g-reward").textContent = fmt(d.reward);
  $("g-swing").textContent = fmt(d.swing) + " rad/s";
  $("g-shield").textContent = d.shield_hits;
  $("g-time").textContent = fmt(d.time, 1) + " s";
  const banner = $("mode-banner");
  banner.classList.toggle("hidden", !(fallback && d.flying));
  banner.textContent = "Flying the " + d.policy_mode + " controller, not the trained network";
  $("btn-stop").disabled = !d.flying;
  buildDisableButtons(d.num_agents);
  drawZmap(d.zmap);
}

// ------------------------------------------------------------------ z map
const zc = $("zmap"), zx = zc.getContext("2d");
let zmapKey = "";
function drawZmap(zm) {
  const key = JSON.stringify(zm || {});
  if (key === zmapKey) return;
  zmapKey = key;
  const W = zc.width, H = zc.height;
  zx.clearRect(0, 0, W, H);
  zx.strokeStyle = "#2a3442";
  zx.beginPath(); zx.moveTo(W / 2, 6); zx.lineTo(W / 2, H - 6); zx.moveTo(6, H / 2); zx.lineTo(W - 6, H / 2); zx.stroke();
  const note = $("zmap-note");
  if (!zm || !zm.available || !zm.points.length) {
    zx.fillStyle = "#8b98a8"; zx.font = "13px system-ui"; zx.textAlign = "center";
    zx.fillText((zm && zm.note) || "z map available after training", W / 2, H / 2 - 8);
    note.textContent = "Missions become points in the policy's task space once a trained model is loaded.";
    return;
  }
  note.textContent = "PCA of z: reference missions (grey) and yours (blue).";
  zx.font = "11px system-ui"; zx.textAlign = "left";
  for (const p of zm.points) {
    const x = W / 2 + p.x * (W / 2 - 40), y = H / 2 - p.y * (H / 2 - 16);
    zx.fillStyle = p.current ? "#4ea8ff" : "#8b98a8";
    zx.beginPath(); zx.arc(x, y, p.current ? 6 : 4, 0, 2 * Math.PI); zx.fill();
    zx.fillStyle = p.current ? "#cfe6ff" : "#aab4c0";
    zx.fillText(p.name, Math.min(x + 7, W - 70), y + 4);
  }
}

// ------------------------------------------------------------------ UI
async function api(path, body) {
  const r = await fetch(path, { method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
function toast(msg) { $("message").textContent = msg; }

function renderExamples(sc) {
  const ex = $("examples");
  ex.innerHTML = "";
  for (const t of EXAMPLES[sc] || []) {
    const s = document.createElement("span");
    s.textContent = t;
    s.onclick = () => { $("mission-text").value = t; };
    ex.appendChild(s);
  }
}

for (const b of document.querySelectorAll("#scene-toggle button")) {
  b.onclick = async () => {
    b.disabled = true;
    try { await api("/api/scene", { scene: b.dataset.scene }); hideCard(); drones.forEach((d) => (d.userData.fresh = true)); }
    catch (e) { toast("Scene switch failed: " + e.message); }
    b.disabled = false;
  };
}

function hideCard() { $("card").classList.add("hidden"); }
function showCard(m) {
  $("card").classList.remove("hidden");
  const src = $("card-source");
  src.textContent = m.source;
  src.className = "tag " + (m.source === "Gemini" ? "gemini" : "fallback");
  const ul = $("card-expl");
  ul.innerHTML = "";
  for (const line of m.explanation || []) { const li = document.createElement("li"); li.textContent = line; ul.appendChild(li); }
  const an = $("card-anchors");
  an.innerHTML = "";
  for (const [k, v] of Object.entries(m.anchors || {})) {
    const div = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = k + " = " + (typeof v.target === "string" ? v.target : JSON.stringify(v.target));
    div.appendChild(b);
    div.appendChild(document.createTextNode(v.description ? "  " + v.description : ""));
    an.appendChild(div);
  }
  $("card-reward").textContent = m.reward;
  $("card-done").textContent = m.done_when || "(none: flies until stopped or time is up)";
  const warn = $("card-warn");
  const ws = (m.warnings || []).filter(Boolean);
  warn.classList.toggle("hidden", !ws.length);
  warn.textContent = ws.length ? "Note: " + ws.join("; ") : "";
}

$("btn-compile").onclick = async () => {
  const text = $("mission-text").value.trim();
  if (!text) { toast("Type a mission first."); return; }
  const b = $("btn-compile");
  b.disabled = true; b.textContent = "Compiling...";
  try { showCard(await api("/api/mission", { text })); }
  catch (e) { toast("Compile failed: " + e.message); }
  b.disabled = false; b.textContent = "Compile";
};
$("mission-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("btn-compile").click(); });
$("btn-approve").onclick = async () => {
  const b = $("btn-approve");
  b.disabled = true;
  try { await api("/api/approve", {}); hideCard(); }
  catch (e) { toast("Approve failed: " + e.message); }
  b.disabled = false;
};
$("btn-cancel").onclick = async () => { hideCard(); try { await api("/api/cancel", {}); } catch (e) { /* ignore */ } };
$("btn-stop").onclick = () => api("/api/stop", {}).catch((e) => toast(e.message));
$("btn-reset").onclick = () => { dropped.clear(); api("/api/reset", {}).catch((e) => toast(e.message)); };
$("btn-gust").onclick = () => {
  const ang = Math.random() * 2 * Math.PI, mag = 5 + Math.random() * 2;
  api("/api/perturb", { kind: "gust", vec: [mag * Math.cos(ang), mag * Math.sin(ang), 0], duration: 2.5 })
    .then(() => toast(`Gust: ${mag.toFixed(1)} m/s for 2.5 s`)).catch((e) => toast(e.message));
};
let waveTimer = null;
$("wave-scale").oninput = (e) => {
  const v = parseFloat(e.target.value);
  $("wave-val").textContent = v.toFixed(1) + "x";
  clearTimeout(waveTimer);
  waveTimer = setTimeout(() => api("/api/perturb", { kind: "waves", scale: v }).catch((er) => toast(er.message)), 80);
};

let clickMode = null;
function setClickMode(m) {
  clickMode = clickMode === m ? null : m;
  $("btn-drop").classList.toggle("on", clickMode === "drop");
  $("btn-move").classList.toggle("on", clickMode === "move");
  canvas.style.cursor = clickMode ? "crosshair" : "";
  if (clickMode) toast(clickMode === "drop" ? "Click the ground to drop an obstacle." : "Click the ground to move the bag.");
}
$("btn-drop").onclick = () => setClickMode("drop");
$("btn-move").onclick = () => setClickMode("move");

let builtDisable = -1;
function buildDisableButtons(n) {
  if (n === builtDisable) return;
  builtDisable = n;
  const row = $("disable-row");
  row.innerHTML = "";
  if (n <= 1) return;
  for (let i = 0; i < n; i++) {
    const b = document.createElement("button");
    b.textContent = "Disable drone " + i;
    b.style.borderColor = "#" + AGENT_COLORS[i % AGENT_COLORS.length].toString(16).padStart(6, "0");
    b.onclick = () => api("/api/perturb", { kind: "disable_drone", agent: i }).then(() => toast(`Drone ${i} knocked out`)).catch((e) => toast(e.message));
    row.appendChild(b);
  }
}

const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
let downAt = null;
canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", (e) => {
  if (!clickMode || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const p = new THREE.Vector3();
  if (!ray.ray.intersectPlane(ground, p)) return;
  if (Math.abs(p.x) > ARENA_HALF || Math.abs(p.y) > ARENA_HALF) { toast("Click inside the arena."); return; }
  const mode = clickMode;
  dropMarker.position.set(p.x, p.y, 0.05);
  dropMarker.visible = true;
  setTimeout(() => (dropMarker.visible = false), 800);
  if (mode === "drop") {
    api("/api/perturb", { kind: "drop_obstacle", pos: [p.x, p.y] })
      .then(() => { dropped.add(curScene === "trash" ? "obstacle_6" : "obstacle_1"); toast("Obstacle dropped"); })
      .catch((er) => toast(er.message));
  } else {
    api("/api/perturb", { kind: "move_payload", pos: [p.x, p.y] }).then(() => toast("Bag moved")).catch((er) => toast(er.message));
  }
  setClickMode(mode);   // one click per activation
});

// ------------------------------------------------------------------ websocket
let ws = null;
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("conn").textContent = "live"; $("conn").className = "pill good"; };
  ws.onclose = () => { $("conn").textContent = "offline, retrying"; $("conn").className = "pill bad"; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "frame") {
      latest = msg.state; latestAt = performance.now();
      applyFrame(msg.state);
      applyDemo(msg.demo);
    } else if (msg.type === "hello" && msg.pending) {
      showCard(msg.pending);
    }
  };
}

// ------------------------------------------------------------------ render loop
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const a = 1 - Math.exp(-dt * 18);
  for (const d of drones) {
    d.position.lerp(d.userData.target, a);
    d.quaternion.slerp(d.userData.qtarget, a);
  }
  if (latest && latest.scene === "ocean" && latest.waves) {
    const t = latest.t + Math.min((performance.now() - latestAt) / 1000, 0.15);
    updateWater(latest.waves, t);
    if (boat.visible && boat.userData.base) {
      const b = boat.userData.base;
      boat.position.set(b[0], b[1], waveHeight(latest.waves, b[0], b[1], t) + 0.1);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
resize();
renderExamples("trash");
connect();
animate();
