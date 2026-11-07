// SkyHand judge demo viewer. Renders env.render_state frames streamed over /ws.
// World frame is z-up, metres (same as the simulator), so we set three.js up = +z.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

let ARENA_HALF = 25;              // replaced by frame.arena_half
const AGENT_COLORS = [0x4ea8ff, 0xffc94e, 0xc77dff, 0x5ce1e6];
const ANCHOR_COLORS = [0xff6b6b, 0x2fbf71, 0xf2b84b, 0xb388ff];
const EXAMPLES = {
  trash: [
    "pick up the trash bag and put it in the bin without hitting the cars",
    "grab the pizza box and drop it in the garbage truck, leave the backpack alone",
    "fly to the bin and hover",
  ],
  delivery: [
    "deliver the organ cooler to the drop-off pad as smoothly as possible and stay out of the no-fly zones",
    "carry the cooler to the clinic pad, avoid the vultures",
    "fly to the drop-off pad",
  ],
  ocean: [
    "net the school closest to the boat without catching any whales or dolphins",
    "gently haul the net over the fish and bring it to the boat",
    "fly to the whale",
  ],
  world: [
    "drone 0 pick up the pizza box and drop it in the garbage truck, drone 1 deliver the organ cooler to the clinic, drone 2 fly to the boat",
    "collect the trash and put it in the garbage truck without hitting the cars",
    "net the school closest to the boat without catching any whales or dolphins",
  ],
};
const ITEM_LABELS = {
  trash_bag: "trash bag", pizza_box: "pizza box", cardboard_box: "cardboard box", bottle: "bottle",
  backpack: "backpack", ball: "ball", flower_pot: "flower pot", cooler: "organ cooler",
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
scene.fog = new THREE.Fog(0x0b1016, 70, 180);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 4000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
function frameCamera() {
  const h = ARENA_HALF;
  camera.position.set(-0.75 * h, -1.25 * h, 1.3 * h);
  controls.target.set(0, 0, 1);
  camera.far = Math.max(600, 10 * h);
  camera.updateProjectionMatrix();
  scene.fog.near = 2.6 * h; scene.fog.far = 6.5 * h;
  controls.maxDistance = 5 * h;
  controls.update();
}

// follow camera: keeps the orbit offset and moves it with the selected drone (default on in big
// arenas). The drone picker selects which drone (multi-drone scenes); Overview frames the whole map.
let follow = false, followIdx = 0;
const FOLLOW_OFFSET = new THREE.Vector3(-28, -40, 24);
function updateCamButtons() {
  const b = document.getElementById("btn-follow");
  if (b) { b.classList.toggle("on", follow); b.textContent = follow ? `Following drone ${followIdx}` : `Follow drone ${followIdx}`; }
  const o = document.getElementById("btn-overview");
  if (o) o.classList.toggle("on", !follow);
  for (const p of document.querySelectorAll("#drone-picker button")) p.classList.toggle("active", follow && +p.dataset.idx === followIdx);
  drones.forEach((d, i) => { d.userData.beacon.material.opacity = (!follow || i === followIdx) ? 0.85 : 0.35; });
  showDroneTask();
}
function setFollow(on, reframe = true) {
  follow = on;
  const d = drones[followIdx];
  if (on && reframe && d) {
    controls.target.copy(d.position);
    camera.position.copy(d.position).add(FOLLOW_OFFSET);
    controls.update();
  } else if (!on && reframe) frameCamera();
  updateCamButtons();
}
// select drone i and follow it; keeps the current viewing offset when already following
function followDrone(i) {
  if (!drones[i]) return;
  const off = camera.position.clone().sub(controls.target);
  const keep = follow && off.length() < 150;
  followIdx = i;
  follow = true;
  controls.target.copy(drones[i].position);
  camera.position.copy(drones[i].position).add(keep ? off : FOLLOW_OFFSET);
  controls.update();
  updateCamButtons();
}
frameCamera();

scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x223322, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(-20, -12, 45);
scene.add(sun);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// ------------------------------------------------------------------ static world (rebuilt when the arena size changes)
const world = new THREE.Group();
scene.add(world);
let flatGround = null, border = null, oceanGroup = null, waterGeo = null, lake = null, sea = null;
const WATER_SEG = 96;
const SEA_SEG = 256, MAX_WAVES = 8;

// same formula as waveHeight() below, evaluated per vertex (height + analytic normal)
function makeSea(h) {
  const geo = new THREE.PlaneGeometry(2 * h + 10, 2 * h + 10, SEA_SEG, SEA_SEG);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x1f5f8b, roughness: 0.3, metalness: 0.1, transparent: true, opacity: 0.62, depthWrite: false,
  });
  const u = {
    uN: { value: 0 }, uT: { value: 0 }, uLevel: { value: 0 },
    uAmp: { value: new Array(MAX_WAVES).fill(0) }, uK: { value: new Array(MAX_WAVES).fill(0) },
    uDir: { value: new Array(MAX_WAVES).fill(0) }, uOmega: { value: new Array(MAX_WAVES).fill(0) },
    uPhase: { value: new Array(MAX_WAVES).fill(0) },
  };
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>
uniform int uN; uniform float uT; uniform float uLevel;
uniform float uAmp[${MAX_WAVES}]; uniform float uK[${MAX_WAVES}]; uniform float uDir[${MAX_WAVES}];
uniform float uOmega[${MAX_WAVES}]; uniform float uPhase[${MAX_WAVES}];`)
      .replace("#include <beginnormal_vertex>", `float skyH = 0.0; vec2 skyG = vec2(0.0);
for (int i = 0; i < ${MAX_WAVES}; i++) {
  if (i >= uN) break;
  vec2 dv = vec2(cos(uDir[i]), sin(uDir[i]));
  float ph = uK[i] * dot(dv, position.xy) - uOmega[i] * uT + uPhase[i];
  skyH += uAmp[i] * cos(ph);
  skyG -= uAmp[i] * uK[i] * sin(ph) * dv;
}
vec3 objectNormal = normalize(vec3(-skyG, 1.0));
#ifdef USE_TANGENT
vec3 objectTangent = vec3(tangent.xyz);
#endif`)
      .replace("#include <begin_vertex>", "vec3 transformed = vec3(position.xy, skyH + uLevel);");
  };
  const m = new THREE.Mesh(geo, mat);
  m.renderOrder = 2;
  m.frustumCulled = false;
  m.visible = false;
  m.userData = { u };
  return m;
}
// water: {level, waves} with both set (world); t: sim time
function updateSea(water, t) {
  const u = sea.userData.u, w = water.waves, n = Math.min(w.amp.length, MAX_WAVES);
  u.uN.value = n; u.uT.value = t; u.uLevel.value = water.level || 0;
  for (let i = 0; i < n; i++) {
    u.uAmp.value[i] = w.amp[i]; u.uK.value[i] = w.k[i]; u.uDir.value[i] = w.dir[i];
    u.uOmega.value[i] = w.omega[i]; u.uPhase.value[i] = w.phase[i];
  }
}
let builtArena = -1;

let arenaChanged = false, maxSpeed = null;
function buildArena(h) {
  if (h === builtArena) return;
  builtArena = h;
  arenaChanged = true;
  ARENA_HALF = h;
  for (const o of [flatGround, border, oceanGroup, lake]) if (o) world.remove(o);
  flatGround = new THREE.Group();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2 * h, 2 * h),
    new THREE.MeshStandardMaterial({ color: 0x2a2f33, roughness: 0.95 }));
  flatGround.add(ground);
  const grid = new THREE.GridHelper(2 * h, Math.round(h * 2 / (h > 60 ? 25 : 2.5)), 0x3b4448, 0x30363a);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.01;
  flatGround.add(grid);
  flatGround.userData = { ground };
  world.add(flatGround);
  border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * h, 2 * h, 0.01)),
    new THREE.LineBasicMaterial({ color: 0x5b6b60 }));
  border.position.z = 0.02;
  world.add(border);
  // ocean: animated surface + dim seabed
  oceanGroup = new THREE.Group();
  waterGeo = new THREE.PlaneGeometry(2 * h + 10, 2 * h + 10, WATER_SEG, WATER_SEG);
  const water = new THREE.Mesh(waterGeo, new THREE.MeshStandardMaterial({
    color: 0x1f5f8b, roughness: 0.35, metalness: 0.1, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false,
  }));
  water.renderOrder = 2;
  oceanGroup.add(water);
  const bed = new THREE.Mesh(new THREE.PlaneGeometry(2 * h + 10, 2 * h + 10), new THREE.MeshStandardMaterial({ color: 0x0c2233, roughness: 1 }));
  bed.position.z = -8;
  oceanGroup.add(bed);
  oceanGroup.userData = { water, bed };
  oceanGroup.visible = false;
  world.add(oceanGroup);
  // world: sea + lakes at the tide level with waves on top, over the whole arena (the terrain
  // hides it on land). Displaced on the GPU, so a fine grid costs nothing per frame.
  if (sea) { world.remove(sea); sea.geometry.dispose(); }
  sea = makeSea(h);
  world.add(sea);
  // lakes: flat plane at water.level, the terrain pokes through it
  lake = new THREE.Mesh(new THREE.PlaneGeometry(2 * h, 2 * h), new THREE.MeshStandardMaterial({
    color: 0x2a78b5, roughness: 0.25, metalness: 0.1, transparent: true, opacity: 0.72, depthWrite: false,
  }));
  lake.renderOrder = 2;
  lake.visible = false;
  world.add(lake);
  frameCamera();
}
buildArena(25);

function updateWater(w, t) {
  const pos = waterGeo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setZ(i, waveHeight(w, pos.getX(i), pos.getY(i), t));
  pos.needsUpdate = true;
  waterGeo.computeVertexNormals();
}

// ------------------------------------------------------------------ terrain heightfield
// heights[i][j] at x = -half + 2*half*j/(res-1), y = -half + 2*half*i/(res-1)
let terrainMesh = null, terrainKey = null, terrainData = null, terrainFetching = null;
let terrainStyle = { level: null, city: false }, terrainStyleKey = "";
const C_MUD = new THREE.Color(0x6b5a3e), C_SAND = new THREE.Color(0xd8c48a), C_GRASS = new THREE.Color(0x4f8a3a),
  C_GRASS2 = new THREE.Color(0x3d6b2e), C_ROCK = new THREE.Color(0x7a746c), C_SNOW = new THREE.Color(0xf4f7fa);

export function terrainColor(h, level, hmax, out) {
  const span = Math.max(hmax - level, 1);
  const g1 = level + Math.max(3.5, 0.4 * span), r1 = level + Math.max(7, 0.68 * span), s1 = level + Math.max(10, 0.85 * span);
  if (h < level - 0.3) return out.copy(C_MUD).lerp(C_SAND, Math.max(0, 1 + (h - level) / 2));
  if (h < level + 0.6) return out.copy(C_SAND);
  if (h < g1) return out.copy(C_SAND).lerp(C_GRASS, Math.min(1, (h - level - 0.6) / 0.6)).lerp(C_GRASS2, Math.max(0, (h - level) / (g1 - level)) * 0.6);
  if (h < r1) return out.copy(C_GRASS2).lerp(C_ROCK, Math.min(1, (h - g1) / Math.max(r1 - g1, 0.5) * 1.6));
  if (h < s1) return out.copy(C_ROCK).lerp(C_SNOW, Math.max(0, (h - s1 + 1.0) / 1.0));
  return out.copy(C_SNOW);
}

// style: {level: water level or null (no lakes: no beaches), city: asphalt instead of nature}
const C_ASPHALT = new THREE.Color(0x2c3034);
function colorTerrain(style) {
  terrainStyle = style;
  if (!terrainMesh || !terrainData) return;
  const geo = terrainMesh.geometry, pos = geo.attributes.position, col = geo.attributes.color;
  const c = new THREE.Color();
  const level = style.level === null ? terrainData.hmin - 1.5 : style.level;
  for (let v = 0; v < pos.count; v++) {
    if (style.city) c.copy(C_ASPHALT);
    else terrainColor(pos.getZ(v), level, terrainData.hmax, c);
    col.setXYZ(v, c.r, c.g, c.b);
  }
  col.needsUpdate = true;
}

function buildTerrain(t) {
  if (!t || !t.heights) return;
  const key = t.key !== undefined ? t.key : "v" + t.version;
  if (key === terrainKey && terrainMesh) return;
  terrainKey = key;
  if (terrainMesh) { world.remove(terrainMesh); terrainMesh.geometry.dispose(); }
  const res = t.res, half = t.half;
  const geo = new THREE.PlaneGeometry(2 * half, 2 * half, res - 1, res - 1);
  const pos = geo.attributes.position;
  let hmax = -Infinity, hmin = Infinity;
  // PlaneGeometry vertex (iy, ix): x = -half + ix*step, y = +half - iy*step  ->  heights row i = res-1-iy
  for (let iy = 0; iy < res; iy++) for (let ix = 0; ix < res; ix++) {
    const h = t.heights[res - 1 - iy][ix];
    pos.setZ(iy * res + ix, h);
    if (h > hmax) hmax = h;
    if (h < hmin) hmin = h;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
  geo.computeVertexNormals();
  terrainMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, flatShading: false }));
  terrainData = { res, half, hmax, hmin, heights: t.heights };
  world.add(terrainMesh);
  colorTerrain(terrainStyle);
}

function fetchTerrain(key) {
  if (terrainFetching === key) return;
  terrainFetching = key;
  fetch("/api/terrain").then((r) => r.json()).then((j) => { if (j.terrain) buildTerrain(j.terrain); })
    .catch(() => {}).finally(() => { setTimeout(() => { if (terrainFetching === key) terrainFetching = null; }, 2000); });
}

// ------------------------------------------------------------------ helpers
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const unitSphere = new THREE.SphereGeometry(1, 20, 14);
const unitCyl = new THREE.CylinderGeometry(0.5, 0.5, 1, 20).rotateX(Math.PI / 2);   // along z
const std = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.7, ...extra });
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

function labelSprite(text, color, w = 64, height = 0.8) {
  const c = document.createElement("canvas");
  c.width = w * 2; c.height = 64;
  const x = c.getContext("2d");
  x.fillStyle = "rgba(10,14,20,0.82)";
  x.fillRect(0, 0, c.width, c.height);
  x.fillStyle = "#" + color.toString(16).padStart(6, "0");
  x.font = "bold 40px sans-serif";
  x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, c.width / 2, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(height * (c.width / 64), height, 1);
  s.renderOrder = 10;
  return s;
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
  // beacon so the 0.7 m drone is findable in a 50 m arena
  const beacon = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.58, 32), new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthTest: false }));
  beacon.renderOrder = 9;
  g.add(beacon);
  g.userData = { mat, gripMat, claws, beacon, target: new THREE.Vector3(), qtarget: new THREE.Quaternion(), fresh: true };
  return g;
}

const drones = [];
const pools = { items: new Map(), boxes: new Map(), spheres: new Map(), zones: new Map(), animals: new Map() };
const anchors = [];
for (let k = 0; k < 4; k++) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.18), new THREE.MeshBasicMaterial({ color: ANCHOR_COLORS[k] }));
  g.add(m);
  const l = labelSprite("a" + k, ANCHOR_COLORS[k]);
  l.position.z = 0.75;
  g.add(l);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1, 48), new THREE.MeshBasicMaterial({
    color: ANCHOR_COLORS[k], transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
  g.add(ring);
  g.userData = { ring };
  g.visible = false;
  dyn.add(g);
  anchors.push(g);
}
let anchorLabel = null, anchorLabelText = "";   // name of the item anchor a0 is bound to

const boat = new THREE.Group();
{
  const hull = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), std(0xd9dee5, { roughness: 0.6 }));
  boat.add(hull);
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.6, 0.6), std(0x8a5a3c));
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
const fishMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.12, 8, 6), std(0xff9f43, { emissive: 0x3a1a00 }), MAX_FISH);
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
const nirPts = makePoints(0x9aa3ad, 0.09);
const greenPts = makePoints(0x39ff6a, 0.15);
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

// pool keyed by name; an object is rebuilt when its signature (kind / type) changes
function syncPool(map, items, sig, create, update) {
  const seen = new Set();
  for (const it of items || []) {
    let o = map.get(it.name);
    const s = sig(it);
    if (o && o.userData.sig !== s) { dyn.remove(o); o = null; }
    if (!o) { o = create(it); o.userData.sig = s; o.userData.name = it.name; map.set(it.name, o); dyn.add(o); }
    update(o, it);
    seen.add(it.name);
  }
  for (const [name, o] of map) if (!seen.has(name)) { dyn.remove(o); map.delete(name); }
}

// ------------------------------------------------------------------ items (graspable)
// Trash looks like trash (drab, crumpled colours) and gets an orange ground ring;
// non-trash is clean and bright. Built at unit size, scaled by item.size.
function itemDims(it) {
  const s = it.size || [0.3, 0.3, 0.3];
  if (it.shape === "sphere") {
    const r = s[1] > 0 ? s[0] / 2 : s[0];      // full extents, or (radius, 0, 0)
    return [2 * r, 2 * r, 2 * r];
  }
  return s;
}

function createItem(it) {
  const g = new THREE.Group();
  const body = new THREE.Group();       // unit-size parts, scaled by dims
  g.add(body);
  const add = (geo, mat, x = 0, y = 0, z = 0, sx = 1, sy = 1, sz = 1) => {
    const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.scale.set(sx, sy, sz); body.add(m); return m;
  };
  const mats = [];
  const M = (c, e = {}) => { const m = std(c, e); mats.push(m); return m; };
  switch (it.type) {
    case "trash_bag":
      add(unitSphere, M(0x1d241d, { roughness: 0.35 }), 0, 0, -0.03, 0.5, 0.47, 0.44);
      add(new THREE.ConeGeometry(0.12, 0.25, 8), M(0x1d241d), 0, 0, 0.47).rotation.x = Math.PI / 2;
      break;
    case "pizza_box":
      add(unitBox, M(0xc9a26b, { roughness: 0.9 }));
      add(unitBox, M(0xa8432f), 0, 0, 0.5, 0.5, 0.5, 0.04);
      add(unitBox, M(0x6e5a3a), 0.2, -0.15, 0.51, 0.25, 0.18, 0.03);   // grease stain
      break;
    case "cardboard_box":
      add(unitBox, M(0x9c6b3a, { roughness: 0.95 }));
      add(unitBox, M(0xcfb27a), 0, 0, 0.5, 0.12, 1.01, 0.02);           // tape
      add(unitBox, M(0x7d5530), 0.3, 0.2, 0.2, 0.5, 1.02, 0.2).rotation.y = 0.3;  // crushed corner
      break;
    case "bottle":
      add(unitCyl, M(0x3a9a5b, { transparent: true, opacity: 0.8, roughness: 0.2 }), 0, 0, -0.1, 1, 1, 0.8);
      add(unitCyl, M(0x3a9a5b, { transparent: true, opacity: 0.8 }), 0, 0, 0.38, 0.4, 0.4, 0.24);
      add(unitCyl, M(0xd0d0d0), 0, 0, 0.21, 1.02, 1.02, 0.2);           // label
      break;
    case "backpack":
      add(unitBox, M(0x2f6fd6, { roughness: 0.6 }), 0, 0, 0, 1, 1, 1);
      add(unitBox, M(0x1f4fa6), 0, -0.55, -0.15, 0.75, 0.2, 0.45);      // front pocket
      add(new THREE.TorusGeometry(0.22, 0.05, 6, 12), M(0x1a1a1a), 0, 0, 0.55).rotation.x = Math.PI / 2;
      break;
    case "ball":
      add(unitSphere, M(0xffd23f, { roughness: 0.4 }), 0, 0, 0, 0.5, 0.5, 0.5);
      add(new THREE.TorusGeometry(0.5, 0.04, 6, 24), M(0xe63946), 0, 0, 0);
      break;
    case "flower_pot":
      add(new THREE.CylinderGeometry(0.5, 0.36, 0.7, 20).rotateX(Math.PI / 2), M(0xc0603a), 0, 0, -0.15);
      add(unitSphere, M(0x3aa655), 0, 0, 0.3, 0.42, 0.42, 0.3);
      add(unitSphere, M(0xff5fa2, { emissive: 0x401020 }), 0.12, 0.1, 0.5, 0.12, 0.12, 0.12);
      break;
    case "cooler": {
      add(unitBox, M(0xeef2f6, { roughness: 0.5 }));
      const red = M(0xe0303f);
      add(unitBox, red, 0, 0, 0.5, 0.45, 0.14, 0.04);
      add(unitBox, red, 0, 0, 0.5, 0.14, 0.45, 0.04);
      add(new THREE.TorusGeometry(0.18, 0.04, 8, 16, Math.PI), M(0x333333), 0, 0, 0.5).rotation.x = Math.PI / 2;
      break;
    }
    default:
      if (it.shape === "sphere") add(unitSphere, M(it.is_trash ? 0x5a5a40 : 0x4ea8ff), 0, 0, 0, 0.5, 0.5, 0.5);
      else add(unitBox, M(it.is_trash ? 0x6b5a3a : 0x4ea8ff));
  }
  for (const m of mats) m.userData.e0 = m.emissive.getHex();
  // trash marker: orange ring on the ground under the item
  let ring = null;
  if (it.is_trash) {
    ring = new THREE.Mesh(new THREE.RingGeometry(0.75, 0.95, 24), new THREE.MeshBasicMaterial({
      color: 0xff8a3d, transparent: true, opacity: 0.85, side: THREE.DoubleSide, depthWrite: false }));
    g.add(ring);
  }
  g.userData = { body, mats, ring };
  return g;
}
function updateItem(g, it) {
  const d = itemDims(it);
  g.position.set(...it.pos);
  g.userData.body.scale.set(d[0], d[1], d[2]);
  g.userData.dims = d;
  g.userData.info = it;
  g.visible = !it.delivered;
  if (g.userData.ring) {
    const r = Math.max(d[0], d[1]) * 0.75 + 0.1;
    g.userData.ring.scale.set(r, r, 1);
    g.userData.ring.position.z = -d[2] / 2 + 0.03;
    g.userData.ring.visible = !it.held;
  }
  for (const m of g.userData.mats) m.emissive.setHex(it.held ? 0x1f6b3f : m.userData.e0);
}

// ------------------------------------------------------------------ boxes by kind
const CAR_COLORS = [0xd94141, 0x2d6cdf, 0xf2c94c, 0xeeeeee, 0x27ae60, 0x8e44ad, 0x222831];
let windowTex = null;
function windowTexture() {
  if (windowTex) return windowTex;
  const c = document.createElement("canvas");
  c.width = 64; c.height = 64;
  const x = c.getContext("2d");
  x.fillStyle = "#8a9099"; x.fillRect(0, 0, 64, 64);
  x.fillStyle = "#3e4a5c";
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) x.fillRect(6 + 32 * i, 8 + 32 * j, 20, 16);
  windowTex = new THREE.CanvasTexture(c);
  windowTex.wrapS = windowTex.wrapT = THREE.RepeatWrapping;
  return windowTex;
}

function createBox(it) {
  const g = new THREE.Group();
  const body = new THREE.Group();
  g.add(body);
  const add = (geo, mat, x = 0, y = 0, z = 0, sx = 1, sy = 1, sz = 1) => {
    const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.scale.set(sx, sy, sz); body.add(m); return m;
  };
  const k = it.kind;
  if (k === "building") {
    const tex = windowTexture().clone();
    tex.needsUpdate = true;
    const shade = 0.85 + 0.25 * ((hash(it.name) % 100) / 100);
    const mat = std(new THREE.Color(0x9aa0a8).multiplyScalar(shade).getHex(), { map: tex, roughness: 0.85 });
    add(unitBox, mat);
    add(unitBox, std(0x5b616a), 0, 0, 0.5, 1.0, 1.0, 0.01);              // roof
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(unitBox), new THREE.LineBasicMaterial({ color: 0x1b2027 }));
    body.add(e);
    g.userData.tex = tex;
  } else if (k === "car") {
    const col = CAR_COLORS[hash(it.name) % CAR_COLORS.length];
    add(unitBox, std(col, { metalness: 0.4, roughness: 0.4 }), 0, 0, -0.15, 1, 1, 0.6);
    add(unitBox, std(0x1d2733, { metalness: 0.3, roughness: 0.2 }), 0, 0, 0.3, 0.85, 0.55, 0.4);   // cabin (long axis fixed below)
    g.userData.cabinIdx = 1;
  } else if (k === "truck") {
    add(unitBox, std(0x2e7d32, { roughness: 0.6 }), 0, -0.1, 0, 1, 0.8, 1);
    add(unitBox, std(0xeeeeee), 0, 0.4, -0.15, 1, 0.2, 0.7);            // cab at +long end
    add(unitBox, std(0x1b5e20), 0, -0.1, 0.5, 1.01, 0.8, 0.02);
    g.userData.cabinIdx = 1;
  } else if (k === "bench") {
    const wood = std(0x8b5a2b, { roughness: 0.9 });
    add(unitBox, wood, 0, 0, 0.1, 1, 1, 0.2);
    add(unitBox, wood, 0, 0.45, 0.6, 1, 0.1, 0.5);
    add(unitBox, std(0x333333), -0.42, 0, -0.3, 0.06, 0.9, 0.6);
    add(unitBox, std(0x333333), 0.42, 0, -0.3, 0.06, 0.9, 0.6);
  } else if (k === "hydrant") {
    const red = std(0xd32f2f, { roughness: 0.5 });
    add(unitCyl, red, 0, 0, -0.1, 0.8, 0.8, 0.8);
    add(unitSphere, red, 0, 0, 0.3, 0.4, 0.4, 0.25);
    add(unitCyl, red, 0, 0, 0.0, 1.1, 0.25, 0.12).rotation.z = Math.PI / 2;
  } else if (k === "bin") {
    add(unitCyl, std(0x3d8bfd, { roughness: 0.6 }), 0, 0, 0, 1, 1, 1);
    add(unitCyl, std(0x1d3f73), 0, 0, 0.5, 1.05, 1.05, 0.06);
  } else if (k === "trunk") {
    add(unitCyl, std(0x6d4c2f, { roughness: 1 }));
  } else if (k === "dropped") {
    add(unitBox, std(0xe05d9a, { emissive: 0x2a0616 }));
    body.add(new THREE.LineSegments(new THREE.EdgesGeometry(unitBox), new THREE.LineBasicMaterial({ color: 0x0b1016 })));
  } else {
    add(unitBox, std(0x6b7a8f, { roughness: 0.8 }));
    body.add(new THREE.LineSegments(new THREE.EdgesGeometry(unitBox), new THREE.LineBasicMaterial({ color: 0x0b1016 })));
  }
  g.userData.body = body;
  return g;
}
function updateBox(g, it) {
  g.visible = !(it.kind === "boat" && boatShown);
  g.position.set(...it.pos);
  let [sx, sy, sz] = it.size;
  // cars / truck are modelled along +y; rotate when the long axis is x
  const alongX = (it.kind === "car" || it.kind === "truck" || it.kind === "bench") && sx > sy;
  g.rotation.z = alongX ? Math.PI / 2 : 0;
  if (alongX) [sx, sy] = [sy, sx];
  g.userData.body.scale.set(sx, sy, sz);
  if (g.userData.tex) g.userData.tex.repeat.set(Math.max(1, Math.round((sx + sy) / 6)), Math.max(1, Math.round(sz / 3)));
}

// ------------------------------------------------------------------ spheres by kind
const rockGeo = new THREE.DodecahedronGeometry(1, 0), canopyGeo = new THREE.IcosahedronGeometry(1, 1);
function createSphere(it) {
  const g = new THREE.Group();
  if (it.kind === "rock") {
    const m = new THREE.Mesh(rockGeo, std(0x7d7f80, { flatShading: true, roughness: 1 }));
    m.rotation.set(hash(it.name) % 7, hash(it.name + "y") % 5, 0);
    g.add(m);
  } else if (it.kind === "canopy") {
    g.add(new THREE.Mesh(canopyGeo, std(0x2f7d32, { flatShading: true, roughness: 0.9 })));
  } else if (it.kind === "bird") {
    const dark = std(0x23262b, { roughness: 0.8 });
    const body = new THREE.Mesh(unitSphere, dark);
    body.scale.set(0.5, 0.22, 0.2);
    g.add(body);
    const wings = [];
    for (const s of [-1, 1]) {
      const w = new THREE.Mesh(new THREE.BoxGeometry(0.25, 0.9, 0.03).translate(0, 0.45 * s, 0), dark);
      g.add(w);
      wings.push([w, s]);
    }
    g.userData.wings = wings;
  } else {
    g.add(new THREE.Mesh(unitSphere, std(0x6b7a8f)));
  }
  g.userData.prev = null;
  return g;
}
function updateSphere(g, it) {
  g.position.set(...it.pos);
  const r = it.radius;
  if (it.kind === "bird") {
    const s = Math.max(r, 0.15) * 2.2;
    g.scale.set(s, s, s);
    const p = g.userData.prev;
    if (p && Math.hypot(it.pos[0] - p[0], it.pos[1] - p[1]) > 1e-3) g.rotation.z = Math.atan2(it.pos[1] - p[1], it.pos[0] - p[0]);
    g.userData.prev = it.pos.slice();
  } else g.scale.set(r, r, it.kind === "canopy" ? r * 0.9 : it.kind === "rock" ? r * 0.7 : r);
}

// ------------------------------------------------------------------ zones by kind
function letterH(color) {
  const g = new THREE.Group();
  const m = new THREE.MeshBasicMaterial({ color });
  for (const [x, y, sx, sy] of [[-0.3, 0, 0.12, 0.9], [0.3, 0, 0.12, 0.9], [0, 0, 0.6, 0.12]]) {
    const b = new THREE.Mesh(new THREE.PlaneGeometry(sx, sy), m);
    b.position.set(x, y, 0);
    g.add(b);
  }
  return g;
}
function createZone(it) {
  const g = new THREE.Group();
  const k = it.kind;
  if (k === "nofly") {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 40, 1, true), new THREE.MeshBasicMaterial({
      color: 0xff4d6d, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }));
    c.rotation.x = Math.PI / 2;
    g.add(c);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 48), new THREE.MeshBasicMaterial({ color: 0xff4d6d, side: THREE.DoubleSide }));
    g.add(ring);
    const top = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 48), new THREE.MeshBasicMaterial({ color: 0xff4d6d, side: THREE.DoubleSide, transparent: true, opacity: 0.5 }));
    g.add(top);
    g.userData = { body: c, ring, top };
  } else if (k === "bin") {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(1, 0.85, 1, 32, 1, true), std(0x3d8bfd, { side: THREE.DoubleSide, roughness: 0.6 }));
    c.rotation.x = Math.PI / 2;
    g.add(c);
    const bottom = new THREE.Mesh(new THREE.CircleGeometry(0.85, 32), std(0x1d3f73));
    bottom.position.z = 0.02;
    g.add(bottom);
    g.userData = { body: c, bottom };
  } else if (k === "pad_pickup" || k === "pad_dropoff") {
    const col = k === "pad_pickup" ? 0x4ea8ff : 0x2fbf71;
    const disc = new THREE.Mesh(new THREE.CircleGeometry(1, 40), std(0x50555c, { roughness: 0.9 }));
    g.add(disc);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.86, 1, 48), new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide }));
    ring.position.z = 0.005;
    g.add(ring);
    const h = letterH(0xffffff);
    h.position.z = 0.01;
    g.add(h);
    const lab = labelSprite(k === "pad_pickup" ? "pickup" : "drop-off", col, 96);
    g.add(lab);
    g.userData = { disc, ring, h, lab };
  } else if (k === "truck") {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 32, 1, true), new THREE.MeshBasicMaterial({
      color: 0x9be36b, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
    c.rotation.x = Math.PI / 2;
    g.add(c);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.93, 1, 48), new THREE.MeshBasicMaterial({ color: 0x9be36b, side: THREE.DoubleSide }));
    g.add(ring);
    g.userData = { body: c, ring };
  } else {   // boat and anything else: a ring on the surface
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1, 64), new THREE.MeshBasicMaterial({
      color: 0x2fbf71, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
    g.add(ring);
    g.userData = { ring };
  }
  return g;
}
function updateZone(g, it) {
  const r = it.radius, h = it.height || 1, z = it.pos[2] || 0, u = g.userData;
  if (it.kind === "nofly") {
    g.position.set(it.pos[0], it.pos[1], z);
    u.body.scale.set(r, h, r);
    u.body.position.z = h / 2;
    u.ring.scale.set(r, r, 1);
    u.ring.position.z = 0.05;
    u.top.scale.set(r, r, 1);
    u.top.position.z = h;
  } else if (it.kind === "bin") {
    g.position.set(it.pos[0], it.pos[1], z);
    u.body.scale.set(r, h, r);
    u.body.position.z = h / 2;
    u.bottom.scale.set(r, r, 1);
  } else if (it.kind === "pad_pickup" || it.kind === "pad_dropoff") {
    g.position.set(it.pos[0], it.pos[1], z + 0.04);
    u.disc.scale.set(r, r, 1);
    u.ring.scale.set(r, r, 1);
    u.h.scale.set(r * 1.1, r * 1.1, 1);
    u.lab.position.z = 1.2;
  } else if (it.kind === "truck") {
    g.position.set(it.pos[0], it.pos[1], z);
    u.body.scale.set(r, h, r);
    u.body.position.z = h / 2;
    u.ring.scale.set(r, r, 1);
    u.ring.position.z = 0.03;
  } else {
    g.position.set(it.pos[0], it.pos[1], z + 0.05);
    u.ring.scale.set(r, r, 1);
  }
}

// ------------------------------------------------------------------ marine animals
const ANIMAL_STYLE = { whale: [0x1f3b73, [2.0, 0.75, 0.6]], dolphin: [0x8a96a3, [1.6, 0.45, 0.45]], turtle: [0x3f8f4a, [1.0, 0.85, 0.35]] };
function createAnimal(it) {
  const g = new THREE.Group();
  const [col, sc] = ANIMAL_STYLE[it.type] || [0x777777, [1, 1, 1]];
  const mat = std(col, { roughness: 0.5, emissive: 0x000000 });
  const body = new THREE.Mesh(unitSphere, mat);
  body.scale.set(...sc);
  g.add(body);
  if (it.type === "turtle") {
    const shell = new THREE.Mesh(unitSphere, std(0x6b4f2a, { roughness: 0.8 }));
    shell.scale.set(0.8, 0.7, 0.4);
    shell.position.z = 0.12;
    g.add(shell);
    for (const [x, y] of [[0.5, 0.6], [0.5, -0.6], [-0.5, 0.5], [-0.5, -0.5]]) {
      const f = new THREE.Mesh(unitSphere, mat);
      f.scale.set(0.3, 0.15, 0.06);
      f.position.set(x, y, 0);
      g.add(f);
    }
  } else {
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.9, 0.08), mat);
    tail.position.set(-sc[0] - 0.05, 0, 0.05);
    g.add(tail);
    const fin = new THREE.Mesh(new THREE.ConeGeometry(0.15, 0.4, 6), mat);
    fin.position.set(-0.1, 0, sc[2] + 0.1);
    g.add(fin);
  }
  const warn = new THREE.Mesh(new THREE.RingGeometry(1.05, 1.25, 40), new THREE.MeshBasicMaterial({
    color: 0xff2d3d, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthTest: false }));
  warn.renderOrder = 8;
  warn.visible = false;
  g.add(warn);
  g.userData = { mat, warn };
  return g;
}
function updateAnimal(g, it) {
  g.position.set(...it.pos);
  const r = Math.max(it.radius || 1, 0.3);
  const s = r / ((ANIMAL_STYLE[it.type] || [0, [1]])[1][0]);
  g.scale.set(s, s, s);
  const v = it.vel || [0, 0, 0];
  if (Math.hypot(v[0], v[1]) > 1e-3) g.rotation.z = Math.atan2(v[1], v[0]);
  g.userData.mat.emissive.setHex(it.tangled ? 0xb0101c : 0x000000);
  g.userData.warn.visible = !!it.tangled;
  g.userData.warn.scale.set(1.6, 1.6, 1);
}

// ------------------------------------------------------------------ frame handling
let latest = null, latestAt = 0, curScene = null;
let boatShown = false;
const $ = (id) => document.getElementById(id);

function applyFrame(st) {
  buildArena(st.arena_half || 25);
  if (st.scene !== curScene) {
    curScene = st.scene;
    for (const s of ["trash", "delivery", "ocean", "world"]) document.body.classList.toggle("scene-" + s, st.scene === s);
    for (const b of document.querySelectorAll("#scene-toggle button")) b.classList.toggle("active", b.dataset.scene === st.scene);
    $("btn-move").textContent = st.scene === "delivery" ? "Move cooler (click)" : "Move item (click)";
    renderExamples(st.scene);
  }
  // ground: terrain heightfield, flat ground (city), or the ocean
  const ter = st.terrain, water = st.water;
  if (ter) {
    if (ter.heights) buildTerrain(ter);
    else if (ter.key !== undefined && ter.key !== terrainKey) fetchTerrain(ter.key);
  }
  const showTerrain = !!ter && !!terrainMesh && (ter.key === undefined || ter.key === terrainKey);
  if (terrainMesh) terrainMesh.visible = showTerrain;
  const waves = !!(water && water.waves);
  // world: terrain + tide level + waves -> the GPU sea at the tide level; ocean: no terrain, CPU waves
  const seaOn = waves && !!ter;
  const ocean = waves && !seaOn;
  oceanGroup.visible = ocean;
  sea.visible = seaOn;
  flatGround.visible = !showTerrain && !waves;
  flatGround.userData.ground.material.color.setHex(st.variant === "city" ? 0x2a2f33 : 0x2c3a2c);
  const lakeOn = !!water && !waves;
  lake.visible = lakeOn;
  if (lakeOn) lake.position.z = water.level;
  const style = { level: (lakeOn || seaOn) ? water.level : null, city: st.variant === "city" };
  const sk = `${terrainKey}|${style.city}|${style.level === null ? "-" : style.level.toFixed(2)}`;
  if (sk !== terrainStyleKey) { terrainStyleKey = sk; colorTerrain(style); }
  document.body.classList.toggle("has-lake", lakeOn || seaOn);    // the tide slider moves lakes and the world's sea
  if (st.max_speed !== undefined) maxSpeed = st.max_speed;
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
    if (d.userData.fresh) {
      d.position.copy(d.userData.target); d.quaternion.copy(d.userData.qtarget); d.userData.fresh = false;
      if (follow && i === followIdx) { controls.target.copy(d.position); camera.position.copy(d.position).add(FOLLOW_OFFSET); }
    }
    d.userData.mat.emissive.setHex(s.holding ? 0x2fbf71 : 0x000000);
    d.userData.gripMat.color.setHex(s.holding ? 0x2fbf71 : s.grip ? 0xf2b84b : 0x9aa3ad);
    for (const [c, sgn] of d.userData.claws) c.position.x = (s.grip ? 0.025 : 0.06) * sgn;
  });
  if (followIdx >= st.drones.length) followIdx = 0;
  buildDronePicker(st.drones.length);
  if (arenaChanged && drones[0]) { arenaChanged = false; setFollow(ARENA_HALF > 60, true); }
  boatShown = !!st.boat;
  syncPool(pools.items, st.items, (it) => it.type + ":" + it.shape + ":" + it.is_trash, createItem, updateItem);
  syncPool(pools.boxes, st.boxes, (it) => it.kind, createBox, updateBox);
  syncPool(pools.spheres, st.spheres, (it) => it.kind, createSphere, updateSphere);
  syncPool(pools.zones, st.zones, (it) => it.kind, createZone, updateZone);
  syncPool(pools.animals, st.animals, (it) => it.type, createAnimal, updateAnimal);
  st.anchors.forEach((a, k) => {
    const g = anchors[k];
    g.visible = a.kind !== "none";
    g.position.set(...a.pos);
    g.userData.ring.visible = a.radius > 0;
    if (a.radius > 0) g.userData.ring.scale.set(a.radius, a.radius, 1);
  });
  labelAnchorItem(st);
  if (st.boat) {
    boat.visible = true;
    const [bx, by, bz] = st.boat.size;
    boat.userData.hull.scale.set(bx, by, bz);
    boat.userData.cabin.position.set(-bx * 0.15, 0, bz / 2 + 0.3);
    boat.userData.cabin.scale.set(bx / 1.2, by / 1.6, 1);
    boat.userData.base = st.boat.pos;
    if (!waves) boat.position.set(...st.boat.pos);
  } else boat.visible = false;
  if (st.net) {
    const n = ensureNet(st.net.rows, st.net.cols);
    n.visible = true;
    const arr = n.geometry.attributes.position.array;
    st.net.nodes.forEach((p, i) => { arr[3 * i] = p[0]; arr[3 * i + 1] = p[1]; arr[3 * i + 2] = p[2]; });
    n.geometry.attributes.position.needsUpdate = true;
    n.geometry.computeBoundingSphere();
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
  const tangled = (st.animals || []).filter((a) => a.tangled).map((a) => a.type);
  $("g-bycatch").textContent = tangled.length ? tangled.join(", ") : "none";
  $("g-bycatch").parentElement.classList.toggle("alert", tangled.length > 0);
  if (latestDemo) $("g-items").textContent = itemsGauge(st);
}

function itemsGauge(st) {
  const its = st.items || [];
  if (!its.length) return "-";
  const trash = its.filter((i) => i.is_trash);
  if ((st.scene === "trash" || st.scene === "world") && trash.length) return `${trash.filter((i) => i.delivered).length}/${trash.length} trash in`;
  return its.some((i) => i.delivered) ? "delivered" : its.some((i) => i.held) ? "carrying" : "waiting";
}

// small label over the item that a handle anchor points at (a0 first)
function labelAnchorItem(st) {
  let best = null, bestK = -1;
  for (let k = 0; k < st.anchors.length && !best; k++) {
    const a = st.anchors[k];
    if (a.kind !== "handle") continue;
    let bd = 0.8;
    for (const it of st.items || []) {
      const d = Math.hypot(it.pos[0] - a.pos[0], it.pos[1] - a.pos[1], it.pos[2] - a.pos[2]);
      if (d < bd) { bd = d; best = it; bestK = k; }
    }
  }
  const text = best ? `a${bestK}: ${ITEM_LABELS[best.type] || best.type}${best.is_trash ? " (trash)" : ""}` : "";
  if (text !== anchorLabelText) {
    if (anchorLabel) dyn.remove(anchorLabel);
    anchorLabel = null;
    anchorLabelText = text;
    if (text) { anchorLabel = labelSprite(text, ANCHOR_COLORS[bestK], 40 + 10 * text.length); dyn.add(anchorLabel); }
  }
  if (anchorLabel && best) {
    const d = itemDims(best);
    anchorLabel.position.set(best.pos[0], best.pos[1], best.pos[2] + d[2] / 2 + 1.0);
  }
}

function fmt(v, d = 2) { return v === null || v === undefined ? "-" : Number(v).toFixed(d); }

let latestDemo = null;
function applyDemo(d) {
  latestDemo = d;
  const fallback = d.policy_mode !== "trained policy";
  const pm = $("policy-mode");
  pm.textContent = "policy: " + d.policy_mode;
  pm.className = "pill " + (fallback ? "fallback" : "good");
  const mm = $("mission-mode");
  mm.textContent = "compiler: " + d.mission_mode;
  mm.className = "pill " + (d.mission_mode === "Gemini" ? "good" : "fallback");
  const sim = $("sim-mode");
  sim.classList.toggle("hidden", !d.sim || d.sim === "simulator");
  sim.textContent = "sim: " + (d.sim || "");
  const s = $("status");
  s.textContent = d.status;
  s.className = "pill status " + d.status.replace(/\s+/g, "-");
  $("message").textContent = d.message || "";
  $("g-reward").textContent = fmt(d.reward);
  $("g-swing").textContent = fmt(d.swing) + " rad/s";
  $("g-speed").textContent = fmt(d.speed, 1) + (maxSpeed ? " / " + maxSpeed.toFixed(0) : "") + " m/s";
  $("g-shield").textContent = d.shield_hits;
  $("g-time").textContent = fmt(d.time, 1) + " s";
  const banner = $("mode-banner");
  banner.classList.toggle("hidden", !(fallback && d.flying));
  banner.textContent = "Flying the " + d.policy_mode + " controller, not the trained network";
  $("btn-stop").disabled = !d.flying;
  // gusts: the env ignores them unless enabled, so only offer the button when they are on
  $("btn-gust").classList.toggle("hidden", !d.gusts);
  document.body.classList.toggle("gusts-on", !!d.gusts);
  buildDisableButtons(d.num_agents);
  showDroneTask();
  drawZmap(d.zmap);
}

// ------------------------------------------------------------------ drone picker (follow camera)
let builtPicker = -1;
function buildDronePicker(n) {
  if (n === builtPicker) return;
  builtPicker = n;
  const row = $("drone-picker");
  row.innerHTML = "";
  row.classList.toggle("hidden", n <= 1);
  for (let i = 0; i < n; i++) {
    const b = document.createElement("button");
    b.dataset.idx = i;
    b.textContent = "Drone " + i;
    b.style.borderBottom = "3px solid #" + AGENT_COLORS[i % AGENT_COLORS.length].toString(16).padStart(6, "0");
    b.onclick = () => followDrone(i);
    row.appendChild(b);
  }
  updateCamButtons();
}
// what the selected drone is doing (per-drone jobs in the world), and whether it is done
function showDroneTask() {
  const el = $("drone-task");
  if (!el) return;
  const tasks = (latestDemo && latestDemo.agent_tasks) || [];
  const done = new Set((latestDemo && latestDemo.agents_done) || []);
  const t = tasks[followIdx];
  const show = follow && drones.length > 1 && !!t;
  el.classList.toggle("hidden", !show);
  if (show) el.textContent = `Drone ${followIdx}: ${t}${done.has(followIdx) ? " (done)" : ""}`;
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
    zx.fillText(p.name, Math.min(x + 7, W - 80), y + 4);
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
function noted(prefix) { return (j) => toast(j && j.note ? prefix + " (" + j.note + ")" : prefix); }

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
    try {
      await api("/api/scene", { scene: b.dataset.scene });
      hideCard();
      drones.forEach((d) => (d.userData.fresh = true));
      $("wave-scale").value = 1; $("wave-val").textContent = "1.0x";
      $("tide").value = 0; $("tide-val").textContent = "+0.0 m";
    } catch (e) { toast("Scene switch failed: " + e.message); }
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
  const ag = $("card-agents");
  ag.innerHTML = "";
  ag.classList.toggle("hidden", !m.agents);
  for (const [i, j] of Object.entries(m.agents || {})) {
    const div = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = `Drone ${i}: `;
    b.style.color = "#" + AGENT_COLORS[+i % AGENT_COLORS.length].toString(16).padStart(6, "0");
    div.appendChild(b);
    div.appendChild(document.createTextNode(j.task || ""));
    const an = Object.entries(j.anchors || {}).map(([k, v]) => `${k} = ${typeof v.target === "string" ? v.target : JSON.stringify(v.target)}`);
    if (an.length) { const s = document.createElement("div"); s.className = "sub"; s.textContent = an.join(", "); div.appendChild(s); }
    const c = document.createElement("code");
    c.textContent = j.reward + (j.done_when ? "\ndone when " + j.done_when : "");
    div.appendChild(c);
    ag.appendChild(div);
  }
  $("card-reward").textContent = m.agents ? "(per drone, see above) fallback: " + m.reward : m.reward;
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
$("btn-reset").onclick = () => api("/api/reset", {}).catch((e) => toast(e.message));
$("btn-gust").onclick = () => {
  const ang = Math.random() * 2 * Math.PI, mag = 5 + Math.random() * 2;
  api("/api/perturb", { kind: "gust", vec: [mag * Math.cos(ang), mag * Math.sin(ang), 0], duration: 2.5 })
    .then(noted(`Gust: ${mag.toFixed(1)} m/s for 2.5 s`)).catch((e) => toast(e.message));
};
function debounced(fn, ms) { let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
const sendWaves = debounced((v) => api("/api/perturb", { kind: "waves", scale: v }).catch((er) => toast(er.message)), 80);
$("wave-scale").oninput = (e) => {
  const v = parseFloat(e.target.value);
  $("wave-val").textContent = v.toFixed(1) + "x";
  sendWaves(v);
};
const sendTide = debounced((v) => api("/api/perturb", { kind: "tide", offset: v }).then(noted(`Tide ${v >= 0 ? "+" : ""}${v.toFixed(1)} m`))
  .catch((er) => toast(er.message)), 80);
$("tide").oninput = (e) => {
  const v = parseFloat(e.target.value);
  $("tide-val").textContent = (v >= 0 ? "+" : "") + v.toFixed(1) + " m";
  sendTide(v);
};

let clickMode = null;
function setClickMode(m) {
  clickMode = clickMode === m ? null : m;
  $("btn-drop").classList.toggle("on", clickMode === "drop");
  $("btn-move").classList.toggle("on", clickMode === "move");
  canvas.style.cursor = clickMode ? "crosshair" : "";
  if (clickMode) toast(clickMode === "drop" ? "Click the ground to drop an obstacle." :
    "Click the ground to move the " + (curScene === "delivery" ? "cooler." : "item the drone is after."));
}
$("btn-drop").onclick = () => setClickMode("drop");
$("btn-follow").onclick = () => setFollow(!follow);
$("btn-overview").onclick = () => setFollow(false);
// keys: 1..9 follow that drone, o / 0 overview (not while typing a mission)
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "o" || e.key === "0") setFollow(false);
  else if (/^[1-9]$/.test(e.key) && drones[+e.key - 1]) followDrone(+e.key - 1);
});
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

// picking: terrain mesh when shown, else the z = 0 plane (water level for lakes / ocean)
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
function pickGround(e) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  if (terrainMesh && terrainMesh.visible) {
    const hit = ray.intersectObject(terrainMesh, false)[0];
    if (hit) return hit.point;
  }
  const p = new THREE.Vector3();
  const lvl = (latest && latest.water && latest.water.level) || 0;
  return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -lvl), p) ? p : null;
}
let downAt = null;
canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", (e) => {
  if (!clickMode || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const p = pickGround(e);
  if (!p) return;
  if (Math.abs(p.x) > ARENA_HALF || Math.abs(p.y) > ARENA_HALF) { toast("Click inside the arena."); return; }
  const mode = clickMode;
  dropMarker.position.set(p.x, p.y, p.z + 0.05);
  dropMarker.visible = true;
  setTimeout(() => (dropMarker.visible = false), 800);
  if (mode === "drop") {
    api("/api/perturb", { kind: "drop_obstacle", pos: [p.x, p.y] }).then(noted("Obstacle dropped")).catch((er) => toast(er.message));
  } else {
    api("/api/perturb", { kind: "move_payload", pos: [p.x, p.y] }).then(noted("Item moved")).catch((er) => toast(er.message));
  }
  setClickMode(mode);   // one click per activation
});

// hover tooltip for items and animals
const tip = $("tip");
let hoverAt = 0;
canvas.addEventListener("pointermove", (e) => {
  const now = performance.now();
  if (now - hoverAt < 60) return;
  hoverAt = now;
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const targets = [...pools.items.values(), ...pools.animals.values()].filter((o) => o.visible);
  const hit = ray.intersectObjects(targets, true)[0];
  let text = "";
  if (hit) {
    let o = hit.object;
    while (o && !pools.items.has(o.userData.name) && !pools.animals.has(o.userData.name)) o = o.parent;
    if (o && pools.items.has(o.userData.name)) {
      const it = o.userData.info;
      text = `${it.name}: ${ITEM_LABELS[it.type] || it.type} · ${it.is_trash ? "trash" : "not trash"}${it.held ? " · held" : ""}`;
    } else if (o) text = o.userData.name.replace("_", " ");
  }
  tip.classList.toggle("hidden", !text);
  if (text) { tip.textContent = text; tip.style.left = (e.clientX - r.left + 14) + "px"; tip.style.top = (e.clientY - r.top + 10) + "px"; }
});
canvas.addEventListener("pointerleave", () => tip.classList.add("hidden"));

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
      applyDemo(msg.demo);
      applyFrame(msg.state);
    } else if (msg.type === "terrain") {
      if (msg.terrain) buildTerrain(msg.terrain);
    } else if (msg.type === "hello") {
      $("btn-gust").classList.toggle("hidden", !msg.gusts);
      if (msg.pending) showCard(msg.pending);
    }
  };
}

// ------------------------------------------------------------------ render loop
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const a = 1 - Math.exp(-dt * 18);
  const fd = drones[followIdx];
  const before = fd ? fd.position.clone() : null;
  for (const d of drones) {
    d.position.lerp(d.userData.target, a);
    d.quaternion.slerp(d.userData.qtarget, a);
  }
  if (follow && before) {
    const delta = fd.position.clone().sub(before);
    camera.position.add(delta);
    controls.target.add(delta);
  }
  // drones stay findable in a 600 m arena: scale the beacon ring with camera distance
  for (const d of drones) {
    const dist = camera.position.distanceTo(d.position);
    d.userData.beacon.scale.setScalar(Math.max(1, dist / 40));
  }
  const tNow = performance.now() / 1000;
  for (const g of pools.spheres.values()) {
    if (g.userData.wings) for (const [w, s] of g.userData.wings) w.rotation.x = s * 0.6 * Math.sin(tNow * 12);
  }
  const waves = latest && latest.water && latest.water.waves;
  if (waves) {
    const t = latest.t + Math.min((performance.now() - latestAt) / 1000, 0.15);
    const level = latest.water.level || 0;           // wave surface offset by the tide level (world)
    if (sea.visible) updateSea(latest.water, t);
    else updateWater(waves, t);
    if (boat.visible && boat.userData.base) {
      const b = boat.userData.base;
      boat.position.set(b[0], b[1], level + waveHeight(waves, b[0], b[1], t) + 0.1);
    }
  }
  controls.update();
  renderer.render(scene, camera);
}
window.__skyhand = { camera, controls, scene, pools, drones, followDrone, setFollow,
  get sea() { return sea; }, get follow() { return { on: follow, idx: followIdx }; } };   // console debugging
resize();
renderExamples("trash");
connect();
animate();
