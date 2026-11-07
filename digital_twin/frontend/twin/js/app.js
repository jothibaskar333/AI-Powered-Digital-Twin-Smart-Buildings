// app.js - 3D digital twin UI.
// Data flow: the backend simulates the building and stores readings; this page only
// renders what GET /api/rooms/live returns (readings, AI outputs, status, equipment
// state) and sends user controls to /api/simulation/*. It never generates values.

import { Renderer, OrbitCamera, attachOrbitControls, rayBox } from "./gl.js";
import { buildBuilding, applyRoomVisuals, STATUS_COLORS } from "./scene.js";

const params = new URLSearchParams(location.search);
const API = (params.get("api") || location.origin).replace(/\/$/, "");
const POLL_MS = 2000;
const $ = (id) => document.getElementById(id);

const st = {
  building: null, live: null, selected: params.get("room"), hover: null, floorView: "all", explode: true,
  labels: true, dirty: true, pollTimer: null, raf: null, failures: 0, history: {}, resetArmed: null,
};
let R, camera, controls, scene;

async function api(path, opts = {}) {
  const res = await fetch(API + path, { cache: "no-store", headers: { "Content-Type": "application/json" }, ...opts });
  let body = null;
  try { body = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) throw new Error((body && body.detail) || `${res.status} ${res.statusText}`);
  return body;
}
const post = (path, data) => api(path, { method: "POST", body: JSON.stringify(data || {}) });

function fatal(html) { $("fatal").innerHTML = `<div>${html}</div>`; $("fatal").hidden = false; }

// ------------------------------------------------------------------ init
async function init() {
  try {
    st.building = await api("/api/building");
  } catch (e) {
    fatal(`<b>Backend unavailable</b><p>Could not load the building from <code>${API}</code>.</p>
      <p>Start it with <code>cd backend</code> then <code>uvicorn main:app --port 8000</code>, then reload this page.</p>
      <p class="hint">${e.message}</p><button class="btn primary" onclick="location.reload()">Retry</button>`);
    return;
  }
  try {
    R = new Renderer($("scene"));
  } catch (e) {
    fatal(`<b>3D view unavailable</b><p>${e.message}</p><p>Enable hardware acceleration / WebGL in the browser settings.</p>`);
    return;
  }
  scene = buildBuilding(R, st.building);
  for (const f of st.building.floors) { R.floorVisible[f] = true; R.floorOffset[f] = 0; }
  camera = new OrbitCamera(scene.center, 42, -0.55, 0.78);
  controls = attachOrbitControls($("scene"), camera, () => { st.dirty = true; });
  setupThresholds();
  setupLabels();
  setupControls();
  setupPicking();
  new ResizeObserver(() => { st.dirty = true; }).observe($("scene"));
  document.addEventListener("visibilitychange", () => { st.dirty = true; });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") select(null); });
  window.addEventListener("pagehide", dispose);
  if (!st.selected || !scene.rooms[st.selected]) st.selected = st.building.rooms[0].id;
  frame();
  poll();
}

function dispose() {
  cancelAnimationFrame(st.raf); clearTimeout(st.pollTimer);
  if (controls) controls.dispose();
  if (R) R.dispose();
}

// ------------------------------------------------------------------ render loop
let explodeCur = 0;
function frame(now = 0) {
  st.raf = requestAnimationFrame(frame);
  if (document.hidden) return;
  let moving = camera.update();
  const explodeGoal = st.explode ? 4.5 : 0;
  if (Math.abs(explodeGoal - explodeCur) > 0.005) { explodeCur += (explodeGoal - explodeCur) * 0.18; moving = true; }
  else explodeCur = explodeGoal;
  for (const f of st.building.floors) R.floorOffset[f] = f * explodeCur;
  // pulse markers of critical rooms
  let pulsing = false;
  if (st.live) for (const e of st.live.rooms) {
    const room = scene.rooms[e.room_id];
    const s = e.status === "critical" ? 0.38 * (1 + 0.25 * Math.sin(now / 160)) : 0.38;
    room.marker.size = [s, s, s];
    if (e.status === "critical") pulsing = true;
  }
  const pulseDue = pulsing && now - (st.lastPulse || 0) > 50;     // ~20 fps is enough for the pulse
  if (pulseDue) st.lastPulse = now;
  if (moving || st.dirty || pulseDue) {
    R.render(camera);
    updateLabels();
    st.dirty = false;
  }
}

// ------------------------------------------------------------------ data
async function poll() {
  clearTimeout(st.pollTimer);
  try {
    st.live = await api("/api/rooms/live");
    st.failures = 0;
    setConn(true);
    applyLive();
    if (st.selected) await loadHistory(st.selected);
  } catch (e) {
    st.failures += 1;
    setConn(false, e.message);
  }
  st.pollTimer = setTimeout(poll, POLL_MS);
}

async function loadHistory(roomId) {
  try {
    st.history[roomId] = await api(`/api/history?room_id=${encodeURIComponent(roomId)}&limit=60`);
    drawSparks();
  } catch (_) { /* sparklines are optional */ }
}

function setConn(ok, msg) {
  const c = $("conn");
  c.className = "conn " + (ok ? "ok" : "err");
  c.textContent = ok ? `Live · updated ${new Date().toLocaleTimeString()}` : `Backend unreachable – retrying (${msg})`;
}

function entryFor(id) { return st.live ? st.live.rooms.find((r) => r.room_id === id) : null; }

function applyLive() {
  for (const [id, room] of Object.entries(scene.rooms)) {
    applyRoomVisuals(room, entryFor(id), { hover: st.hover === id, selected: st.selected === id });
  }
  const sim = st.live.simulation;
  if (sim) {
    $("sim-time").textContent = sim.sim_time.replace("T", " ");
    const rs = $("run-state");
    rs.textContent = sim.running ? `RUNNING ${sim.speed}×` : "PAUSED";
    rs.className = "pill " + (sim.running ? "running" : "paused");
    $("btn-run").textContent = sim.running ? "Pause" : "Start";
    $("btn-run").className = "btn " + (sim.running ? "" : "primary");
    $("auto-control").checked = sim.auto_control;
    for (const b of $("speed").children) b.classList.toggle("on", Number(b.dataset.v) === sim.speed);
  }
  updateLabelContent();
  renderPanel();
  st.dirty = true;
}

// ------------------------------------------------------------------ labels
const labelEls = {};
function setupLabels() {
  for (const r of st.building.rooms) {
    const el = document.createElement("div");
    el.className = "rlabel";
    el.dataset.room = r.id;
    el.innerHTML = `<span class="d"></span><span class="n">${r.name}</span><span class="v"></span>`;
    el.addEventListener("click", () => select(r.id));
    $("labels").appendChild(el);
    labelEls[r.id] = el;
  }
}
function updateLabelContent() {
  for (const [id, el] of Object.entries(labelEls)) {
    const e = entryFor(id);
    el.querySelector(".d").style.background = STATUS_COLORS[e ? e.status : "unknown"];
    el.querySelector(".v").textContent = e && e.reading ? `${num(e.reading.temperature)} °C` : "–";
    el.classList.toggle("sel", st.selected === id);
  }
}
function updateLabels() {
  for (const [id, el] of Object.entries(labelEls)) {
    const room = scene.rooms[id];
    const visible = st.labels && R.floorVisible[room.floor] !== false;
    const a = room.labelAnchor;
    const p = visible ? R.project([a[0], a[1] + (R.floorOffset[room.floor] || 0), a[2]]) : null;
    if (!p || p.x < -50 || p.y < -20 || p.x > R.canvas.clientWidth + 50 || p.y > R.canvas.clientHeight + 20) { el.style.display = "none"; continue; }
    el.style.display = "flex";
    el.style.left = `${p.x}px`; el.style.top = `${p.y}px`;
    el.style.zIndex = String(1000 - Math.round(p.depth * 500));
  }
}

// ------------------------------------------------------------------ picking
function pickRoom(clientX, clientY) {
  if (!R.viewProj) return null;
  const ray = R.ray(clientX, clientY);
  let best = null;
  for (const [id, room] of Object.entries(scene.rooms)) {
    if (R.floorVisible[room.floor] === false) continue;
    const off = R.floorOffset[room.floor] || 0;
    const t = rayBox(ray, [room.min[0], room.min[1] + off, room.min[2]], [room.max[0], room.max[1] + off, room.max[2]]);
    if (t !== null && (!best || t < best.t)) best = { id, t };
  }
  return best ? best.id : null;
}
function setupPicking() {
  const cv = $("scene");
  let down = null;
  cv.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, b: e.button }; });
  cv.addEventListener("pointerup", (e) => {
    if (down && down.b === 0 && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) select(pickRoom(e.clientX, e.clientY));
    down = null;
  });
  let pending = false;
  cv.addEventListener("pointermove", (e) => {
    if (e.buttons || pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const id = pickRoom(e.clientX, e.clientY);
      if (id !== st.hover) {
        st.hover = id;
        cv.classList.toggle("pointer", !!id);
        if (st.live) applyLive(); else st.dirty = true;
      }
    });
  });
  cv.addEventListener("pointerleave", () => { if (st.hover) { st.hover = null; if (st.live) applyLive(); } });
}

function select(id) {
  st.selected = id;
  if (st.live) applyLive();
  if (id) loadHistory(id);
  renderPanel();
}

// ------------------------------------------------------------------ room panel
const fmtTime = (ts) => (ts ? ts.replace("T", " ") : "–");
// readings are floats; print them the way Python does (0.0, not 0) so the 3D view and
// the Streamlit dashboard display identical text for the same value
const num = (v) => (typeof v === "number" && Number.isInteger(v) ? v.toFixed(1) : String(v));
function renderPanel() {
  const id = st.selected;
  $("room-empty").hidden = !!id;
  $("room-body").hidden = !id;
  if (!id) return;
  const cfg = st.building.rooms.find((r) => r.id === id);
  const e = entryFor(id);
  $("room-name").textContent = cfg.name;
  $("room-sub").textContent = `${id} · ${st.building.floor_names[cfg.floor] || "Floor " + cfg.floor}`;
  const status = e ? e.status : "unknown";
  $("room-status").textContent = status;
  $("room-status").className = "status " + status;
  $("room-reasons").innerHTML = (e ? e.reasons : ["Waiting for data"]).map((r) => `<li>${r}</li>`).join("");
  const rd = e && e.reading;
  // same labels / units / formatting as the Streamlit dashboard's sensor cards
  const fmt = {
    temperature: (v) => `${num(v)} °C`, humidity: (v) => `${num(v)} %`, occupancy: (v) => (v ? "Occupied" : "Empty"),
    light_level: (v) => `${num(v)} lux`, power_consumption: (v) => `${num(v)} kW`,
  };
  $("sensor-table").innerHTML = st.building.sensors.map((s) =>
    `<tr><td>${s.label}</td><td data-sensor="${s.key}">${rd ? fmt[s.key](rd[s.key]) : "–"}</td></tr>`).join("");
  $("reading-time").textContent = rd ? `Reading at ${fmtTime(rd.timestamp)} (simulated time)${rd.is_anomaly ? " · fault scenario active" : ""}` : "";
  const fp = e ? e.fault_probability : null;
  const th = st.building.thresholds.fault_probability;
  $("ai-table").innerHTML =
    `<tr><td>Fault / anomaly probability<br><span class="muted">Isolation Forest · critical above ${Math.round(th * 100)}%</span></td>` +
    `<td data-ai="fault">${fp === null || fp === undefined ? "model unavailable" : (fp * 100).toFixed(1) + "%"}</td></tr>` +
    `<tr><td>Predicted power for these conditions<br><span class="muted">Random Forest</span></td>` +
    `<td data-ai="power">${e && e.predicted_power_kw !== null ? e.predicted_power_kw + " kW" : "model unavailable"}</td></tr>`;
  const s = e && e.state;
  $("equip-table").innerHTML = s ? (
    `<tr><td>HVAC</td><td>${s.hvac.toUpperCase()} <span class="muted">(${s.hvac_mode})</span></td></tr>` +
    `<tr><td>Lights</td><td>${s.lights.toUpperCase()} <span class="muted">(${s.lights_mode})</span></td></tr>`) : "";
  if (s) {
    if (document.activeElement !== $("hvac-mode")) $("hvac-mode").value = s.hvac_mode;
    if (document.activeElement !== $("lights-mode")) $("lights-mode").value = s.lights_mode;
  }
  const sc = s && s.scenario;
  $("scenario-active").hidden = !sc;
  if (sc) $("scenario-active").innerHTML = `<b>${sc.kind.replace("_", " ")}</b> (${sc.source}) – ${sc.description}. ${sc.remaining_minutes} simulated min left.`;
  $("btn-clear").disabled = !sc;
  drawSparks();
}

function drawSpark(canvas, values, color, threshold) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  if (!values.length) return;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (threshold !== undefined) { lo = Math.min(lo, threshold); hi = Math.max(hi, threshold); }
  if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
  const y = (v) => h - 4 - ((v - lo) / (hi - lo)) * (h - 8);
  if (threshold !== undefined) { g.strokeStyle = "#e3a008"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, y(threshold)); g.lineTo(w, y(threshold)); g.stroke(); g.setLineDash([]); }
  g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
  values.forEach((v, i) => { const x = (i / Math.max(1, values.length - 1)) * (w - 2) + 1; i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v)); });
  g.stroke();
}
function drawSparks() {
  const h = st.history[st.selected] || [];
  drawSpark($("spark-temp"), h.map((r) => r.temperature), "#2563eb", st.building.thresholds.comfort_temp_c);
  drawSpark($("spark-power"), h.map((r) => r.power_consumption), "#7c3aed", st.building.thresholds.power_alert_kw);
}

// ------------------------------------------------------------------ controls
function setupThresholds() {
  const t = st.building.thresholds;
  document.querySelectorAll(".th-power").forEach((e) => (e.textContent = t.power_alert_kw));
  document.querySelectorAll(".th-temp").forEach((e) => (e.textContent = t.comfort_temp_c));
  document.querySelectorAll(".th-fault").forEach((e) => (e.textContent = Math.round(t.fault_probability * 100) + "%"));
  $("scenario-kind").innerHTML = Object.entries(st.building.scenarios)
    .map(([k, d]) => `<option value="${k}" title="${d}">${k.replace("_", " ")}</option>`).join("");
}

async function control(fn) {
  try { await fn(); await poll(); } catch (e) { setConn(false, e.message); }
}

function setupControls() {
  $("btn-run").addEventListener("click", () => control(() =>
    post(st.live && st.live.simulation && st.live.simulation.running ? "/api/simulation/pause" : "/api/simulation/start")));
  $("btn-reset").addEventListener("click", () => {
    const b = $("btn-reset");
    if (!st.resetArmed) {          // two-step confirmation instead of a blocking dialog
      b.textContent = "Confirm reset"; b.classList.add("danger");
      st.resetArmed = setTimeout(() => { b.textContent = "Reset"; b.classList.remove("danger"); st.resetArmed = null; }, 3000);
      return;
    }
    clearTimeout(st.resetArmed); st.resetArmed = null; b.textContent = "Reset"; b.classList.remove("danger");
    control(() => post("/api/simulation/reset"));
  });
  const speeds = (st.live && st.live.simulation && st.live.simulation.speeds) || [1, 10, 60, 360];
  $("speed").innerHTML = speeds.map((v) => `<button type="button" data-v="${v}">${v}×</button>`).join("");
  for (const b of $("speed").children) b.addEventListener("click", () => control(() => post("/api/simulation/speed", { speed: Number(b.dataset.v) })));
  $("auto-control").addEventListener("change", (e) => control(() => post("/api/simulation/auto_control", { enabled: e.target.checked })));

  const views = [["all", "All"], ...st.building.floors.map((f) => [String(f), f === 0 ? "Ground" : `Floor ${f}`])];
  $("floor-view").innerHTML = views.map(([v, l]) => `<button type="button" data-v="${v}" class="${v === "all" ? "on" : ""}">${l}</button>`).join("");
  for (const b of $("floor-view").children) b.addEventListener("click", () => {
    st.floorView = b.dataset.v;
    for (const x of $("floor-view").children) x.classList.toggle("on", x === b);
    for (const f of st.building.floors) R.floorVisible[f] = st.floorView === "all" || Number(st.floorView) === f;
    if (st.selected && !R.floorVisible[scene.rooms[st.selected].floor]) select(null);
    st.dirty = true;
  });
  $("explode").addEventListener("change", (e) => { st.explode = e.target.checked; st.dirty = true; });
  $("show-labels").addEventListener("change", (e) => { st.labels = e.target.checked; st.dirty = true; });
  $("btn-camera").addEventListener("click", () => { camera.reset(); st.dirty = true; });

  $("hvac-mode").addEventListener("change", (e) => control(() => post("/api/simulation/room_control", { room_id: st.selected, hvac_mode: e.target.value })));
  $("lights-mode").addEventListener("change", (e) => control(() => post("/api/simulation/room_control", { room_id: st.selected, lights_mode: e.target.value })));
  $("btn-scenario").addEventListener("click", () => control(() => post("/api/simulation/scenario",
    { room_id: st.selected, kind: $("scenario-kind").value, duration_minutes: Number($("scenario-min").value) })));
  $("btn-clear").addEventListener("click", () => control(() => post("/api/simulation/scenario/clear", { room_id: st.selected })));
}

// expose a small read-only hook for automated browser tests
window.__twin = {
  get live() { return st.live; }, get selected() { return st.selected; },
  pick: (x, y) => pickRoom(x, y), project: (id) => {
    const room = scene.rooms[id]; const c = room.center;
    return R.project([c[0], c[1] + (R.floorOffset[room.floor] || 0), c[2]]);
  },
};

init();
