// scene.js - builds the smart building programmatically from GET /api/building.
// Every room's position, size and ID comes from the backend's building.py, so the
// rooms in the scene are the same rooms that produce the readings.

import { Mesh, hexToRgb, mix } from "./gl.js";

export const STATUS_COLORS = { normal: "#2e9d63", warning: "#e3a008", critical: "#d64545", unknown: "#9aa3ad" };

const C = {
  slab: hexToRgb("#c9ccd1"), slabEdge: hexToRgb("#9aa0a8"), wall: hexToRgb("#eef0f2"), partition: hexToRgb("#e3e6ea"),
  glass: hexToRgb("#9fc6e6"), frame: hexToRgb("#5b6470"), wood: hexToRgb("#a0714f"), desk: hexToRgb("#d9c9ae"),
  chair: hexToRgb("#3d4a5c"), monitor: hexToRgb("#22272e"), plantPot: hexToRgb("#8c6a4f"), plant: hexToRgb("#4f9a5a"),
  hvac: hexToRgb("#f7f8f9"), core: hexToRgb("#b7bcc4"), corridor: hexToRgb("#d6d9de"), ground: hexToRgb("#dfe7dc"),
  pavement: hexToRgb("#cfd3d6"), sofa: hexToRgb("#4f7d6b"), bench: hexToRgb("#cfd8e3"), equip: hexToRgb("#8b97a6"),
  counter: hexToRgb("#b8a48a"), lightOff: hexToRgb("#8a8f98"),
};
const FLOORING = { office: "#b9c1cc", meeting: "#c8a27a", lab: "#d4dbe2", cafeteria: "#e2d6c4", lobby: "#cdb393" };
const SKIN = ["#e0b48f", "#c68a5d", "#8d5a3b", "#f1c9a5"];
const SHIRTS = ["#3f6fb0", "#b0533f", "#4f8a5b", "#7d5ba6", "#c08a2d", "#2f7f86"];

export function buildBuilding(R, B) {
  const FH = B.floor_height, SLAB = 0.25, WH = FH - SLAB, W = B.width, D = B.depth, RD = B.room_depth;
  const T = 0.2, TP = 0.15;
  // Building coordinates use z towards the north (glazed facade at z = 0). The scene
  // mirrors z (world z = depth - z) so that, seen from the glazed side, rooms run
  // left to right in the same order as their x coordinates.
  const Z = (z) => D - z;
  const add = (kind, opts) => {
    if (opts.pos) opts = { ...opts, pos: [opts.pos[0], opts.pos[1], Z(opts.pos[2])] };
    return R.add(new Mesh(kind, opts));
  };
  const box = (x0, y0, z0, x1, y1, z1, color, o = {}) => add("box", {
    pos: [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], size: [Math.abs(x1 - x0), Math.abs(y1 - y0), Math.abs(z1 - z0)], color, ...o });

  // ground and pavement
  add("box", { pos: [W / 2, -0.36, D / 2], size: [90, 0.2, 70], color: C.ground });
  add("box", { pos: [W / 2, -0.245, -3.5], size: [W + 4, 0.03, 7], color: C.pavement });   // in front of the glazed facade

  // wall along x (fixed z) or along z (fixed x), with openings [{a,b,y0,y1,glass,door}]
  function wall(axis, fixed, from, to, y0, height, thick, color, floor, openings = []) {
    const segs = [];
    const ops = [...openings].sort((p, q) => p.a - q.a);
    let cur = from;
    const put = (a, b, ya, yb, col, o = {}) => {
      if (b - a < 0.01 || yb - ya < 0.01) return;
      if (axis === "x") box(a, ya, fixed - thick / 2, b, yb, fixed + thick / 2, col, { floor, ...o });
      else box(fixed - thick / 2, ya, a, fixed + thick / 2, yb, b, col, { floor, ...o });
    };
    for (const op of ops) {
      put(cur, op.a, y0, y0 + height, color);
      put(op.a, op.b, y0, y0 + op.y0, color);                       // below opening
      put(op.a, op.b, y0 + op.y1, y0 + height, color);              // above opening
      if (op.glass) {
        const gt = 0.05;
        if (axis === "x") box(op.a, y0 + op.y0, fixed - gt / 2, op.b, y0 + op.y1, fixed + gt / 2, C.glass, { floor, opacity: 0.25, edges: C.frame });
        else box(fixed - gt / 2, y0 + op.y0, op.a, fixed + gt / 2, y0 + op.y1, op.b, C.glass, { floor, opacity: 0.25, edges: C.frame });
        for (let m = op.a + 2; m < op.b - 0.5; m += 2) put(m - 0.04, m + 0.04, y0 + op.y0, y0 + op.y1, C.frame);   // mullions
      }
      cur = op.b;
    }
    put(cur, to, y0, y0 + height, color);
    return segs;
  }

  const floors = B.floors;
  for (const f of floors) {
    const fy = f * FH;
    // slab (floor of this storey)
    box(0, fy - SLAB, 0, W, fy, D, C.slab, { floor: f, edges: C.slabEdge });
    // corridor flooring + core
    box(0.1, fy + 0.001, RD + TP / 2, W - 0.1, fy + 0.03, D - 0.1, C.corridor, { floor: f });
    box(20.6, fy, RD + 0.5, 23.7, fy + WH, D - 0.4, C.core, { floor: f, edges: C.slabEdge });
    if (f === 0) for (let s = 0; s < 10; s++) box(0.4, fy, RD + 0.5 + s * 0.32, 3.2, fy + (s + 1) * 0.32, RD + 0.5 + (s + 1) * 0.32, C.slab, { floor: f });

    // exterior walls
    const roomsHere = B.rooms.filter((r) => r.floor === f);
    const south = roomsHere.map((r) => ({ a: r.x + 0.4, b: r.x + r.w - 0.4, y0: 0.9, y1: 2.9, glass: true }));
    if (f === 0) {
      const lobby = roomsHere.find((r) => r.furniture === "lobby");
      if (lobby) { // main entrance: replace the lobby window with a full-height glass door + side glazing
        const i = south.findIndex((o) => o.a === lobby.x + 0.4);
        south.splice(i, 1, { a: lobby.x + 0.4, b: lobby.x + 2.6, y0: 0.9, y1: 2.9, glass: true },
          { a: lobby.x + 2.8, b: lobby.x + 4.8, y0: 0.0, y1: 2.6, glass: true },
          { a: lobby.x + 5.0, b: lobby.x + lobby.w - 0.4, y0: 0.9, y1: 2.9, glass: true });
      }
    }
    wall("x", 0, 0, W, fy, WH, T, C.wall, f, south);
    wall("x", D, 0, W, fy, WH, T, C.wall, f, [{ a: 5, b: 9, y0: 1.0, y1: 2.6, glass: true }, { a: 12, b: 16, y0: 1.0, y1: 2.6, glass: true }]);
    wall("z", 0, 0, D, fy, WH, T, C.wall, f, [{ a: 2.5, b: 7.5, y0: 0.9, y1: 2.8, glass: true }]);
    wall("z", W, 0, D, fy, WH, T, C.wall, f, [{ a: 2.5, b: 7.5, y0: 0.9, y1: 2.8, glass: true }]);
    // interior partitions between rooms and corridor wall with doors
    const xs = [...new Set(roomsHere.map((r) => r.x).filter((x) => x > 0))];
    for (const x of xs) wall("z", x, T / 2, RD, fy, WH, TP, C.partition, f);
    const doors = roomsHere.map((r) => ({ a: r.x + r.w - 2.2, b: r.x + r.w - 1.2, y0: 0, y1: 2.15 }));
    wall("x", RD, 0, W, fy, WH, TP, C.partition, f, doors);
    for (const d of doors) box(d.a + 0.02, fy, RD - 1.0, d.a + 0.07, fy + 2.1, RD - 0.05, C.wood, { floor: f });
  }

  // rooms: flooring tile (status coloured), furniture, lights, HVAC, sensor node, people, pick volume
  const rooms = {};
  for (const r of B.rooms) {
    const f = r.floor, fy = f * FH, x0 = r.x, z0 = r.z, x1 = r.x + r.w, z1 = r.z + r.d;
    const base = hexToRgb(FLOORING[r.furniture] || "#c8ccd2");
    const tile = box(x0 + 0.1, fy + 0.001, z0 + 0.12, x1 - 0.1, fy + 0.045, z1 - 0.1, base, { floor: f, tag: r.id });
    const volume = box(x0 + 0.12, fy + 0.05, z0 + 0.14, x1 - 0.12, fy + WH - 0.02, z1 - 0.12, [1, 1, 1], { floor: f, opacity: 0.0, visible: false });
    const people = [];
    const person = (px, pz, i) => {
      const shirt = hexToRgb(SHIRTS[(i + r.id.charCodeAt(3)) % SHIRTS.length]), skin = hexToRgb(SKIN[(i * 3 + r.id.charCodeAt(2)) % SKIN.length]);
      const body = add("cyl", { pos: [px, fy + 0.62, pz], size: [0.42, 1.15, 0.42], color: shirt, floor: f, visible: false });
      const head = add("sphere", { pos: [px, fy + 1.42, pz], size: [0.32, 0.34, 0.32], color: skin, floor: f, visible: false });
      people.push(body, head);
    };
    const plant = (px, pz) => {
      add("cyl", { pos: [px, fy + 0.25, pz], size: [0.4, 0.5, 0.4], color: C.plantPot, floor: f });
      add("sphere", { pos: [px, fy + 0.85, pz], size: [0.75, 0.9, 0.75], color: C.plant, floor: f });
    };
    const desk = (cx, cz, w = 1.5, d = 0.75) => {
      box(cx - w / 2, fy + 0.7, cz - d / 2, cx + w / 2, fy + 0.76, cz + d / 2, C.desk, { floor: f });
      box(cx - w / 2 + 0.05, fy, cz - d / 2 + 0.05, cx - w / 2 + 0.1, fy + 0.7, cz + d / 2 - 0.05, C.frame, { floor: f });
      box(cx + w / 2 - 0.1, fy, cz - d / 2 + 0.05, cx + w / 2 - 0.05, fy + 0.7, cz + d / 2 - 0.05, C.frame, { floor: f });
    };
    const chair = (cx, cz) => {
      box(cx - 0.25, fy + 0.42, cz - 0.25, cx + 0.25, fy + 0.5, cz + 0.25, C.chair, { floor: f });
      box(cx - 0.25, fy + 0.5, cz + 0.2, cx + 0.25, fy + 1.0, cz + 0.27, C.chair, { floor: f });
      add("cyl", { pos: [cx, fy + 0.21, cz], size: [0.08, 0.42, 0.08], color: C.frame, floor: f });
    };
    let seats = [];
    if (r.furniture === "office") {
      for (const [dx, dz] of [[2.2, 3.0], [5.8, 3.0], [2.2, 6.2], [5.8, 6.2]]) {
        desk(x0 + dx, z0 + dz);
        box(x0 + dx - 0.3, fy + 0.76, z0 + dz - 0.3, x0 + dx + 0.3, fy + 1.12, z0 + dz - 0.26, C.monitor, { floor: f });
        chair(x0 + dx, z0 + dz + 0.75); seats.push([x0 + dx, z0 + dz + 0.75]);
      }
      box(x1 - 0.7, fy, z0 + 7.8, x1 - 0.2, fy + 1.6, z0 + 9.6, C.desk, { floor: f, edges: C.frame });
      plant(x0 + 0.7, z0 + 9.2);
    } else if (r.furniture === "meeting") {
      box(x0 + 1.9, fy + 0.72, z0 + 4.2, x0 + 6.1, fy + 0.78, z0 + 5.8, C.wood, { floor: f });
      box(x0 + 3.6, fy, z0 + 4.7, x0 + 4.4, fy + 0.72, z0 + 5.3, C.frame, { floor: f });
      for (const dx of [2.5, 3.5, 4.5, 5.5]) { chair(x0 + dx, z0 + 3.6); chair(x0 + dx, z0 + 6.4); seats.push([x0 + dx, z0 + 3.6], [x0 + dx, z0 + 6.4]); }
      box(x1 - 0.25, fy + 1.0, z0 + 3.4, x1 - 0.18, fy + 2.3, z0 + 6.6, C.monitor, { floor: f });
      plant(x0 + 0.7, z0 + 0.9);
    } else if (r.furniture === "lab") {
      box(x0 + 0.3, fy, z0 + 1.5, x0 + 1.2, fy + 0.9, z0 + 8.5, C.bench, { floor: f, edges: C.frame });
      box(x0 + 3.2, fy, z0 + 2.8, x0 + 4.6, fy + 0.9, z0 + 7.2, C.bench, { floor: f, edges: C.frame });
      for (const z of [2.2, 4.4, 6.6]) box(x0 + 0.4, fy + 0.9, z0 + z, x0 + 1.0, fy + 1.35, z0 + z + 0.8, C.equip, { floor: f });
      box(x0 + 3.5, fy + 0.9, z0 + 3.4, x0 + 4.3, fy + 1.2, z0 + 4.2, C.equip, { floor: f });
      box(x1 - 1.4, fy, z0 + 1.0, x1 - 0.3, fy + 2.3, z0 + 2.8, C.equip, { floor: f, edges: C.frame });   // fume hood
      seats = [[x0 + 1.7, z0 + 3.5], [x0 + 2.7, z0 + 5.5], [x0 + 5.2, z0 + 4.8], [x0 + 1.7, z0 + 7.0]];
    } else if (r.furniture === "cafeteria") {
      for (const [dx, dz] of [[2.2, 3.0], [5.2, 3.0], [2.2, 6.6], [5.2, 6.6]]) {
        add("cyl", { pos: [x0 + dx, fy + 0.75, z0 + dz], size: [1.2, 0.05, 1.2], color: C.desk, floor: f });
        add("cyl", { pos: [x0 + dx, fy + 0.37, z0 + dz], size: [0.12, 0.74, 0.12], color: C.frame, floor: f });
        for (const [sx, sz] of [[-0.9, 0], [0.9, 0], [0, -0.9], [0, 0.9]]) {
          add("cyl", { pos: [x0 + dx + sx, fy + 0.23, z0 + dz + sz], size: [0.4, 0.46, 0.4], color: C.chair, floor: f });
        }
        seats.push([x0 + dx - 0.9, z0 + dz], [x0 + dx + 0.9, z0 + dz]);
      }
      box(x1 - 1.1, fy, z0 + 1.0, x1 - 0.3, fy + 1.0, z0 + 8.8, C.counter, { floor: f, edges: C.frame });
      plant(x0 + 0.7, z0 + 9.2);
    } else if (r.furniture === "lobby") {
      box(x0 + 2.5, fy, z0 + 6.4, x0 + 5.7, fy + 1.1, z0 + 7.2, C.wood, { floor: f, edges: C.frame });
      box(x0 + 5.7, fy, z0 + 5.2, x0 + 6.5, fy + 1.1, z0 + 7.2, C.wood, { floor: f, edges: C.frame });
      box(x0 + 0.5, fy, z0 + 1.8, x0 + 1.4, fy + 0.45, z0 + 4.8, C.sofa, { floor: f });
      box(x0 + 0.5, fy + 0.45, z0 + 1.8, x0 + 0.75, fy + 0.95, z0 + 4.8, C.sofa, { floor: f });
      box(x0 + 3.0, fy + 0.005, z0 + 0.3, x0 + 4.6, fy + 0.05, z0 + 1.5, C.chair, { floor: f });        // entrance mat
      plant(x0 + 0.8, z0 + 0.8); plant(x0 + 7.2, z0 + 0.8);
      seats = [[x0 + 4.1, z0 + 7.8], [x0 + 1.9, z0 + 3.0], [x0 + 3.8, z0 + 2.6]];
    }
    seats.slice(0, 4).forEach(([px, pz], i) => person(px, pz, i));

    // ceiling light panels
    const lights = [];
    for (const fx of [0.3, 0.7]) for (const fz of [0.3, 0.68]) {
      lights.push(add("box", { pos: [x0 + r.w * fx, fy + WH - 0.25, z0 + r.d * fz], size: [1.4, 0.06, 0.55], color: C.lightOff, floor: f }));
    }
    // wall-mounted HVAC unit on the corridor wall, with a status LED
    const hx = x0 + r.w * 0.35, hz = RD - TP / 2 - 0.16;
    box(hx - 0.75, fy + 2.45, hz - 0.15, hx + 0.75, fy + 2.85, hz + 0.15, C.hvac, { floor: f, edges: C.frame });
    const hvacLed = box(hx + 0.45, fy + 2.5, hz - 0.18, hx + 0.65, fy + 2.58, hz - 0.15, C.lightOff, { floor: f, unlit: true });
    const hvacVent = box(hx - 0.6, fy + 2.44, hz - 0.17, hx + 0.3, fy + 2.47, hz - 0.15, C.frame, { floor: f });
    // sensor node on the room's west wall: housing + status indicator sphere
    const sx = x0 + (x0 === 0 ? T / 2 : TP / 2) + 0.06, sz = z0 + r.d * 0.5;
    box(sx - 0.05, fy + 2.0, sz - 0.18, sx + 0.06, fy + 2.4, sz + 0.18, C.hvac, { floor: f, edges: C.frame });
    const marker = add("sphere", { pos: [sx + 0.25, fy + 2.2, sz], size: [0.38, 0.38, 0.38], color: hexToRgb(STATUS_COLORS.unknown), floor: f, unlit: false });

    rooms[r.id] = {
      cfg: r, floor: f, tile, volume, marker, lights, hvacLed, hvacVent, people, base,
      min: [x0, fy, Z(z1)], max: [x1, fy + WH, Z(z0)],
      labelAnchor: [x0 + r.w / 2, fy + WH + 0.35, Z(z0 + r.d / 2)],
      center: [x0 + r.w / 2, fy + 1.2, Z(z0 + r.d / 2)],
    };
  }
  return { rooms, center: [W / 2, FH * 0.9, D / 2], FH, WH };
}

// apply one /api/rooms/live entry to the room's meshes
export function applyRoomVisuals(room, entry, highlight) {
  const status = entry ? entry.status : "unknown";
  const sc = hexToRgb(STATUS_COLORS[status] || STATUS_COLORS.unknown);
  room.tile.color = mix(room.base, sc, status === "normal" ? 0.28 : 0.55);
  room.tile.emissive = highlight.hover || highlight.selected ? [0.06, 0.06, 0.06] : [0, 0, 0];
  room.marker.color = sc;
  room.marker.emissive = mix([0, 0, 0], sc, 0.35);
  room.volume.visible = !!(highlight.hover || highlight.selected);
  room.volume.color = highlight.selected ? hexToRgb("#3b82f6") : sc;
  room.volume.opacity = highlight.selected ? 0.10 : 0.07;
  room.volume.edges = highlight.selected ? hexToRgb("#2563eb") : (highlight.hover ? hexToRgb("#64748b") : null);
  const state = entry && entry.state;
  const lights = state ? state.lights : "off";
  for (const l of room.lights) {
    l.color = lights === "on" ? [1, 0.97, 0.82] : lights === "dim" ? [0.86, 0.83, 0.7] : C.lightOff;
    l.emissive = lights === "on" ? [0.35, 0.33, 0.25] : lights === "dim" ? [0.12, 0.11, 0.08] : [0, 0, 0];
  }
  const hvac = state ? state.hvac : "off";
  room.hvacLed.color = hexToRgb(hvac === "on" ? "#22c55e" : hvac === "fault" ? "#ef4444" : "#9ca3af");
  room.hvacVent.color = hvac === "on" ? hexToRgb("#60a5fa") : C.frame;
  const occupied = !!(entry && entry.reading && entry.reading.occupancy === 1);
  for (const p of room.people) p.visible = occupied;
}
