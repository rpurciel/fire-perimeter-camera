// Front end of a fire workspace (camreg/server.py): calibration and analysis spaces for several cameras.
// Pixel coordinates are 1-based like MATLAB (pixel centres at 1..nx, 1..ny).
// The camera pose on a still is its info-bar X/Y/Z plus one fixed mount correction per camera,
// K.c = [d_azimuth, d_tilt, roll, hfov_at_Z1, lean_east, lean_north], from the panorama calibration or
// fitted by Optimize. K is the active camera; every camera keeps its own state (S.cams).
'use strict';

const D2R = Math.PI / 180, R_EARTH = 6371000;
const $ = (id) => document.getElementById(id);
const COLORS = ['#e11d9b', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#0891b2'];
const COMBO = '#ff3b1f';

const S = {
  ws: null, cams: [], space: 'calibration',
  mode: null, draft: [], pending: null, hover: null, probe: null, busy: false,
  dirty: false, combo: null, centre: null, settingCentre: false,
  tl: { start: null, end: null, bin: 0, t: null, events: [] },    // shared timeline (see setTime)
};
let K = null;           // active camera: { id, name, meta, cfg, c, calib, gcp, gcpStills, names, skyObs, skyStill, edges, ... }

// ------------------------------------------------------------------ server --
async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch('/api/' + path, opt);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}

function log(msg, err) {
  const li = document.createElement('li');
  li.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  if (err) li.className = 'err';
  $('log').prepend(li);
  (err ? console.error : console.log)(msg);
}

async function busy(fn) {
  if (S.busy) return;
  S.busy = true;
  document.querySelectorAll('aside button').forEach((b) => (b.disabled = true));
  try { await fn(); } catch (e) { log(e.message, true); }
  finally {
    S.busy = false;
    document.querySelectorAll('aside button').forEach((b) => (b.disabled = false));
  }
}

async function waitJob(id, onLog) {
  let seen = 0;
  for (;;) {
    const j = (await api('jobs?ids=' + id)).jobs[0];
    j.log.slice(seen).forEach((l) => onLog && onLog(l));
    seen = j.log.length;
    if (j.state === 'done') return j.result;
    if (j.state === 'error') throw new Error(j.error);
    await new Promise((ok) => setTimeout(ok, 700));
  }
}

function markDirty() { S.dirty = true; $('dirty').hidden = false; }

// ---------------------------------------------------------------- geometry --
const stillByFile = (f, k = K) => k.cfg.stills.find((s) => s.file === f);
const currentStill = (k = K) => k.cfg.stills[k.still];

// Same as core.pose_from_bar: zoom treated as magnification of the Z1 field of view;
// the lean tips the pan axis lean_east toward east and lean_north toward north.
// po: the camera's per-position offsets ({poseKey: {pan, tilt, roll, scale}}: pan/tilt from the edge calibration,
// as the camera doesn't return exactly to the pan/tilt its bar reports; roll and scale (image magnification)
// from a hand adjustment in the calibration step), as core.pose_from_bar.
const poseKey = (st) => `${(st.X + 0).toFixed(2)},${(st.Y + 0).toFixed(2)},${(st.Z + 0).toFixed(1)}`;
function poseFor(still, c = K.c, po = K ? K.calib.pose_offsets : null) {
  const o = po && po[poseKey(still)], X = still.X + (o ? o.pan || 0 : 0), Y = still.Y + (o ? o.tilt || 0 : 0);
  const hfov = 2 * Math.atan(Math.tan(c[3] / 2 * D2R) / still.Z) / D2R;
  let q = [((X + c[0]) % 360 + 360) % 360, Y + c[1], c[2], hfov];
  const le = c[4] || 0, ln = c[5] || 0;
  if (le || ln) {
    const A = camAxes(q), M = leanMatrix(le, ln);
    const f = mul(M, A.f), r = mul(M, A.right);
    const hd = Math.atan2(f[0], f[1]), ti = Math.asin(Math.max(-1, Math.min(1, f[2])));
    const r0 = [Math.cos(hd), -Math.sin(hd), 0], u0 = cross3(r0, f);
    q = [((hd / D2R) % 360 + 360) % 360, ti / D2R, Math.atan2(dot(r, u0), dot(r, r0)) / D2R, hfov];
  }
  const roll = o ? o.roll || 0 : 0, scale = o ? o.scale || 1 : 1;     // as core.turn_pose
  return roll || scale !== 1 ? [q[0], q[1], q[2] + roll, 2 * Math.atan(Math.tan(q[3] / 2 * D2R) / scale) / D2R] : q;
}
function leanMatrix(le, ln) {       // R_y(lean_east) * R_x(-lean_north), as core._lean_matrix
  const a = le * D2R, b = -ln * D2R, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  return [[ca, sa * sb, sa * cb], [0, cb, -sb], [-sa, ca * sb, ca * cb]];
}
const mul = (M, v) => M.map((row) => dot(row, v));
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const fullC = (c) => [...c, 0, 0, 0, 0, 0, 0].slice(0, 6);
const currentPose = (k = K) => poseFor(currentStill(k), k.c, k.calib.pose_offsets);
const po = (k) => k.calib.pose_offsets || null;          // sent with every correction to the server
const samePTZ = (a, b) => a && b && a.X === b.X && a.Y === b.Y && a.Z === b.Z;

function camAxes(q) {
  const hd = q[0] * D2R, ti = q[1] * D2R, ro = q[2] * D2R;
  const f = [Math.cos(ti) * Math.sin(hd), Math.cos(ti) * Math.cos(hd), Math.sin(ti)];
  const r0 = [Math.cos(hd), -Math.sin(hd), 0];
  const u0 = cross3(r0, f);
  const right = r0.map((x, i) => Math.cos(ro) * x + Math.sin(ro) * u0[i]);
  const up = r0.map((x, i) => -Math.sin(ro) * x + Math.cos(ro) * u0[i]);
  return { f, right, up };
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function project(rays, q, cfg = K.cfg) {
  const { nx, ny } = cfg, A = camAxes(q);
  const fp = (nx / 2) / Math.tan(q[3] / 2 * D2R);
  return rays.map((r) => {
    const d = dot(r, A.f);
    return { u: (nx + 1) / 2 + fp * dot(r, A.right) / d, v: (ny + 1) / 2 - fp * dot(r, A.up) / d, front: d > 0 };
  });
}

function groundRay(lat, lon, z, cfg = K.cfg) {
  const [lc, oc, zc] = cfg.cam;
  const north = (lat - lc) * D2R * R_EARTH, east = (lon - oc) * D2R * R_EARTH * Math.cos(lc * D2R);
  const dist = Math.hypot(east, north), up = z - zc - dist * dist / (2 * R_EARTH);
  const n = Math.hypot(east, north, up);
  return [east / n, north / n, up / n];
}

function pixelAzimuth(u, q, cfg = K.cfg) {         // azimuth of the ray through (u, image centre row)
  const { nx } = cfg, A = camAxes(q), fp = (nx / 2) / Math.tan(q[3] / 2 * D2R);
  const r = A.f.map((x, i) => x + ((u - (nx + 1) / 2) / fp) * A.right[i]);
  return Math.atan2(r[0], r[1]) / D2R;
}

// Prediction of landmark i on the still it was clicked on.
function gcpPrediction(i) {
  const g = K.gcp[i], st = stillByFile(K.gcpStills[i]);
  if (!st) return { u: NaN, v: NaN, front: false };
  return project([groundRay(g[2], g[3], g[4])], poseFor(st))[0];
}
function rms() {
  if (!K.gcp.length) return NaN;
  let s = 0;
  K.gcp.forEach((g, i) => { const p = gcpPrediction(i); s += (p.u - g[0]) ** 2 + (p.v - g[1]) ** 2; });
  return Math.sqrt(s / K.gcp.length);
}

// ------------------------------------------------------------------ viewer --
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const toScreen = (u, v) => [K.view.ox + (u - 0.5) * K.view.s, K.view.oy + (v - 0.5) * K.view.s];
const toImage = (x, y) => [(x - K.view.ox) / K.view.s + 0.5, (y - K.view.oy) / K.view.s + 0.5];

function fitView() {
  if (!K) return;
  const { nx, ny } = K.cfg, w = canvas.clientWidth, h = canvas.clientHeight;
  const s = Math.min(w / nx, h / ny);
  K.view = { s, ox: (w - nx * s) / 2, oy: (h - ny * s) / 2 };
  draw();
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
}

function pathLine(pts) {
  let pen = false;
  ctx.beginPath();
  for (const p of pts) {
    if (!p) { pen = false; continue; }
    const [x, y] = toScreen(p[0], p[1]);
    pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    pen = true;
  }
}

function draw() {
  if (!K) return;
  const dpr = window.devicePixelRatio || 1, { nx, ny } = K.cfg, { s, ox, oy } = K.view;
  const st = currentStill(), q = currentPose(), cal = S.space === 'calibration';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (K.img && K.img.complete && K.img.naturalWidth) {
    ctx.imageSmoothingEnabled = s < 2;
    ctx.drawImage(K.img, 0, 0, nx, ny, ox, oy, nx * s, ny * s);   // info bar cropped
  }
  ctx.lineJoin = 'round';
  // Red: modeled skyline (only the in-frame part, like the MATLAB figure).
  const sky = project(K.cfg.sky_rays, q).map((r) =>
    r.front && r.u >= 1 && r.u <= nx && r.v >= 1 && r.v <= ny ? [r.u, r.v] : null);
  ctx.strokeStyle = cal ? '#ff2a2a' : 'rgba(255,42,42,.55)'; ctx.lineWidth = cal ? 1.6 : 1; pathLine(sky); ctx.stroke();
  // Yellow: traced skyline, on the still it was traced on.
  if (cal && K.skyStill === st.file) {
    ctx.fillStyle = '#ffe600';
    for (const [u, v] of K.skyObs) { const [x, y] = toScreen(u, v); ctx.fillRect(x - 1.8, y - 1.8, 3.6, 3.6); }
  }
  // Fire edges: this still bright; other stills at the same pan/tilt/zoom faint (all faint while calibrating).
  for (const e of K.edges) {
    const mine = e.still === st.file;
    if (!mine && !samePTZ(stillByFile(e.still), st)) continue;
    const bright = mine && !cal;
    ctx.strokeStyle = bright ? '#ff2bd6' : 'rgba(255,43,214,.35)'; ctx.lineWidth = bright ? 2 : 1.2;
    pathLine(e.uv); ctx.stroke();
  }
  // Landmarks: cyan x = where the model puts it on this still; green circle + link
  // where it was clicked (only on the still it was clicked on).
  if (cal) {
    ctx.font = '11px sans-serif';
    K.gcp.forEach((g, i) => {
      const pr = project([groundRay(g[2], g[3], g[4])], q)[0];
      if (!pr.front) return;
      const [px, py] = toScreen(pr.u, pr.v);
      ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(px - 6, py - 6); ctx.lineTo(px + 6, py + 6);
      ctx.moveTo(px - 6, py + 6); ctx.lineTo(px + 6, py - 6); ctx.stroke();
      let lx = px, ly = py;
      if (K.gcpStills[i] === st.file) {
        const [x, y] = toScreen(g[0], g[1]);
        ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(px, py); ctx.stroke();
        ctx.strokeStyle = '#22e04a'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
        lx = x; ly = y;
      }
      ctx.fillStyle = '#22e04a'; ctx.fillText(K.names[i], lx + 8, ly - 8);
    });
  }
  drawHover(st);
  if (S.pending) cross(S.pending, '#22e04a', 9);
  if (S.probe) { const r = project([groundRay(...S.probe.geo)], q)[0]; if (r.front) cross([r.u, r.v], '#00e5ff', 10); }
  if (K.lastPoint && K.lastPoint.still === st.file) {
    const [x, y] = toScreen(K.lastPoint.u, K.lastPoint.v);
    ctx.fillStyle = '#ff2bd6'; ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
  }
  if (S.draft.length) {
    ctx.strokeStyle = S.mode === 'sky' ? '#ffe600' : '#ff2bd6'; ctx.lineWidth = 1.5;
    pathLine(S.draft); ctx.stroke();
  }
}

function cross([u, v], color, r) {
  const [x, y] = toScreen(u, v);
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); ctx.stroke();
}

// ----------------------------------------------------------- interaction --
const PROMPTS = {
  gcp: 'Click landmark in PHOTO (Esc to cancel)',
  sky: 'Draw visible terrain skyline; double-click to finish',
  terrain: 'Click a terrain pixel; Esc cancels',
  freehand: 'Trace visible fire edge freehand; release mouse to finish',
  polyline: 'Click fire edge vertices; double-click to finish',
};
const ACT_MODE = { gcp: 'addGCP', sky: 'traceSky', terrain: 'clickTerrain', freehand: 'freehandEdge', polyline: 'polylineEdge' };

function setMode(mode) {
  S.mode = mode; S.draft = [];
  $('prompt').hidden = !PROMPTS[mode];
  $('prompt').textContent = PROMPTS[mode] || '';
  canvas.classList.toggle('tool', !!mode);
  document.querySelectorAll('[data-act]').forEach((b) => b.classList.toggle('active', !!mode && b.dataset.act === ACT_MODE[mode]));
  canvas.focus();
  draw();
}

let drag = null, spaceDown = false;
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (!K) return;
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  const [u, v] = toImage(x, y);
  const inImage = u >= 0.5 && u <= K.cfg.nx + 0.5 && v >= 0.5 && v <= K.cfg.ny + 0.5;
  if (e.button !== 0 || !S.mode || spaceDown || S.mode === 'gcp_entry') {
    drag = { x, y, ox: K.view.ox, oy: K.view.oy };
    canvas.classList.add('panning');
    return;
  }
  if (!inImage) return;
  if (S.mode === 'gcp') { S.pending = [u, v]; showGcpForm(); setMode('gcp_entry'); return; }
  if (S.mode === 'terrain') { setMode(null); clickTerrain(u, v); return; }
  if (S.mode === 'sky' || S.mode === 'polyline') {
    if (e.detail >= 2 && S.draft.length) { finishDraft(); return; }
    S.draft.push([u, v]); draw(); return;
  }
  if (S.mode === 'freehand') { S.draft = [[u, v]]; S.mode = 'freehand_drawing'; }
});
window.addEventListener('mousemove', (e) => {
  if (!K) return;
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  if (drag) {
    K.view.ox = drag.ox + x - drag.x; K.view.oy = drag.oy + y - drag.y; draw(); return;
  }
  const [u, v] = toImage(x, y);
  const inImage = u >= 0.5 && u <= K.cfg.nx + 0.5 && v >= 0.5 && v <= K.cfg.ny + 0.5;
  if (inImage) {
    $('cursor').textContent = `u ${u.toFixed(1)}  v ${v.toFixed(1)}`;
    if (S.mode === 'freehand_drawing') { S.draft.push([u, v]); draw(); }
  }
  if (e.target === canvas && inImage && !S.mode) setHover(edgeNearPixel(u, v), 'image');
  else if (S.hover && S.hover.src === 'image') setHover(null);
});
window.addEventListener('mouseup', () => {
  if (drag) { drag = null; canvas.classList.remove('panning'); }
  if (S.mode === 'freehand_drawing') finishDraft();
});
canvas.addEventListener('wheel', (e) => {
  if (!K) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  const fit = Math.min(canvas.clientWidth / K.cfg.nx, canvas.clientHeight / K.cfg.ny);
  const s = Math.min(fit * 40, Math.max(fit * 0.5, K.view.s * Math.exp(-e.deltaY * 0.0015)));
  K.view.ox = x - (x - K.view.ox) * s / K.view.s;
  K.view.oy = y - (y - K.view.oy) * s / K.view.s;
  K.view.s = s; draw();
}, { passive: false });

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveWorkspace(); return; }
  const el = document.activeElement;                // the timeline slider keeps its focus after a click,
  const typing = /INPUT|SELECT|TEXTAREA/.test(el.tagName) && el.type !== 'range';   // but the keys stay ours
  if (e.key === ' ' && !typing) { spaceDown = true; e.preventDefault(); return; }
  if (typing || !K) return;
  if (e.key === 'Escape') cancel();
  else if (e.key === 'Enter' && (S.mode === 'sky' || S.mode === 'polyline')) finishDraft();
  else if (e.key === 'Backspace' && S.draft.length) { e.preventDefault(); S.draft.pop(); draw(); }
  else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? stepCamera(-1) : stepTime(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? stepCamera(1) : stepTime(1); }
  else if (/^Arrow(Up|Down)$|^Page(Up|Down)$|^Home$|^End$/.test(e.key) && el.type === 'range') e.preventDefault();
  else if (e.key === 'f' || e.key === 'F') fitView();
  else if (/^[1-9]$/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const k = S.cams[parseInt(e.key, 10) - 1];      // same numbers as the camera tabs
    if (k && k.cfg) switchCamera(k.id);
  }
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') spaceDown = false; });

function cancel() {
  S.pending = null; $('gcpForm').hidden = true; S.settingCentre = false; setMode(null);
}

async function finishDraft() {
  const mode = S.mode, pts = S.draft.slice(), still = currentStill().file, k = K;
  setMode(null);
  if (pts.length < 2) return;
  await busy(async () => {
    if (mode === 'sky') {
      // Uniformly resample drawn polyline, ~8 pixels spacing.
      const r = await api('resample', { pts, spacing: 8 });
      if (r.uv.length) { k.skyObs = r.uv; k.skyStill = still; markDirty(); }
    } else {
      const r = await api('resample', { pts, spacing: 4, exact: true });
      if (!r.uv.length) return;
      const geo = (await api('map', { camera: k.id, c: k.c, po: po(k), items: [{ still, uv: r.uv }] })).geo[0];
      saveEdgesForUndo(k);
      k.edges.push({ uv: r.uv, geo, kind: mode === 'polyline' ? 'polyline' : 'freehand', still });
      markDirty();
      const ok = geo.filter((x) => x[2] !== null).length;
      log(`${k.name} edge ${k.edges.length}: ${geo.length} samples, ${ok} valid terrain intersections.`);
    }
  });
  refresh();
}

async function clickTerrain(u, v) {
  const still = currentStill().file, k = K;
  await busy(async () => {
    const q = (await api('map', { camera: k.id, c: k.c, po: po(k), items: [{ still, uv: [[u, v]] }] })).geo[0][0];
    k.lastPoint = { u, v, geo: q, still };
    const out = $('terrainOut');
    out.hidden = false;
    if (q[2] !== null) {
      out.textContent = `Pixel (${u.toFixed(1)}, ${v.toFixed(1)})\nLat ${q[2].toFixed(7)}\nLon ${q[3].toFixed(7)}\nElevation ${q[4].toFixed(1)} m\nRange ${(q[5] / 1000).toFixed(2)} km`;
      log(`Pixel (${u.toFixed(1)}, ${v.toFixed(1)}): ${q[2].toFixed(7)} N, ${q[3].toFixed(7)} E, ${q[4].toFixed(1)} m; range ${(q[5] / 1000).toFixed(2)} km`);
    } else {
      out.textContent = 'No terrain intersection in available DEM / range.';
    }
  });
  refresh();
}

// ---------------------------------------------------------------- actions --
const fmt = (a) => a.map((x) => x.toFixed(3)).join(', ');
function setCorrection(c, label) {        // a new mount correction for the active camera
  K.c = fullC(c); if (label !== undefined) K.calib = label;
  markDirty(); refresh(); remapEdges(K);
}
const ACTIONS = {
  addGCP: () => setMode('gcp'),
  traceSky: () => setMode('sky'),
  autoSky: () => busy(async () => {
    const still = currentStill().file;
    const r = await api('autosky', { camera: K.id, still, c: K.c, po: po(K) });
    K.skyObs = r.uv; K.skyStill = still; markDirty();
    const off = r.median_offset_px;
    log(`Auto skyline: ${r.uv.length} points across ${Math.round(100 * Math.min(1, r.columns))}% of the width` +
      (off === null ? '' : `; median ${Math.abs(off).toFixed(1)} px ${off > 0 ? 'below' : 'above'} the modelled skyline`));
    if (off !== null && Math.abs(off) > 10)
      log('The detected skyline is far from the modelled one. Haze or smoke may hide distant ridges, or the ' +
        'correction is still rough. Check the yellow points before optimizing.', true);
    refresh();
  }),
  removeGCP: () => { K.gcp.pop(); K.gcpStills.pop(); K.names.pop(); markDirty(); refresh(); },
  clearSky: () => { K.skyObs = []; K.skyStill = ''; markDirty(); refresh(); },
  clickTerrain: () => setMode('terrain'),
  freehandEdge: () => setMode('freehand'),
  polylineEdge: () => setMode('polyline'),
  clearEdges: () => {                    // only the current still's edges
    const st = currentStill(), n = K.edges.filter((e) => e.still === st.file).length;
    if (!n) { log(`No edges drawn on this still (${st.time.slice(11, 19)}).`); return; }
    if (!confirm(`Remove the ${n} edge${n > 1 ? 's' : ''} drawn on the ${K.name} ${st.time.slice(11, 19)} still? (Undo brings them back.)`)) return;
    saveEdgesForUndo(K); K.edges = K.edges.filter((e) => e.still !== st.file); markDirty();
    log(`Cleared ${n} edge${n > 1 ? 's' : ''} on the ${st.time.slice(11, 19)} still; ${K.edges.length} left on other stills.`);
    refresh();
  },
  undo: () => undo(),
  optimize: () => busy(async () => {
    $('prompt').hidden = false; $('prompt').textContent = 'Optimizing mount correction...';
    const r = await api('optimize', { camera: K.id, c: K.c, po: po(K), gcp: K.gcp, gcp_stills: K.gcpStills, sky_obs: K.skyObs,
      sky_still: K.skyStill, fit_lean: $('fitLean').checked });
    const before = K.c.slice();
    const src = K.calib.source.endsWith(' + Optimize') ? K.calib.source : K.calib.source + ' + Optimize';
    log(`Optimize: ${r.message} ${r.nit} iterations, ${r.nfev} evaluations, score ${r.score.toFixed(3)}\n` +
      `correction [${fmt(before)}] -> [${fmt(fullC(r.c))}]`);
    setCorrection(r.c, { ...K.calib, source: src });
  }).finally(() => { $('prompt').hidden = !PROMPTS[S.mode]; }),
  importMat: () => busy(async () => {
    if (!$('importName').value) throw new Error('No calibration of this camera found in the workspace or output/.');
    const c = await api('calib/import', { camera: K.id, key: $('importName').value });
    saveEdgesForUndo(K);
    K.gcp = c.gcp; K.gcpStills = c.gcp_stills; K.names = c.names; K.skyObs = c.sky_obs; K.skyStill = c.sky_still;
    const added = c.edges.length;
    K.edges = K.edges.concat(c.edges.map((e) => ({ ...e, geo: [] })));
    log(`Imported ${c.name}: correction [${fmt(fullC(c.c))}], ${K.gcp.length} landmarks, ${K.skyObs.length} skyline points, ` +
      `${added} edges added` + (c.note ? ` (${c.note})` : ''));
    if (c.image_file) goTo(c.image_file);
    setCorrection(c.c, { source: 'imported ' + c.name, note: c.note, report: null });
  }),
  exportMat: () => busy(async () => {
    const r = await api('calib/export', { camera: K.id, name: $('exportMatName').value, c: K.c, po: po(K), gcp: K.gcp,
      gcp_stills: K.gcpStills, names: K.names, sky_obs: K.skyObs, sky_still: K.skyStill,
      edges: K.edges.map((e) => ({ uv: e.uv, kind: e.kind, still: e.still })), image_file: currentStill().file });
    log('Saved ' + r.saved); loadCalibList();
  }),
  exportAll: () => busy(async () => {
    const cams = {};
    for (const k of readyCams()) cams[k.id] = { c: k.c, po: po(k), edges: k.edges.map((e) => ({ uv: e.uv, kind: e.kind, still: e.still })) };
    const close = $('closeRing').checked;
    const r = await api('export_all', { name: $('exportName').value, cameras: cams, close, time_stamps: $('timeStamps').checked,
      centre: S.ws.combine.centre, max_age_min: S.ws.combine.max_age_min });
    log(`Saved ${r.saved.map((p) => p.split(/[\\/]/).pop()).join(', ')} in exports/: ${r.n_composites} combined ` +
      `perimeters (${close ? 'closed rings with connectors' : 'observed parts only'}); times: ${r.zone}`);
    if (r.overlap.length) log('Overlap check: ' + overlapText(r.overlap));
    S.ws.export_name = $('exportName').value; markDirty();
  }),
  setCentre: () => { S.settingCentre = true; $('prompt').hidden = false; $('prompt').textContent = 'Click the map at the fire centre (Esc cancels)'; },
  autoCentre: () => { S.ws.combine.centre = null; markDirty(); scheduleCombine(); },
};
document.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => ACTIONS[b.dataset.act]()));
$('resetCorr').addEventListener('click', () => {
  setCorrection(K.cfg.c0, { source: 'none (raw bar values)', note: '', report: null });
  log(`${K.name}: mount correction reset to raw bar values`);
});

// Undo: while a line is being drawn, its last vertex; otherwise the last change to the edge list
// (edge added, removed, cleared, or edges added by an import).
function saveEdgesForUndo(k) {
  k.edgeUndo.push(k.edges.slice());
  if (k.edgeUndo.length > 100) k.edgeUndo.shift();
}
function undo() {
  if (S.draft.length && (S.mode === 'polyline' || S.mode === 'sky')) { S.draft.pop(); canvas.focus(); draw(); return; }
  if (!K.edgeUndo.length) { log('Nothing to undo.'); return; }
  const before = K.edges.length;
  K.edges = K.edgeUndo.pop(); markDirty();
  const n = K.edges.length - before;
  log(`Undo: ${n < 0 ? `removed ${-n} edge${n < -1 ? 's' : ''}` : n > 0 ? `restored ${n} edge${n > 1 ? 's' : ''}` : 'edges restored'}; ${K.edges.length} now.`);
  refresh(); remapEdges(K);      // restored edges may predate a correction change
}

function showGcpForm() {
  $('gcpForm').hidden = false;
  $('gcpPix').textContent = `(${S.pending[0].toFixed(1)}, ${S.pending[1].toFixed(1)})`;
  $('gcpLat').focus();
  draw();
}
$('gcpForm').addEventListener('submit', (e) => {
  e.preventDefault();
  if (!S.pending) return;
  let q;
  try { q = readNums([['gcpLat', 'Latitude'], ['gcpLon', 'Longitude'], ['gcpZ', 'Elevation']]); }
  catch (err) { log(err.message, true); return; }
  K.gcp.push([S.pending[0], S.pending[1], ...q]);
  K.gcpStills.push(currentStill().file);
  K.names.push($('gcpName').value || 'Landmark');
  S.pending = null; $('gcpForm').hidden = true; setMode(null); markDirty(); refresh();
});
$('gcpCancel').addEventListener('click', cancel);
splitPairInto('gcpLat', 'gcpLon');
splitPairInto('cLat', 'cLon');

// Keep each edge's geolocation in step with its camera's mount correction (export does the same).
async function remapEdges(k) {
  if (!k.edges.length) { scheduleCombine(); return; }
  try {
    const geo = (await api('map', { camera: k.id, c: k.c, po: po(k), items: k.edges.map((e) => ({ still: e.still, uv: e.uv })) })).geo;
    k.edges.forEach((e, i) => { e.geo = geo[i]; });
    refresh();
  } catch (err) { log(err.message, true); }
}

// ------------------------------------------------------------- workspace --
async function saveWorkspace() {
  if (!S.ws) return;
  const cameras = {};
  for (const k of S.cams) {
    if (!k.cfg) continue;
    cameras[k.id] = {
      calibration: { c: k.c, source: k.calib.source, note: k.calib.note || '', report: k.calib.report || null,
        pose_offsets: k.calib.pose_offsets || null },
      rev: k.rev,
      landmarks: { gcp: k.gcp, stills: k.gcpStills, names: k.names }, skyline: { uv: k.skyObs, still: k.skyStill },
      edges: k.edges.map((e) => ({ uv: e.uv, kind: e.kind, still: e.still })),
      ui: { still: currentStill(k).file },
    };
  }
  try {
    const r = await api('workspace/save', { cameras, space: S.space, active_camera: K ? K.id : '',
      combine: S.ws.combine, export_name: $('exportName').value,
      timeline: { start: S.tl.start, end: S.tl.end, bin_s: S.tl.bin, t: S.tl.t } });
    for (const k of S.cams) if (r.revs && k.id in r.revs) k.rev = r.revs[k.id];
    S.dirty = false; $('dirty').hidden = true;
    log('Workspace saved (' + r.modified + ')');
    if (r.stale && r.stale.length)
      log(`The calibration or landmarks of ${r.stale.join(', ')} were changed in the calibration step after this page ` +
        'was opened, so this page did not overwrite them. Reload the page to see the new calibration.', true);
  } catch (e) { log('Save failed: ' + e.message, true); }
}
$('saveWs').addEventListener('click', saveWorkspace);
window.addEventListener('beforeunload', (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });

const readyCams = () => S.cams.filter((k) => k.cfg);

function camState(c, color) {
  const st = c.state, cal = st.calibration || {};
  st.landmarks = st.landmarks || {}; st.skyline = st.skyline || {};
  const k = { id: c.id, name: c.name, meta: c.meta, cfg: c.cfg, color,
    c: fullC(cal.c || (c.cfg ? c.cfg.c0 : [0, 0, 0, 60])),
    calib: { source: cal.c ? (cal.source || 'saved') : 'none (raw bar values)', note: cal.note || '', report: cal.report || null,
      pose_offsets: cal.c ? cal.pose_offsets || null : null },
    rev: st.rev || 0,
    gcp: st.landmarks.gcp || [], gcpStills: st.landmarks.stills || [], names: st.landmarks.names || [],
    skyObs: st.skyline.uv || [], skyStill: st.skyline.still || '',
    edges: (st.edges || []).map((e) => ({ ...e, geo: [] })), edgeUndo: [], lastPoint: null,
    still: 0, img: null, view: { s: 1, ox: 0, oy: 0 }, ui: st.ui || {} };
  if (k.cfg) {
    const i = k.cfg.stills.findIndex((s) => s.file === k.ui.still);
    k.still = Math.max(0, i);
    const known = new Set(k.cfg.stills.map((s) => s.file)), n = k.edges.length;
    k.edges = k.edges.filter((e) => known.has(e.still));
    if (k.edges.length < n) log(`${k.name}: ${n - k.edges.length} edges are on stills no longer in the stills folder; left out.`, true);
  }
  return k;
}

function renderTabs() {
  $('camTabs').innerHTML = '';
  S.cams.forEach((k, i) => {
    const b = document.createElement('button');
    b.textContent = `${i + 1} · ${k.name}`;
    b.style.setProperty('--cam', k.color);
    b.className = 'cam-tab' + (K && k.id === K.id ? ' on' : '');
    b.disabled = !k.cfg;
    b.title = k.cfg ? `${k.name} (${k.cfg.stills.length} stills)` : `${k.name} is not loaded: load it in the portal`;
    b.addEventListener('click', () => switchCamera(k.id));
    $('camTabs').appendChild(b);
  });
  document.querySelectorAll('[data-space-btn]').forEach((b) => b.classList.toggle('on', b.dataset.spaceBtn === S.space));
  document.querySelectorAll('.camname').forEach((e) => { e.textContent = K ? K.name : ''; e.style.color = K ? K.color : ''; });
}

function switchCamera(id) {               // every camera already shows its still for the timeline's time
  const k = S.cams.find((c) => c.id === id);
  if (!k || !k.cfg || k === K) return;
  cancel(); S.hover = null; S.probe = null;
  K = k;
  $('exportMatName').value = k.id + '_registration.mat';
  renderTabs(); renderCameraCard(); renderPanorama(); loadCalibList();
  resizeCanvas(); fitView(); setTime(S.tl.t);
  map && map.fitBounds(footprint(K, 0.36, 8), { padding: [10, 10] });
}

function setSpace(space) {
  S.space = space; document.body.dataset.space = space;
  cancel(); renderTabs(); refresh();
  if (map) setTimeout(() => map.invalidateSize(), 0);
}
document.querySelectorAll('[data-space-btn]').forEach((b) => b.addEventListener('click', () => setSpace(b.dataset.spaceBtn)));

// ------------------------------------------------------- calibration space --
function renderCameraCard() {
  const m = K.meta;
  $('cName').value = K.name; $('cLat').value = m.lat; $('cLon').value = m.lon; $('cElev').value = m.elev_m.toFixed(2);
  $('cStills').value = m.stills_dir; $('cHfov').value = (m.hfov0 || 60).toFixed(1);
  const ix = m.index || {};
  $('cIndex').textContent = [
    `${K.cfg.stills.length} stills · ${ix.n_poses ?? '?'} poses · ${ix.n_review ?? 0} for review (review.csv)`,
    `${(m.image_size || []).join('x')} · bar ${m.bar_px} px${m.glyphs ? ' · bar font ' + m.glyphs : ''}` +
      (ix.elev_ft ? ` · bar elevation ${ix.elev_ft} ft = ${(ix.elev_ft * 0.3048).toFixed(1)} m` : ''),
    `DEM tiles: ${(m.tiles || []).join(', ') || 'none'}` + ((m.tiles_missing || []).length ? ` · missing: ${m.tiles_missing.join(', ')}` : ''),
  ].join('\n');
}

async function refreshCamera(k) {           // after its site, folder or bar readings changed on the server
  const r = await api('camera/config', { camera: k.id });
  k.meta = r.meta;
  if (r.cfg) {
    k.cfg = r.cfg; k.img = null;
    const known = new Set(k.cfg.stills.map((s) => s.file)), n = k.edges.length;
    k.edges = k.edges.filter((e) => known.has(e.still));
    if (k.edges.length < n) log(`${k.name}: ${n - k.edges.length} edges were on stills that are gone; removed.`, true);
  }
  setWindow(S.tl.start, S.tl.end, S.tl.bin);     // its stills may have changed
  if (k === K) { renderCameraCard(); renderPanorama(); fitView(); }
  remapEdges(k);
}

async function reloadCamera(reindex) {
  const k = K;
  const r = await api('camera/reload', { camera: k.id, reindex });
  await waitJob(r.job, (l) => log(`${k.name}: ${l}`));
  await refreshCamera(k);
}

$('cApply').addEventListener('click', () => busy(async () => {
  const k = K, [lat, lon, elev_m, hfov0] = readNums([['cLat', 'Latitude'], ['cLon', 'Longitude'], ['cElev', 'Height'], ['cHfov', 'Field of view']]);
  const upd = { id: k.id, name: $('cName').value, lat, lon, elev_m, stills_dir: $('cStills').value, hfov0 };
  const r = await api('camera/update', upd);
  k.name = r.camera.name; k.meta = r.camera; renderTabs();
  log(`${k.name}: camera details saved` + (r.unloaded ? '; reloading it...' : ''));
  if (r.unloaded) await reloadCamera(false);
}));
$('cReindex').addEventListener('click', () => busy(async () => {
  if (!confirm(`Read the info bars of all of ${K.name}'s stills again? This rewrites stills_index.csv in its stills folder.`)) return;
  await reloadCamera(true);
}));

function renderPanorama() {
  const m = K.meta;
  $('panoList').innerHTML = '';
  (m.panoramas || []).forEach((p) => {
    const li = document.createElement('li');
    const a = document.createElement('a'); a.href = '/ws/panoramas/' + encodeURIComponent(p); a.target = '_blank'; a.textContent = p;
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'Take this panorama off the list';
    del.onclick = () => busy(async () => { const r = await api('panorama/remove', { camera: K.id, name: p }); K.meta.panoramas = r.panoramas; renderPanorama(); });
    li.append(a, del); $('panoList').appendChild(li);
  });
  if (!(m.panoramas || []).length) $('panoList').innerHTML = '<li class="muted">No panorama yet.</li>';
  $('toSetup').href = '/setup?camera=' + encodeURIComponent(K.id);
  const box = $('panoResult'), rep = K.calib.report;
  box.innerHTML = '';
  if (m.has_proposal) {
    const p = document.createElement('p'); p.className = 'small warn-text';
    p.textContent = 'A proposed calibration is waiting in the calibration step: check, adjust and accept it there.';
    box.appendChild(p);
  }
  if (!rep) { box.insertAdjacentHTML('beforeend', '<p class="muted small">No calibration report yet. Use the calibration step.</p>'); return; }
  const div = document.createElement('div');
  div.className = 'pano-report';
  const c = fullC(K.c), lean = Math.hypot(c[4], c[5]), deg = (x) => x === null || x === undefined ? '–' : (x >= 0 ? '+' : '') + x.toFixed(2);
  const head = document.createElement('p');
  head.className = 'small';
  head.innerHTML = `<b></b> ${rep.when || ''}<br><span class="mono">correction [${fmt(c)}]<br>lean ${lean.toFixed(2)}° toward ` +
    `${(((Math.atan2(c[4], c[5]) / D2R) % 360) + 360) % 360 | 0}°</span>`;
  head.querySelector('b').textContent = 'In use: ' + K.calib.source;
  div.appendChild(head);
  const tb = document.createElement('table'); tb.className = 'pose mono';
  if (rep.method === 'edges') {
    tb.innerHTML = '<thead><tr><th>Frame</th><th>right</th><th>down</th><th>edge</th></tr></thead>';
    for (const f of rep.frames || []) {
      const tr = document.createElement('tr'), th = document.createElement('th');
      th.textContent = f.label;
      tr.appendChild(th);
      if (!f.used) { const td = document.createElement('td'); td.colSpan = 3; td.className = 'muted'; td.textContent = f.note; tr.appendChild(td); }
      else for (const x of [deg(f.shift_after_deg[0]), deg(f.shift_after_deg[1]), (f.contrast_after ?? NaN).toFixed(0)]) {
        const td = document.createElement('td'); td.textContent = x; tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
    const note = document.createElement('p'); note.className = 'muted small';
    note.textContent = 'Remaining offset of the real silhouette from the modelled skyline, in degrees, after the calibration.';
    div.append(tb, note);
    const po = Object.entries(rep.pose_offsets || {});
    if (po.length) {
      const p = document.createElement('p'); p.className = 'small mono';
      p.textContent = po.map(([key, o]) => `pose ${key}: pan ${deg(o.pan)}°, tilt ${deg(o.tilt)}°` +
        (o.roll || (o.scale && o.scale !== 1) ? `, roll ${deg(o.roll || 0)}°, stretch ×${(o.scale || 1).toFixed(4)}` : '') + ` (${o.how})`).join('\n');
      p.style.whiteSpace = 'pre-wrap'; div.appendChild(p);
    }
    for (const l of rep.landmarks || []) {
      const p = document.createElement('p'); p.className = 'small mono';
      p.textContent = `landmark ${l.name}: ${l.error_deg === null ? '–' : l.error_deg.toFixed(3)}° (${l.error_px === null ? '–' : l.error_px.toFixed(1)} px)`;
      div.appendChild(p);
    }
  } else {                                         // an older panorama skyline fit
    tb.innerHTML = '<thead><tr><th>Frame</th><th>X</th><th>median</th><th>90%</th></tr></thead>';
    for (const f of rep.frames || []) {
      const tr = document.createElement('tr');
      tr.innerHTML = f.note ? `<th></th><td colspan="3" class="muted"></td>` :
        `<th></th><td>${f.bar[0].toFixed(2)}</td><td>${(f.median_px ?? NaN).toFixed(1)} px</td><td>${(f.p90_px ?? NaN).toFixed(1)} px</td>`;
      tr.querySelector('th').textContent = f.name;
      if (f.note) tr.querySelector('td').textContent = f.note;
      tb.appendChild(tr);
    }
    div.appendChild(tb);
  }
  if (rep.overlay) {
    const a = document.createElement('a'); a.href = '/ws/' + rep.overlay + '?t=' + encodeURIComponent(rep.when || ''); a.target = '_blank';
    const img = document.createElement('img'); img.src = a.href; img.className = 'overlay'; img.alt = 'Calibration overlay';
    img.title = rep.method === 'edges' ? 'Red: the skyline before; green: with this calibration; circles: landmarks (x: where the calibration puts them). Click to open.'
      : 'Yellow: detected skyline; orange: points left out; red: DEM skyline with the fitted correction. Click to open.';
    a.appendChild(img); div.appendChild(a);
  }
  box.appendChild(div);
}

$('panoAdd').addEventListener('change', (e) => busy(async () => {
  for (const f of e.target.files) {
    const data = await new Promise((ok, fail) => {
      const rd = new FileReader(); rd.onload = () => ok(rd.result); rd.onerror = () => fail(new Error('Could not read ' + f.name));
      rd.readAsDataURL(f);
    });
    const r = await api('panorama/upload', { camera: K.id, name: f.name, data });
    K.meta.panoramas = r.panoramas;
    log(`${K.name}: added panorama ${f.name}`);
  }
  e.target.value = ''; renderPanorama();
}));
async function loadCalibList() {
  try {
    const r = await api('calib/list', { camera: K.id });
    const sel = $('importName'); sel.innerHTML = '';
    r.files.forEach((f) => sel.add(new Option(f.label, f.key)));
  } catch (e) { log(e.message, true); }
}

// ------------------------------------------------------------------ status --
function refresh() {
  if (!K) return;
  if (S.hover && !(K.edges[S.hover.e] && S.hover.k < K.edges[S.hover.e].uv.length)) S.hover = null;
  const st = currentStill(), q = currentPose(), r = rms();
  const set = (id, x) => { $(id).textContent = x; };
  set('barX', st.X.toFixed(2)); set('barY', st.Y.toFixed(2)); set('barZ', 'Z ' + st.Z.toFixed(1));
  K.c.slice(0, 4).forEach((x, j) => set('c' + j, (j < 3 && x >= 0 ? '+' : '') + x.toFixed(3)));
  const lean = Math.hypot(K.c[4], K.c[5]), toward = ((Math.atan2(K.c[4], K.c[5]) / D2R) % 360 + 360) % 360;
  const m = K.meta || {}, demOf = (l) => ({ srtm: 'SRTM 30 m', '3dep10': 'USGS 3DEP 10 m' })[(l || 'srtm').split(':')[0]];
  const demNote = m.dem_mismatch ? `calibrated on ${demOf(m.calibration_dem)}, but the workspace now uses ${demOf(m.dem)}: calibrate it again` : '';
  set('calibLine', 'Calibration: ' + K.calib.source + (K.calib.note ? ` — ${K.calib.note}` : '') + (demNote ? ` — ${demNote}` : '') +
    (m.dem ? ` · DEM: ${demOf(m.dem)}` : ''));
  $('calibLine').classList.toggle('warn', K.calib.source.startsWith('none') || !!K.calib.note || !!demNote);
  set('leanLine', lean ? `Pan-axis lean ${lean.toFixed(2)}° toward azimuth ${toward.toFixed(0)}° ` +
    `(east ${K.c[4] >= 0 ? '+' : ''}${K.c[4].toFixed(3)}, north ${K.c[5] >= 0 ? '+' : ''}${K.c[5].toFixed(3)})`
    : 'Pan-axis lean: none (vertical)');
  q.forEach((x, j) => set('q' + j, x.toFixed(3)));
  $('status').textContent = (S.space === 'calibration'
    ? 'Green circles: clicked landmarks\nCyan crosses: model predictions\nRed: modeled skyline; yellow: traced or detected skyline\n\n'
    : 'Pink: fire edges on this camera\nMap: each camera\'s lines; red-orange: combined (observed parts)\n\n') +
    `${K.name}: azimuth ${q[0].toFixed(3)}, tilt ${q[1].toFixed(3)}, roll ${q[2].toFixed(3)}, HFOV ${q[3].toFixed(3)} deg\n` +
    `${K.gcp.length} landmarks; ${K.skyObs.length} skyline points; landmark RMS ${isNaN(r) ? 'NaN' : r.toFixed(2)} px\n` +
    readyCams().map((k) => `${k.name}: ${k.edges.length} edges on ${new Set(k.edges.map((e) => e.still)).size} stills`).join('\n');
  renderLists();
  drawLanes();
  draw();
  updateMap();
  scheduleCombine();
}

const timeOf = (file, k = K) => { const s = stillByFile(file, k); return s ? s.time.slice(11, 19) : '?'; };
function goTo(file) {                        // the time of one of the active camera's stills, for all cameras
  const st = stillByFile(file);
  if (!st) return;
  if (st.t < S.tl.start || st.t > S.tl.end) { const [a, b] = dataRange(); S.tl.t = st.t; setWindow(a, b, S.tl.bin); }
  setTime(st.t);
}

function renderLists() {
  const cur = currentStill().file;
  $('gcpList').innerHTML = '';
  K.gcp.forEach((g, i) => {
    const li = document.createElement('li');
    const p = gcpPrediction(i), res = Math.hypot(p.u - g[0], p.v - g[1]);
    if (K.gcpStills[i] !== cur) li.className = 'other';
    const label = document.createElement('span');
    const where = K.gcpStills[i].startsWith('pano:') ? 'panorama ' + K.gcpStills[i].split('#').pop() : timeOf(K.gcpStills[i]);
    label.textContent = `${K.names[i]} · ${where} (${g[2].toFixed(4)}, ${g[3].toFixed(4)}, ${g[4].toFixed(0)} m)`;
    label.className = 'go'; label.title = 'Show the still it was clicked on'; label.onclick = () => goTo(K.gcpStills[i]);
    const err = document.createElement('span'); err.className = 'mono'; err.textContent = res.toFixed(1) + ' px';
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'Remove landmark';
    del.onclick = () => { K.gcp.splice(i, 1); K.gcpStills.splice(i, 1); K.names.splice(i, 1); markDirty(); refresh(); };
    li.append(label, err, del); $('gcpList').appendChild(li);
  });
  $('edgeList').innerHTML = '';
  K.edges.forEach((e, i) => {
    const li = document.createElement('li');
    const ok = e.geo.filter((x) => x[2] !== null);
    if (e.still !== cur) li.className = 'other';
    const rng = ok.length ? ` · ${(Math.min(...ok.map((x) => x[5])) / 1000).toFixed(1)}–${(Math.max(...ok.map((x) => x[5])) / 1000).toFixed(1)} km` : '';
    const label = document.createElement('span');
    label.textContent = `Edge ${i + 1} · ${timeOf(e.still)} · ${ok.length}/${e.geo.length} on terrain${rng}`;
    label.className = 'go'; label.title = 'Show this still'; label.onclick = () => goTo(e.still);
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'Remove edge';
    del.onclick = () => { saveEdgesForUndo(K); K.edges.splice(i, 1); markDirty(); refresh(); };
    li.append(label, del); $('edgeList').appendChild(li);
  });
}

// ---------------------------------------------------------------- timeline --
// One time axis for all cameras (seconds since 1970 UTC, as each still's t). At the selected time every camera
// shows its most recent still at or before that time. The slider runs from Start to End, in steps of Bin
// (blank: from one still of any camera to the next), so long recordings can be worked through in windows.
// Times are entered and shown on the camera clock (the bar's time; the workspace's UTC offset).
const tzOffset = () => {
  const m = /^([+-])(\d{1,2}):?(\d{2})$/.exec((S.ws.utc_offset || '').trim());
  return m ? (m[1] === '-' ? -1 : 1) * (parseInt(m[2], 10) * 3600 + parseInt(m[3], 10) * 60) : 0;
};
const clock = (t) => new Date(Math.floor(t + tzOffset()) * 1000).toISOString().slice(0, 19).replace('T', ' ');
function parseClock(text) {
  const m = /^\s*(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)(?::(\d\d(?:\.\d*)?))?\s*$/.exec(String(text).replace(/[\u2010-\u2015\u2212]/g, '-'));
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) / 1000 + parseFloat(m[6] || '0') - tzOffset() : NaN;
}
const fmtDt = (s) => { const a = Math.abs(s), m = Math.floor(a / 60); return `${s < 0 ? '−' : '+'}${m}:${String(Math.round(a - 60 * m)).padStart(2, '0')}`; };

function stillAt(k, t) {                       // index of k's most recent still at or before t, -1 if none
  const st = k.cfg.stills;
  let lo = 0, hi = st.length - 1, best = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (st[m].t <= t + 1e-3) { best = m; lo = m + 1; } else hi = m - 1; }
  return best;
}
function dataRange() {
  const ks = readyCams();
  return [Math.min(...ks.map((k) => k.cfg.stills[0].t)), Math.max(...ks.map((k) => k.cfg.stills[k.cfg.stills.length - 1].t))];
}
function setWindow(start, end, bin) {          // the timeline's window and step; keeps the selected time inside
  const [lo, hi] = dataRange();
  S.tl.start = Math.max(lo - 3600, Math.min(start, end));
  S.tl.end = Math.min(hi + 3600, Math.max(start, end));
  S.tl.bin = bin > 0 ? bin : 0;
  S.tl.events = [...new Set(readyCams().flatMap((k) => k.cfg.stills.map((s) => s.t)))]
    .filter((t) => t >= S.tl.start - 1e-3 && t <= S.tl.end + 1e-3).sort((a, b) => a - b);
  $('tlStart').value = clock(S.tl.start); $('tlEnd').value = clock(S.tl.end);
  $('tlBin').value = S.tl.bin ? +(S.tl.bin / 60).toFixed(3) : '';
  const slider = $('tlSlider');
  slider.max = Math.max(1, Math.ceil(S.tl.end - S.tl.start));
  slider.step = S.tl.bin || 'any';
  setTime(Math.max(S.tl.start, Math.min(S.tl.end, S.tl.t ?? S.tl.start)));
}
function snapTime(t) {                         // a slider position -> a bin boundary, or the latest still time
  t = Math.max(S.tl.start, Math.min(S.tl.end, t));
  if (S.tl.bin) return Math.min(S.tl.end, S.tl.start + Math.round((t - S.tl.start) / S.tl.bin) * S.tl.bin);
  const ev = S.tl.events;
  let best = ev.length ? ev[0] : t;
  for (const x of ev) { if (x <= t + 1e-3) best = x; else break; }
  return best;
}
function stepTime(dir) {                       // one bin, or to the previous/next still of any camera
  const t = S.tl.t;
  if (S.tl.bin) {
    const k = (t - S.tl.start) / S.tl.bin;
    return setTime(S.tl.start + (dir > 0 ? Math.floor(k + 1e-6) + 1 : Math.ceil(k - 1e-6) - 1) * S.tl.bin);
  }
  const ev = S.tl.events, x = dir > 0 ? ev.find((e) => e > t + 1e-3) : [...ev].reverse().find((e) => e < t - 1e-3);
  if (x !== undefined) setTime(x);
}
function stepCamera(dir) {                     // the active camera's previous/next still
  const i = Math.max(0, stillAt(K, S.tl.t) + (dir > 0 ? 1 : K.early ? 0 : -1));
  if (K.cfg.stills[i]) setTime(K.cfg.stills[i].t);
}

function setTime(t) {
  t = Math.max(S.tl.start, Math.min(S.tl.end, t));
  S.tl.t = t;
  for (const k of readyCams()) {
    const i = stillAt(k, t);
    k.early = i < 0;                           // no still yet at this time: its first one is shown
    k.still = Math.max(0, i);
  }
  $('tlSlider').value = t - S.tl.start;
  const st = currentStill();
  $('stillTime').textContent = clock(t) + (S.tl.bin ? ` · bins of ${+(S.tl.bin / 60).toFixed(3)} min` : '');
  $('stillBar').textContent = `${K.name}: bar X ${st.X.toFixed(2)} Y ${st.Y.toFixed(2)} Z ${st.Z.toFixed(1)} · ${st.pose} · still ${K.still + 1}/${K.cfg.stills.length}`;
  const box = $('otherTime');
  box.innerHTML = '';
  for (const k of readyCams()) {
    const o = currentStill(k), span = document.createElement('span');
    span.textContent = k.early ? `${k.name}: no still yet (first ${o.time.slice(11, 19)})` : `${k.name} ${o.time.slice(11, 19)} (${fmtDt(o.t - t)})`;
    span.style.color = k.color;
    span.className = (k === K ? 'cur' : '') + (k.early ? ' early' : '');
    box.appendChild(span);
  }
  $('stillWarn').classList.toggle('off', st.Z === 1 && !K.early);
  $('stillWarn').textContent = K.early ? `${K.name} has no still at or before this time; its first still is shown`
    : `Z ${st.Z}: field of view assumed = Z1 value / ${st.Z} (untested)`;
  loadImage();
  refresh();
}

function loadImage() {                         // the active camera's current still (and its neighbours, cached)
  const k = K, i = k.still, st = currentStill(k);
  if (k.img && k.imgFile === st.file) return;
  const img = new Image();
  img.onload = () => { if (k.img === img && k === K) draw(); };
  img.src = `/still/${encodeURIComponent(k.id)}/${encodeURIComponent(st.file)}`;
  k.img = img; k.imgFile = st.file;
  for (const j of [i - 1, i + 1]) if (k.cfg.stills[j]) new Image().src = `/still/${encodeURIComponent(k.id)}/${encodeURIComponent(k.cfg.stills[j].file)}`;
}

// Lanes under the slider, one per camera: a tick at each still, a tall mark where fire edges are drawn, a dot
// at the still shown, and the selected time across all of them. Drawn on a canvas (thousands of stills).
const LANE = 11;
function drawLanes() {
  const cv = $('tlLanes'), ks = readyCams(), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = ks.length * LANE + 2;
  cv.style.height = h + 'px';
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d'), span = S.tl.end - S.tl.start || 1, X = (t) => (t - S.tl.start) / span * w;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  ks.forEach((k, row) => {
    const y = 1 + row * LANE;
    g.fillStyle = getComputedStyle(document.body).getPropertyValue('--btn') || '#eee';
    g.fillRect(0, y, w, LANE - 2);
    g.globalAlpha = 0.45; g.fillStyle = k.color;
    for (const s of k.cfg.stills) if (s.t >= S.tl.start && s.t <= S.tl.end) g.fillRect(Math.round(X(s.t)), y + 3, 1, LANE - 8);
    g.globalAlpha = 1;
    const withEdges = new Set(k.edges.map((e) => e.still));
    for (const s of k.cfg.stills) if (withEdges.has(s.file) && s.t >= S.tl.start && s.t <= S.tl.end) g.fillRect(Math.round(X(s.t)) - 1, y, 3, LANE - 2);
    if (!k.early) {
      const cur = currentStill(k);
      g.fillStyle = k === K ? '#111' : '#555';
      g.beginPath(); g.arc(X(cur.t), y + (LANE - 2) / 2, k === K ? 3 : 2.2, 0, 2 * Math.PI); g.fill();
    }
  });
  g.fillStyle = getComputedStyle(document.body).getPropertyValue('--accent') || '#2463eb';
  g.fillRect(Math.round(X(S.tl.t)) - 1, 0, 2, h);
}
function laneAt(e) {                           // (camera, time) under the mouse on the lanes
  const cv = $('tlLanes'), r = cv.getBoundingClientRect();
  const t = S.tl.start + (e.clientX - r.left) / r.width * (S.tl.end - S.tl.start);
  return { k: readyCams()[Math.floor((e.clientY - r.top - 1) / LANE)], t };
}
$('tlLanes').addEventListener('click', (e) => {   // the camera's still nearest the click (clicks on a mark land on it)
  const { k, t } = laneAt(e);
  if (!k) return;
  const st = k.cfg.stills, i = stillAt(k, t), j = Math.min(st.length - 1, i + 1);
  const pick = i < 0 ? st[0] : Math.abs(st[j].t - t) < Math.abs(st[i].t - t) ? st[j] : st[i];
  setTime(pick.t);
});
$('tlLanes').addEventListener('mousemove', (e) => {
  const { k, t } = laneAt(e);
  if (!k) return;
  const i = stillAt(k, t), s = k.cfg.stills[Math.max(0, i)], n = k.edges.filter((x) => x.still === s.file).length;
  $('tlLanes').title = `${k.name} · ${clock(t)} · still ${s.time.slice(11, 19)}${n ? ` · ${n} edge${n > 1 ? 's' : ''}` : ''}`;
});

$('tlSlider').addEventListener('input', (e) => setTime(snapTime(S.tl.start + parseFloat(e.target.value))));
$('prev').addEventListener('click', () => stepTime(-1));
$('next').addEventListener('click', () => stepTime(1));
function applyWindow() {
  const a = parseClock($('tlStart').value), b = parseClock($('tlEnd').value), bin = parseNum($('tlBin').value);
  if (!isFinite(a) || !isFinite(b)) { log('Enter Start and End as YYYY-MM-DD HH:MM[:SS] (camera clock).', true); setWindow(S.tl.start, S.tl.end, S.tl.bin); return; }
  if (b <= a) { log('End must be after Start.', true); setWindow(S.tl.start, S.tl.end, S.tl.bin); return; }
  setWindow(a, b, isFinite(bin) && bin > 0 ? bin * 60 : 0);
}
['tlStart', 'tlEnd', 'tlBin'].forEach((id) => $(id).addEventListener('change', applyWindow));
$('tlAll').addEventListener('click', () => { const [a, b] = dataRange(); setWindow(a, b, S.tl.bin); });
new ResizeObserver(() => { if (S.tl.start !== null) drawLanes(); }).observe($('tlLanes'));

$('fit').addEventListener('click', fitView);

// ------------------------------------------------------------ fire lines --
// All edges drawn on one still form one line for that time: their terrain points in drawing order,
// joined across gaps (no terrain hit) and between edges. Each next edge is reversed if that makes
// the join shorter. Same rule as core.fire_lines (export).
function fireLines(k = K) {
  const byStill = new Map();
  k.edges.forEach((e, i) => { if (!byStill.has(e.still)) byStill.set(e.still, []); byStill.get(e.still).push(i); });
  const lines = [];
  for (const [still, idx] of byStill) {
    const pts = [];
    for (const i of idx) {
      let run = [];
      k.edges[i].geo.forEach((g, j) => { if (g[2] !== null) run.push({ e: i, k: j, ll: [g[2], g[3]] }); });
      if (pts.length && run.length > 1) {
        const end = pts[pts.length - 1].ll;
        if (geoDist(end, run[run.length - 1].ll) < geoDist(end, run[0].ll)) run = run.reverse();
      }
      pts.push(...run);
    }
    lines.push({ still, pts });
  }
  return lines.sort((a, b) => (stillByFile(a.still, k)?.time || '').localeCompare(stillByFile(b.still, k)?.time || ''));
}
const geoDist = (a, b) => Math.hypot(a[0] - b[0], (a[1] - b[1]) * Math.cos(a[0] * D2R));

// ---------------------------------------------------------- linked hover --
// Pointing at an edge on the photo marks the same point on the map, and the other way round (active camera).
const edgeVisibleHere = (e, st = currentStill()) => e.still === st.file || samePTZ(stillByFile(e.still), st);

function edgeNearPixel(u, v) {                // nearest edge sample within 8 screen px
  let best = null, lim = 8 / K.view.s;
  K.edges.forEach((e, i) => {
    if (!edgeVisibleHere(e)) return;
    e.uv.forEach(([x, y], k) => { const d = Math.hypot(x - u, y - v); if (d < lim) { lim = d; best = { e: i, k, t: 0 }; } });
  });
  return best;
}

function edgeNearMapPoint(p) {                // nearest point on the active camera's map lines within 10 px
  if (!S.mapLines) return null;
  let best = null, lim = 10;
  for (const line of S.mapLines) {
    const c = line.pts.map((q) => map.latLngToContainerPoint(q.ll));
    for (let j = 0; j < c.length; j++) {
      const a = c[j], b = c[j + 1] || a, dx = b.x - a.x, dy = b.y - a.y, L2 = dx * dx + dy * dy;
      const t = L2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / L2)) : 0;
      const d = Math.hypot(a.x + t * dx - p.x, a.y + t * dy - p.y);
      if (d >= lim) continue;
      lim = d;
      const P = line.pts[j], Q = line.pts[j + 1];
      const ll = [P.ll[0] + t * (Q ? Q.ll[0] - P.ll[0] : 0), P.ll[1] + t * (Q ? Q.ll[1] - P.ll[1] : 0)];
      if (Q && Q.e === P.e && Q.k === P.k + 1)          // within one edge: interpolate between samples
        best = { e: P.e, k: P.k, t, ll };
      else if (Q && Q.e === P.e && Q.k === P.k - 1)     // same, on an edge joined in reverse
        best = { e: P.e, k: Q.k, t: 1 - t, ll };
      else {                                           // a join across a gap: snap to the nearer end
        const R = Q && t > 0.5 ? Q : P;
        best = { e: R.e, k: R.k, t: 0, ll: R.ll };
      }
    }
  }
  return best;
}

function setHover(h, src) {
  const same = (a, b) => (!a && !b) || (a && b && a.e === b.e && a.k === b.k && Math.abs(a.t - b.t) < 1e-3);
  if (same(S.hover, h)) return;
  S.hover = h ? { ...h, src } : null;
  draw(); showMapHover();
}

function hoverPoint(h) {                      // pixel (u, v) and terrain point [lat, lon, elev, range] of a hover
  const e = K.edges[h.e], k2 = Math.min(h.k + 1, e.uv.length - 1), mix = (a, b) => a + h.t * (b - a);
  const uv = [mix(e.uv[h.k][0], e.uv[k2][0]), mix(e.uv[h.k][1], e.uv[k2][1])];
  const g = e.geo[h.k], g2 = e.geo[k2];
  if (!g) return { uv, geo: null };
  const geo = g[2] === null ? null : h.ll ? [h.ll[0], h.ll[1], mix(g[4], (g2[2] === null ? g : g2)[4]), mix(g[5], (g2[2] === null ? g : g2)[5])]
    : [g[2], g[3], g[4], g[5]];
  return { uv, geo };
}

function hoverLabel(h, geo) {
  const where = geo ? `${geo[0].toFixed(5)}, ${geo[1].toFixed(5)} · ${geo[2].toFixed(0)} m · ${(geo[3] / 1000).toFixed(1)} km`
    : 'no terrain intersection';
  return `${K.name} edge ${h.e + 1} · ${timeOf(K.edges[h.e].still)} · ${where}`;
}

function drawHover(st) {
  if (!S.hover || !K.edges[S.hover.e]) return;
  const h = S.hover, e = K.edges[h.e];
  if (!edgeVisibleHere(e, st)) return;
  const { uv, geo } = hoverPoint(h), [x, y] = toScreen(uv[0], uv[1]);
  ctx.lineWidth = 3; ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 1.5; ctx.strokeStyle = '#ff2bd6'; ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
  if (h.src === 'map') {                      // the label is on the map when hovering the photo
    const text = hoverLabel(h, geo);
    ctx.font = '12px sans-serif';
    const w = ctx.measureText(text).width + 10, lx = Math.min(x + 12, canvas.clientWidth - w - 4), ly = Math.max(y - 26, 4);
    ctx.fillStyle = 'rgba(0,0,0,.72)'; ctx.fillRect(lx, ly, w, 18);
    ctx.fillStyle = '#fff'; ctx.fillText(text, lx + 5, ly + 13);
  }
}

function showMapHover() {
  if (!hoverLayer) return;
  hoverLayer.clearLayers();
  const h = S.hover;
  if (!h || !K.edges[h.e]) return;
  const { geo } = hoverPoint(h), e = K.edges[h.e];
  if (!geo) return;
  const note = edgeVisibleHere(e) ? '' : ` (drawn on the ${timeOf(e.still)} still)`;
  const m = L.circleMarker([geo[0], geo[1]], { radius: 7, color: '#fff', weight: 3, fillColor: '#ff2bd6', fillOpacity: 1, interactive: false })
    .addTo(hoverLayer);
  if (h.src === 'image' || note) m.bindTooltip(hoverLabel(h, geo) + note, { permanent: true, direction: 'top', offset: [0, -8] }).openTooltip();
}

// --------------------------------------------------------------------- map --
let map = null, mapLayer = null, hoverLayer = null, comboLayer = null;
const layers = {};

function footprint(k, radiusDeg, n) {
  const [lat, lon] = k.cfg.cam, cos = Math.cos(lat * D2R), q = currentPose(k), pts = [[lat, lon]];
  for (let j = 0; j <= n; j++) {
    const az = pixelAzimuth(1 + (k.cfg.nx - 1) * j / n, q, k.cfg) * D2R;
    pts.push([lat + radiusDeg * Math.cos(az), lon + radiusDeg * Math.sin(az) / cos]);
  }
  return pts;
}

function initMap() {
  if (!window.L) { $('map').textContent = 'Map unavailable (Leaflet did not load; needs internet).'; return; }
  const [lat, lon] = K.cfg.cam;
  map = L.map('map', { zoomSnap: 0.25 }).setView([lat, lon], 10);
  const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { maxZoom: 17,
    attribution: 'Map data © OpenStreetMap contributors, SRTM | Style © OpenTopoMap (CC-BY-SA)' });
  const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18, attribution: 'Imagery © Esri' });
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' });
  topo.addTo(map);
  mapLayer = L.layerGroup().addTo(map);
  comboLayer = L.layerGroup().addTo(map);
  hoverLayer = L.layerGroup().addTo(map);
  const overlays = { 'Combined perimeter (observed parts)': comboLayer };
  for (const k of readyCams()) { layers[k.id] = L.layerGroup().addTo(map); overlays[`${k.name} lines`] = layers[k.id]; }
  L.control.layers({ Topographic: topo, Satellite: sat, Streets: osm }, overlays).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);
  map.fitBounds(footprint(K, 0.36, 8), { padding: [10, 10] });   // camera's field of view, ~40 km
  map.on('mousemove', (e) => setHover(S.mode ? null : edgeNearMapPoint(e.containerPoint), 'map'));
  map.on('mouseout', () => { if (S.hover && S.hover.src === 'map') setHover(null); });
  map.on('click', async (e) => {
    if (S.settingCentre) {
      S.settingCentre = false; $('prompt').hidden = !PROMPTS[S.mode];
      S.ws.combine.centre = [e.latlng.lat, e.latlng.lng]; markDirty();
      log(`Fire centre set to ${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`);
      scheduleCombine(); return;
    }
    try {
      const z = (await api('elevation', { camera: K.id, lat: [e.latlng.lat], lon: [e.latlng.lng] })).z[0];
      if (z === null) { log('Outside the DEM.', true); return; }
      if (S.mode === 'gcp_entry') {
        $('gcpLat').value = e.latlng.lat.toFixed(6); $('gcpLon').value = e.latlng.lng.toFixed(6); $('gcpZ').value = z.toFixed(1);
      }
      S.probe = { geo: [e.latlng.lat, e.latlng.lng, z] };
      const r = project([groundRay(e.latlng.lat, e.latlng.lng, z)], currentPose())[0];
      log(`Map point ${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}, ${z.toFixed(0)} m -> ${K.name} ` +
        (r.front ? `pixel (${r.u.toFixed(1)}, ${r.v.toFixed(1)})` : 'behind camera'));
      draw(); updateMap();
    } catch (err) { log(err.message, true); }
  });
}

function updateMap() {
  if (!map) return;
  mapLayer.clearLayers();
  for (const k of readyCams()) {
    const [lat, lon] = k.cfg.cam, active = k === K;
    L.circleMarker([lat, lon], { radius: active ? 7 : 5, color: k.color, fillOpacity: 1 }).bindTooltip(k.name).addTo(mapLayer);
    const fov = footprint(k, 0.54, 24);
    fov.push([lat, lon]);
    L.polyline(fov, { color: k.color, weight: active ? 1.6 : 0.8, opacity: active ? 1 : .6, dashArray: '4 4' }).addTo(mapLayer);
    // One line per still (time) of each camera; the current still's line bold.
    if (!layers[k.id]) layers[k.id] = L.layerGroup().addTo(map);
    layers[k.id].clearLayers();
    const lines = fireLines(k), cur = currentStill(k).file;
    if (active) S.mapLines = lines;
    for (const line of lines) {
      const mine = line.still === cur;
      if (line.pts.length > 1) L.polyline(line.pts.map((p) => p.ll), { color: k.color, weight: mine ? 3 : 1.5,
        opacity: mine ? 1 : .45, interactive: false }).addTo(layers[k.id]);
    }
  }
  if (S.space === 'calibration')
    K.gcp.forEach((g, i) => L.circleMarker([g[2], g[3]], { radius: 5, color: '#16a34a', fillOpacity: .8 })
      .bindTooltip(K.names[i]).addTo(mapLayer));
  showMapHover();
  if (K.lastPoint && K.lastPoint.geo[2] !== null)
    L.circleMarker([K.lastPoint.geo[2], K.lastPoint.geo[3]], { radius: 5, color: '#e11d9b', fillOpacity: 1 }).addTo(mapLayer);
  if (S.probe) L.circleMarker([S.probe.geo[0], S.probe.geo[1]], { radius: 5, color: '#00b8d4', fill: false }).addTo(mapLayer);
}

// -------------------------------------------------------- combined preview --
// The combined perimeter at the current still's time: from each camera, its latest line at that time;
// where they see the same part of the fire, the newest line. Computed by the server (combine.py).
let comboTimer = null, comboSeq = 0;
function scheduleCombine() {
  clearTimeout(comboTimer);
  comboTimer = setTimeout(runCombine, 250);
}

function overlapText(list) {
  const pairs = new Map();
  for (const o of list) {
    const key = o.cameras.map((c) => S.cams.find((k) => k.id === c)?.name || c).join(' vs ');
    if (!pairs.has(key)) pairs.set(key, []);
    pairs.get(key).push(o);
  }
  return [...pairs].map(([key, os]) => {
    const med = os.map((o) => o.median_m).sort((a, b) => a - b)[Math.floor(os.length / 2)];
    const deg = os.reduce((s, o) => s + o.degrees, 0) / os.length;
    return `${key}: lines ${med.toFixed(0)} m apart (median) over ${deg.toFixed(0)}° of the perimeter` +
      (os.length > 1 ? ` (${os.length} times)` : `, ${Math.abs(os[0].dt_s / 60).toFixed(1)} min apart; 90% within ${os[0].p90_m.toFixed(0)} m`);
  }).join(' · ');
}

async function runCombine() {
  if (!K || S.space !== 'analysis') { comboLayer && comboLayer.clearLayers(); return; }
  const seq = ++comboSeq, cameras = {};
  for (const k of readyCams()) if (k.edges.length) cameras[k.id] = k.edges.map((e) => ({ still: e.still, geo: e.geo }));
  const ma = $('maxAge').value.trim();
  S.ws.combine.max_age_min = ma === '' || !isFinite(parseNum(ma)) ? null : parseNum(ma);
  try {
    const r = Object.keys(cameras).length ? await api('combine', { cameras, at: S.tl.t, centre: S.ws.combine.centre,
      max_age_min: S.ws.combine.max_age_min }) : { composite: null, centre: null };
    if (seq !== comboSeq) return;
    S.combo = r.composite; S.centre = r.centre;
    renderCombo();
  } catch (e) { log(e.message, true); }
}

function renderCombo() {
  if (!comboLayer) return;
  comboLayer.clearLayers();
  const c = S.combo;
  const nameOf = (id) => S.cams.find((k) => k.id === id)?.name || id;
  if (!c) {
    $('comboLine').textContent = 'No lines at or before this time yet.'; $('overlapLine').textContent = ''; return;
  }
  for (const run of c.runs) {
    L.polyline(run.pts, { color: '#fff', weight: 7, opacity: .8, interactive: false }).addTo(comboLayer);
    L.polyline(run.pts, { color: COMBO, weight: 4, opacity: 1, interactive: false }).addTo(comboLayer);
  }
  if (S.centre) L.circleMarker(S.centre, { radius: 4, color: COMBO, fill: false, weight: 2 })
    .bindTooltip(S.ws.combine.centre ? 'Fire centre (set)' : 'Fire centre (auto)').addTo(comboLayer);
  $('comboLine').textContent = `At ${clock(S.tl.t).slice(11)}: ` + c.sources.map((s) => {
    const k = S.cams.find((x) => x.id === s.camera);
    return `${nameOf(s.camera)} ${k ? timeOf(s.still, k) : ''} (${s.age_s < 1 ? 'now' : (s.age_s / 60).toFixed(0) + ' min old'}, ` +
      `${s.owned_deg.toFixed(0)}° used)`;
  }).join(' + ') + ` · ${(c.observed_m / 1000).toFixed(2)} km observed`;
  $('overlapLine').textContent = c.overlap.length ? 'Overlap check: ' + overlapText(c.overlap) : '';
}
$('maxAge').addEventListener('change', () => { markDirty(); scheduleCombine(); });

// -------------------------------------------------------------------- init --
(async function init() {
  let cfg;
  try {
    cfg = await api('config');
  } catch (e) { log('Could not open the workspace: ' + e.message + ' (open it from the portal).', true); return; }
  S.ws = cfg.workspace;
  S.ws.combine = S.ws.combine || { centre: null, max_age_min: null, close: true };
  S.cams = cfg.cameras.map((c, i) => camState(c, COLORS[i % COLORS.length]));
  document.title = S.ws.name; $('title').textContent = S.ws.name;
  $('exportName').value = S.ws.export_name || S.ws.slug;
  $('maxAge').value = S.ws.combine.max_age_min ?? '';
  const ready = readyCams();
  if (!ready.length) { log('No camera of this workspace is loaded. Load them in the portal first.', true); return; }
  S.space = new URLSearchParams(location.search).get('space') || S.ws.space || 'calibration';
  document.body.dataset.space = S.space;
  K = ready.find((k) => k.id === S.ws.active_camera) || ready[0];
  $('exportMatName').value = K.id + '_registration.mat';
  resizeCanvas();
  renderTabs(); renderCameraCard(); renderPanorama(); loadCalibList();
  fitView();
  // timeline: as saved, else all the stills of all cameras, at the active camera's last still
  const tl = S.ws.timeline || {}, [lo, hi] = dataRange();
  S.tl.t = tl.t ?? (K.ui.still ? currentStill().t : lo);
  setWindow(tl.start ?? lo, tl.end ?? hi, tl.bin_s || 0);
  initMap();
  for (const k of ready) await remapEdges(k);
  refresh();
  new ResizeObserver(() => { resizeCanvas(); fitView(); if (map) map.invalidateSize(); }).observe(canvas);
  for (const k of S.cams) if (k.calib.note) log(`${k.name}: ${k.calib.note}`, true);
  const missing = S.cams.filter((k) => !k.cfg).map((k) => k.name);
  log(`Ready: ${ready.map((k) => `${k.name} (${k.cfg.stills.length} stills, ${k.cfg.nx}×${k.cfg.ny})`).join(', ')}` +
    (missing.length ? `; not loaded: ${missing.join(', ')}` : ''));
  S.dirty = false; $('dirty').hidden = true;
})();
