// Front end for camreg/server.py: port of the verdi_combined_v3_fixed.m figure.
// Pixel coordinates are 1-based like MATLAB (pixel centres at 1..nx, 1..ny).
// The camera pose on a still is its info-bar X/Y/Z plus one fixed mount
// correction S.c = [d_azimuth, d_tilt, roll, hfov_at_Z1, lean_east, lean_north],
// fitted by Optimize (the lean usually comes from a panorama calibration).
'use strict';

const D2R = Math.PI / 180, R_EARTH = 6371000;
const $ = (id) => document.getElementById(id);

const S = {
  cfg: null, c: [0, 0, 0, 60, 0, 0], calib: 'none (raw bar values)', gcp: [], gcpStills: [], names: [], skyObs: [], skyStill: '', edges: [],
  lastPoint: null, probe: null, mode: null, draft: [], pending: null, edgeUndo: [],
  hover: null,      // edge point under the mouse on either panel: { e, k, t, src, ll }
  still: 0, img: null, view: { s: 1, ox: 0, oy: 0 }, busy: false,
};

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

// ---------------------------------------------------------------- geometry --
const stillByFile = (f) => S.cfg.stills.find((s) => s.file === f);
const currentStill = () => S.cfg.stills[S.still];

// Same as core.pose_from_bar: zoom treated as magnification of the Z1 field of view;
// the lean tips the pan axis lean_east toward east and lean_north toward north.
function poseFor(still, c = S.c) {
  const hfov = 2 * Math.atan(Math.tan(c[3] / 2 * D2R) / still.Z) / D2R;
  const q = [((still.X + c[0]) % 360 + 360) % 360, still.Y + c[1], c[2], hfov];
  const le = c[4] || 0, ln = c[5] || 0;
  if (!le && !ln) return q;
  const A = camAxes(q), M = leanMatrix(le, ln);
  const f = mul(M, A.f), r = mul(M, A.right);
  const hd = Math.atan2(f[0], f[1]), ti = Math.asin(Math.max(-1, Math.min(1, f[2])));
  const r0 = [Math.cos(hd), -Math.sin(hd), 0], u0 = cross3(r0, f);
  return [((hd / D2R) % 360 + 360) % 360, ti / D2R, Math.atan2(dot(r, u0), dot(r, r0)) / D2R, hfov];
}
function leanMatrix(le, ln) {       // R_y(lean_east) * R_x(-lean_north), as core._lean_matrix
  const a = le * D2R, b = -ln * D2R, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  return [[ca, sa * sb, sa * cb], [0, cb, -sb], [-sa, ca * sb, ca * cb]];
}
const mul = (M, v) => M.map((row) => dot(row, v));
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const fullC = (c) => [...c, 0, 0, 0, 0, 0, 0].slice(0, 6);
const currentPose = () => poseFor(currentStill());
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

function project(rays, q) {
  const { nx, ny } = S.cfg, A = camAxes(q);
  const fp = (nx / 2) / Math.tan(q[3] / 2 * D2R);
  return rays.map((r) => {
    const d = dot(r, A.f);
    return { u: (nx + 1) / 2 + fp * dot(r, A.right) / d, v: (ny + 1) / 2 - fp * dot(r, A.up) / d, front: d > 0 };
  });
}

function groundRay(lat, lon, z) {
  const [lc, oc, zc] = S.cfg.cam;
  const north = (lat - lc) * D2R * R_EARTH, east = (lon - oc) * D2R * R_EARTH * Math.cos(lc * D2R);
  const dist = Math.hypot(east, north), up = z - zc - dist * dist / (2 * R_EARTH);
  const n = Math.hypot(east, north, up);
  return [east / n, north / n, up / n];
}

function pixelAzimuth(u, q) {         // azimuth of the ray through (u, image centre row)
  const { nx } = S.cfg, A = camAxes(q), fp = (nx / 2) / Math.tan(q[3] / 2 * D2R);
  const r = A.f.map((x, i) => x + ((u - (nx + 1) / 2) / fp) * A.right[i]);
  return Math.atan2(r[0], r[1]) / D2R;
}

// Prediction of landmark i on the still it was clicked on.
function gcpPrediction(i) {
  const g = S.gcp[i];
  return project([groundRay(g[2], g[3], g[4])], poseFor(stillByFile(S.gcpStills[i])))[0];
}
function rms() {
  if (!S.gcp.length) return NaN;
  let s = 0;
  S.gcp.forEach((g, i) => { const p = gcpPrediction(i); s += (p.u - g[0]) ** 2 + (p.v - g[1]) ** 2; });
  return Math.sqrt(s / S.gcp.length);
}

// ------------------------------------------------------------------ viewer --
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const toScreen = (u, v) => [S.view.ox + (u - 0.5) * S.view.s, S.view.oy + (v - 0.5) * S.view.s];
const toImage = (x, y) => [(x - S.view.ox) / S.view.s + 0.5, (y - S.view.oy) / S.view.s + 0.5];

function fitView() {
  const { nx, ny } = S.cfg, w = canvas.clientWidth, h = canvas.clientHeight;
  const s = Math.min(w / nx, h / ny);
  S.view = { s, ox: (w - nx * s) / 2, oy: (h - ny * s) / 2 };
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
  if (!S.cfg) return;
  const dpr = window.devicePixelRatio || 1, { nx, ny } = S.cfg, { s, ox, oy } = S.view;
  const st = currentStill(), q = currentPose();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (S.img && S.img.complete && S.img.naturalWidth) {
    ctx.imageSmoothingEnabled = s < 2;
    ctx.drawImage(S.img, 0, 0, nx, ny, ox, oy, nx * s, ny * s);   // info bar cropped
  }
  ctx.lineJoin = 'round';
  // Red: modeled skyline (only the in-frame part, like the MATLAB figure).
  const sky = project(S.cfg.sky_rays, q).map((r) =>
    r.front && r.u >= 1 && r.u <= nx && r.v >= 1 && r.v <= ny ? [r.u, r.v] : null);
  ctx.strokeStyle = '#ff2a2a'; ctx.lineWidth = 1.6; pathLine(sky); ctx.stroke();
  // Yellow: traced skyline, on the still it was traced on.
  if (S.skyStill === st.file) {
    ctx.fillStyle = '#ffe600';
    for (const [u, v] of S.skyObs) { const [x, y] = toScreen(u, v); ctx.fillRect(x - 1.8, y - 1.8, 3.6, 3.6); }
  }
  // Fire edges: this still bright; other stills at the same pan/tilt/zoom faint.
  for (const e of S.edges) {
    const mine = e.still === st.file;
    if (!mine && !samePTZ(stillByFile(e.still), st)) continue;
    ctx.strokeStyle = mine ? '#ff2bd6' : 'rgba(255,43,214,.35)'; ctx.lineWidth = mine ? 2 : 1.2;
    pathLine(e.uv); ctx.stroke();
  }
  // Landmarks: cyan x = where the model puts it on this still; green circle + link
  // where it was clicked (only on the still it was clicked on).
  ctx.font = '11px sans-serif';
  S.gcp.forEach((g, i) => {
    const pr = project([groundRay(g[2], g[3], g[4])], q)[0];
    if (!pr.front) return;
    const [px, py] = toScreen(pr.u, pr.v);
    ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(px - 6, py - 6); ctx.lineTo(px + 6, py + 6);
    ctx.moveTo(px - 6, py + 6); ctx.lineTo(px + 6, py - 6); ctx.stroke();
    let lx = px, ly = py;
    if (S.gcpStills[i] === st.file) {
      const [x, y] = toScreen(g[0], g[1]);
      ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(px, py); ctx.stroke();
      ctx.strokeStyle = '#22e04a'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
      lx = x; ly = y;
    }
    ctx.fillStyle = '#22e04a'; ctx.fillText(S.names[i], lx + 8, ly - 8);
  });
  drawHover(st);
  if (S.pending) cross(S.pending, '#22e04a', 9);
  if (S.probe) { const r = project([groundRay(...S.probe.geo)], q)[0]; if (r.front) cross([r.u, r.v], '#00e5ff', 10); }
  if (S.lastPoint && S.lastPoint.still === st.file) {
    const [x, y] = toScreen(S.lastPoint.u, S.lastPoint.v);
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
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  const [u, v] = toImage(x, y);
  const inImage = u >= 0.5 && u <= S.cfg.nx + 0.5 && v >= 0.5 && v <= S.cfg.ny + 0.5;
  if (e.button !== 0 || !S.mode || spaceDown || S.mode === 'gcp_entry') {
    drag = { x, y, ox: S.view.ox, oy: S.view.oy };
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
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  if (drag) {
    S.view.ox = drag.ox + x - drag.x; S.view.oy = drag.oy + y - drag.y; draw(); return;
  }
  const [u, v] = toImage(x, y);
  const inImage = S.cfg && u >= 0.5 && u <= S.cfg.nx + 0.5 && v >= 0.5 && v <= S.cfg.ny + 0.5;
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
  e.preventDefault();
  const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
  const fit = Math.min(canvas.clientWidth / S.cfg.nx, canvas.clientHeight / S.cfg.ny);
  const s = Math.min(fit * 40, Math.max(fit * 0.5, S.view.s * Math.exp(-e.deltaY * 0.0015)));
  S.view.ox = x - (x - S.view.ox) * s / S.view.s;
  S.view.oy = y - (y - S.view.oy) * s / S.view.s;
  S.view.s = s; draw();
}, { passive: false });

document.addEventListener('keydown', (e) => {
  const typing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName);
  if (e.key === ' ' && !typing) { spaceDown = true; e.preventDefault(); return; }
  if (typing) return;
  if (e.key === 'Escape') cancel();
  else if (e.key === 'Enter' && (S.mode === 'sky' || S.mode === 'polyline')) finishDraft();
  else if (e.key === 'Backspace' && S.draft.length) { e.preventDefault(); S.draft.pop(); draw(); }
  else if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
  else if (e.key === 'ArrowLeft') setStill(S.still - 1);
  else if (e.key === 'ArrowRight') setStill(S.still + 1);
  else if (e.key === 'f' || e.key === 'F') fitView();
});
document.addEventListener('keyup', (e) => { if (e.key === ' ') spaceDown = false; });

function cancel() {
  S.pending = null; $('gcpForm').hidden = true; setMode(null);
}

async function finishDraft() {
  const mode = S.mode, pts = S.draft.slice(), still = currentStill().file;
  setMode(null);
  if (pts.length < 2) return;
  await busy(async () => {
    if (mode === 'sky') {
      // Uniformly resample drawn polyline, ~8 pixels spacing.
      const r = await api('resample', { pts, spacing: 8 });
      if (r.uv.length) { S.skyObs = r.uv; S.skyStill = still; }
    } else {
      const r = await api('resample', { pts, spacing: 4, exact: true });
      if (!r.uv.length) return;
      const geo = (await api('map', { c: S.c, items: [{ still, uv: r.uv }] })).geo[0];
      saveEdgesForUndo();
      S.edges.push({ uv: r.uv, geo, kind: mode === 'polyline' ? 'polyline' : 'freehand', still });
      const ok = geo.filter((x) => x[2] !== null).length;
      log(`Edge ${S.edges.length}: ${geo.length} samples, ${ok} valid terrain intersections.`);
    }
  });
  refresh();
}

async function clickTerrain(u, v) {
  const still = currentStill().file;
  await busy(async () => {
    const q = (await api('map', { c: S.c, items: [{ still, uv: [[u, v]] }] })).geo[0][0];
    S.lastPoint = { u, v, geo: q, still };
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
const ACTIONS = {
  addGCP: () => setMode('gcp'),
  traceSky: () => setMode('sky'),
  autoSky: () => busy(async () => {
    const still = currentStill().file;
    const r = await api('autosky', { still, c: S.c });
    S.skyObs = r.uv; S.skyStill = still;
    const off = r.median_offset_px;
    log(`Auto skyline: ${r.uv.length} points across ${Math.round(100 * Math.min(1, r.columns))}% of the width` +
      (off === null ? '' : `; median ${Math.abs(off).toFixed(1)} px ${off > 0 ? 'below' : 'above'} the modelled skyline`));
    if (off !== null && Math.abs(off) > 10)
      log('The detected skyline is far from the modelled one. Haze or smoke may hide distant ridges, or the ' +
        'correction is still rough. Check the yellow points before optimizing.', true);
    refresh();
  }),
  removeGCP: () => { S.gcp.pop(); S.gcpStills.pop(); S.names.pop(); refresh(); },
  clearSky: () => { S.skyObs = []; S.skyStill = ''; refresh(); },
  clickTerrain: () => setMode('terrain'),
  freehandEdge: () => setMode('freehand'),
  polylineEdge: () => setMode('polyline'),
  clearEdges: () => {                    // only the current still's edges
    const st = currentStill(), n = S.edges.filter((e) => e.still === st.file).length;
    if (!n) { log(`No edges drawn on this still (${st.time.slice(11, 19)}).`); return; }
    if (!confirm(`Remove the ${n} edge${n > 1 ? 's' : ''} drawn on the ${st.time.slice(11, 19)} still? (Undo brings them back.)`)) return;
    saveEdgesForUndo(); S.edges = S.edges.filter((e) => e.still !== st.file);
    log(`Cleared ${n} edge${n > 1 ? 's' : ''} on the ${st.time.slice(11, 19)} still; ${S.edges.length} left on other stills.`);
    refresh();
  },
  undo: () => undo(),
  optimize: () => busy(async () => {
    $('prompt').hidden = false; $('prompt').textContent = 'Optimizing mount correction...';
    const r = await api('optimize', { c: S.c, gcp: S.gcp, gcp_stills: S.gcpStills, sky_obs: S.skyObs, sky_still: S.skyStill,
      fit_lean: $('fitLean').checked });
    const before = S.c.slice();
    S.c = r.c;
    if (!S.calib.endsWith('+ Optimize (unsaved)')) S.calib += ' + Optimize (unsaved)';
    log(`Optimize: ${r.message} ${r.nit} iterations, ${r.nfev} evaluations, score ${r.score.toFixed(3)}\n` +
      `correction [${fmt(before)}] -> [${fmt(S.c)}]`);
    refresh(); remapEdges();
  }).finally(() => { $('prompt').hidden = !PROMPTS[S.mode]; }),
  saveFit: () => busy(async () => {
    const r = await api('save', { name: $('calibName').value, c: S.c, gcp: S.gcp, gcp_stills: S.gcpStills,
      names: S.names, sky_obs: S.skyObs, sky_still: S.skyStill, edges: S.edges, image_file: currentStill().file });
    log('Saved ' + r.saved); setFiles(r.files); S.calib = r.saved.split(/[\/]/).pop(); refresh();
  }),
  loadFit: () => busy(async () => {
    if (!$('loadName').value) throw new Error('No .mat calibration in the output folder yet.');
    const c = await api('load', { name: $('loadName').value });
    saveEdgesForUndo();
    applyCalibration(c);
  }),
  exportEdges: () => busy(async () => {
    if (!S.edges.length) throw new Error('Draw at least one edge first.');
    const r = await api('export', { name: $('exportName').value, c: S.c, edges: S.edges, time_stamps: $('timeStamps').checked });
    log('Saved ' + r.saved.join(' and ') + ' (' + r.note + ')'); setFiles(r.files);
  }),
};
document.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => ACTIONS[b.dataset.act]()));
$('resetCorr').addEventListener('click', () => {
  S.c = fullC(S.cfg.c0); S.calib = 'none (raw bar values)';
  log('Mount correction reset to raw bar values'); refresh(); remapEdges();
});

// Undo: while a line is being drawn, its last vertex; otherwise the last change to the edge list
// (edge added, removed, cleared, or replaced by loading a calibration).
function saveEdgesForUndo() {
  S.edgeUndo.push(S.edges.slice());
  if (S.edgeUndo.length > 100) S.edgeUndo.shift();
}
function undo() {
  if (S.draft.length && (S.mode === 'polyline' || S.mode === 'sky')) { S.draft.pop(); canvas.focus(); draw(); return; }
  if (!S.edgeUndo.length) { log('Nothing to undo.'); return; }
  const before = S.edges.length;
  S.edges = S.edgeUndo.pop();
  const n = S.edges.length - before;
  log(`Undo: ${n < 0 ? `removed ${-n} edge${n < -1 ? 's' : ''}` : n > 0 ? `restored ${n} edge${n > 1 ? 's' : ''}` : 'edges restored'}; ${S.edges.length} now.`);
  refresh(); remapEdges();      // restored edges may predate a correction change
}

function applyCalibration(c) {
  S.c = fullC(c.c); S.calib = c.name; S.gcp = c.gcp; S.gcpStills = c.gcp_stills; S.names = c.names;
  S.skyObs = c.sky_obs; S.skyStill = c.sky_still; S.edges = c.edges;
  const i = S.cfg.stills.findIndex((s) => s.file === c.image_file);
  if (i >= 0) setStill(i);
  log(`Loaded ${c.name}: correction [${fmt(S.c)}], ${S.gcp.length} landmarks, ${S.skyObs.length} skyline points, ${S.edges.length} edges` +
    (c.note ? ` (${c.note})` : ''));
  refresh(); remapEdges();
}

function showGcpForm() {
  $('gcpForm').hidden = false;
  $('gcpPix').textContent = `(${S.pending[0].toFixed(1)}, ${S.pending[1].toFixed(1)})`;
  $('gcpLat').focus();
  draw();
}
$('gcpForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = ['gcpLat', 'gcpLon', 'gcpZ'].map((id) => parseFloat($(id).value));
  if (!S.pending) return;
  if (q.some((x) => !isFinite(x))) { log('Enter valid latitude, longitude and elevation.', true); return; }
  S.gcp.push([S.pending[0], S.pending[1], ...q]);
  S.gcpStills.push(currentStill().file);
  S.names.push($('gcpName').value || 'Landmark');
  S.pending = null; $('gcpForm').hidden = true; setMode(null); refresh();
});
$('gcpCancel').addEventListener('click', cancel);

// Keep each edge's geolocation in step with the mount correction (export does the same).
async function remapEdges() {
  if (!S.edges.length) return;
  try {
    const geo = (await api('map', { c: S.c, items: S.edges.map((e) => ({ still: e.still, uv: e.uv })) })).geo;
    S.edges.forEach((e, k) => { e.geo = geo[k]; });
    refresh();
  } catch (err) { log(err.message, true); }
}

// ------------------------------------------------------------------ status --
function refresh() {
  if (S.hover && !(S.edges[S.hover.e] && S.hover.k < S.edges[S.hover.e].uv.length)) S.hover = null;
  const st = currentStill(), q = currentPose(), r = rms();
  const set = (id, x) => { $(id).textContent = x; };
  set('barX', st.X.toFixed(2)); set('barY', st.Y.toFixed(2)); set('barZ', 'Z ' + st.Z.toFixed(1));
  S.c.slice(0, 4).forEach((x, j) => set('c' + j, (j < 3 && x >= 0 ? '+' : '') + x.toFixed(3)));
  const lean = Math.hypot(S.c[4], S.c[5]), toward = ((Math.atan2(S.c[4], S.c[5]) / D2R) % 360 + 360) % 360;
  set('calibLine', 'Calibration: ' + S.calib);
  $('calibLine').classList.toggle('warn', S.calib.startsWith('none'));
  set('leanLine', lean ? `Pan-axis lean ${lean.toFixed(2)}° toward azimuth ${toward.toFixed(0)}° ` +
    `(east ${S.c[4] >= 0 ? '+' : ''}${S.c[4].toFixed(3)}, north ${S.c[5] >= 0 ? '+' : ''}${S.c[5].toFixed(3)})`
    : 'Pan-axis lean: none (vertical)');
  q.forEach((x, j) => set('q' + j, x.toFixed(3)));
  $('status').textContent =
    'Green circles: clicked landmarks\nCyan crosses: model predictions\nRed: modeled skyline; yellow: traced or detected skyline\n\n' +
    `Azimuth ${q[0].toFixed(3)} deg\nTilt ${q[1].toFixed(3)} deg\nRoll ${q[2].toFixed(3)} deg\nHFOV ${q[3].toFixed(3)} deg\n\n` +
    `${S.gcp.length} landmarks; ${S.skyObs.length} skyline points\nLandmark RMS: ${isNaN(r) ? 'NaN' : r.toFixed(2)} pixels\n${S.edges.length} mapped edges`;
  renderLists();
  renderEdgeMarks();
  draw();
  updateMap();
}

const timeOf = (file) => { const s = stillByFile(file); return s ? s.time.slice(11, 19) : '?'; };
function goTo(file) { const i = S.cfg.stills.findIndex((s) => s.file === file); if (i >= 0) setStill(i); }

function renderLists() {
  const cur = currentStill().file;
  $('gcpList').innerHTML = '';
  S.gcp.forEach((g, i) => {
    const li = document.createElement('li');
    const p = gcpPrediction(i), res = Math.hypot(p.u - g[0], p.v - g[1]);
    if (S.gcpStills[i] !== cur) li.className = 'other';
    const label = document.createElement('span');
    label.textContent = `${S.names[i]} · ${timeOf(S.gcpStills[i])} (${g[2].toFixed(4)}, ${g[3].toFixed(4)}, ${g[4].toFixed(0)} m)`;
    label.className = 'go'; label.title = 'Show the still it was clicked on'; label.onclick = () => goTo(S.gcpStills[i]);
    const err = document.createElement('span'); err.className = 'mono'; err.textContent = res.toFixed(1) + ' px';
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'Remove landmark';
    del.onclick = () => { S.gcp.splice(i, 1); S.gcpStills.splice(i, 1); S.names.splice(i, 1); refresh(); };
    li.append(label, err, del); $('gcpList').appendChild(li);
  });
  $('edgeList').innerHTML = '';
  S.edges.forEach((e, i) => {
    const li = document.createElement('li');
    const ok = e.geo.filter((x) => x[2] !== null);
    if (e.still !== cur) li.className = 'other';
    const rng = ok.length ? ` · ${(Math.min(...ok.map((x) => x[5])) / 1000).toFixed(1)}–${(Math.max(...ok.map((x) => x[5])) / 1000).toFixed(1)} km` : '';
    const label = document.createElement('span');
    label.textContent = `Edge ${i + 1} · ${timeOf(e.still)} · ${ok.length}/${e.geo.length} on terrain${rng}`;
    label.className = 'go'; label.title = 'Show this still'; label.onclick = () => goTo(e.still);
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'Remove edge';
    del.onclick = () => { saveEdgesForUndo(); S.edges.splice(i, 1); refresh(); };
    li.append(label, del); $('edgeList').appendChild(li);
  });
}

// Pink marks under the slider at the stills that have fire edges; click one to go there.
let marksKey = '';
function renderEdgeMarks() {
  const n = S.cfg.stills.length, count = new Map();
  S.edges.forEach((e) => count.set(e.still, (count.get(e.still) || 0) + 1));
  const idx = [...count.keys()].map((f) => S.cfg.stills.findIndex((s) => s.file === f)).filter((i) => i >= 0).sort((a, b) => a - b);
  const key = idx.map((i) => `${i}:${count.get(S.cfg.stills[i].file)}`).join(',') + '|' + S.still;
  if (key === marksKey) return;
  marksKey = key;
  const box = $('edgeMarks');
  box.innerHTML = '';
  for (const i of idx) {
    const st = S.cfg.stills[i], k = count.get(st.file), m = document.createElement('span');
    m.style.left = (n > 1 ? 100 * i / (n - 1) : 50) + '%';
    m.style.width = Math.max(100 / n, 0) + '%';
    m.title = `${st.time.slice(11, 19)} · ${k} edge${k > 1 ? 's' : ''}`;
    if (i === S.still) m.classList.add('current');
    m.addEventListener('click', () => setStill(i));
    box.appendChild(m);
  }
}

function setFiles(files) {
  const sel = $('loadName'), cur = sel.value;
  sel.innerHTML = '';
  files.filter((f) => f.toLowerCase().endsWith('.mat')).forEach((f) => sel.add(new Option(f, f)));
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
}

// ------------------------------------------------------------------ stills --
function setStill(i) {
  i = Math.max(0, Math.min(S.cfg.stills.length - 1, i));
  S.still = i;
  const st = currentStill();
  $('stillSlider').value = i;
  $('stillTime').textContent = st.time.slice(0, 22);
  $('stillBar').textContent = `bar X ${st.X.toFixed(2)} Y ${st.Y.toFixed(2)} Z ${st.Z.toFixed(1)} · ${st.pose} · ${i + 1}/${S.cfg.stills.length}`;
  $('stillWarn').classList.toggle('off', st.Z === 1);
  $('stillWarn').textContent = `Z ${st.Z}: field of view assumed = Z1 value / ${st.Z} (untested)`;
  const img = new Image();
  img.onload = () => { if (S.img === img) draw(); };
  img.src = '/still/' + encodeURIComponent(st.file);
  S.img = img;
  for (const k of [i - 1, i + 1]) if (S.cfg.stills[k]) new Image().src = '/still/' + encodeURIComponent(S.cfg.stills[k].file);
  refresh();
}
$('stillSlider').addEventListener('input', (e) => setStill(parseInt(e.target.value, 10)));
$('prev').addEventListener('click', () => setStill(S.still - 1));
$('next').addEventListener('click', () => setStill(S.still + 1));

// ------------------------------------------------------------ fire lines --
// All edges drawn on one still form one line for that time: their terrain points in drawing order,
// joined across gaps (no terrain hit) and between edges. Each next edge is reversed if that makes
// the join shorter. Same rule as core.fire_lines (export).
function fireLines() {
  const byStill = new Map();
  S.edges.forEach((e, i) => { if (!byStill.has(e.still)) byStill.set(e.still, []); byStill.get(e.still).push(i); });
  const lines = [];
  for (const [still, idx] of byStill) {
    const pts = [];
    for (const i of idx) {
      let run = [];
      S.edges[i].geo.forEach((g, k) => { if (g[2] !== null) run.push({ e: i, k, ll: [g[2], g[3]] }); });
      if (pts.length && run.length > 1) {
        const end = pts[pts.length - 1].ll;
        if (geoDist(end, run[run.length - 1].ll) < geoDist(end, run[0].ll)) run = run.reverse();
      }
      pts.push(...run);
    }
    lines.push({ still, pts });
  }
  return lines.sort((a, b) => (stillByFile(a.still)?.time || '').localeCompare(stillByFile(b.still)?.time || ''));
}
const geoDist = (a, b) => Math.hypot(a[0] - b[0], (a[1] - b[1]) * Math.cos(a[0] * D2R));

// ---------------------------------------------------------- linked hover --
// Pointing at an edge on the photo marks the same point on the map, and the other way round.
const edgeVisibleHere = (e, st = currentStill()) => e.still === st.file || samePTZ(stillByFile(e.still), st);

function edgeNearPixel(u, v) {                // nearest edge sample within 8 screen px
  let best = null, lim = 8 / S.view.s;
  S.edges.forEach((e, i) => {
    if (!edgeVisibleHere(e)) return;
    e.uv.forEach(([x, y], k) => { const d = Math.hypot(x - u, y - v); if (d < lim) { lim = d; best = { e: i, k, t: 0 }; } });
  });
  return best;
}

function edgeNearMapPoint(p) {                // nearest point on a map fire line within 10 px
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
  const e = S.edges[h.e], k2 = Math.min(h.k + 1, e.uv.length - 1), mix = (a, b) => a + h.t * (b - a);
  const uv = [mix(e.uv[h.k][0], e.uv[k2][0]), mix(e.uv[h.k][1], e.uv[k2][1])];
  const g = e.geo[h.k], g2 = e.geo[k2];
  const geo = g[2] === null ? null : h.ll ? [h.ll[0], h.ll[1], mix(g[4], (g2[2] === null ? g : g2)[4]), mix(g[5], (g2[2] === null ? g : g2)[5])]
    : [g[2], g[3], g[4], g[5]];
  return { uv, geo };
}

function hoverLabel(h, geo) {
  const where = geo ? `${geo[0].toFixed(5)}, ${geo[1].toFixed(5)} · ${geo[2].toFixed(0)} m · ${(geo[3] / 1000).toFixed(1)} km`
    : 'no terrain intersection';
  return `Edge ${h.e + 1} · ${timeOf(S.edges[h.e].still)} · ${where}`;
}

function drawHover(st) {
  if (!S.hover || !S.edges[S.hover.e]) return;
  const h = S.hover, e = S.edges[h.e];
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
  if (!h || !S.edges[h.e]) return;
  const { geo } = hoverPoint(h), e = S.edges[h.e];
  if (!geo) return;
  const note = edgeVisibleHere(e) ? '' : ` (drawn on the ${timeOf(e.still)} still)`;
  const m = L.circleMarker([geo[0], geo[1]], { radius: 7, color: '#fff', weight: 3, fillColor: '#ff2bd6', fillOpacity: 1, interactive: false })
    .addTo(hoverLayer);
  if (h.src === 'image' || note) m.bindTooltip(hoverLabel(h, geo) + note, { permanent: true, direction: 'top', offset: [0, -8] }).openTooltip();
}

// --------------------------------------------------------------------- map --
let map = null, mapLayer = null, hoverLayer = null;

function footprint(radiusDeg, n) {
  const [lat, lon] = S.cfg.cam, cos = Math.cos(lat * D2R), q = currentPose(), pts = [[lat, lon]];
  for (let k = 0; k <= n; k++) {
    const az = pixelAzimuth(1 + (S.cfg.nx - 1) * k / n, q) * D2R;
    pts.push([lat + radiusDeg * Math.cos(az), lon + radiusDeg * Math.sin(az) / cos]);
  }
  return pts;
}

function initMap() {
  if (!window.L) { $('map').textContent = 'Map unavailable (Leaflet did not load; needs internet).'; return; }
  const [lat, lon] = S.cfg.cam;
  map = L.map('map', { zoomSnap: 0.25 }).setView([lat, lon], 10);
  const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { maxZoom: 17,
    attribution: 'Map data © OpenStreetMap contributors, SRTM | Style © OpenTopoMap (CC-BY-SA)' });
  const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18, attribution: 'Imagery © Esri' });
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' });
  topo.addTo(map);
  L.control.layers({ Topographic: topo, Satellite: sat, Streets: osm }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);
  mapLayer = L.layerGroup().addTo(map);
  hoverLayer = L.layerGroup().addTo(map);
  map.fitBounds(footprint(0.36, 8), { padding: [10, 10] });   // camera's field of view, ~40 km
  map.on('mousemove', (e) => setHover(S.mode ? null : edgeNearMapPoint(e.containerPoint), 'map'));
  map.on('mouseout', () => { if (S.hover && S.hover.src === 'map') setHover(null); });
  map.on('click', async (e) => {
    try {
      const z = (await api('elevation', { lat: [e.latlng.lat], lon: [e.latlng.lng] })).z[0];
      if (z === null) { log('Outside the DEM.', true); return; }
      if (S.mode === 'gcp_entry') {
        $('gcpLat').value = e.latlng.lat.toFixed(6); $('gcpLon').value = e.latlng.lng.toFixed(6); $('gcpZ').value = z.toFixed(1);
      }
      S.probe = { geo: [e.latlng.lat, e.latlng.lng, z] };
      const r = project([groundRay(e.latlng.lat, e.latlng.lng, z)], currentPose())[0];
      log(`Map point ${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}, ${z.toFixed(0)} m -> ` +
        (r.front ? `pixel (${r.u.toFixed(1)}, ${r.v.toFixed(1)})` : 'behind camera'));
      draw(); updateMap();
    } catch (err) { log(err.message, true); }
  });
}

function updateMap() {
  if (!map) return;
  mapLayer.clearLayers();
  const [lat, lon] = S.cfg.cam, cur = currentStill().file;
  L.circleMarker([lat, lon], { radius: 6, color: '#00b8d4', fillOpacity: 1 }).bindTooltip('Camera').addTo(mapLayer);
  const fov = footprint(0.54, 24);
  fov.push([lat, lon]);
  L.polyline(fov, { color: '#00b8d4', weight: 1, dashArray: '4 4' }).addTo(mapLayer);
  S.gcp.forEach((g, i) => L.circleMarker([g[2], g[3]], { radius: 5, color: '#16a34a', fillOpacity: .8 })
    .bindTooltip(S.names[i]).addTo(mapLayer));
  // One line per still (time): all its edges joined, across gaps (see fireLines).
  S.mapLines = fireLines();
  for (const line of S.mapLines) {
    const mine = line.still === cur;
    if (line.pts.length > 1) L.polyline(line.pts.map((p) => p.ll), { color: '#e11d9b', weight: mine ? 3 : 1.5,
      opacity: mine ? 1 : .45, interactive: false }).addTo(mapLayer);
  }
  showMapHover();
  if (S.lastPoint && S.lastPoint.geo[2] !== null)
    L.circleMarker([S.lastPoint.geo[2], S.lastPoint.geo[3]], { radius: 5, color: '#e11d9b', fillOpacity: 1 }).addTo(mapLayer);
  if (S.probe) L.circleMarker([S.probe.geo[0], S.probe.geo[1]], { radius: 5, color: '#00b8d4', fill: false }).addTo(mapLayer);
}

// -------------------------------------------------------------------- init --
(async function init() {
  try {
    S.cfg = await api('config');
  } catch (e) { log('Could not reach the server: ' + e.message, true); return; }
  const c = S.cfg;
  document.title = c.title; $('title').textContent = c.title;
  S.c = fullC(c.c0);
  $('calibName').value = c.calib_name; $('exportName').value = c.export_name;
  setFiles(c.files);
  $('stillSlider').max = c.stills.length - 1;
  resizeCanvas();
  fitView();
  setStill(Math.max(0, c.stills.findIndex((s) => s.file === c.default_still)));
  if (c.initial) applyCalibration(c.initial);
  initMap();
  refresh();
  new ResizeObserver(() => { resizeCanvas(); fitView(); if (map) map.invalidateSize(); }).observe(canvas);
  log(`Ready: ${c.stills.length} stills, image ${c.nx}×${c.ny} (${c.bar_px} px bar cropped); pose from each frame's info bar`);
})();
