// Calibration step (camreg/server.py /setup): representative frames of each camera, optional landmarks clicked
// on them, and the edge-based calibration (camreg/edgefit.py) run per camera. Its result is a proposal, which can
// be moved, rotated and stretched by hand before it is accepted. The server projects the skylines and landmarks
// (/api/setup/predict) and turns a hand adjustment (a 2-D similarity drawn here) into camera angles
// (/api/setup/adjust), so this page has no camera maths of its own.
// Pixel coordinates are 1-based like the rest of the app (pixel centres at 1..w, 1..h).
'use strict';

const $ = (id) => document.getElementById(id);
const P = {
  ws: null, cams: [], K: null,          // K: { id, name, frames, landmarks, calibration, rev, site }
  frame: null, img: null, view: { s: 1, ox: 0, oy: 0 }, lines: {}, pred: [], fov: null,
  mode: null, pending: null, busy: false,
  // hand adjustment of the proposal: transforms [a, b, tx, c, d, ty] (u' = a u + b v + tx, v' = c u + d v + ty)
  // not yet sent (pend), being turned into angles by the server (infl) and being dragged (dragT)
  scope: 'camera', pend: null, infl: null, dragT: null, adj: null, flushTimer: null, lmErr: null,
  // suggested landmarks (reference points the camera can see, /api/reference/candidates) and the one being placed
  cands: null, cand: null,
};

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
  if (P.busy) return;
  P.busy = true;
  document.querySelectorAll('aside button').forEach((b) => (b.disabled = true));
  try { await fn(); } catch (e) { log(e.message, true); }
  finally {
    P.busy = false; document.querySelectorAll('aside button').forEach((b) => (b.disabled = false));
    if (P.K) { renderAdjust(); draw(); }
  }
}

const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); kids.forEach((k) => e.append(k)); return e; };
const deg = (x, d = 2) => (x === null || x === undefined || !isFinite(x) ? '–' : (x >= 0 ? '+' : '') + x.toFixed(d));

// ------------------------------------------------------------------ viewer --
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const toScreen = (u, v) => [P.view.ox + (u - 0.5) * P.view.s, P.view.oy + (v - 0.5) * P.view.s];
const toImage = (x, y) => [(x - P.view.ox) / P.view.s + 0.5, (y - P.view.oy) / P.view.s + 0.5];
const frameSize = () => (P.frame && P.frame.size) || [1920, 1080];

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr; canvas.height = canvas.clientHeight * dpr;
}
function fitView() {
  const [w, h] = frameSize(), s = Math.min(canvas.clientWidth / w, canvas.clientHeight / h);
  P.view = { s, ox: (canvas.clientWidth - w * s) / 2, oy: (canvas.clientHeight - h * s) / 2 };
  draw();
}

function polyline(pts) {
  ctx.beginPath();
  pts.forEach(([u, v], i) => { const [x, y] = toScreen(u, v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
}
function stroke(pts, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); polyline(pts); ctx.stroke(); ctx.setLineDash([]);
}

// ------------------------------------------------------ 2-D similarities --
const ID = () => [1, 0, 0, 0, 1, 0];
const tmul = (B, A) => [B[0] * A[0] + B[1] * A[3], B[0] * A[1] + B[1] * A[4], B[0] * A[2] + B[1] * A[5] + B[2],
  B[3] * A[0] + B[4] * A[3], B[3] * A[1] + B[4] * A[4], B[3] * A[2] + B[4] * A[5] + B[5]];      // B after A
const tapply = (T, [u, v]) => [T[0] * u + T[1] * v + T[2], T[3] * u + T[4] * v + T[5]];
const isId = (T) => T.every((x, i) => Math.abs(x - ID()[i]) < 1e-9);
function similarity([pu, pv], rotDeg = 0, scale = 1, du = 0, dv = 0) {   // about (pu, pv); positive = clockwise on screen
  const t = rotDeg * Math.PI / 180, a = scale * Math.cos(t), b = -scale * Math.sin(t), c = scale * Math.sin(t), d = a;
  return [a, b, pu - a * pu - b * pv + du, c, d, pv - c * pu - d * pv + dv];
}
// What the proposed line looks like now: its last server projection with the adjustments not yet turned into angles.
const shownT = () => tmul(P.dragT || ID(), tmul(P.pend || ID(), P.infl || ID()));
const shownLine = () => (P.lines.proposal ? (() => { const T = shownT(); return P.lines.proposal.map((p) => tapply(T, p)); })() : null);

function draw() {
  const dpr = window.devicePixelRatio || 1, [w, h] = frameSize(), { s, ox, oy } = P.view;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (P.img && P.img.complete && P.img.naturalWidth) {
    ctx.imageSmoothingEnabled = s < 2;
    ctx.drawImage(P.img, 0, 0, w, h, ox, oy, w * s, h * s);       // the info bar (below h) is not shown
  }
  ctx.lineJoin = 'round';
  const L = P.lines, prop = P.K && P.K.proposal;
  if (prop) {                                      // in use red; automatic result yellow; proposed green
    if (L.calibration) stroke(L.calibration, 'rgba(255,40,40,.9)', 1.2);
    if (L.auto && prop.adjusted) stroke(L.auto, 'rgba(255,214,0,.95)', 1.2, [6, 4]);
    if (L.proposal) stroke(shownLine(), '#22e04a', 1.6);
  } else {
    if (L.previous) stroke(L.previous, 'rgba(255,40,40,.9)', 1.2);
    if (L.calibration) stroke(L.calibration, edgeCalibrated() ? '#22e04a' : '#ff2a2a', 1.6);
  }
  ctx.font = '12px sans-serif';
  const T = prop ? shownT() : ID();
  const lm = P.K ? P.K.landmarks : { gcp: [], stills: [], names: [] };
  lm.gcp.forEach((g, i) => {
    if (lm.stills[i] !== P.frame.id) return;
    const [x, y] = toScreen(g[0], g[1]);
    ctx.strokeStyle = '#22e04a'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    const p = P.pred.find((q) => q[0] === i);
    if (p) {
      const [px, py] = toScreen(...tapply(T, [p[1], p[2]]));
      ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(px - 6, py - 6); ctx.lineTo(px + 6, py + 6); ctx.moveTo(px - 6, py + 6); ctx.lineTo(px + 6, py - 6); ctx.stroke();
      ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(px, py); ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.lineWidth = 3;
    ctx.strokeText(lm.names[i], x + 9, y - 9); ctx.fillText(lm.names[i], x + 9, y - 9);
  });
  if (P.cands && $('candShow').checked) {
    const used = new Set((lm.info || []).map((x) => x && x.ref_id).filter(Boolean));
    for (const c of P.cands) {
      if (!c.in_frame || used.has(c.id)) continue;
      const [x, y] = toScreen(...tapply(T, [c.u, c.v])), on = P.cand && P.cand.id === c.id, r = on ? 7 : 5;
      ctx.strokeStyle = CAND_COLOR[c.source] || '#ffb300'; ctx.lineWidth = on ? 2.4 : 1.4;
      ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); ctx.stroke();
      if (on || P.view.s > 1.2) {                   // names when zoomed in (the list has them all)
        ctx.fillStyle = '#fff'; ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.lineWidth = 3;
        ctx.strokeText(c.name, x + 8, y + 4); ctx.fillText(c.name, x + 8, y + 4);
      }
    }
  }
  if (P.pending) {
    const [x, y] = toScreen(...P.pending);
    ctx.strokeStyle = '#22e04a'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();
  }
  const g = adjustable() && (P.adj ? P.adj.g : gizmo());
  if (g) drawGizmo(g);
}

// ------------------------------------------------- hand adjustment: gizmo --
// The ⊕ pivot sits on the proposed line in the middle of the view; round handles (rotate) and square ones
// (stretch) on a bar through it. Dragging the line moves it.
const adjustable = () => !!(P.K && P.K.proposal && P.frame && P.lines.proposal && !P.busy && !P.mode &&
  (P.scope === 'camera' || P.frame.kind === 'still'));
function lineV(line, u) {                          // v of a line (sorted by u) at column u
  let i = 1;
  while (i < line.length - 1 && line[i][0] < u) i++;
  const [a, b] = [line[i - 1], line[i]], t = b[0] === a[0] ? 0 : (u - a[0]) / (b[0] - a[0]);
  return a[1] + t * (b[1] - a[1]);
}
function gizmo() {
  const line = shownLine();
  if (!line || line.length < 2) return null;
  const [w] = frameSize(), [u0] = toImage(0, 0), [u1] = toImage(canvas.clientWidth, 0);
  const lo = Math.max(line[0][0], u0, 1), hi = Math.min(line[line.length - 1][0], u1, w);
  if ((hi - lo) * P.view.s < 40) return null;
  const pivot = [(lo + hi) / 2, lineV(line, (lo + hi) / 2)], [px, py] = toScreen(...pivot);
  return { pivot, px, py, L: Math.max(50, Math.min(360, 0.32 * (hi - lo) * P.view.s)) };
}
function handles(g) {                              // screen positions; while dragging they follow the drag
  const at = (dx) => [g.px + dx, g.py], pts = { pivot: at(0), rot: [at(-g.L), at(g.L)], str: [at(-0.55 * g.L), at(0.55 * g.L)] };
  if (!P.dragT) return pts;
  const f = (p) => toScreen(...tapply(P.dragT, toImage(...p)));
  return { pivot: f(pts.pivot), rot: pts.rot.map(f), str: pts.str.map(f) };
}
function drawGizmo(g) {
  const h = handles(g);
  ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(...h.rot[0]); ctx.lineTo(...h.rot[1]); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#fff'; ctx.strokeStyle = '#111'; ctx.lineWidth = 1.5;
  for (const [x, y] of h.rot) { ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  for (const [x, y] of h.str) { ctx.fillRect(x - 5, y - 5, 10, 10); ctx.strokeRect(x - 5, y - 5, 10, 10); }
  const [x, y] = h.pivot;
  ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
}
function hitTest(x, y) {                           // screen point -> 'rotate' | 'stretch' | 'move' | null
  const g = gizmo();
  if (!g) return null;
  const h = handles(g), near = (p, r) => Math.hypot(p[0] - x, p[1] - y) <= r;
  if (h.rot.some((p) => near(p, 9))) return { kind: 'rotate', g };
  if (h.str.some((p) => near(p, 9))) return { kind: 'stretch', g };
  if (near(h.pivot, 9)) return { kind: 'move', g };
  const pts = shownLine().map((p) => toScreen(...p));
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i], dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    if (Math.hypot(ax + t * dx - x, ay + t * dy - y) <= 6) return { kind: 'move', g };
  }
  return null;
}
function dragTransform(a) {
  const { g } = a;
  if (a.kind === 'move') return similarity(g.pivot, 0, 1, (a.vx - a.sx) / P.view.s, (a.vy - a.sy) / P.view.s);
  if (a.kind === 'rotate') {
    const ang = Math.atan2(a.vy - g.py, a.vx - g.px) - Math.atan2(a.sy - g.py, a.sx - g.px);
    return similarity(g.pivot, ((ang * 180 / Math.PI + 540) % 360) - 180);
  }
  const k = (a.vx - g.px) / (a.sx - g.px);
  return similarity(g.pivot, 0, Math.max(0.5, Math.min(2, isFinite(k) ? k : 1)));
}

const edgeCalibrated = () => P.K && P.K.calibration && P.K.calibration.report && P.K.calibration.report.method === 'edges';

let drag = null, moved = false;
canvas.addEventListener('mousedown', (e) => {
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const hit = e.button === 0 && adjustable() ? hitTest(x, y) : null;
  if (hit) {                                       // a hand adjustment (Shift: a tenth of the mouse motion)
    P.adj = { kind: hit.kind, g: hit.g, sx: x, sy: y, vx: x, vy: y, lx: x, ly: y };
    e.preventDefault();
    return;
  }
  drag = { x: e.clientX, y: e.clientY, ox: P.view.ox, oy: P.view.oy }; moved = false;
});
window.addEventListener('mousemove', (e) => {
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (P.frame) {
    const [u, v] = toImage(x, y), [w, h] = frameSize();
    if (u >= 0.5 && v >= 0.5 && u <= w + 0.5 && v <= h + 0.5) $('cursor').textContent = `u ${u.toFixed(1)}  v ${v.toFixed(1)}`;
  }
  if (P.adj) {
    const a = P.adj, f = e.shiftKey ? 0.1 : 1;
    a.vx += (x - a.lx) * f; a.vy += (y - a.ly) * f; a.lx = x; a.ly = y;
    P.dragT = dragTransform(a); draw();
    return;
  }
  if (!drag && e.target === canvas) {
    const hit = adjustable() ? hitTest(x, y) : null;
    canvas.classList.toggle('adj-move', !!hit && hit.kind === 'move');
    canvas.classList.toggle('adj-rotate', !!hit && hit.kind === 'rotate');
    canvas.classList.toggle('adj-stretch', !!hit && hit.kind === 'stretch');
  }
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
  if (moved) { P.view.ox = drag.ox + dx; P.view.oy = drag.oy + dy; draw(); }
});
window.addEventListener('mouseup', (e) => {
  if (P.adj) {
    if (P.dragT) P.pend = tmul(P.dragT, P.pend || ID());
    P.adj = null; P.dragT = null;
    flushAdjust();
    return;
  }
  const was = drag; drag = null;
  if (was && !moved && e.target === canvas && !P.mode && P.cands && $('candShow').checked) {
    const r0 = canvas.getBoundingClientRect(), c = candAt(e.clientX - r0.left, e.clientY - r0.top);
    if (c) { placeCandidate(c); return; }
  }
  if (!was || moved || e.target !== canvas || P.mode !== 'add') return;
  const r = canvas.getBoundingClientRect(), [u, v] = toImage(e.clientX - r.left, e.clientY - r.top), [w, h] = frameSize();
  if (u < 0.5 || v < 0.5 || u > w + 0.5 || v > h + 0.5) return;
  P.pending = [u, v]; P.mode = 'entry';
  $('lmPix').textContent = `(${u.toFixed(1)}, ${v.toFixed(1)})`; $('lmFrame').textContent = P.frame.label;
  $('lmForm').hidden = false;
  if (P.cand) {                                    // a suggested landmark: its location is known
    const c = P.cand;
    $('lmName').value = c.name; $('lmLat').value = c.lat.toFixed(7); $('lmLon').value = c.lon.toFixed(7); $('lmZ').value = c.elev.toFixed(1);
    $('prompt').textContent = `${c.name}: check the location below and confirm`;
  } else {
    $('prompt').textContent = 'Now give its location: type it, or click it on the map'; $('lmLat').focus();
  }
  draw();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, [w, h] = frameSize();
  const fit = Math.min(canvas.clientWidth / w, canvas.clientHeight / h);
  const s = Math.min(fit * 40, Math.max(fit * 0.5, P.view.s * Math.exp(-e.deltaY * 0.0015)));
  P.view.ox = x - (x - P.view.ox) * s / P.view.s; P.view.oy = y - (y - P.view.oy) * s / P.view.s; P.view.s = s; draw();
}, { passive: false });
const NUDGE_KEYS = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down', BracketLeft: 'ccw',
  BracketRight: 'cw', Minus: 'shrink', NumpadSubtract: 'shrink', Equal: 'grow', NumpadAdd: 'grow' };
document.addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === 'Escape') cancelLandmark();
  else if (e.key === 'f' || e.key === 'F') fitView();
  else if (NUDGE_KEYS[e.code] && adjustable() && !e.metaKey && !e.ctrlKey) { e.preventDefault(); nudge(NUDGE_KEYS[e.code], e.shiftKey); }
});
$('fit').addEventListener('click', fitView);

// -------------------------------------------------------------- cameras --
function renderTabs() {
  $('camTabs').innerHTML = '';
  P.cams.forEach((c, i) => {
    const b = el('button', { textContent: `${i + 1} · ${c.name}`, className: 'cam-tab' + (P.K && P.K.id === c.id ? ' on' : ''),
      disabled: !c.ready, title: c.ready ? c.name : `${c.name} is not loaded: load it in the portal` });
    b.addEventListener('click', () => busy(() => openCamera(c.id)));
    $('camTabs').appendChild(b);
  });
  document.querySelectorAll('.camname').forEach((e) => { e.textContent = P.K ? P.K.name : ''; });
}

async function openCamera(id, frameId) {
  cancelLandmark();
  await settle();
  const keepCands = !!(P.cands && P.K && P.K.id === id);
  const r = await api('setup/frames', { camera: id });
  P.K = { id, ...r };
  P.pend = P.infl = P.dragT = null; P.lmErr = null; P.cands = null; P.cand = null;
  $('candBox').hidden = true; $('suggest').textContent = 'Suggest landmarks';
  renderTabs(); renderStrip(); renderLandmarks(); renderResult(); renderAdjust();
  const f = P.K.frames.find((x) => x.id === frameId) || P.K.frames.find((x) => x.url && x.bar && !x.note) || P.K.frames[0];
  if (f) await selectFrame(f);
  if (keepCands) { await fetchCands(false); draw(); }
  history.replaceState(null, '', '/setup?camera=' + encodeURIComponent(id));
  drawMap();
}

function renderStrip() {
  $('strip').innerHTML = '';
  for (const f of P.K.frames) {
    const n = P.K.landmarks.stills.filter((s) => s === f.id).length;
    const b = el('button', { className: 'thumb' + (P.frame && P.frame.id === f.id ? ' on' : ''), disabled: !f.url || !f.bar,
      title: f.note || f.label });
    if (f.url) b.append(el('img', { src: f.url, alt: f.label, loading: 'lazy' }));
    b.append(el('span', { textContent: f.label + (n ? ` · ${n} landmark${n > 1 ? 's' : ''}` : '') + (f.note ? ` · ${f.note}` : '') }));
    b.addEventListener('click', () => busy(() => selectFrame(f)));
    $('strip').appendChild(b);
  }
}

async function selectFrame(f) {
  await settle();
  P.frame = f;
  renderAdjust();
  const img = new Image();
  img.onload = () => { if (P.img === img) draw(); };
  img.src = f.url;
  P.img = img;
  renderStrip(); fitView();
  await predict();
}

// Lines: with a proposal, the calibration in use, the proposal and (once adjusted) its automatic result; else
// the calibration in use and the one before it. Landmarks are projected with the proposal if there is one.
async function fetchPredict() {
  const prop = P.K.proposal;
  const lines = prop ? ['calibration', 'proposal'].concat(prop.adjusted ? ['auto'] : []) : ['calibration', 'previous'];
  return api('setup/predict', { camera: P.K.id, frame: P.frame.id, lines, landmarks_with: prop ? 'proposal' : 'calibration' });
}
function applyPredict(r) { P.lines = r.lines; P.pred = r.landmarks; P.fov = r.fov; P.lmErr = r.landmark_errors; }
async function predict() {
  if (!P.frame || !P.K) return;
  const id = P.K.id, fid = P.frame.id, r = await fetchPredict();
  if (!P.K || P.K.id !== id || P.frame.id !== fid) return;
  applyPredict(r);
  if (P.cands) await fetchCands(false);
  draw(); drawMap(); renderLandmarks();
}

// ------------------------------------------------------------ landmarks --
$('addLm').addEventListener('click', () => {
  if (!P.frame) return;
  P.mode = 'add'; P.pending = null; P.cand = null;
  $('prompt').hidden = false; $('prompt').textContent = 'Click the landmark in the image (Esc cancels)';
  canvas.classList.add('tool');
});
function cancelLandmark() {
  P.mode = null; P.pending = null; P.cand = null; $('lmForm').hidden = true; $('prompt').hidden = true; canvas.classList.remove('tool');
  renderCands(); draw();
}
$('lmCancel').addEventListener('click', cancelLandmark);
splitPairInto('lmLat', 'lmLon');

$('lmSnap').addEventListener('click', () => busy(async () => {
  const [lat, lon] = readNums([['lmLat', 'Latitude'], ['lmLon', 'Longitude']]);
  const r = await api('setup/summit', { camera: P.K.id, lat, lon });
  $('lmLat').value = r.lat.toFixed(6); $('lmLon').value = r.lon.toFixed(6); $('lmZ').value = r.elev.toFixed(1);
  log(`Snapped to the highest DEM point nearby (${r.moved_m.toFixed(0)} m away): ${r.elev.toFixed(0)} m`);
  probe = [r.lat, r.lon]; drawMap();
}));

$('lmForm').addEventListener('submit', (e) => {
  e.preventDefault();
  busy(async () => {
    const [lat, lon, z] = readNums([['lmLat', 'Latitude'], ['lmLon', 'Longitude'], ['lmZ', 'Elevation']]);
    const lm = P.K.landmarks, c = P.cand;
    // a suggested point keeps its source and accuracy, unless its location was edited
    const same = c && Math.abs(c.lat - lat) < 1e-6 && Math.abs(c.lon - lon) < 1e-6 && Math.abs(c.elev - z) < 0.5;
    const next = { gcp: [...lm.gcp, [P.pending[0], P.pending[1], lat, lon, z]], stills: [...lm.stills, P.frame.id],
      names: [...lm.names, $('lmName').value || 'Landmark'],
      info: [...lmInfo(lm), same ? { source: c.source, ref_id: c.id, acc_m: c.acc_m } : {}] };
    await saveLandmarks(next);
    log(`${P.K.name}: landmark ${next.names[next.names.length - 1]} added on ${P.frame.label}`);
    cancelLandmark();
  });
});

async function saveLandmarks(lm) {
  const r = await api('setup/landmarks', { camera: P.K.id, landmarks: lm });
  P.K.landmarks = r.landmarks; P.K.rev = r.rev;
  renderLandmarks(); renderStrip(); await predict();
}

function renderLandmarks() {
  const lm = P.K.landmarks, rep = (P.K.calibration || {}).report || {};
  const live = P.lmErr ? Object.fromEntries(P.lmErr.map(([i, px, d]) => [i, { error_px: px, error_deg: d }])) : null;
  $('lmList').innerHTML = '';
  if (!lm.gcp.length) $('lmList').append(el('li', { className: 'muted', textContent: 'No landmarks (optional).' }));
  lm.gcp.forEach((g, i) => {
    const f = P.K.frames.find((x) => x.id === lm.stills[i]);
    const res = live ? live[i] : (rep.landmarks || []).find((x) => x.name === lm.names[i] && x.frame === lm.stills[i]);
    const inf = lmInfo(lm)[i];
    const label = el('span', { className: 'go', textContent: `${lm.names[i]} · ${f ? f.label : lm.stills[i]} (${g[2].toFixed(4)}, ${g[3].toFixed(4)}, ${g[4].toFixed(0)} m)` +
      (inf.source ? ` · ${inf.source} ±${inf.acc_m.toFixed(0)} m` : ''), title: 'Show its frame' });
    label.onclick = () => f && busy(() => selectFrame(f));
    const err = el('span', { className: 'mono', textContent: res && res.error_deg !== null ? `${res.error_deg.toFixed(2)}° (${res.error_px.toFixed(0)} px)` : '' });
    const del = el('button', { textContent: '×', title: 'Remove landmark' });
    del.onclick = () => busy(async () => {
      await saveLandmarks({ gcp: lm.gcp.filter((_, j) => j !== i), stills: lm.stills.filter((_, j) => j !== i), names: lm.names.filter((_, j) => j !== i),
        info: lmInfo(lm).filter((_, j) => j !== i) });
    });
    $('lmList').append(el('li', {}, label, err, del));
  });
}

// ------------------------------------------------- suggested landmarks --
// Reference points with known coordinates the camera can see (camreg/reference.py): NOAA NGS survey marks
// (surveyed: lookouts, towers, peaks), OpenStreetMap peaks and towers, FAA obstacles. They are fetched once per
// camera (a job); here they are projected with the proposal (or the calibration in use) onto the frame. Placing
// one is click-to-confirm: its coordinates and accuracy come with it.
const CAND_COLOR = { NGS: '#00e5ff', OSM: '#ffb300', FAA: '#ff4fd8' };
const lmInfo = (lm) => lm.gcp.map((_, i) => (lm.info && lm.info[i]) || {});

async function fetchCands(refresh) {
  const id = P.K.id, fid = P.frame && P.frame.id;
  let r = await api('reference/candidates', { camera: id, frame: fid, with: P.K.proposal ? 'proposal' : 'calibration' });
  if (!r.fetched || refresh) {
    log(`${P.K.name}: fetching reference points (NGS, OpenStreetMap, FAA; about a minute)...`);
    const j = await api('reference/fetch', { camera: id });
    const [job] = await waitJobs([j.job]);
    if (job.state === 'error') throw new Error('Reference points: ' + job.error);
    Object.entries(job.result.sources).forEach(([n, x]) => log(`  ${n}: ${x.error ? 'not available (' + x.error + ')' : x.n + ' points'}`, !!x.error));
    r = await api('reference/candidates', { camera: id, frame: fid, with: P.K.proposal ? 'proposal' : 'calibration' });
  }
  if (!P.K || P.K.id !== id || (P.frame && P.frame.id) !== fid) return;
  P.cands = r.candidates;
  P.candMeta = r;
  renderCands();
}
function renderCands() {
  if (!P.cands) return;
  $('candBox').hidden = false;
  $('suggest').textContent = 'Fetch points again';
  const used = new Set(lmInfo(P.K.landmarks).map((x) => x.ref_id).filter(Boolean));
  const inFrame = P.cands.filter((c) => c.in_frame && !used.has(c.id));
  inFrame.sort((a, b) => (b.on_skyline - a.on_skyline) || (a.sigma_deg - b.sigma_deg));
  $('candInfo').textContent = `${inFrame.length} of ${P.cands.length} visible reference points fall in this frame (fetched ${P.candMeta.fetched}).` +
    (P.candMeta.note ? ' ' + P.candMeta.note : '') + ' Bold: on the skyline, the easiest to find. ±: how well its position is known, seen from the camera. ' +
    'Click one (in the list or its ◇ on the frame; zoom in for their names), then click the feature itself in the image.';
  $('candList').innerHTML = '';
  for (const c of inFrame.slice(0, 60)) {
    const go = el('span', { className: 'go', textContent: `${c.name} · ${c.kind}, ${(c.range_m / 1000).toFixed(1)} km`,
      title: c.note + (c.moved_m ? ` · moved ${c.moved_m.toFixed(0)} m to the DEM summit` : '') });
    go.onclick = () => placeCandidate(c);
    $('candList').append(el('li', { className: (c.on_skyline ? 'sky' : '') + (P.cand && P.cand.id === c.id ? ' on' : '') },
      el('span', { className: 'cand-src', textContent: c.source }), go,
      el('span', { className: 'mono small', textContent: `±${c.sigma_deg.toFixed(c.sigma_deg < 0.01 ? 3 : 2)}°` })));
  }
  if (!inFrame.length) $('candList').append(el('li', { className: 'muted', textContent: 'None in this frame.' }));
}
function candAt(x, y) {                            // the suggested point under a screen position
  const T = P.K.proposal ? shownT() : ID();
  let best = null, bd = 9;
  for (const c of P.cands || []) {
    if (!c.in_frame) continue;
    const [cx, cy] = toScreen(...tapply(T, [c.u, c.v])), d = Math.hypot(cx - x, cy - y);
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}
function placeCandidate(c) {
  cancelLandmark();
  P.cand = c; P.mode = 'add'; P.pending = null;
  $('prompt').hidden = false;
  $('prompt').textContent = `Click ${c.name} itself in the image (◇: where the calibration puts it; Esc cancels)`;
  canvas.classList.add('tool');
  renderCands(); draw(); drawMap();
}
$('suggest').addEventListener('click', () => busy(() => fetchCands(!!P.cands)));
$('candShow').addEventListener('change', () => { draw(); drawMap(); });

// ------------------------------------------------------------ calibrate --
async function waitJobs(ids) {
  const seen = {};
  for (;;) {
    const r = await api('jobs?ids=' + ids.join(','));
    $('jobs').innerHTML = '';
    for (const j of r.jobs) {
      const cam = P.cams.find((c) => c.id === j.camera);
      j.log.slice(seen[j.id] || 0).forEach((l) => { if (/WARNING|Correction|pose /.test(l)) log(`${cam ? cam.name : j.camera}: ${l}`, l.startsWith('WARNING')); });
      seen[j.id] = j.log.length;
      $('jobs').append(el('div', { className: 'job ' + j.state },
        el('div', { className: 'row-head' }, el('b', { textContent: cam ? cam.name : j.camera }),
          el('span', { className: 'small muted', textContent: j.state === 'error' ? 'failed' : j.state })),
        el('pre', { className: 'mono small joblog', textContent: j.log.slice(-6).join('\n') + (j.error ? '\n' + j.error : '') })));
    }
    if (r.jobs.every((j) => j.state === 'done' || j.state === 'error')) return r.jobs;
    await new Promise((ok) => setTimeout(ok, 1000));
  }
}

async function calibrate(ids) {
  await settle();
  P.cams = (await api('workspace')).cameras;
  const pend = P.cams.filter((c) => ids.includes(c.id) && c.has_proposal).map((c) => c.name);
  if (pend.length && !confirm(`This replaces the proposed calibration of ${pend.join(', ')}, and any hand adjustments made to it. Calibrate anyway?`)) return;
  const r = await api('calibrate', { cameras: ids });
  const jobs = await waitJobs(r.jobs);
  const failed = jobs.filter((j) => j.state === 'error');
  failed.forEach((j) => log(`${j.camera}: ${j.error}`, true));
  log(`Calibrated ${jobs.length - failed.length} camera${jobs.length - failed.length === 1 ? '' : 's'}` +
    (failed.length ? `; ${failed.length} failed` : '') + '. Check the green lines on the frames, adjust them if needed, then accept each camera\'s calibration.');
  const keep = P.frame && P.frame.id;
  await openCamera(P.K.id, keep);
}
$('calOne').addEventListener('click', () => busy(() => calibrate([P.K.id])));
$('calAll').addEventListener('click', () => busy(() => calibrate(P.cams.filter((c) => c.ready).map((c) => c.id))));

const RAW = [0, 0, 0, 60, 0, 0];
function renderResult() {
  const box = $('result'), cal = P.K.calibration || {}, prop = P.K.proposal;
  box.innerHTML = '';
  box.append(el('p', { className: 'small', textContent: 'In use: ' + (cal.c ? cal.source : 'none (raw bar values)') }));
  const shown = prop && prop.report && prop.report.method === 'edges' ? prop : !prop && cal.report && cal.report.method === 'edges' ? cal : null;
  if (prop) box.append(el('p', { className: 'small', textContent: `Proposed (not in use yet): ${prop.source}` +
    (prop.adjusted ? '; adjusted by hand. The frame offsets below are for the automatic result: they are measured again when you accept.' : '') }));
  if (!shown) return;
  const rep = shown.report;
  const names = ['d_azimuth', 'd_tilt', 'roll', 'HFOV', 'lean E', 'lean N'];
  const cols = prop ? [['in use', cal.c || RAW]].concat(prop.adjusted ? [['automatic', prop.auto.c]] : [], [['proposed', prop.c]])
    : [['before', rep.c_previous || rep.c_start], ['in use', cal.c]];
  const tb0 = el('table', { className: 'pose mono' });
  tb0.innerHTML = '<thead><tr><th></th>' + cols.map(([n]) => `<th>${n}</th>`).join('') + '</tr></thead>';
  names.forEach((n, i) => { const tr = el('tr'); tr.append(el('th', { textContent: n }), ...cols.map(([, c]) => el('td', { textContent: deg((c || RAW)[i], 3) }))); tb0.append(tr); });
  box.append(tb0);
  const tb = el('table', { className: 'pose mono' });
  tb.innerHTML = '<thead><tr><th>Frame</th><th colspan="2">offset before</th><th colspan="2">after (right, down °)</th></tr></thead>';
  for (const f of rep.frames) {
    const tr = el('tr');
    tr.append(el('th', { textContent: f.label }));
    if (!f.used) tr.append(el('td', { colSpan: 4, className: 'muted', textContent: f.note }));
    else [...(f.shift_before_deg || [NaN, NaN]), ...(f.shift_after_deg || [NaN, NaN])].forEach((x) => tr.append(el('td', { textContent: deg(x) })));
    tb.append(tr);
  }
  box.append(el('p', { className: 'small muted', textContent: 'Where the real silhouette is relative to the modelled skyline, before and after; 0 = on it.' }), tb);
  for (const [key, o] of Object.entries(shown.pose_offsets || {}))
    box.append(el('p', { className: 'small mono', textContent: `position ${key}: pan ${deg(o.pan)}°, tilt ${deg(o.tilt)}°` +
      (o.roll || (o.scale && o.scale !== 1) ? `, roll ${deg(o.roll || 0)}°, stretch ×${(o.scale || 1).toFixed(4)}` : '') + ` · ${o.how}` }));
  if (rep.overlay) {
    const a = el('a', { href: '/ws/' + rep.overlay + '?t=' + encodeURIComponent(rep.when || ''), target: '_blank' });
    a.append(el('img', { src: a.href, className: 'overlay', alt: 'Calibration overlay', title: 'Red: before; green: after. Click to open.' }));
    box.append(a);
  }
}

// ----------------------------------------------------- adjust & accept --
const poseKey = (bar) => `${(bar[0] + 0).toFixed(2)},${(bar[1] + 0).toFixed(2)},${(bar[2] + 0).toFixed(1)}`;
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

function renderAdjust() {
  const prop = P.K && P.K.proposal, cal = (P.K && P.K.calibration) || {};
  $('adjBody').hidden = !prop; $('startWrap').hidden = !!prop;
  const st = $('propStatus');
  st.innerHTML = '';
  if (!P.K) return;
  if (!prop) {
    st.textContent = `In use: ${cal.c ? cal.source : 'none (raw bar values)'}. Calibrate (step 2) to propose a new calibration, or adjust the one in use by hand.`;
    return;
  }
  st.append(el('b', { className: 'pending', textContent: 'Proposed, not in use yet: ' }),
    `${prop.source}${prop.adjusted ? ' + your adjustment' : ''}. Check the green line on the frames; drag it into place if needed, then Accept.`);
  const still = P.frame && P.frame.kind === 'still';
  if (!still && P.scope === 'position') P.scope = 'camera';
  document.querySelectorAll('#scopeSeg button').forEach((b) => {
    b.classList.toggle('on', b.dataset.scope === P.scope);
    if (b.dataset.scope === 'position') b.disabled = !still || P.busy;
  });
  const key = P.frame && P.frame.bar ? poseKey(P.frame.bar) : '';
  $('scopeHint').textContent = P.scope === 'camera'
    ? 'Changes the camera\'s correction (azimuth, tilt, roll, field of view): the line moves on every frame and every still. Check the other frames afterwards.'
      + (still ? '' : ' Panorama frames are drawn with their own small pan error from the fit.')
    : `Changes only the stills taken at X, Y, Z = ${key}: their own pan, tilt, roll and stretch.`;
  // values: automatic and now, the "now" column editable
  const tb = $('adjVals'), auto = prop.auto || prop, po = prop.pose_offsets || {}, apo = auto.pose_offsets || {};
  tb.innerHTML = '<thead><tr><th></th><th>automatic</th><th>now</th></tr></thead>';
  const row = (label, a, now, id, d = 3) => {
    const tr = el('tr'), inp = el('input', { id, value: now.toFixed(d), inputMode: 'decimal' });
    inp.addEventListener('change', () => setValues(id.startsWith('cv') ? 'camera' : 'position'));
    const td = el('td'); td.append(inp);
    tr.append(el('th', { textContent: label }), el('td', { textContent: a.toFixed(d) }), td);
    tb.append(tr);
  };
  tb.append(el('tr', {}, el('th', { textContent: 'Whole camera', className: 'sub' })));
  ['d_azimuth (°)', 'd_tilt (°)', 'roll (°)', 'HFOV at Z 1 (°)'].forEach((n, i) => row(n, auto.c[i], prop.c[i], 'cv' + i));
  if (still) {
    const o = po[key] || {}, a = apo[key] || {};
    tb.append(el('tr', {}, el('th', { textContent: `This position ${key}`, className: 'sub' })));
    row('pan (°)', a.pan || 0, o.pan || 0, 'pv0'); row('tilt (°)', a.tilt || 0, o.tilt || 0, 'pv1');
    row('roll (°)', a.roll || 0, o.roll || 0, 'pv2'); row('stretch (×)', a.scale || 1, o.scale || 1, 'pv3', 4);
  }
}

document.querySelectorAll('#scopeSeg button').forEach((b) => b.addEventListener('click', () => {
  P.scope = b.dataset.scope; renderAdjust(); draw();
}));

// A typed value: the whole row of four is sent, the lean kept.
function setValues(which) {
  busy(async () => {
    await settle();
    const prop = P.K.proposal;
    let body;
    if (which === 'camera') {
      const v = readNums([0, 1, 2, 3].map((i) => ['cv' + i, ['d_azimuth', 'd_tilt', 'roll', 'HFOV'][i]]));
      body = { c: [...v, prop.c[4] || 0, prop.c[5] || 0] };
    } else {
      const v = readNums([0, 1, 2, 3].map((i) => ['pv' + i, ['pan', 'tilt', 'roll', 'stretch'][i]]));
      body = { offset: { key: poseKey(P.frame.bar), pan: v[0], tilt: v[1], roll: v[2], scale: v[3] } };
    }
    const r = await api('setup/proposal', { camera: P.K.id, action: 'set', ...body });
    P.K.proposal = r.proposal;
    await predict(); renderAdjust(); renderResult();
  });
}

function nudge(kind, big) {
  const g = gizmo(), [w, h] = frameSize(), piv = g ? g.pivot : [(w + 1) / 2, (h + 1) / 2];
  const px = big ? 10 : 1, rot = big ? 0.5 : 0.05, sc = big ? 0.01 : 0.001;
  const T = { left: similarity(piv, 0, 1, -px, 0), right: similarity(piv, 0, 1, px, 0), up: similarity(piv, 0, 1, 0, -px),
    down: similarity(piv, 0, 1, 0, px), cw: similarity(piv, rot), ccw: similarity(piv, -rot),
    grow: similarity(piv, 0, 1 + sc), shrink: similarity(piv, 0, 1 / (1 + sc)) }[kind];
  P.pend = tmul(T, P.pend || ID());
  draw();
  clearTimeout(P.flushTimer);
  P.flushTimer = setTimeout(flushAdjust, 300);
}
document.querySelectorAll('[data-nudge]').forEach((b) => b.addEventListener('click', (e) => {
  if (adjustable()) nudge(b.dataset.nudge, e.shiftKey);
}));

// Turn the adjustments drawn so far into camera angles (server), then redraw from the new angles.
async function flushAdjust() {
  clearTimeout(P.flushTimer);
  if (P.infl || !P.pend) return;
  if (isId(P.pend)) { P.pend = null; return; }
  P.infl = P.pend; P.pend = null;
  const id = P.K.id, fid = P.frame.id, A = P.infl;
  try {
    const r = await api('setup/adjust', { camera: id, frame: fid, scope: P.scope, A: [A.slice(0, 3), A.slice(3)] });
    if (P.K.id === id) {
      P.K.proposal = r.proposal;
      const pr = await fetchPredict();
      if (P.K.id === id && P.frame.id === fid) { applyPredict(pr); if (P.cands) await fetchCands(false); }
    }
  } catch (e) { log(e.message, true); }
  P.infl = null;
  draw(); renderAdjust(); renderLandmarks(); drawMap();
  if (P.pend) flushAdjust();
}
async function settle() {                          // wait until every adjustment has been sent
  while (P.infl || P.pend || P.adj) {
    if (!P.infl && P.pend && !P.adj) flushAdjust();
    await sleep(30);
  }
}

$('accept').addEventListener('click', () => busy(async () => {
  await settle();
  const name = P.K.name, r = await api('setup/accept', { camera: P.K.id });
  const jobs = await waitJobs(r.jobs);
  if (jobs[0].state === 'error') throw new Error(`${name}: ${jobs[0].error}`);
  log(`${name}: calibration accepted (${jobs[0].result.calibration.source}).`);
  await openCamera(P.K.id, P.frame && P.frame.id);
}));
$('resetAdj').addEventListener('click', () => busy(async () => {
  await settle();
  P.K.proposal = (await api('setup/proposal', { camera: P.K.id, action: 'reset' })).proposal;
  log(`${P.K.name}: back to the automatic result.`);
  await predict(); renderAdjust(); renderResult();
}));
$('discard').addEventListener('click', () => busy(async () => {
  await settle();
  if (P.K.proposal.adjusted && !confirm('Discard the proposed calibration and your adjustments? The calibration in use stays.')) return;
  await api('setup/proposal', { camera: P.K.id, action: 'discard' });
  P.K.proposal = null;
  log(`${P.K.name}: proposal discarded; the calibration in use stays.`);
  await predict(); renderAdjust(); renderResult();
}));
$('startAdj').addEventListener('click', () => busy(async () => {
  P.K.proposal = (await api('setup/proposal', { camera: P.K.id, action: 'start' })).proposal;
  log(`${P.K.name}: the calibration in use is proposed for adjusting; accept it when done.`);
  await predict(); renderAdjust(); renderResult();
}));
$('openApp').addEventListener('click', (e) => {
  e.preventDefault();
  busy(async () => {
    await settle();
    const pend = (await api('workspace')).cameras.filter((c) => c.has_proposal).map((c) => c.name);
    if (pend.length && !confirm(`${pend.join(', ')}: the proposed calibration is not accepted yet, so the workspace uses the ` +
      'calibration in use. The proposal is kept for later. Open the workspace anyway?')) return;
    location.href = $('openApp').href;
  });
});

// ------------------------------------------------------------------ map --
let map = null, layer = null, probe = null;
function initMap() {
  if (!window.L) { $('map').textContent = 'Map unavailable (needs internet).'; return; }
  map = L.map('map', { zoomSnap: 0.25 }).setView([39.5, -119.9], 10);
  const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { maxZoom: 17,
    attribution: 'Map data © OpenStreetMap contributors, SRTM | Style © OpenTopoMap (CC-BY-SA)' });
  const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 18, attribution: 'Imagery © Esri' });
  topo.addTo(map);
  L.control.layers({ Topographic: topo, Satellite: sat }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);
  layer = L.layerGroup().addTo(map);
  map.on('click', (e) => busy(async () => {
    probe = [e.latlng.lat, e.latlng.lng];
    if (P.mode === 'entry') {
      const z = (await api('elevation', { camera: P.K.id, lat: [e.latlng.lat], lon: [e.latlng.lng] })).z[0];
      if (z === null) throw new Error('Outside the DEM.');
      $('lmLat').value = e.latlng.lat.toFixed(6); $('lmLon').value = e.latlng.lng.toFixed(6); $('lmZ').value = z.toFixed(1);
    }
    drawMap();
  }));
}
let fitted = '';
function drawMap() {
  if (!map || !P.K) return;
  layer.clearLayers();
  const [lat, lon] = P.K.site, cos = Math.cos(lat * Math.PI / 180);
  L.circleMarker([lat, lon], { radius: 6, color: '#00b8d4', fillOpacity: 1 }).bindTooltip(P.K.name).addTo(layer);
  if (P.fov) {
    const pts = [[lat, lon]];
    let a0 = P.fov[0], a1 = P.fov[2];
    if (a1 < a0) a1 += 360;
    for (let k = 0; k <= 16; k++) {
      const az = (a0 + (a1 - a0) * k / 16) * Math.PI / 180;
      pts.push([lat + 0.4 * Math.cos(az), lon + 0.4 * Math.sin(az) / cos]);
    }
    pts.push([lat, lon]);
    L.polyline(pts, { color: '#00b8d4', weight: 1.2, dashArray: '4 4' }).addTo(layer);
  }
  const lm = P.K.landmarks;
  lm.gcp.forEach((g, i) => L.circleMarker([g[2], g[3]], { radius: 5, color: lm.stills[i] === (P.frame && P.frame.id) ? '#16a34a' : '#6b7280',
    fillOpacity: .8 }).bindTooltip(lm.names[i]).addTo(layer));
  if (P.cands && $('candShow').checked)
    for (const c of P.cands.filter((x) => x.in_frame))
      L.circleMarker([c.lat, c.lon], { radius: P.cand && P.cand.id === c.id ? 6 : 3, color: CAND_COLOR[c.source] || '#ffb300', weight: 1.5, fillOpacity: .5 })
        .bindTooltip(`${c.name} (${c.source}, ±${c.acc_m.toFixed(0)} m)`).on('click', () => placeCandidate(c)).addTo(layer);
  if (probe) L.circleMarker(probe, { radius: 5, color: '#e11d9b', fill: false }).addTo(layer);
  if (fitted !== P.K.id) { map.setView([lat + 0.15, lon], 10); fitted = P.K.id; }
}

// ----------------------------------------------------------------- init --
(async function init() {
  try {
    const r = await api('workspace');
    P.ws = r.workspace; P.cams = r.cameras;
    $('title').textContent = `${P.ws.name} · calibration step`; document.title = `${P.ws.name} · Calibration`;
  } catch (e) { log('Could not open the workspace: ' + e.message + ' (open it from the portal).', true); return; }
  resizeCanvas(); initMap();
  new ResizeObserver(() => { resizeCanvas(); fitView(); if (map) map.invalidateSize(); }).observe(canvas);
  const ready = P.cams.filter((c) => c.ready);
  if (!ready.length) { log('No camera is loaded yet: Load cameras in the portal first.', true); renderTabs(); return; }
  const want = new URLSearchParams(location.search).get('camera');
  await busy(() => openCamera((ready.find((c) => c.id === want) || ready[0]).id));
  log('Add landmarks if you like (optional), then Calibrate (about a minute per camera), check and adjust the result, and accept it.');
})();
