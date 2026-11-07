// Portal: open or create a fire workspace, add its cameras, and load & calibrate them (camreg/server.py).
'use strict';

const $ = (id) => document.getElementById(id);
const FT = 0.3048;
const P = { ws: null, cams: [], map: null, layer: null, pick: null, browsePath: '.' };

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

async function guard(fn) {
  try { await fn(); } catch (e) { log(e.message, true); }
}

const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  kids.forEach((k) => e.append(k));
  return e;
};

// -------------------------------------------------------------- workspaces --
async function showPicker() {
  const r = await api('workspaces');
  $('wsDir').textContent = 'Workspaces in ' + r.ws_root;
  $('pickView').hidden = false; $('wsView').hidden = true;
  $('wsList').innerHTML = '';
  $('wsNone').hidden = r.workspaces.length > 0;
  for (const w of r.workspaces) {
    const open = el('button', { textContent: 'Open', className: 'primary' });
    open.onclick = () => guard(async () => { await api('workspace/open', { slug: w.slug }); await showWorkspace(true); });
    $('wsList').append(el('li', {},
      el('span', { className: 'go', textContent: w.name, onclick: () => open.click() }),
      el('span', { className: 'muted small', textContent: `${w.cameras.join(', ') || 'no cameras'} · ${w.edges} edges · saved ${w.modified}` }),
      open));
  }
}

$('newWs').addEventListener('submit', (e) => {
  e.preventDefault();
  guard(async () => {
    await api('workspace/new', { name: $('wsName').value, utc_offset: $('wsOffset').value });
    log(`Created workspace ${$('wsName').value}`);
    await showWorkspace(false);
  });
});
$('switchWs').addEventListener('click', () => guard(showPicker));

async function showWorkspace(autoload) {
  const r = await api('workspace');
  P.ws = r.workspace; P.cams = r.cameras;
  $('pickView').hidden = true; $('wsView').hidden = false;
  $('wsTitle').textContent = P.ws.name;
  $('wsDir').textContent = r.dir;
  $('wsInfo').textContent = `Camera clock: ${P.ws.utc_offset ? 'UTC ' + P.ws.utc_offset : 'no UTC offset set'} · DEM tiles from ${P.ws.dem_dir} · last saved ${P.ws.modified}`;
  renderCams();
  initMap();
  await showDem();
  if (autoload && P.cams.length) loadAll();
}

function camStatus(c) {
  const s = c.stills;
  const stills = !s.exists ? 'stills folder not found' :
    s.needs_index ? `${s.n_images} images, bars not read yet` :
    `${s.n_indexed} stills with bar readings` + (s.n_images > s.n_indexed ? ` (${s.n_images - s.n_indexed} other images)` : '');
  const pano = c.panoramas.length ? `${c.panoramas.length} panorama${c.panoramas.length > 1 ? 's' : ''}` : 'no panorama';
  return `${stills} · ${pano} · calibration: ${c.calibration_source}` + (c.ready ? ' · loaded' : '');
}
const DEM_NAME = { srtm: 'SRTM 30 m', '3dep10': 'USGS 3DEP 10 m' };
const demName = (label) => DEM_NAME[(label || 'srtm').split(':')[0]];

function renderCams() {
  $('camList').innerHTML = '';
  $('camNone').hidden = P.cams.length > 0;
  for (const c of P.cams) {
    const addPano = el('input', { type: 'file', accept: 'image/*', multiple: true, className: 'small' });
    addPano.onchange = () => guard(async () => { await uploadPanoramas(c.id, addPano.files); await showWorkspace(false); });
    const del = el('button', { textContent: '×', title: 'Remove this camera from the workspace' });
    del.onclick = () => guard(async () => {
      if (!confirm(`Remove ${c.name} from this workspace? Its calibration and edges are removed too; its stills folder is not touched.`)) return;
      await api('camera/remove', { id: c.id }); await showWorkspace(false);
    });
    const elev = c.index && c.index.elev_ft ? ` (bar: ${c.index.elev_ft} ft)` : '';
    const check = el('button', { textContent: 'Check site', className: 'small',
      title: 'Surveyed points (NOAA NGS) near this site, the DEM ground there and the terrain right around the camera' });
    const panel = el('div', { className: 'site-panel' });
    check.onclick = () => guard(() => siteCheck(c, panel, check));
    const mism = c.dem_mismatch ? el('div', { className: 'small warn', textContent:
      `Calibrated on ${demName(c.calibration_dem)}; the workspace now uses ${demName(c.dem)}. Calibrate it again in the calibration step.` }) : '';
    $('camList').append(el('li', { className: 'cam' },
      el('div', { className: 'cam-body' },
        el('b', { textContent: c.name }),
        el('div', { className: 'muted small mono', textContent:
          `${c.lat.toFixed(6)}, ${c.lon.toFixed(6)} · ${c.elev_m.toFixed(1)} m = ${(c.elev_m / FT).toFixed(0)} ft${elev} · ${c.stills_dir}` }),
        el('div', { className: 'small', textContent: camStatus(c) }), mism,
        el('div', { className: 'btns' }, el('label', { className: 'small muted', textContent: 'Add panorama ' }, addPano), check),
        panel),
      del));
  }
  $('loadAll').disabled = !P.cams.length;
  $('openApp').hidden = $('openSetup').hidden = !P.cams.length || !P.cams.every((c) => c.ready);
  drawCams();
}

// ------------------------------------------------------------- site check --
// NGS survey marks near the entered site, the DEM ground there and the terrain right around the camera: with the
// 10 m DEM a site a few metres off a summit, or a height below the ground, hides the view in the model.
const m1 = (x) => (x === null || x === undefined ? '–' : x.toFixed(1));
async function siteCheck(c, panel, btn) {
  btn.disabled = true;
  try {
    const r = await api('site/check', { camera: c.id });
    panel.innerHTML = '';
    const g = r.ground_m, up = r.camera_above_ground_m, n = r.near;
    panel.append(el('p', { className: 'small', textContent: `DEM (${r.dem}) ground at the site: ${m1(g)} m; the camera height ${c.elev_m.toFixed(1)} m is ` +
      (up === null ? '–' : `${Math.abs(up).toFixed(1)} m ${up >= 0 ? 'above' : 'BELOW'} it.`) }));
    if (n) {
      const s = n.summit, blocked = n.max_el_deg > r.near_block_deg && n.blocked_share >= 0.1;
      panel.append(el('p', { className: 'small' + (blocked ? ' warn' : ''), textContent:
        `Highest DEM point within 150 m: ${s.distance_m.toFixed(0)} m away, ${s.elev.toFixed(1)} m (camera ${s.camera_above_m >= 0 ? '+' : ''}${s.camera_above_m.toFixed(1)} m above it). ` +
        `Nearby ground (within 300 m) rises to ${n.max_el_deg >= 0 ? '+' : ''}${n.max_el_deg.toFixed(1)}° from the camera (azimuth ${n.az_deg.toFixed(0)}°, ${n.dist_m.toFixed(0)} m away)` +
        (n.max_el_deg > r.near_block_deg ? `; it hides the view in ${(100 * n.blocked_share).toFixed(0)}% of directions` : '') +
        (blocked ? '. A camera on a summit or mast looks down on its surroundings: the site or height is probably off.' : '.') }));
      panel.append(siteRow(c, 'DEM summit', s.lat, s.lon, `${s.distance_m.toFixed(0)} m away, ground ${s.elev.toFixed(1)} m`, s.elev));
    }
    panel.append(el('p', { className: 'small muted', textContent: r.marks.length ? 'NOAA NGS survey marks within 1.5 km (a lookout or tower mark is often the camera\'s own site):' :
      'No NGS survey marks within 1.5 km.' }));
    for (const m of r.marks.slice(0, 8))
      panel.append(siteRow(c, `${m.pid} ${m.name}`, m.lat, m.lon, `${m.type || 'mark'} · ${m.distance_m.toFixed(0)} m away · position ${m.position} · ` +
        `DEM ground ${m1(m.ground_m)} m` + (m.height ? ` · mark height ${m.height.toFixed(1)} m (${m.height_source})` : ''), m.ground_m));
    const h = el('input', { value: c.elev_m.toFixed(1), inputMode: 'decimal', size: 7 });
    const lat = el('input', { value: c.lat.toFixed(7), inputMode: 'decimal', size: 11 });
    const lon = el('input', { value: c.lon.toFixed(7), inputMode: 'decimal', size: 12 });
    panel._fill = (la, lo, ground) => { lat.value = la.toFixed(7); lon.value = lo.toFixed(7);
      if (ground !== null && ground !== undefined && parseNum(h.value) < ground + 1) h.value = (ground + 5).toFixed(1); };
    const apply = el('button', { textContent: 'Apply', className: 'primary small' });
    apply.onclick = () => guard(async () => {
      const v = [parseNum(lat.value), parseNum(lon.value), parseNum(h.value)];
      if (v.some((x) => !isFinite(x))) throw new Error('Enter the latitude, longitude and height as numbers.');
      const r2 = await api('camera/update', { id: c.id, lat: v[0], lon: v[1], elev_m: v[2] });
      log(`${c.name}: site set to ${v[0].toFixed(6)}, ${v[1].toFixed(6)}, ${v[2].toFixed(1)} m` + (r2.unloaded ? ' (load the cameras again)' : ''));
      await showWorkspace(false);
    });
    panel.append(el('div', { className: 'adj-row small' }, 'Site', lat, lon, 'height (m)', h, apply));
    panel.append(el('p', { className: 'small muted', textContent: 'Use puts a point\'s position in the boxes (and, if the height is below its ground, the ground + 5 m); check the height (the info bar\'s elevation, or the mast above the ground) and Apply.' }));
  } finally { btn.disabled = false; }
}
function siteRow(c, name, lat, lon, info, ground) {
  const use = el('button', { textContent: 'Use', className: 'small' });
  const row = el('div', { className: 'site-row small' }, el('span', { className: 'mono', textContent: `${lat.toFixed(7)}, ${lon.toFixed(7)}` }),
    el('b', { textContent: ' ' + name + ' ' }), el('span', { className: 'muted', textContent: info }), use);
  use.onclick = () => { const p = row.parentElement; if (p && p._fill) p._fill(lat, lon, ground); };
  return row;
}

// -------------------------------------------------------------------- DEM --
async function showDem() {
  const r = await api('dem/status', {});
  P.dem = r;
  document.querySelectorAll('#demSeg button').forEach((b) => b.classList.toggle('on', b.dataset.dem === r.source));
  $('demRadius').value = r.radius_km;
  const g = r.grid, cov = r.cameras.filter((c) => c.covered).length;
  $('demStatus').textContent = `In use: ${r.label}. ` + (r.box ? `3DEP area: ${r.area_km2.toFixed(0)} km² (${r.size_mb.toFixed(0)} MB). ` : '') +
    (g ? `Downloaded ${g.fetched} (${g.file}); it covers ${cov} of ${r.cameras.length} camera${r.cameras.length === 1 ? '' : 's'}.` :
      cov ? `A downloaded grid covers ${cov} of ${r.cameras.length} cameras.` : 'Not downloaded yet.') +
    (r.source === '3dep10' && cov < r.cameras.length ? ' Cameras it does not cover use SRTM: download again.' : '');
  if (!r.box) $('demStatus').textContent = `In use: ${r.label}. Add a camera first: the 3DEP area is drawn around the cameras.`;
  $('demFetch').textContent = g ? 'Download 3DEP again' : 'Download 3DEP';
  $('demFetch').disabled = !r.box;
}
document.querySelectorAll('#demSeg button').forEach((b) => b.addEventListener('click', () => guard(async () => {
  if (b.dataset.dem === '3dep10' && !(P.dem && (P.dem.grid || P.dem.cameras.some((c) => c.covered)))) {
    log('Download the 3DEP grid first (Download 3DEP).', true); return;
  }
  const r = await api('dem/set', { source: b.dataset.dem, radius_km: parseNum($('demRadius').value) });
  log(`DEM: ${r.label}` + (r.unloaded ? '. The cameras were unloaded: Load cameras again.' : ''));
  await showWorkspace(false);
})));
$('demFetch').addEventListener('click', () => guard(async () => {
  const radius = parseNum($('demRadius').value);
  if (!isFinite(radius)) throw new Error('Enter the 3DEP radius in km.');
  const st = await api('dem/status', { radius_km: radius });
  if (!st.box) throw new Error('Add a camera first: the 3DEP area is drawn around the cameras.');
  if (st.area_km2 > st.max_area_km2) throw new Error(`The 3DEP area would be ${st.area_km2.toFixed(0)} km² (limit ${st.max_area_km2.toFixed(0)}): use a smaller radius.`);
  if (!confirm(`Download about ${st.size_mb.toFixed(0)} MB of USGS 3DEP elevation (${st.area_km2.toFixed(0)} km² around the cameras)? It takes a few minutes; the workspace then uses it.`)) return;
  $('demFetch').disabled = true;
  try {
    const r = await api('dem/fetch', { radius_km: radius, refresh: !!st.grid });
    const [j] = await waitJobs([r.job], (jobs) => renderJobs(jobs.map((x) => ({ ...x, camera: x.camera || 'DEM' }))));
    if (j.state === 'error') throw new Error('3DEP download failed: ' + j.error);
    log('3DEP downloaded; the workspace uses it now. Load the cameras again.');
    await showWorkspace(false);
  } finally { $('demFetch').disabled = false; }
}));

// ------------------------------------------------------------- add camera --
async function uploadPanoramas(camera, files) {
  for (const f of files) {
    const data = await new Promise((ok, fail) => {
      const rd = new FileReader();
      rd.onload = () => ok(rd.result); rd.onerror = () => fail(new Error('Could not read ' + f.name));
      rd.readAsDataURL(f);
    });
    log(`Uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)...`);
    await api('panorama/upload', { camera, name: f.name, data });
  }
}

$('addCam').addEventListener('submit', (e) => {
  e.preventDefault();
  guard(async () => {
    const [lat, lon, h] = readNums([['camLat', 'Latitude'], ['camLon', 'Longitude'], ['camElev', 'Height']]);
    const r = await api('camera/add', { name: $('camName').value, lat, lon,
      elev_m: $('camElevUnit').value === 'ft' ? h * FT : h, stills_dir: $('camStills').value });
    await uploadPanoramas(r.camera.id, $('camPano').files);
    log(`Added ${r.camera.name}`);
    $('addCam').reset(); $('browser').hidden = true; P.pick = null;
    await showWorkspace(false);
  });
});

splitPairInto('camLat', 'camLon');

// Folder picker: subfolders of the project folder with their image counts.
async function browse(path) {
  const r = await api('folders', { path });
  P.browsePath = r.path;
  $('brPath').textContent = `${r.path} (${r.images} images)`;
  $('brList').innerHTML = '';
  if (r.parent !== null) $('brList').append(el('li', {}, el('span', { className: 'go', textContent: '↑ ..', onclick: () => guard(() => browse(r.parent)) })));
  for (const d of r.dirs)
    $('brList').append(el('li', {}, el('span', { className: 'go', textContent: d.name, onclick: () => guard(() => browse(d.path)) }),
      el('span', { className: 'muted small', textContent: d.images === null ? '' : `${d.images} images` })));
  $('browser').hidden = false;
}
$('browse').addEventListener('click', () => guard(() => browse($('camStills').value || '.')));
$('brUse').addEventListener('click', () => { $('camStills').value = P.browsePath; $('browser').hidden = true; });

// ------------------------------------------------------------------- map --
function initMap() {
  if (P.map || !window.L) { if (!window.L) $('pmap').textContent = 'Map unavailable (needs internet).'; else P.map.invalidateSize(); drawCams(); return; }
  P.map = L.map('pmap').setView(P.cams.length ? [P.cams[0].lat, P.cams[0].lon] : [39.5, -119.8], 9);
  L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', { maxZoom: 17,
    attribution: 'Map data © OpenStreetMap contributors, SRTM | Style © OpenTopoMap (CC-BY-SA)' }).addTo(P.map);
  P.layer = L.layerGroup().addTo(P.map);
  P.map.on('click', (e) => {
    $('camLat').value = e.latlng.lat.toFixed(6); $('camLon').value = e.latlng.lng.toFixed(6);
    P.pick = [e.latlng.lat, e.latlng.lng]; drawCams();
  });
  drawCams();
}

function drawCams() {
  if (!P.layer) return;
  P.layer.clearLayers();
  const pts = [];
  for (const c of P.cams) {
    L.circleMarker([c.lat, c.lon], { radius: 6, color: '#00b8d4', fillOpacity: 1 }).bindTooltip(c.name, { permanent: true, direction: 'right' }).addTo(P.layer);
    pts.push([c.lat, c.lon]);
  }
  if (P.pick) L.circleMarker(P.pick, { radius: 6, color: '#e11d9b', fillOpacity: .3 }).bindTooltip('New camera').addTo(P.layer);
  if (pts.length > 1) P.map.fitBounds(pts, { padding: [40, 40], maxZoom: 11 });
  else if (pts.length) P.map.setView(pts[0], 10);
}

// ------------------------------------------------------- load & calibrate --
async function waitJobs(ids, render) {
  for (;;) {
    const r = await api('jobs?ids=' + ids.join(','));
    render(r.jobs);
    if (r.jobs.every((j) => j.state === 'done' || j.state === 'error')) return r.jobs;
    await new Promise((ok) => setTimeout(ok, 700));
  }
}

function renderJobs(jobs) {
  $('jobs').innerHTML = '';
  for (const j of jobs) {
    const cam = P.cams.find((c) => c.id === j.camera);
    const pct = j.total ? Math.round(100 * j.done / j.total) : (j.state === 'done' ? 100 : 0);
    $('jobs').append(el('div', { className: 'job ' + j.state },
      el('div', { className: 'row-head' }, el('b', { textContent: cam ? cam.name : j.camera }),
        el('span', { className: 'small muted', textContent: j.state === 'error' ? 'failed' : j.state + (j.total ? ` · ${j.done}/${j.total} stills read` : '') })),
      el('div', { className: 'bar' }, el('span', { style: `width:${pct}%` })),
      el('pre', { className: 'mono small joblog', textContent: j.log.slice(-8).join('\n') + (j.error ? '\n' + j.error : '') })));
  }
}

async function loadAll() {
  $('loadAll').disabled = true;
  try {
    const r = await api('ingest', {});
    const jobs = await waitJobs(r.jobs, renderJobs);
    const failed = jobs.filter((j) => j.state === 'error'), warnings = jobs.flatMap((j) => j.warnings || []);
    await showWorkspace(false);
    if (failed.length) { log(`${failed.length} camera(s) could not be loaded; see above.`, true); return; }
    if (warnings.length) {          // stay here until they have been seen; the links go on
      warnings.forEach((w) => log(w, true));
      log('All cameras loaded, with the warnings above. Go on to the calibration step when ready.');
      return;
    }
    // next: the calibration step, unless every camera is calibrated already
    const calibrated = P.cams.every((c) => c.calibration_source && !c.calibration_source.startsWith('none'));
    log(calibrated ? 'All cameras loaded. Opening the workspace...' : 'All cameras loaded. Opening the calibration step...');
    location.href = calibrated ? '/app' : '/setup';
  } catch (e) { log(e.message, true); }
  finally { $('loadAll').disabled = !P.cams.length; }
}
$('loadAll').addEventListener('click', loadAll);

// -------------------------------------------------------------------- init --
(async function init() {
  try {
    const r = await api('workspaces');
    if (r.open) await showWorkspace(false); else await showPicker();
  } catch (e) { log('Could not reach the server: ' + e.message, true); }
})();
