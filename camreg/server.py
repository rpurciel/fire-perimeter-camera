"""Local web server: fire workspaces, camera calibration and fire-edge mapping on camera stills.

Pages: / is the portal (open or create a workspace, add its cameras, load and calibrate them) and /app holds
the calibration and analysis spaces of the open workspace (camreg/web/). The browser keeps the drawing
state and does the cheap projections itself; this server does the DEM work (optimization, pixel-to-terrain
mapping, elevation lookups), the bar reading and the panorama calibration (as background jobs), the
multi-camera combination (combine.py) and the file I/O.

The camera pose on each still is its info-bar pan/tilt/zoom plus one fixed mount correction per camera
(core.pose_from_bar; 6 values including the pan-axis lean); there is no free pose.

Listens on 127.0.0.1 only. Writes only inside the open workspace's folder (workspace.py) and, for the bar
readings, the index CSVs in a camera's stills folder (camstills.index).
"""
import base64
import contextlib
import csv
import io
import itertools
import json
import math
import mimetypes
import os
import queue
import shutil
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from camstills import index as stills_index
from camstills.bar import bar_geometry, default_bar_px
from . import combine, core, dem3dep, edgefit, panorama, reference, skyline, tiles
from .workspace import DEFAULT_DEM, DEFAULT_HFOV, DEFAULT_HORIZON_KM, Workspace, list_workspaces, now, safe_file, slug

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
# Limits of the mount correction for Optimize: d_azimuth, d_tilt, roll, HFOV at Z1, lean east, lean north.
# The bar's pan X is the azimuth from north to a few degrees, so the azimuth offset stays small.
CORRECTION_LIM = ((-20, 20), (-20, 20), (-12, 12), (15, 110), (-10, 10), (-10, 10))
NEAR_KM = 40.0                 # terrain tiles this close to a camera are loaded at full resolution, the rest at 3 arc-seconds
HORIZON_KM_LIM = (20.0, 250.0)  # the horizon distance a camera can be given (setup_horizon)
SAME_SITE_M = 500.0            # a calibration file belongs to a camera if its site is this close
COVER_SLACK_KM = 1.0           # a 3DEP grid still serves a camera whose radius it misses by up to this
MAIN_POSE_SHARE = 0.05         # a camera position with this share of the stills counts as a main one
MIN_ABOVE_GROUND_M = 1.0       # a camera lower than this above the DEM ground at its site gets a warning
DEM_LABELS = {'srtm': 'SRTM 30 m', '3dep10': 'USGS 3DEP 10 m + SRTM'}


def _safe(x):
    """JSON-safe copy: numpy -> Python, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _safe(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _arr(rows, ncol):
    a = np.array([[np.nan if x is None else x for x in r] for r in (rows or [])], float)
    return a.reshape(-1, ncol)


def _lm_info(new, old=None):
    """The landmarks' 'info' list ({source, ref_id, acc_m} per landmark, {} = typed or clicked by hand) aligned
    with new. A client that sends landmarks without it (the workspace page) keeps the info of unchanged ones."""
    n = len(new.get('gcp') or [])
    info = new.get('info')
    if isinstance(info, list) and len(info) == n:
        return [dict(x) if isinstance(x, dict) else {} for x in info]
    old = old or {}
    key = lambda g, f, nm: (f, nm, round(float(g[2]), 7), round(float(g[3]), 7))
    old_info = old.get('info') or []
    have = {key(g, f, nm): (old_info[i] if i < len(old_info) else {})
            for i, (g, f, nm) in enumerate(zip(old.get('gcp') or [], old.get('stills') or [], old.get('names') or []))}
    return [dict(have.get(key(g, f, nm)) or {}) for g, f, nm in zip(new.get('gcp') or [], new.get('stills') or [],
                                                                       new.get('names') or [])]


def _site_m(a, b):
    """Horizontal distance in metres between (lat, lon, ...) a and b."""
    dn = math.radians(a[0] - b[0]) * core.R_EARTH
    de = math.radians(a[1] - b[1]) * core.R_EARTH * math.cos(math.radians(a[0]))
    return math.hypot(dn, de)


# ------------------------------------------------------------------ camera --

class CameraSession:
    """One camera of the open workspace: its stills with their bar poses, and its camera model on the DEM."""

    def __init__(self, cam, stills_dir, dem, utc_offset=''):
        self.id = cam['id']
        self.stills_dir = stills_dir
        self.cam = (float(cam['lat']), float(cam['lon']), float(cam['elev_m']))
        with open(os.path.join(stills_dir, 'stills_index.csv')) as f:
            rows = [r for r in csv.DictReader(f) if os.path.isfile(os.path.join(stills_dir, r['file']))]
        if not rows:
            raise ValueError('None of the stills in %s/stills_index.csv are in that folder' % stills_dir)
        self.stills = sorted(({'file': r['file'], 'time': r['timestamp'], 'X': float(r['pan_X_deg']),
                               'Y': float(r['tilt_Y_deg']), 'Z': float(r['zoom_Z']), 'pose': r.get('pose_id', ''),
                               't': combine.bar_time(r['timestamp'], utc_offset)} for r in rows),
                             key=lambda s: s['t'])
        self.bars = {s['file']: (s['X'], s['Y'], s['Z']) for s in self.stills}
        self.times = {s['file']: s['time'] for s in self.stills}
        self.t = {s['file']: s['t'] for s in self.stills}
        w, h = Image.open(os.path.join(stills_dir, self.stills[0]['file'])).size
        self.bar_px = int(cam.get('bar_px') or 0)
        self.nx, self.ny = w, h - self.bar_px
        self.lim = core.full_limits(CORRECTION_LIM)
        self.c0 = core.full_correction([0, 0, 0, cam.get('hfov0') or DEFAULT_HFOV, 0, 0])
        self.model = core.CameraModel(self.cam, dem, self.nx, self.ny)
        # Skyline for every azimuth: the pan comes from each frame's bar and can point anywhere.
        self.sky_rays = self.model.horizon_rays(0, 360, 0.1)

    def pose(self, still, c, po=None):
        """Model pose on a still: its bar X/Y/Z (plus its position's offsets po, see core.pose_from_bar) plus
        mount correction c."""
        if still not in self.bars:
            raise ValueError('Unknown still: %s' % still)
        return core.pose_from_bar(self.bars[still], core.full_correction(c), po)

    def config(self):
        return {'cam': list(self.cam), 'c0': self.c0.tolist(), 'lim': self.lim.tolist(), 'nx': self.nx,
                'ny': self.ny, 'bar_px': self.bar_px, 'stills': self.stills,
                'sky_rays': np.round(self.sky_rays, 7).tolist()}

    def optimize(self, req):
        gcp = _arr(req.get('gcp'), 5)
        stills = req.get('gcp_stills', [])
        sky = _arr(req.get('sky_obs'), 2)
        if len(gcp) < 3 and len(sky) < 10:
            return {'error': 'Add >=3 landmarks or trace a skyline before optimizing.'}
        groups = [(self.bars[s], gcp[[i for i, t in enumerate(stills) if t == s]])
                  for s in dict.fromkeys(stills) if s in self.bars]
        sky_term = (self.bars[req['sky_still']], sky) if len(sky) and req.get('sky_still') in self.bars else None
        c, res = core.optimize_correction(core.full_correction(req['c']), self.model, self.sky_rays,
                                          groups, sky_term, self.lim, verbose=False,
                                          fit_lean=bool(req.get('fit_lean')), offsets=req.get('po'))
        return {'c': c.tolist(), 'score': float(res.fun), 'nit': int(res.nit),
                'nfev': int(res.nfev), 'message': str(res.message)}

    def autosky(self, req):
        """Detect the skyline on a still (skyline.py); compared with the current model as a haze check."""
        still = req.get('still')
        if still not in self.bars:
            raise ValueError('Unknown still: %s' % still)
        img = np.asarray(Image.open(os.path.join(self.stills_dir, still)).convert('RGB'))[:self.ny, :self.nx]
        uv = skyline.skyline_points(img, spacing=int(req.get('spacing', 8)))
        if len(uv) < 10:
            return {'error': 'No clear skyline found on this still (too little open sky above the terrain).'}
        u, v, front = self.model.project(self.sky_rays, self.pose(still, req['c'], req.get('po')))
        ok = front & np.isfinite(u) & (u > -self.nx) & (u < 2 * self.nx)
        offset = None
        if ok.sum() > 1:
            o = np.argsort(u[ok])
            d = uv[:, 1] - np.interp(uv[:, 0], u[ok][o], v[ok][o], left=np.nan, right=np.nan)
            if np.isfinite(d).any():
                offset = float(np.nanmedian(d))
        return {'uv': uv.tolist(), 'columns': float(len(uv) * int(req.get('spacing', 8)) / self.nx),
                'median_offset_px': offset}

    def map(self, req):
        """items: [{still, uv}] -> geo rows per item, each mapped with its own still's pose."""
        return {'geo': [self.model.map_pixels(_arr(it['uv'], 2), self.pose(it['still'], req['c'], req.get('po')))
                        for it in req.get('items', [])]}

    def lines(self, edges, geos):
        return combine.camera_lines(self.id, geos, [e['still'] for e in edges], self.t)


# ------------------------------------------------------------------- jobs ---

class Job:
    _ids = itertools.count(1)

    def __init__(self, kind, camera, fn):
        self.id = str(next(Job._ids))
        self.kind, self.camera, self.fn = kind, camera, fn
        self.state, self.done, self.total, self.log, self.result, self.error = 'queued', 0, 0, [], None, ''
        self.warnings = []

    def say(self, *parts):
        msg = ' '.join(str(p) for p in parts)
        self.log.extend(msg.split('\n'))
        print('[%s %s] %s' % (self.kind, self.camera, msg))

    def warn(self, msg):
        """Something the person must see (shown by the portal before it opens the workspace)."""
        self.warnings.append(msg)
        self.say('WARNING: ' + msg)

    def progress(self, done, total):
        self.done, self.total = done, total

    def status(self):
        return {'id': self.id, 'kind': self.kind, 'camera': self.camera, 'state': self.state, 'done': self.done,
                'total': self.total, 'log': self.log[-200:], 'result': self.result, 'error': self.error,
                'warnings': self.warnings}


# ----------------------------------------------------------------- portal ---

class Portal:
    def __init__(self, project_root, ws_root, dem_dir):
        self.project_root = os.path.abspath(project_root)
        self.ws_root = os.path.abspath(ws_root)
        self.dem_dir = os.path.abspath(dem_dir)
        os.makedirs(self.ws_root, exist_ok=True)
        self.ws = None
        self.sessions = {}
        self.dems = {}
        self.jobs = {}
        self.frame_cache = {}       # camera id -> (key, representative frames) for the calibration step
        self.ref_cache = {}         # camera id -> (key, landmark candidates)
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ jobs --
    def _worker(self):
        while True:
            job = self.queue.get()
            job.state = 'running'
            try:
                job.result = job.fn(job)
                job.state = 'done'
            except Exception as err:            # reported to the page; the server keeps running
                job.state, job.error = 'error', '%s: %s' % (type(err).__name__, err)
                job.say('Failed: ' + job.error)
                traceback.print_exc()

    def _submit(self, kind, camera, fn):
        job = Job(kind, camera, fn)
        self.jobs[job.id] = job
        self.queue.put(job)
        return job.id

    def jobs_status(self, req):
        ids = req.get('ids')
        jobs = [self.jobs[i] for i in ids if i in self.jobs] if ids else list(self.jobs.values())[-30:]
        return {'jobs': [j.status() for j in jobs]}

    # ------------------------------------------------------- workspace --
    def _need_ws(self):
        if self.ws is None:
            raise ValueError('No workspace is open. Open or create one in the portal (/).')
        return self.ws

    def _resolve(self, p):
        p = os.path.expanduser(str(p or '.'))
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(self.project_root, p))

    def _store(self, p):
        rel = os.path.relpath(os.path.abspath(p), self.project_root)
        return os.path.abspath(p) if rel.startswith('..') else rel

    def workspaces(self, _):
        return {'workspaces': list_workspaces(self.ws_root), 'open': self.ws.slug if self.ws else None,
                'ws_root': self.ws_root, 'dem_dir': self.dem_dir}

    def workspace_new(self, req):
        offset = str(req.get('utc_offset') or '').strip()
        combine.parse_offset(offset)                      # validates
        ws = Workspace.create(self.ws_root, self.project_root, req.get('name', ''), offset,
                              self._store(self.dem_dir))
        self._open(ws)
        return self.workspace(None)

    def workspace_open(self, req):
        self._open(Workspace.load(self.ws_root, self.project_root, req['slug']))
        return self.workspace(None)

    def _open(self, ws):
        with self.lock:
            self.ws = ws
            self.sessions = {}

    def workspace(self, _):
        ws = self._need_ws()
        return {'workspace': {k: v for k, v in ws.data.items() if k != 'cameras'},
                'dir': ws.dir, 'cameras': [self._camera_summary(c) for c in ws.cameras]}

    def _camera_summary(self, cam):
        meta = {k: cam.get(k) for k in ('id', 'name', 'lat', 'lon', 'elev_m', 'stills_dir', 'glyphs', 'bar_px',
                                        'image_size', 'hfov0', 'panoramas', 'index', 'tiles', 'tiles_missing',
                                        'tiles_downloaded')}
        meta['horizon_km'] = self._horizon_km(cam)
        meta['calibration_source'] = cam['calibration'].get('source', 'none')
        meta['dem'] = self._dem_label(cam)
        meta['calibration_dem'] = (cam['calibration'].get('dem') or 'srtm') if cam['calibration'].get('c') else None
        meta['dem_mismatch'] = bool(meta['calibration_dem']) and meta['calibration_dem'].split(':')[0] != meta['dem'].split(':')[0]
        meta['n_landmarks'] = len(cam['landmarks'].get('gcp') or [])
        meta['has_proposal'] = bool(cam.get('proposal'))
        meta['n_edges'] = len(cam.get('edges', []))
        meta['stills'] = self._stills_state(self.ws.resolve(cam['stills_dir']))
        meta['ready'] = cam['id'] in self.sessions
        return meta

    def _stills_state(self, d):
        st = {'folder': d, 'exists': os.path.isdir(d)}
        if not st['exists']:
            return st
        imgs = stills_index.list_images(d)
        names = set(imgs)
        idx = os.path.join(d, 'stills_index.csv')
        st.update(n_images=len(imgs), has_index=os.path.isfile(idx), n_indexed=0)
        if st['has_index']:
            with open(idx) as f:
                st['n_indexed'] = sum(r['file'] in names for r in csv.DictReader(f))
        st['needs_index'] = not st['has_index'] or st['n_indexed'] == 0
        return st

    def save(self, req):
        """Store the page's per-camera state (calibration, landmarks, skyline, edges, view) and settings.

        Each camera's calibration and landmarks carry a revision ('rev'), raised whenever they change. A page that
        sends an older revision than the workspace has (it was opened before the calibration step changed them)
        does not overwrite them; the camera is listed in 'stale'. Its other state is saved."""
        ws = self._need_ws()
        stale = []
        with self.lock:
            for cid, st in (req.get('cameras') or {}).items():
                cam = ws.camera(cid)
                for k in ('skyline', 'edges', 'ui'):
                    if k in st:
                        cam[k] = _safe(st[k])
                if 'rev' in st and int(st['rev']) != cam.get('rev', 0):
                    if any(k in st for k in ('calibration', 'landmarks')):
                        stale.append(cam['name'])
                    continue
                changed = False
                if isinstance(st.get('landmarks'), dict):
                    st['landmarks'] = dict(st['landmarks'], info=_lm_info(st['landmarks'], cam.get('landmarks')))
                for k in ('calibration', 'landmarks'):
                    if k in st and _safe(st[k]) != _safe(cam.get(k)):
                        cam[k] = _safe(st[k])
                        changed = True
                if changed:
                    cam['rev'] = cam.get('rev', 0) + 1
            for k in ('space', 'active_camera', 'combine', 'export_name', 'timeline'):
                if k in req:
                    ws.data[k] = _safe(req[k])
            ws.save()
        return {'saved': ws.path('workspace.json'), 'modified': ws.data['modified'], 'stale': stale,
                'revs': {c['id']: c.get('rev', 0) for c in ws.cameras}}

    # ---------------------------------------------------------- cameras --
    def _site(self, req):
        lat, lon, elev = float(req['lat']), float(req['lon']), float(req['elev_m'])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Latitude/longitude out of range.')
        return lat, lon, elev

    def camera_add(self, req):
        ws = self._need_ws()
        lat, lon, elev = self._site(req)
        with self.lock:
            cam = ws.add_camera(req.get('name', ''), lat, lon, elev, self._resolve(req.get('stills_dir', '')))
            ws.save()
        return {'camera': self._camera_summary(cam)}

    def camera_update(self, req):
        """Change a camera's name, site, stills folder or starting HFOV. A changed site or folder unloads
        the camera until it is loaded again (Load cameras, or the app's Apply)."""
        ws = self._need_ws()
        with self.lock:
            cam = ws.camera(req['id'])
            if 'name' in req and str(req['name']).strip():
                cam['name'] = str(req['name']).strip()
            changed = False
            if any(k in req for k in ('lat', 'lon', 'elev_m')):
                site = self._site({k: req.get(k, cam[k]) for k in ('lat', 'lon', 'elev_m')})
                # the page shows rounded values: only a real change (> 1 cm) counts
                if _site_m(site, (cam['lat'], cam['lon'])) > 0.01 or abs(site[2] - cam['elev_m']) > 0.01:
                    cam['lat'], cam['lon'], cam['elev_m'] = site
                    changed = True
            if req.get('stills_dir'):
                d = ws.store(self._resolve(req['stills_dir']))
                if d != cam['stills_dir']:
                    cam.update(stills_dir=d, bar_px=None, glyphs='', image_size=None, index=None)
                    changed = True
            if req.get('hfov0'):
                h = float(req['hfov0'])
                if not CORRECTION_LIM[3][0] <= h <= CORRECTION_LIM[3][1]:
                    raise ValueError('The field of view must be %g to %g deg.' % CORRECTION_LIM[3])
                cam['hfov0'] = h
            if changed:
                self.sessions.pop(cam['id'], None)
            ws.save()
        return {'camera': self._camera_summary(cam), 'unloaded': changed}

    def camera_remove(self, req):
        ws = self._need_ws()
        with self.lock:
            cam = ws.camera(req['id'])
            ws.cameras.remove(cam)
            self.sessions.pop(cam['id'], None)
            ws.save()
        return {'removed': cam['id']}

    def panorama_upload(self, req):
        """Copy an uploaded panorama image (base64, optionally a data: URL) into the workspace."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        data = str(req.get('data', ''))
        data = base64.b64decode(data.split(',', 1)[1] if data.startswith('data:') else data)
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise ValueError('%s is not an image this server can read' % req.get('name'))
        name = safe_file('%s_%s' % (cam['id'], safe_file(req.get('name', 'panorama.jpg'))))
        with open(ws.path('panoramas', name), 'wb') as f:
            f.write(data)
        with self.lock:
            if name not in cam['panoramas']:
                cam['panoramas'].append(name)
            ws.save()
        return {'panoramas': cam['panoramas']}

    def panorama_remove(self, req):
        """Take a panorama off the camera's list (the file stays in panoramas/)."""
        ws = self._need_ws()
        with self.lock:
            cam = ws.camera(req['camera'])
            if req.get('name') in cam['panoramas']:
                cam['panoramas'].remove(req['name'])
            ws.save()
        return {'panoramas': cam['panoramas']}

    def folders(self, req):
        """Subfolders of a folder (default: the project folder) with their image counts, for the folder picker."""
        d = self._resolve(req.get('path') or '.')
        if not os.path.isdir(d):
            raise ValueError('Not a folder: %s' % d)
        dirs = []
        for e in sorted(os.scandir(d), key=lambda e: e.name.lower()):
            if e.is_dir() and not e.name.startswith(('.', '__')):
                try:
                    n = sum(1 for f in os.scandir(e.path) if f.name.lower().endswith(stills_index.IMAGE_EXT))
                except OSError:
                    n = None
                dirs.append({'name': e.name, 'path': self._store(e.path), 'images': n})
        here = sum(1 for f in os.scandir(d) if f.name.lower().endswith(stills_index.IMAGE_EXT))
        parent = os.path.dirname(d)
        return {'path': self._store(d), 'abs': d, 'parent': self._store(parent) if parent != d else None,
                'dirs': dirs, 'images': here}

    # -------------------------------------------------------- ingestion --
    def ingest(self, req):
        """Load cameras: one job per camera (bar readings if missing, DEM tiles, camera model; calibrating is the
        next step, /setup)."""
        ws = self._need_ws()
        ids = req.get('cameras') or [c['id'] for c in ws.cameras]
        return {'jobs': [self._submit('load', cid, lambda job, cid=cid: self._ingest(job, cid, bool(req.get('reindex'))))
                         for cid in ids]}

    def _ingest(self, job, cid, reindex=False):
        """Load a camera: bar readings, DEM tiles and camera model. Calibrating is the next step (/setup)."""
        ws = self._need_ws()
        cam = ws.camera(cid)
        self._read_bars(job, cam, reindex)
        calib = cam['calibration']
        job.say('Calibration: ' + calib['source'] if calib.get('c') else 'Not calibrated yet (the calibration step is next).')
        self._load_session(job, cam)
        self.frame_cache.pop(cid, None)
        with self.lock:
            ws.save()
        return {'camera': self._camera_summary(cam)}

    def _read_bars(self, job, cam, force=False):
        """Bar readings of the camera's stills folder: read if missing (or forced), and the bar height."""
        d = self.ws.resolve(cam['stills_dir'])
        st = self._stills_state(d)
        if not st['exists']:
            raise ValueError('Stills folder not found: %s' % d)
        if force or st['needs_index']:
            job.say('Reading the info bars of %d stills in %s' % (st.get('n_images', 0), cam['stills_dir']))
            s = stills_index.index_folder(d, 'auto', progress=job.progress, log=job.say)
            cam.update(glyphs=s['glyphs'], bar_px=s['bar_px'], image_size=s['image_size'], index=s)
            return
        with open(os.path.join(d, 'stills_index.csv')) as f:
            rows = [r for r in csv.DictReader(f) if os.path.isfile(os.path.join(d, r['file']))]
        n_review = 0
        if os.path.isfile(os.path.join(d, 'review.csv')):
            with open(os.path.join(d, 'review.csv')) as f:
                n_review = sum(1 for _ in csv.DictReader(f))
        elev = sorted(float(r['elev_ft']) for r in rows if r.get('elev_ft'))
        times = sorted(r['timestamp'] for r in rows)
        job.say('Bar readings found: %d stills (%d images in the folder)' % (len(rows), st['n_images']))
        if not cam.get('bar_px'):
            sample = stills_index.spread([r['file'] for r in rows], stills_index.N_SAMPLE_TOP)
            grays = [stills_index.bottom_gray(os.path.join(d, f)) for f in sample]
            size = grays[0][1]
            top, height = bar_geometry([g for g, s in grays if s == size], default_bar_px(size[1]))
            cam.update(bar_px=height, image_size=list(size))
            job.say('Stills are %dx%d; black bar is the bottom %d px' % (size[0], size[1], height))
        cam['index'] = dict(cam.get('index') or {}, n_images=st['n_images'], n_stills=len(rows),
                            n_poses=len({r.get('pose_id') for r in rows}), n_review=n_review,
                            first=times[0] if times else '', last=times[-1] if times else '',
                            elev_ft=elev[len(elev) // 2] if elev else None)

    def _horizon_km(self, cam):
        return float(cam.get('horizon_km') or DEFAULT_HORIZON_KM)

    def _tiles(self, cam, job=None, horizon_km=None):
        """(near, far) paths of the terrain tiles around the camera: those within NEAR_KM, and the rest within its
        horizon distance. Missing tiles are downloaded (tiles.fetch); ones still missing are noted on the camera."""
        d = self._dem_dir()
        km = horizon_km or self._horizon_km(cam)
        names = tiles.tile_names(cam['lat'], cam['lon'], km + 5)
        found, missing = tiles.find(names, d)
        missing = [n for n in missing if n not in tiles.unavailable(d)]
        if missing:
            if job:
                job.say('Downloading %d terrain tile%s for %.0f km around the camera (once; about 10 MB each)...'
                        % (len(missing), '' if len(missing) == 1 else 's', km))
            found += tiles.fetch(missing, d, log=(lambda m: job.warn('%s: %s' % (cam['name'], m[9:])) if m.startswith('WARNING: ')
                                                  else job.say(m)) if job else (lambda m: None))
            found, missing = tiles.find(names, d)
        near_names = {n.lower() for n in tiles.tile_names(cam['lat'], cam['lon'], NEAR_KM)}
        near = [p for p in found if tiles.tile_name(p).lower() in near_names]
        far = [p for p in found if tiles.tile_name(p).lower() not in near_names]
        if horizon_km is None:
            cam['tiles'] = sorted(tiles.tile_name(p) for p in found)
            cam['tiles_missing'] = missing
            cam['tiles_downloaded'] = sum(1 for p in found if os.path.dirname(p) == tiles.download_dir(d))
        if job:
            job.say('Terrain to %.0f km: %d tiles (%d at full resolution)%s' % (km, len(found), len(near),
                    ('; none for %s (sea, or outside the tile set): not modelled' % ', '.join(missing)) if missing else ''))
        if not near:
            raise ValueError('No terrain tile at the camera in %s or its downloads (it needs %s)'
                             % (d, ', '.join(sorted(near_names))))
        return near, far

    def _dem_dir(self):
        return self.ws.resolve(self.ws.data.get('dem_dir') or self.dem_dir) if self.ws else self.dem_dir

    def _fine_box(self, sites=None, radius_km=None):
        """The workspace's 3DEP box: dem radius_km (or the given radius) around every camera site (or the given
        sites); None without cameras."""
        sites = sites if sites is not None else [(c['lat'], c['lon']) for c in self.ws.cameras]
        return dem3dep.box_around(sites, radius_km or self.ws.dem['radius_km']) if sites else None

    def _fine_path(self, cam):
        """The cached 3DEP grid covering this camera's dem radius (None if the workspace uses SRTM or none is there)."""
        if self.ws.dem['source'] != '3dep10':
            return None
        return self._fine_cover(cam)

    def _fine_cover(self, cam):
        """A downloaded 3DEP grid covering the dem radius around the camera (less COVER_SLACK_KM, so a small site
        change keeps it), or None."""
        r = max(1.0, self.ws.dem['radius_km'] - COVER_SLACK_KM)
        return dem3dep.find(dem3dep.box_around([(cam['lat'], cam['lon'])], r), os.path.join(self._dem_dir(), 'downloads', '3dep'))

    def _dem_label(self, cam):
        """Which DEM the camera's model uses: 'srtm', or '3dep10' with the grid's file name."""
        p = self._fine_path(cam)
        return '3dep10:' + os.path.basename(p) if p else 'srtm'

    def _grid(self, key, make, job=None, msg=None):
        """A DEM grid from the cache, made if needed (grids are shared by the cameras that use the same tiles)."""
        with self.lock:
            g = self.dems.get(key)
        if g is None:
            if job and msg:
                job.say(msg)
            g = make()
            with self.lock:
                self.dems[key] = g
        return g

    def _dem(self, cam, job=None, horizon_km=None):
        """The camera's terrain (core.Terrain): the workspace's 3DEP grid if it uses one, the full-resolution tiles
        within NEAR_KM and the coarse ones out to the horizon distance (or horizon_km)."""
        near, far = self._tiles(cam, job, horizon_km)
        layers = [self._grid(('near',) + tuple(sorted(near)), lambda: core.DEM(sorted(near)), job,
                             'Loading terrain near the camera (%d tiles)...' % len(near))]
        if far:
            layers.append(self._grid(('far',) + tuple(sorted(far)), lambda: core.DEM(sorted(far), coarse=True), job,
                                     'Loading distant terrain (%d tiles at 3 arc-seconds)...' % len(far)))
        fine = self._fine_path(cam)
        if self.ws.dem['source'] == '3dep10' and not fine and job:
            job.warn('%s: the workspace uses the 3DEP 10 m DEM, but no downloaded 3DEP grid covers %.0f km around this '
                     'camera; using SRTM. Download it again in the portal (DEM card).' % (cam['name'], self.ws.dem['radius_km']))
        if fine:
            layers.insert(0, self._grid(('3dep', fine), lambda: dem3dep.load(fine), job,
                                        'DEM: USGS 3DEP 10 m (%s) within %.0f km, SRTM beyond'
                                        % (os.path.basename(fine), self.ws.dem['radius_km'])))
        return core.Terrain(layers, max_range_m=1000 * (horizon_km or self._horizon_km(cam)),
                            fine_step_m=10.0 if fine else None, name=os.path.basename(fine) if fine else 'srtm')

    def _load_session(self, job, cam):
        dem = self._dem(cam, job)
        g = float(dem(cam['lat'], cam['lon']))
        if np.isfinite(g) and cam['elev_m'] - g < MIN_ABOVE_GROUND_M:
            job.warn('%s: the camera height (%.1f m) is %.1f m %s the DEM ground at its site (%.1f m). Rays from the camera '
                     'then start in the terrain. Check the site (Check site in the portal) and the height.'
                     % (cam['name'], cam['elev_m'], abs(cam['elev_m'] - g), 'below' if cam['elev_m'] < g else 'above', g))
        job.say('Computing the 360-degree skyline...')
        s = CameraSession(cam, self.ws.resolve(cam['stills_dir']), dem, self.ws.data.get('utc_offset', ''))
        if np.isfinite(g) and cam['elev_m'] - g >= MIN_ABOVE_GROUND_M:
            self._near_warning(job, cam, s, dem)
        s.dem_label = self._dem_label(cam)
        with self.lock:
            self.sessions[cam['id']] = s
        job.say('Ready: %d stills, image %dx%d (%d px bar cropped)' % (len(s.stills), s.nx, s.ny + s.bar_px, s.bar_px))

    @staticmethod
    def _near_terrain_in_view(cam, s, dem):
        """Ground near the camera (reference.near_terrain) that rises above the middle of the view of the camera's
        main positions (MAIN_POSE_SHARE of its stills or more; one-off views aimed at the lookout itself don't count): the
        camera would then see mostly the ground next to it, so its site or height is off. Returns (near, worst
        azimuth, worst angle above the view's centre line) or None."""
        near = reference.near_terrain(dem, cam['lat'], cam['lon'], cam['elev_m'])
        c = core.full_correction(cam['calibration'].get('c') or s.c0)
        worst = None
        count = {}
        for b in s.bars.values():
            count[core.pose_key(b)] = count.get(core.pose_key(b), 0) + 1
        for key in [k for k, n in count.items() if n >= MAIN_POSE_SHARE * len(s.bars)]:   # the main positions only
            q = core.pose_from_bar(tuple(float(x) for x in key.split(',')), c)
            half = q[3] / 2
            for a in range(int(np.floor(q[0] - half)), int(np.ceil(q[0] + half)) + 1):
                over = near['by_az'][a % 360] - q[1]
                if worst is None or over > worst[2]:
                    worst = (near, a % 360, over)
        return worst if worst and worst[2] > 0 else None

    def _near_warning(self, job, cam, s, dem):
        hit = self._near_terrain_in_view(cam, s, dem)
        if hit:
            near, az, over = hit
            sm = near['summit']
            job.warn('%s: in the view of the stills (azimuth %d) the DEM ground within %.0f m of the camera rises %.1f deg above the '
                     'middle of the image, so the model sees mostly nearby ground there. The site or height is probably off: '
                     'the highest DEM point within 150 m is %.0f m away at %.1f m (camera %+.1f m above it). Use Check site '
                     'in the portal.' % (cam['name'], az, reference.NEAR_RADIUS_M, over, sm['distance_m'], sm['elev'],
                                         sm['camera_above_m']))

    def _panorama(self, job, cam):
        """Calibrate from the camera's panoramas; returns the calibration (not applied here)."""
        paths = [self.ws.path('panoramas', p) for p in cam['panoramas']]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise ValueError('Panorama file missing: %s' % ', '.join(missing))
        job.say('Panorama calibration from %s' % ', '.join(cam['panoramas']))
        frames = []

        def log(msg):                       # the calibration's warnings are shown by the portal and the page
            if msg.startswith('WARNING: '):
                job.warn('%s: %s' % (cam['name'], msg[9:]))
            else:
                job.say(msg)
        c, mat, overlay = panorama.run(paths, (cam['lat'], cam['lon'], cam['elev_m']), self._tiles(cam)[0],
                                       self.ws.path('calibration'), cam['id'], hfov0=cam.get('hfov0') or DEFAULT_HFOV,
                                       glyphs='auto', log=log, frames_out=frames, dem=self._dem(cam))
        mag, toward = core.lean_description(c)
        report = {'files': list(cam['panoramas']), 'when': now(),
                  'mat': 'calibration/' + os.path.basename(mat), 'overlay': 'calibration/' + os.path.basename(overlay),
                  'lean_deg': mag, 'lean_toward_deg': toward,
                  'frames': [{'name': fr.name, 'bar': list(fr.bar) if fr.bar else None, 'time': fr.time,
                              'note': fr.note, 'median_px': fr.stats.get('median_abs_px'),
                              'p90_px': fr.stats.get('p90_abs_px'), 'kept': fr.stats.get('kept'), 'width': fr.nx}
                             for fr in frames]}
        return _safe({'c': list(c), 'source': 'panorama (%s)' % ', '.join(cam['panoramas']), 'note': '',
                      'report': report, 'dem': self._dem_label(cam)})

    # ------------------------------------------------------- app: setup --
    def _session(self, req):
        cid = req.get('camera')
        if cid not in self.sessions:
            raise ValueError('Camera %r is not loaded (run Load cameras in the portal)' % cid)
        return self.sessions[cid]

    def config(self, _):
        ws = self._need_ws()
        cams = []
        for cam in ws.cameras:
            s = self.sessions.get(cam['id'])
            cams.append({'id': cam['id'], 'name': cam['name'], 'meta': self._camera_summary(cam),
                         'cfg': s.config() if s else None,
                         'state': {k: cam.get(k) for k in ('calibration', 'landmarks', 'skyline', 'edges', 'ui', 'rev')}})
        return {'workspace': {k: v for k, v in ws.data.items() if k != 'cameras'}, 'cameras': cams}

    def camera_config(self, req):
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        s = self.sessions.get(cam['id'])
        return {'meta': self._camera_summary(cam), 'cfg': s.config() if s else None}

    def reload(self, req):
        """Job: re-read the bars (if asked) and rebuild one camera's model, e.g. after its site changed."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])

        def run(job):
            self._read_bars(job, cam, bool(req.get('reindex')))
            self._load_session(job, cam)
            with self.lock:
                ws.save()
            return {'camera': cam['id']}
        return {'job': self._submit('reload', cam['id'], run)}

    def panorama_run(self, req):
        """Job: run the panorama calibration again; the page decides whether to use the result."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        if not cam['panoramas']:
            raise ValueError('This camera has no panorama. Add one first.')
        return {'job': self._submit('panorama', cam['id'], lambda job: self._panorama(job, cam))}

    # ------------------------------------------------- calibration step --
    # /setup: a few representative frames per camera (its panorama frames and some stills), landmarks clicked
    # on them, and the calibration (edgefit) run per camera as a job.

    def _frame_file(self, cam, name):
        return self.ws.path('calibration', 'frames', cam['id'], name)

    def _frames(self, cam):
        """Representative frames of a camera (cached): every panorama frame, and a few stills (edgefit.representative)."""
        s = self.sessions.get(cam['id'])
        if s is None:
            raise ValueError('%s is not loaded (Load cameras in the portal first).' % cam['name'])
        key = (tuple(cam['panoramas']), len(s.stills), s.stills[0]['file'])
        hit = self.frame_cache.get(cam['id'])
        if hit and hit[0] == key:
            return hit[1]
        out = []
        os.makedirs(self._frame_file(cam, ''), exist_ok=True)
        for p in cam['panoramas']:
            try:
                pf = panorama.load_frames([self.ws.path('panoramas', p)], glyphs='auto')
            except Exception as err:
                out.append({'id': 'pano:' + p, 'kind': 'panorama', 'label': 'Panorama %s' % p, 'bar': None, 'size': None,
                            'url': '', 'note': 'not readable: %s' % err})
                continue
            for fr in pf:
                name = '%s_%d.jpg' % (slug(os.path.splitext(p)[0]), fr.index + 1)
                if not os.path.isfile(self._frame_file(cam, name)):
                    Image.fromarray(fr.rgb).save(self._frame_file(cam, name), quality=90)
                out.append({'id': 'pano:' + fr.name, 'kind': 'panorama', 'bar': list(fr.bar) if fr.bar else None,
                            'label': 'Panorama frame %d · X %s' % (fr.index + 1, '%+.2f' % fr.bar[0] if fr.bar else '?'),
                            'size': [fr.nx, fr.ny], 'time': fr.time, 'note': fr.note,
                            'url': '/ws/frames/%s/%s' % (cam['id'], name)})

        for st in edgefit.representative(s.stills, self._bright(s)):
            out.append({'id': st['file'], 'kind': 'still', 'bar': [st['X'], st['Y'], st['Z']], 'size': [s.nx, s.ny],
                        'label': 'Still %s · X %+.2f' % (st['time'][11:19], st['X']), 'time': st['time'], 'note': '',
                        'url': '/still/%s/%s' % (cam['id'], st['file'])})
        self.frame_cache[cam['id']] = (key, out)
        return out

    @staticmethod
    def _bright(s):
        """is_bright(still) for edgefit.representative: a daylight image, judged on a thumbnail above the bar.
        Night images are greyscale (infrared mode, with auto gain, so often not dark): daylight needs colour."""
        def bright(st):
            im = Image.open(os.path.join(s.stills_dir, st['file'])).convert('RGB')
            im.thumbnail((320, 320))
            a = np.asarray(im, float)[: int(im.height * s.ny / (s.ny + s.bar_px))]
            return a.mean() > 50 and (a.max(2) - a.min(2)).mean() > 10
        return bright

    def _frame_info(self, cam, fid):
        """bar and size of a frame id (a representative frame, or any of the camera's stills)."""
        s = self._session({'camera': cam['id']})
        if fid in s.bars:
            return {'id': fid, 'bar': list(s.bars[fid]), 'size': [s.nx, s.ny], 'kind': 'still'}
        for f in self._frames(cam):
            if f['id'] == fid and f['bar']:
                return f
        raise ValueError('Unknown frame %r' % fid)

    def setup_frames(self, req):
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        return {'frames': self._frames(cam), 'landmarks': cam['landmarks'], 'calibration': cam['calibration'],
                'proposal': cam.get('proposal'), 'rev': cam.get('rev', 0), 'site': [cam['lat'], cam['lon'], cam['elev_m']], 'name': cam['name'],
                'horizon': {'km': self._horizon_km(cam), 'terrain': {'tiles': len(cam.get('tiles') or []),
                                                                     'downloaded': cam.get('tiles_downloaded') or 0,
                                                                     'missing': cam.get('tiles_missing') or []}}}

    def setup_horizon(self, req):
        """Job: set how far out a camera's skyline and rays follow the terrain (km), and reload the camera with it
        (terrain tiles it needs are downloaded). A calibration fitted with another distance stays in use; calibrate
        again to fit the new skyline."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        km = float(req['km'])
        if not HORIZON_KM_LIM[0] <= km <= HORIZON_KM_LIM[1]:
            raise ValueError('The horizon distance must be %g to %g km.' % HORIZON_KM_LIM)

        def run(job):
            old = self._horizon_km(cam)
            with self.lock:
                cam['horizon_km'] = km
            try:
                self._load_session(job, cam)
            except Exception:
                with self.lock:
                    cam['horizon_km'] = old
                raise
            self.frame_cache.pop(cam['id'], None)
            self.ref_cache.pop(cam['id'], None)
            with self.lock:
                ws.save()
            return {'camera': cam['id'], 'horizon_km': km}
        return {'job': self._submit('horizon', cam['id'], run)}

    def setup_landmarks(self, req):
        """Replace a camera's landmarks: {gcp: [[u, v, lat, lon, elev]], stills: [frame ids], names: [...], info:
        [{source, ref_id, acc_m}, ...] (optional; from the reference points, see reference_candidates)}."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        lm = req.get('landmarks') or {}
        gcp, ids, names = _arr(lm.get('gcp'), 5), list(lm.get('stills') or []), list(lm.get('names') or [])
        if not (len(gcp) == len(ids) == len(names)) or not np.isfinite(gcp).all():
            raise ValueError('Each landmark needs a pixel, latitude, longitude, elevation, frame and name.')
        with self.lock:
            new = {'gcp': gcp.tolist(), 'stills': ids, 'names': [str(n) for n in names]}
            new['info'] = _lm_info(dict(new, info=lm.get('info')), cam.get('landmarks'))
            cam['landmarks'] = _safe(new)
            cam['rev'] = cam.get('rev', 0) + 1
            ws.save()
        return {'landmarks': cam['landmarks'], 'rev': cam['rev']}

    def _specs(self, cam):
        """The calibrations a frame can be drawn with, by name: (correction, pose offsets, report).
        'calibration' is the one in use, 'previous' the one before it and 'proposal' the proposed one."""
        cal, prop = cam['calibration'], cam.get('proposal')
        out = {'calibration': (cal['c'], cal.get('pose_offsets'), cal.get('report')) if cal.get('c') else
               ([0, 0, 0, cam.get('hfov0') or DEFAULT_HFOV, 0, 0], None, None)}
        prev = (cal.get('report') or {}).get('c_previous')
        if cal.get('c') and prev:
            out['previous'] = (prev, None, None)
        if prop:
            out['proposal'] = (prop['c'], prop.get('pose_offsets'), prop.get('report'))
        return out

    @staticmethod
    def _frame_pose(f, c, po, report):
        """Pose of a frame: a still with its position's offsets (po); a panorama frame with its own pan/tilt error
        from the edge fit that made report (every panorama frame gets one, edgefit.EdgeFit.run)."""
        if f['kind'] == 'panorama':
            row = next((r for r in (report or {}).get('frames') or [] if r.get('id') == f['id']), None)
            off = row.get('offset') if row else None
            po = {core.pose_key(f['bar']): list(off[:2])} if off else None
        return core.pose_from_bar(f['bar'], c, po)

    def setup_predict(self, req):
        """For one frame: the DEM skyline of each calibration named in lines (see _specs; or {'c', 'po'}), the
        camera's landmarks on that frame projected with the calibration named landmarks_with, every landmark's
        error with it (each on its own frame), and the frame's left/centre/right azimuths."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        s = self._session(req)
        f = self._frame_info(cam, req['frame'])
        specs = self._specs(cam)

        def spec(x):
            if isinstance(x, dict):
                return (x['c'], x.get('po'), None) if x.get('c') else None
            return specs.get(x)
        models = {}

        def model(size):
            if tuple(size) not in models:
                models[tuple(size)] = core.CameraModel(s.cam, s.model.dem, *size)
            return models[tuple(size)]
        w, h = f['size']
        m = model(f['size'])
        out = {'lines': {}, 'landmarks': [], 'landmark_errors': [], 'fov': None}
        for name in req.get('lines') or ['calibration']:
            sp = spec(name)
            if sp is None:
                continue
            u, v, front = m.project(s.sky_rays, self._frame_pose(f, *sp))
            ok = front & (u >= 1) & (u <= w) & (v >= 1) & (v <= h)
            o = np.argsort(u[ok])
            pts = np.column_stack([u[ok][o], v[ok][o]])
            out['lines'][name if isinstance(name, str) else 'custom'] = pts[:: max(1, len(pts) // 2000)]
        sp = spec(req.get('landmarks_with') or 'calibration')
        if sp is not None:
            q = self._frame_pose(f, *sp)
            lm = cam['landmarks']
            for i, (g, fid) in enumerate(zip(lm['gcp'], lm['stills'])):
                try:
                    fi = f if fid == f['id'] else self._frame_info(cam, fid)
                except ValueError:
                    continue
                mi = model(fi['size'])
                qi = q if fi is f else self._frame_pose(fi, *sp)
                u, v, front = mi.project(mi.ground_rays([g[2:5]]), qi)
                if not front[0] or not np.isfinite(u[0]):
                    continue
                if fid == f['id']:
                    out['landmarks'].append([i, float(u[0]), float(v[0])])
                px = float(np.hypot(u[0] - g[0], v[0] - g[1]))
                ppd = (fi['size'][0] / 2) / np.tan(np.radians(qi[3] / 2)) * np.pi / 180
                out['landmark_errors'].append([i, px, px / ppd])
            rays = m.pixel_rays([[1, (h + 1) / 2], [(w + 1) / 2, (h + 1) / 2], [w, (h + 1) / 2]], q)
            out['fov'] = [float(np.degrees(np.arctan2(r[0], r[1])) % 360) for r in rays]
        return out

    # The proposal: the calibrate job proposes a calibration (cam['proposal']) instead of applying it; it is
    # checked on the frames and applied by setup_accept (or dropped). It is kept in workspace.json.

    def _proposal(self, cam):
        prop = cam.get('proposal')
        if not prop:
            raise ValueError('%s has no proposed calibration: run Calibrate first.'
                             % cam['name'])
        return prop

    def setup_proposal(self, req):
        """action 'discard': drop the camera's proposal (the calibration in use stays)."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        if req.get('action') != 'discard':
            raise ValueError('Unknown action %r' % req.get('action'))
        with self.lock:
            cam.pop('proposal', None)
            ws.save()
        return {'proposal': None}

    def setup_accept(self, req):
        """Job: apply a camera's proposed calibration. An adjusted proposal is first measured again on the
        calibration frames (remaining offsets, landmark errors, overlay image)."""
        ws = self._need_ws()
        cid = req['camera']
        self._proposal(ws.camera(cid))
        return {'jobs': [self._submit('accept', cid, lambda job: self._accept(job, cid))]}

    def setup_summit(self, req):
        """The highest DEM point within radius_m (default 300 m) of lat/lon: landmarks are mostly peaks."""
        s = self._session(req)
        la, lo, z, moved = reference.summit(s.model.dem, float(req['lat']), float(req['lon']), float(req.get('radius_m', 300)))
        return {'lat': la, 'lon': lo, 'elev': z, 'moved_m': moved}

    # ------------------------------------------- reference points & DEM --
    # Points with known coordinates around a camera (NGS survey marks, OpenStreetMap peaks and towers, FAA
    # obstacles; reference.py), fetched once per camera and kept in reference/<camera>.json. They are offered as
    # landmark candidates on the frames and as a check of the camera's site.

    def _ref_path(self, cid):
        return self.ws.path('reference', '%s.json' % cid)

    def reference_fetch(self, req):
        """Job: fetch (or refresh) the reference points around a camera."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])

        def run(job):
            box = reference.box_around(cam['lat'], cam['lon'])
            def log(msg):
                job.warn('%s: %s' % (cam['name'], msg[9:])) if msg.startswith('WARNING: ') else job.say(msg)
            lib = reference.fetch(box, os.path.join(self._dem_dir(), 'downloads', 'faa'), log=log)
            lib['site'] = [cam['lat'], cam['lon']]
            os.makedirs(ws.path('reference'), exist_ok=True)
            with open(self._ref_path(cam['id']) + '.tmp', 'w') as f:
                json.dump(lib, f)
            os.replace(self._ref_path(cam['id']) + '.tmp', self._ref_path(cam['id']))
            self.ref_cache.pop(cam['id'], None)
            job.say('%d reference points around %s' % (len(lib['points']), cam['name']))
            return {'camera': cam['id'], 'n': len(lib['points']), 'sources': lib['sources']}
        return {'job': self._submit('reference', cam['id'], run)}

    def _candidates(self, cam, s):
        """The camera's landmark candidates (reference.candidates), cached per site, DEM and fetch."""
        path = self._ref_path(cam['id'])
        if not os.path.isfile(path):
            return None, []
        with open(path) as f:
            lib = json.load(f)
        key = (tuple(s.cam), getattr(s, 'dem_label', ''), lib.get('fetched'))
        hit = self.ref_cache.get(cam['id'])
        if hit and hit[0] == key:
            return lib, hit[1]
        cands = reference.candidates(s.model, lib['points'])
        self.ref_cache[cam['id']] = (key, cands)
        return lib, cands

    def reference_candidates(self, req):
        """Landmark candidates of a camera: every visible reference point with its azimuth, elevation angle, range
        and angular accuracy, and, for a frame, where the calibration named by 'with' (see _specs) puts it."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        s = self._session(req)
        lib, cands = self._candidates(cam, s)
        if lib is None:
            return {'fetched': None, 'candidates': [], 'sources': {}}
        out = [dict(c) for c in cands]
        if req.get('frame'):
            f = self._frame_info(cam, req['frame'])
            sp = self._specs(cam).get(req.get('with') or 'calibration')
            w, h = f['size']
            m = core.CameraModel(s.cam, s.model.dem, w, h)
            if out and sp is not None:
                q = self._frame_pose(f, *sp)
                u, v, front = m.project(m.ground_rays([[c['lat'], c['lon'], c['elev']] for c in out]), q)
                for c, uu, vv, fr in zip(out, u, v, front):
                    ok = bool(fr and np.isfinite(uu) and -0.1 * w <= uu <= 1.1 * w and -0.1 * h <= vv <= 1.1 * h)
                    c.update(u=float(uu) if ok else None, v=float(vv) if ok else None, in_frame=ok)
        far = (lib.get('site') and _site_m(lib['site'], (cam['lat'], cam['lon'])) > 5000)
        return {'fetched': lib.get('fetched'), 'sources': lib.get('sources'), 'candidates': _safe(out),
                'note': 'The camera moved more than 5 km since these points were fetched: fetch them again.' if far else ''}

    def site_check(self, req):
        """NGS survey marks near a camera's site (or the given lat/lon), with the DEM ground there and the camera's
        height above it (reference.site_check)."""
        ws = self._need_ws()
        cam = ws.camera(req['camera']) if req.get('camera') else None
        lat = float(req['lat']) if req.get('lat') is not None else cam['lat']
        lon = float(req['lon']) if req.get('lon') is not None else cam['lon']
        elev = float(req['elev_m']) if req.get('elev_m') is not None else (cam['elev_m'] if cam else None)
        s = self.sessions.get(cam['id']) if cam else None
        dem = (s.model.dem if s and _site_m(s.cam, (lat, lon)) < 1 else
               self._dem(dict(cam or {}, lat=lat, lon=lon, name='site'), horizon_km=NEAR_KM))
        g = float(dem(lat, lon))
        marks = reference.site_check(lat, lon, dem, elev, float(req.get('radius_m') or 1500))
        label = self._dem_label(dict(cam or {}, lat=lat, lon=lon))
        near = reference.near_terrain(dem, lat, lon, elev) if elev is not None and np.isfinite(g) else None
        return _safe({'site': [lat, lon, elev], 'ground_m': g if np.isfinite(g) else None,
                      'camera_above_ground_m': elev - g if elev is not None and np.isfinite(g) else None,
                      'dem': DEM_LABELS[label.split(':')[0]], 'marks': marks, 'near': near,
                      'near_block_deg': reference.NEAR_BLOCK_DEG})

    def dem_status(self, req):
        """The workspace's DEM setting, its 3DEP box (size, whether a grid is downloaded) and which cameras it covers.
        radius_km: the size of the box for this radius instead of the saved one (the page asks before downloading)."""
        ws = self._need_ws()
        d = ws.dem
        box = self._fine_box(radius_km=float(req['radius_km']) if req.get('radius_km') else None)
        found = dem3dep.find(box, os.path.join(self._dem_dir(), 'downloads', '3dep')) if box else None
        cams = [{'id': c['id'], 'name': c['name'], 'covered': bool(self._fine_cover(c))} for c in ws.cameras]
        out = {'source': d['source'], 'radius_km': d['radius_km'], 'label': DEM_LABELS[d['source']], 'cameras': cams,
               'box': list(box) if box else None}
        if box:
            ny, nx = dem3dep.box_shape(box)
            out.update(area_km2=dem3dep.box_area_km2(box), size_mb=nx * ny * 4 / 1e6, max_area_km2=dem3dep.MAX_AREA_KM2)
        if found:
            out['grid'] = dict(dem3dep.meta(found), file=os.path.basename(found))
        return _safe(out)

    def _unload_all(self):
        with self.lock:
            self.sessions = {}
            self.frame_cache = {}
            self.ref_cache = {}

    def dem_set(self, req):
        """Choose the DEM ('srtm' or '3dep10') and the 3DEP radius. Unloads the cameras (Load cameras again)."""
        ws = self._need_ws()
        src = str(req.get('source') or ws.dem['source'])
        if src not in DEM_LABELS:
            raise ValueError('Unknown DEM %r' % src)
        r = float(req.get('radius_km') or ws.dem['radius_km'])
        if not 2 <= r <= 60:
            raise ValueError('The 3DEP radius must be 2 to 60 km.')
        changed = dict(ws.dem) != dict(DEFAULT_DEM, source=src, radius_km=r)
        with self.lock:
            ws.data['dem'] = {'source': src, 'radius_km': r}
            ws.save()
        if changed:
            self._unload_all()
        return dict(self.dem_status(req), unloaded=changed)

    def dem_fetch(self, req):
        """Job: download the 3DEP 10 m grid of the workspace's box (dem3dep.fetch), then use it."""
        ws = self._need_ws()
        if req.get('radius_km'):
            with self.lock:
                ws.data['dem'] = dict(ws.dem, radius_km=float(req['radius_km']))
                ws.save()
        box = self._fine_box()
        if not box:
            raise ValueError('Add the cameras first: the 3DEP area is around them.')
        if dem3dep.box_area_km2(box) > dem3dep.MAX_AREA_KM2:
            raise ValueError('The 3DEP area would be %.0f km2 (limit %.0f): use a smaller radius.'
                             % (dem3dep.box_area_km2(box), dem3dep.MAX_AREA_KM2))

        def run(job):
            path = dem3dep.find(box, os.path.join(self._dem_dir(), 'downloads', '3dep'))
            if not path and all(self._fine_cover(c) for c in ws.cameras):
                path = self._fine_cover(ws.cameras[0])
            if path and not req.get('refresh'):
                job.say('Already downloaded: %s' % os.path.basename(path))
            else:
                path = dem3dep.fetch(box, os.path.join(self._dem_dir(), 'downloads', '3dep'), log=job.say)
            with self.lock:
                ws.data['dem'] = dict(ws.dem, source='3dep10')
                ws.save()
            self._unload_all()
            job.say('The workspace now uses the 3DEP 10 m DEM; load the cameras again.')
            return {'file': os.path.basename(path)}
        return {'job': self._submit('dem', '', run)}

    def calibrate(self, req):
        """Jobs: calibrate cameras (default: all loaded ones) from their edges and landmarks; each result is a
        proposal (setup_accept applies it)."""
        ws = self._need_ws()
        ids = req.get('cameras') or [c['id'] for c in ws.cameras if c['id'] in self.sessions]
        return {'jobs': [self._submit('calibrate', cid, lambda job, cid=cid: self._calibrate(job, cid)) for cid in ids]}

    def _edge_frames(self, job, cam, s):
        """A camera's calibration frames: its panorama frames (panorama.load_frames; detect notes towers and roofs),
        its representative stills, more stills per pose when it has a panorama (for the poses' own offsets) and
        the stills with landmarks. Returns (panorama frames, EdgeFrames, Landmarks)."""
        ws = self.ws
        pano = []
        for p in cam['panoramas']:
            try:
                pano += panorama.load_frames([ws.path('panoramas', p)], glyphs='auto')
            except Exception as err:
                job.warn('%s: panorama %s not used (%s)' % (cam['name'], p, err))
        if pano:
            panorama.detect(pano)
        usable = [fr for fr in pano if fr.bar and not fr.note]
        frames = [edgefit.EdgeFrame.from_rgb('pano:' + fr.name, 'Panorama frame %d · X %+.2f' % (fr.index + 1, fr.bar[0]),
                                             fr.rgb, fr.bar, 'panorama', fr.note) for fr in pano if fr.bar]
        lm = cam['landmarks']
        still_ids = [f['id'] for f in self._frames(cam) if f['kind'] == 'still']
        if usable:                                 # more stills per pose to measure the poses' own offsets
            still_ids += [st['file'] for st in edgefit.representative(s.stills, self._bright(s), per_pose=6)
                          if st['file'] not in still_ids]
        still_ids += [fid for fid in dict.fromkeys(lm['stills']) if fid in s.bars and fid not in still_ids]
        for fid in still_ids:
            rgb = np.asarray(Image.open(os.path.join(s.stills_dir, fid)).convert('RGB'))[:s.ny, :s.nx]
            frames.append(edgefit.EdgeFrame.from_rgb(fid, 'Still %s · X %+.2f' % (s.times[fid][11:19], s.bars[fid][0]),
                                                     rgb, s.bars[fid]))
        ids = {fr.id for fr in frames}
        landmarks = [edgefit.Landmark(fid, *g, name=n, sigma_deg=self._lm_sigma(s, g, inf))
                     for g, fid, n, inf in zip(lm['gcp'], lm['stills'], lm['names'], _lm_info(lm, lm))]
        for l in landmarks:
            if l.frame not in ids:
                job.warn('%s: landmark %s is on %s, which is not loaded; not used.' % (cam['name'], l.name, l.frame))
        return pano, frames, landmarks

    @staticmethod
    def _lm_sigma(s, g, info):
        """Angular accuracy (deg) of a landmark's coordinates seen from the camera: atan(acc_m / range); 0 (the
        default weight) for a landmark without a known accuracy."""
        acc = (info or {}).get('acc_m')
        if not acc:
            return 0.0
        lat, lon = s.cam[0], s.cam[1]
        r = math.hypot((g[2] - lat) * reference.K_M_PER_DEG, (g[3] - lon) * reference.K_M_PER_DEG * math.cos(math.radians(lat)))
        return float(math.degrees(math.atan2(float(acc), max(r, 1.0))))

    def _calibrate(self, job, cid):
        """Job: the edge fit of one camera, stored as its proposal (cam['proposal']; see setup_accept)."""
        ws = self._need_ws()
        cam = ws.camera(cid)
        s = self._session({'camera': cid})
        hfov0 = cam.get('hfov0') or DEFAULT_HFOV
        job.say('Calibrating %s' % cam['name'])
        pano, frames, landmarks = self._edge_frames(job, cam, s)
        usable = [fr for fr in pano if fr.bar and not fr.note]
        c_old = cam['calibration'].get('c')
        wide = False
        if usable:
            job.say('Starting point: the skyline fit of the panorama (%d frames)...' % len(usable))
            c_start, _ = panorama.calibrate(pano, s.cam, [], hfov0=hfov0, verbose=False, dem=s.model.dem)
            start = 'panorama skyline fit'
        elif c_old:
            c_start, start = c_old, 'previous calibration'
        else:
            c_start, start, wide = [0, 0, 0, hfov0, 0, 0], 'raw bar values', True
        ids = {fr.id for fr in frames}
        fit = edgefit.EdgeFit(frames, s.cam, s.model.dem, landmarks)
        if not usable:
            fit.landmark_weight = edgefit.STILLS_LANDMARK_WEIGHT
        if not usable and fit.landmarks:          # no panorama: the landmarks alone give the starting point
            c_lm, fitted = fit.landmark_start(c_start, log=job.say)
            if fitted:
                c_start, start, wide = c_lm, 'landmark fit (%s; from %s)' % (', '.join(fitted), start), False
        spread = panorama._pan_spread(usable) if len(usable) > 1 else 0.0
        if len(usable) >= 2 and spread >= panorama.LEAN_MIN_SPREAD:
            free = range(6)
        else:
            free = [0, 1, 2, 3] if len([l for l in landmarks if l.frame in ids]) >= 3 else [0, 1, 2]
        c, offsets, rep = fit.run(c_start, free, wide, log=job.say)
        overlay = 'calibration/%s_proposal_overlay.jpg' % cid
        edgefit.overlay(fit, core.full_correction(c_old if c_old else c_start),
                        cam['calibration'].get('pose_offsets') if c_old else None, c, ws.path(*overlay.split('/')))
        rep.update(when=now(), start=start, overlay=overlay, c_previous=list(core.full_correction(c_old)) if c_old else None,
                   horizon_km=self._horizon_km(cam))
        if np.isfinite(rep['worst_still_offset_deg']) and rep['worst_still_offset_deg'] > 0.3:
            job.warn('%s: one still is %.2f deg off its silhouette after the calibration. Smoke or haze along the skyline '
                     'often does that to a single still; check it, shorten the horizon distance or add a landmark on that '
                     'camera position.' % (cam['name'], rep['worst_still_offset_deg']))
        for l in rep['landmarks']:
            if l['error_deg'] is not None and l['error_deg'] > 0.2:
                job.warn('%s: landmark %s is %.2f deg (%.0f px) from where the calibration puts it; check the click and '
                         'its coordinates.' % (cam['name'], l['name'], l['error_deg'], l['error_px']))
        n_lm = rep['n_landmarks']
        with self.lock:
            cam['proposal'] = _safe({'c': list(c), 'pose_offsets': offsets, 'report': rep, 'adjusted': False,
                                     'auto': {'c': list(c), 'pose_offsets': offsets}, 'from': 'edge fit', 'when': now(),
                                     'dem': s.dem_label,
                                     'source': 'edge fit (%d frames%s)' % (rep['n_frames'], ', %d landmarks' % n_lm if n_lm else '')})
            ws.save()
        job.say('Proposed; check it, then accept it.')
        return {'camera': cid, 'proposal': cam['proposal']}

    def _accept(self, job, cid):
        """Job: apply the proposal (setup_accept)."""
        ws = self._need_ws()
        cam = ws.camera(cid)
        s = self._session({'camera': cid})
        prop = json.loads(json.dumps(self._proposal(cam)))
        c, po = core.full_correction(prop['c']), prop.get('pose_offsets') or {}
        rep = json.loads(json.dumps(prop.get('report') or {}))
        cal_old = cam['calibration']
        c_old = core.full_correction(cal_old['c'] if cal_old.get('c') else [0, 0, 0, cam.get('hfov0') or DEFAULT_HFOV, 0, 0])
        overlay = 'calibration/%s_calibration_overlay.jpg' % cid
        auto_overlay = rep.get('overlay') if prop.get('from') == 'edge fit' else None
        if prop.get('adjusted') or not auto_overlay or not os.path.isfile(ws.path(*auto_overlay.split('/'))):
            job.say('Measuring the edges with the %s calibration...' % ('adjusted' if prop.get('adjusted') else 'proposed'))
            _, frames, landmarks = self._edge_frames(job, cam, s)
            rows = {r['id']: r for r in rep.get('frames') or [] if r.get('id')} if rep.get('method') == 'edges' else {}
            if rows:                               # the frames the fit used, and its notes
                frames = [fr for fr in frames if fr.id in rows]
                for fr in frames:
                    fr.note = rows[fr.id].get('note') or ''
            for fr in frames:
                if fr.kind == 'panorama':          # its own pan/tilt error, from the fit
                    off = (rows.get(fr.id) or {}).get('offset')
                    fr.off = [float(x) for x in off[:2]] if off else [0.0, 0.0]
                else:
                    o = po.get(core.pose_key(fr.bar)) or {}
                    fr.off = [o.get('pan') or 0.0, o.get('tilt') or 0.0]
                    fr.turn = (o.get('roll') or 0.0, o.get('scale') or 1.0)
            fit = edgefit.EdgeFit(frames, s.cam, s.model.dem, landmarks)
            out = []
            for fr in frames:
                row = dict(rows.get(fr.id) or dict(id=fr.id, label=fr.label, kind=fr.kind, bar=list(fr.bar)))
                row.update(offset=list(fr.off), used=not fr.note, note=fr.note)
                if not fr.note:
                    best, _, _, dxd, dyd = fit.shift_search(fr, c, measure=True)
                    if 'shift_after_deg' in row:
                        row['shift_auto_deg'] = row['shift_after_deg']
                    row.update(shift_after_deg=[dxd, dyd], contrast_after=fit.contrast(fr, c), best_contrast=best)
                    job.say('  %-40s remaining offset %+.2f deg right, %+.2f deg down' % (fr.label, dxd, dyd))
                out.append(row)
            rep['frames'] = out
            rep['landmarks'] = [dict(name=lm.name, frame=lm.frame, error_deg=e, error_px=px, predicted=[u, v])
                                for lm, (e, px, u, v) in zip(fit.landmarks, fit.landmark_errors(c))]
            rep['n_landmarks'] = len(rep['landmarks'])
            rep['n_frames'] = len([r for r in out if r['used']])
            st = [r for r in out if r['used'] and r['kind'] == 'still']
            rep['worst_still_offset_deg'] = max((max(abs(x) for x in r['shift_after_deg']) for r in st), default=np.nan)
            edgefit.overlay(fit, c_old, cal_old.get('pose_offsets') if cal_old.get('c') else None, c, ws.path(*overlay.split('/')))
            rep['method'] = 'edges'
        else:
            shutil.copyfile(ws.path(*auto_overlay.split('/')), ws.path(*overlay.split('/')))
        if prop.get('adjusted'):
            rep['manual'] = {'c_auto': prop['auto']['c'], 'c': list(c), 'when': now(),
                             'positions': {k: {x: o.get(x) for x in ('pan', 'tilt', 'roll', 'scale')}
                                           for k, o in po.items() if o.get('manual')}}
        mag, toward = core.lean_description(c)
        rep.update(method=rep.get('method') or 'edges', c=list(c), pose_offsets=po, lean_deg=mag, lean_toward_deg=toward,
                   overlay=overlay, accepted=now(), c_previous=list(c_old) if cal_old.get('c') else None)
        rep.setdefault('c_start', list(c_old))
        rep.setdefault('frames', [])
        rep.setdefault('landmarks', [])
        rep['mat'] = self._write_mat(cam, s, c, po, rep)
        source = prop['source'] + (' + manual adjustment' if prop.get('adjusted') else '')
        with self.lock:
            if _safe(cam.get('proposal')) != _safe(prop):
                raise ValueError('The proposal was changed while it was being accepted; accept it again.')
            cam['calibration'] = _safe({'c': list(c), 'pose_offsets': po, 'note': '', 'report': rep, 'source': source,
                                        'dem': s.dem_label if prop.get('adjusted') else (prop.get('dem') or s.dem_label)})
            cam['rev'] = cam.get('rev', 0) + 1
            cam.pop('proposal', None)
            ws.save()
        job.say('Accepted: %s' % source)
        return {'camera': cid, 'calibration': cam['calibration'], 'rev': cam['rev']}

    def _write_mat(self, cam, s, c, po, rep):
        """The camera's calibration as a .mat (core.save_calibration), with the pose of its first calibration still."""
        lm = cam['landmarks']
        stills = [r['id'] for r in rep.get('frames') or [] if r.get('kind') == 'still' and r['id'] in s.bars]
        first = stills[0] if stills else s.stills[0]['file']
        mat = 'calibration/%s_calibration.mat' % cam['id']
        core.save_calibration(self.ws.path(*mat.split('/')), s.cam, s.pose(first, c, po), _arr(lm['gcp'], 5), lm['names'],
                              np.zeros((0, 2)), [], (s.ny, s.nx), first, correction=c, landmark_stills=lm['stills'],
                              extra_fields={'calibration_method': rep.get('method') or 'edges',
                                            'calibration_frames': core._cellstr([r['id'] for r in rep.get('frames') or [] if r.get('used')]),
                                            'pose_offsets_json': json.dumps(_safe(po))})
        return mat

    # ------------------------------------------------- app: registration --
    def optimize(self, req):
        return self._session(req).optimize(req)

    def autosky(self, req):
        return self._session(req).autosky(req)

    def map(self, req):
        return self._session(req).map(req)

    def resample(self, req):
        out = core.resample_polyline(_arr(req.get('pts'), 2), float(req.get('spacing', 8)), bool(req.get('exact', False)))
        return {'uv': [] if out is None else out.tolist()}

    def elevation(self, req):
        s = self._session(req)
        return {'z': s.model.dem(np.asarray(req['lat'], float), np.asarray(req['lon'], float))}

    def calib_list(self, req):
        """.mat calibrations whose camera site is this camera's: in the workspace and in the project's output/."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        out = []
        for where, d in (('workspace', ws.path('calibration')), ('workspace', ws.path('exports')),
                         ('output', os.path.join(self.project_root, 'output'))):
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith('.mat'):
                    continue
                try:
                    site = core.load_calibration(os.path.join(d, f))['site']
                except Exception:
                    continue
                if site is not None and _site_m(site, (cam['lat'], cam['lon'])) <= SAME_SITE_M:
                    rel = os.path.relpath(os.path.join(d, f), ws.dir if where == 'workspace' else d)
                    out.append({'key': '%s:%s' % (where, rel), 'label': '%s/%s' % (where, rel)})
        return {'files': out}

    def _calib_path(self, key):
        """'workspace:calibration/x.mat' (in the workspace) or 'output:x.mat' (the project's output/)."""
        where, _, rel = str(key).partition(':')
        if where not in ('workspace', 'output'):
            raise ValueError('Unknown calibration %r' % key)
        base = self.ws.dir if where == 'workspace' else os.path.join(self.project_root, 'output')
        parts = rel.replace('\\', '/').split('/')
        return os.path.join(base, *[safe_file(p) for p in parts])

    def calib_import(self, req):
        """Landmarks, skyline trace, edges and mount correction of a .mat calibration of this camera."""
        ws = self._need_ws()
        cam = ws.camera(req['camera'])
        s = self._session(req)
        c = core.load_calibration(self._calib_path(req['key']))
        if c['site'] is not None and _site_m(c['site'], s.cam) > SAME_SITE_M:
            raise ValueError('That calibration is for a camera %.1f km from %s.' % (_site_m(c['site'], s.cam) / 1000,
                                                                                 cam['name']))
        image = os.path.basename(c['image_file'])
        note = ''
        if c['correction'] is not None:
            corr = core.full_correction(c['correction'])
        elif image in s.bars:            # MATLAB file: recover the correction from its pose on its image
            corr = core.correction_from_pose(c['p'], s.bars[image])
            note = 'mount correction derived from the saved pose on %s' % image
        else:
            corr = s.c0
            note = 'saved image %r is not one of the stills; pose ignored' % image
        n = len(c['gcp'])
        stills = c['landmark_stills'] if len(c['landmark_stills']) == n else [image] * n
        known = lambda f: f if f in s.bars else ''
        return {'name': os.path.basename(self._calib_path(req['key'])), 'c': corr.tolist(), 'note': note,
                'image_file': known(image), 'gcp': c['gcp'].reshape(-1, 5), 'gcp_stills': [known(f) for f in stills],
                'names': c['names'], 'sky_obs': c['sky_obs'].reshape(-1, 2), 'sky_still': known(c['skyline_still'] or image),
                'edges': [{'uv': e['uv'], 'kind': e['kind'], 'still': e['still'] or image} for e in c['edges']
                          if (e['still'] or image) in s.bars]}

    def calib_export(self, req):
        """Write the camera's calibration (and its edges) as a .mat in the MATLAB script's layout."""
        ws = self._need_ws()
        s = self._session(req)
        name = safe_file(req['name'] if str(req['name']).lower().endswith('.mat') else str(req['name']) + '.mat')
        image = req.get('image_file') or s.stills[0]['file']
        edges = [self._edge(e) for e in req.get('edges', [])]
        po = req.get('po')
        for e in edges:
            e['geo'] = s.model.map_pixels(e['uv'], s.pose(e['still'], req['c'], po))
        path = ws.path('exports', name)
        core.save_calibration(path, s.cam, s.pose(image, req['c'], po), _arr(req.get('gcp'), 5), req.get('names', []),
                              _arr(req.get('sky_obs'), 2), edges, (s.ny, s.nx), image, correction=req['c'],
                              landmark_stills=req.get('gcp_stills', []), skyline_still=req.get('sky_still', ''))
        return {'saved': path}

    # ------------------------------------------------------ app: combine --
    def _edge(self, e):
        return {'uv': _arr(e['uv'], 2), 'geo': _arr(e.get('geo'), 6), 'kind': e.get('kind', ''),
                'still': e.get('still', '')}

    @staticmethod
    def _max_age(req):
        m = req.get('max_age_min')
        return float(m) * 60 if m not in (None, '') else None

    def combine(self, req):
        """Preview: the composite (observed parts only) at time `at`, from the edges' current terrain points."""
        lines = []
        for cid, edges in (req.get('cameras') or {}).items():
            if cid in self.sessions and edges:            # the page sends each edge's still and terrain points
                lines += self.sessions[cid].lines(edges, [_arr(e.get('geo'), 6) for e in edges])
        lines = [l for l in lines if len(l.pts) >= 2]
        if not lines:
            return {'composite': None, 'centre': None}
        centre = tuple(req['centre']) if req.get('centre') else combine.default_centre(lines)
        comp = combine.composite(lines, float(req['at']), centre, self._max_age(req))
        if comp:
            comp['runs'] = [{'camera': r['camera'], 'still': r['still'], 't': r['t'], 'pts': r['pts'][:, :2]}
                            for r in comp['runs']]
        return {'composite': comp, 'centre': list(centre)}

    def export_all(self, req):
        """Per-camera CSV + KML (core.export_edges) and the combined perimeter (combine.write_combined)."""
        ws = self._need_ws()
        base = safe_file(req.get('name') or ws.data.get('export_name') or ws.slug)
        base = base[:-4] if base.lower().endswith(('.kml', '.csv')) else base
        utc = ws.data.get('utc_offset', '')
        stamps = bool(req.get('time_stamps', True))
        close = bool(req.get('close'))
        names = {c['id']: c['name'] for c in ws.cameras}
        written, lines = [], []
        with self.lock:
            for cid, st in (req.get('cameras') or {}).items():
                edges = [self._edge(e) for e in st.get('edges', [])]
                if not edges or cid not in self.sessions:
                    continue
                s = self.sessions[cid]
                # Recompute using the CURRENT mount correction and each edge's own frame.
                poses = [s.pose(e['still'], st['c'], st.get('po')) for e in edges]
                p = ws.path('exports', '%s_%s' % (base, cid))
                core.export_edges(edges, s.model, poses, p, '%s fire edges' % names[cid],
                                  times=[s.times.get(e['still'], '') for e in edges], utc_offset=utc,
                                  time_stamps=stamps)
                written += [p + '.csv', p + '.kml']
                lines += s.lines(edges, [s.model.map_pixels(e['uv'], q) for e, q in zip(edges, poses)])
            if not written:
                return {'error': 'Draw at least one edge first.'}
            comps, centre = combine.series(lines, req.get('centre'), self._max_age(req), close)
            p = ws.path('exports', base + '_combined')
            combine.write_combined(p, comps, names, utc, close, '%s combined perimeter' % ws.data['name'])
            written += [p + '.csv', p + '.kml']
        over = [dict(o, t=c['t']) for c in comps for o in c['overlap']]
        return {'saved': written, 'n_composites': len(comps), 'centre': centre, 'overlap': over,
                'closed': close, 'zone': utc or 'camera clock (no UTC offset set for this workspace)'}


ROUTES = {
    # portal
    '/api/workspaces': 'workspaces', '/api/workspace/new': 'workspace_new', '/api/workspace/open': 'workspace_open',
    '/api/workspace': 'workspace', '/api/workspace/save': 'save',
    '/api/camera/add': 'camera_add', '/api/camera/update': 'camera_update', '/api/camera/remove': 'camera_remove',
    '/api/panorama/upload': 'panorama_upload', '/api/panorama/remove': 'panorama_remove', '/api/folders': 'folders',
    '/api/ingest': 'ingest', '/api/jobs': 'jobs_status',
    # app
    '/api/config': 'config', '/api/camera/config': 'camera_config', '/api/camera/reload': 'reload',
    '/api/panorama/run': 'panorama_run', '/api/optimize': 'optimize', '/api/autosky': 'autosky', '/api/map': 'map',
    '/api/resample': 'resample', '/api/elevation': 'elevation', '/api/calib/list': 'calib_list',
    '/api/calib/import': 'calib_import', '/api/calib/export': 'calib_export', '/api/combine': 'combine',
    '/api/export_all': 'export_all',
    # calibration step
    '/api/setup/frames': 'setup_frames', '/api/setup/landmarks': 'setup_landmarks', '/api/setup/predict': 'setup_predict',
    '/api/setup/summit': 'setup_summit', '/api/calibrate': 'calibrate', '/api/setup/horizon': 'setup_horizon',
    '/api/setup/proposal': 'setup_proposal', '/api/setup/accept': 'setup_accept',
    # reference points, site check, DEM
    '/api/reference/fetch': 'reference_fetch', '/api/reference/candidates': 'reference_candidates',
    '/api/site/check': 'site_check', '/api/dem/status': 'dem_status', '/api/dem/set': 'dem_set', '/api/dem/fetch': 'dem_fetch',
}
GET_ROUTES = ('/api/workspaces', '/api/workspace', '/api/config', '/api/jobs')


def make_handler(portal: Portal):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not self.path.startswith(('/still/', '/static/', '/ws/', '/api/jobs')):
                super().log_message(fmt, *args)

        def _send(self, code, body, ctype='application/json', cache='no-store'):
            if isinstance(body, (dict, list)):
                body = json.dumps(_safe(body), allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', cache)
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path, cache='no-cache'):
            # Page code is revalidated on every load ('no-cache'), so an updated app.js is never run
            # from the browser cache against a newer server; the stills never change and are cached.
            if not os.path.isfile(path):
                return self._send(404, {'error': 'not found'})
            with open(path, 'rb') as f:
                self._send(200, f.read(), mimetypes.guess_type(path)[0] or 'application/octet-stream', cache)

        def do_GET(self):
            url = urlparse(self.path)
            path = unquote(url.path)
            if path in ('/', '/index.html', '/portal'):
                return self._file(os.path.join(WEB_DIR, 'portal.html'))
            if path == '/app':
                return self._file(os.path.join(WEB_DIR, 'index.html'))
            if path == '/setup':
                return self._file(os.path.join(WEB_DIR, 'setup.html'))
            if path.startswith('/static/'):
                return self._file(os.path.join(WEB_DIR, os.path.basename(path)))
            if path.startswith('/still/'):
                parts = path.split('/')            # '', 'still', camera, file
                s = portal.sessions.get(parts[2]) if len(parts) == 4 else None
                if s is None or parts[3] not in s.bars:
                    return self._send(404, {'error': 'unknown still'})
                return self._file(os.path.join(s.stills_dir, parts[3]), cache='max-age=3600')
            if path.startswith('/ws/') and portal.ws:
                parts = path.split('/')            # '', 'ws', panoramas|calibration, file  or  '', 'ws', 'frames', camera, file
                try:
                    if len(parts) == 4 and parts[2] in ('panoramas', 'calibration'):
                        return self._file(portal.ws.path(parts[2], safe_file(parts[3])))
                    if len(parts) == 5 and parts[2] == 'frames':
                        return self._file(portal.ws.path('calibration', 'frames', safe_file(parts[3]), safe_file(parts[4])),
                                          cache='max-age=3600')
                except ValueError:
                    pass
                return self._send(404, {'error': 'not found'})
            if path in GET_ROUTES:
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if 'ids' in q:
                    q['ids'] = q['ids'].split(',')
                return self._api(path, q)
            self._send(404, {'error': 'not found'})

        def do_POST(self):
            path = urlparse(self.path).path
            # JSON-only POSTs: cross-site pages cannot send these without a CORS preflight.
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self._send(415, {'error': 'JSON required'})
            n = int(self.headers.get('Content-Length', 0))
            try:
                req = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self._send(400, {'error': 'bad JSON'})
            self._api(path, req)

        def _api(self, path, req):
            if path not in ROUTES:
                return self._send(404, {'error': 'not found'})
            try:
                with portal.lock if path in ('/api/calib/export',) else contextlib.nullcontext():
                    self._send(200, getattr(portal, ROUTES[path])(req))
            except (ValueError, KeyError, OSError, TypeError) as err:
                self._send(400, {'error': '%s: %s' % (type(err).__name__, err)})
    return Handler


def serve(project_root, ws_root, dem_dir, port=8790, open_browser=True, host='127.0.0.1'):
    portal = Portal(project_root, ws_root, dem_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(portal))
    # 127.0.0.1, not localhost: macOS resolves localhost to ::1 first, where another program may hold the port
    url = 'http://%s:%d/' % ('127.0.0.1' if host in ('127.0.0.1', '0.0.0.0') else host, port)
    print('Serving %s  (workspaces in %s; Ctrl-C to stop)' % (url, portal.ws_root))
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        httpd.server_close()
