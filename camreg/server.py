"""Local web backend for camera registration and fire-edge mapping.

Serves the HTML/JS front end (camreg/web/) and a small JSON API over
camreg.core. The browser keeps the session state (pose, landmarks, skyline,
edges) and does the cheap projections itself; this server does the DEM work:
optimization, pixel-to-terrain mapping, elevation lookups and file I/O.

The camera pose on each still is its info-bar pan/tilt/zoom plus one fixed
mount correction (core.pose_from_bar; 6 values including the pan-axis lean);
there is no free pose.

Listens on 127.0.0.1 only. Files are read and written only inside out_dir,
by bare file name.
"""
import contextlib
import csv
import json
import math
import mimetypes
import os
import re
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from . import core, skyline

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
SAFE_NAME = re.compile(r'^[\w.\- ()]+$')


@dataclass
class Config:
    cam: tuple                  # lat, lon, elevation_m
    correction0: tuple          # starting mount correction: d_azimuth, d_tilt, roll, hfov at Z1[, lean_east, lean_north]
    correction_lim: tuple       # 4 or 6 x (min, max) for the correction (lean defaults to +-10 deg)
    tiles: tuple                # .hgt paths
    stills_dir: str
    stills_index: str           # CSV: file,timestamp,pan_X_deg,tilt_Y_deg,zoom_Z,pose_id,...
    out_dir: str
    bar_px: int = 0
    default_still: str = ''
    initial_calibration: str = ''   # file name in out_dir loaded at start, optional
    title: str = 'Camera registration and fire-edge mapping'
    kml_name: str = 'Fire edges'
    calib_name: str = 'registration.mat'
    export_name: str = 'fire_edges'
    utc_offset: str = ''        # time zone of the bar times for the KML time stamps, e.g. '-07:00'


def _clean(a):
    """numpy -> JSON-safe nested lists, NaN/inf -> None."""
    a = np.asarray(a, float)
    return [[None if not math.isfinite(x) else x for x in row] for row in np.atleast_2d(a).tolist()] \
        if a.ndim > 1 else [None if not math.isfinite(x) else x for x in a.tolist()]


def _arr(rows, ncol):
    a = np.array([[np.nan if x is None else x for x in r] for r in (rows or [])], float)
    return a.reshape(-1, ncol)


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(cfg.stills_index) as f:
            self.stills = [{'file': r['file'], 'time': r['timestamp'], 'X': float(r['pan_X_deg']),
                            'Y': float(r['tilt_Y_deg']), 'Z': float(r['zoom_Z']), 'pose': r['pose_id']}
                           for r in csv.DictReader(f)]
        self.bars = {s['file']: (s['X'], s['Y'], s['Z']) for s in self.stills}
        self.times = {s['file']: s['time'] for s in self.stills}
        w, h = Image.open(os.path.join(cfg.stills_dir, self.stills[0]['file'])).size
        self.nx, self.ny = w, h - cfg.bar_px
        self.lim = core.full_limits(cfg.correction_lim)
        self.c0 = core.full_correction(cfg.correction0)
        self.model = core.CameraModel(cfg.cam, core.DEM(list(cfg.tiles)), self.nx, self.ny)
        # Skyline for every azimuth: the pan comes from each frame's bar and can point anywhere.
        self.sky_rays = self.model.horizon_rays(0, 360, 0.1)
        self.lock = threading.Lock()

    def pose(self, still, c):
        """Model pose on a still: its bar X/Y/Z plus mount correction c."""
        if still not in self.bars:
            raise ValueError('Unknown still: %s' % still)
        return core.pose_from_bar(self.bars[still], core.full_correction(c))

    # ---------------------------------------------------------------- API --
    def config(self, _):
        c = self.cfg
        initial = None
        if c.initial_calibration and os.path.isfile(self._path(c.initial_calibration)):
            initial = self.load({'name': c.initial_calibration})
        return {'title': c.title, 'cam': list(c.cam), 'c0': self.c0.tolist(), 'lim': self.lim.tolist(),
                'nx': self.nx, 'ny': self.ny, 'bar_px': c.bar_px, 'stills': self.stills,
                'default_still': c.default_still or self.stills[0]['file'],
                'sky_rays': np.round(self.sky_rays, 7).tolist(), 'calib_name': c.calib_name,
                'export_name': c.export_name, 'files': self.files(None)['files'],
                'initial': initial}

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
                                          fit_lean=bool(req.get('fit_lean')))
        return {'c': c.tolist(), 'score': float(res.fun), 'nit': int(res.nit),
                'nfev': int(res.nfev), 'message': str(res.message)}

    def autosky(self, req):
        """Detect the skyline on a still (skyline.py); compared with the current model as a haze check."""
        still = req.get('still')
        if still not in self.bars:
            raise ValueError('Unknown still: %s' % still)
        img = np.asarray(Image.open(os.path.join(self.cfg.stills_dir, still)).convert('RGB'))[:self.ny, :self.nx]
        uv = skyline.skyline_points(img, spacing=int(req.get('spacing', 8)))
        if len(uv) < 10:
            return {'error': 'No clear skyline found on this still (too little open sky above the terrain).'}
        u, v, front = self.model.project(self.sky_rays, self.pose(still, req['c']))
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
        c = req['c']
        return {'geo': [_clean(self.model.map_pixels(_arr(it['uv'], 2), self.pose(it['still'], c)))
                        for it in req.get('items', [])]}

    def resample(self, req):
        pts = _arr(req.get('pts'), 2)
        out = core.resample_polyline(pts, float(req.get('spacing', 8)), bool(req.get('exact', False)))
        return {'uv': [] if out is None else out.tolist()}

    def elevation(self, req):
        return {'z': _clean(self.model.dem(np.asarray(req['lat'], float), np.asarray(req['lon'], float)))}

    def files(self, _):
        names = sorted(f for f in os.listdir(self.cfg.out_dir)
                       if f.lower().endswith(('.mat', '.kml', '.csv')))
        return {'files': names, 'out_dir': self.cfg.out_dir}

    def save(self, req):
        name = self._name(req['name'], '.mat')
        image = req.get('image_file', '')
        edges = [self._edge(e) for e in req.get('edges', [])]
        core.save_calibration(self._path(name), self.cfg.cam, self.pose(image, req['c']),
                              _arr(req.get('gcp'), 5), req.get('names', []), _arr(req.get('sky_obs'), 2),
                              edges, (self.ny, self.nx), image, correction=req['c'],
                              landmark_stills=req.get('gcp_stills', []),
                              skyline_still=req.get('sky_still', ''))
        return {'saved': self._path(name), 'files': self.files(None)['files']}

    def load(self, req):
        name = self._name(req['name'], '.mat')
        c = core.load_calibration(self._path(name))
        image = os.path.basename(c['image_file'])
        note = ''
        if c['correction'] is not None:
            corr = core.full_correction(c['correction'])
        elif image in self.bars:     # MATLAB file: recover the correction from its pose on its image
            corr = core.correction_from_pose(c['p'], self.bars[image])
            note = 'mount correction derived from the saved pose on %s' % image
        else:
            corr = self.c0
            note = 'saved image %r is not a known still; pose ignored, landmarks/skyline assigned to it' % image
        n = len(c['gcp'])
        stills = c['landmark_stills'] if len(c['landmark_stills']) == n else [image] * n
        return {'name': name, 'c': corr.tolist(), 'note': note, 'image_file': image,
                'gcp': _clean(c['gcp'].reshape(-1, 5)), 'gcp_stills': stills, 'names': c['names'],
                'sky_obs': _clean(c['sky_obs'].reshape(-1, 2)), 'sky_still': c['skyline_still'] or image,
                'edges': [{'uv': _clean(e['uv']), 'geo': _clean(e['geo']), 'kind': e['kind'],
                           'still': e['still'] or image} for e in c['edges']]}

    def export(self, req):
        base = req['name']
        base = base[:-4] if base.lower().endswith(('.kml', '.csv')) else base
        base = self._name(base, '')
        edges = [self._edge(e) for e in req.get('edges', [])]
        if not edges:
            return {'error': 'Draw at least one edge first.'}
        # Recompute using the CURRENT mount correction and each edge's own frame.
        poses = [self.pose(e['still'], req['c']) for e in edges]
        stamps = bool(req.get('time_stamps', True))
        core.export_edges(edges, self.model, poses, self._path(base), self.cfg.kml_name,
                          times=[self.times.get(e['still'], '') for e in edges], utc_offset=self.cfg.utc_offset,
                          time_stamps=stamps)
        note = ('time-stamped for Google Earth' + ('' if self.cfg.utc_offset else
                ' (no UTC offset set for this camera, so the times carry no time zone)')) if stamps else 'no time stamps'
        return {'saved': [self._path(base) + '.csv', self._path(base) + '.kml'], 'note': note,
                'files': self.files(None)['files']}

    # ------------------------------------------------------------ helpers --
    def _edge(self, e):
        return {'uv': _arr(e['uv'], 2), 'geo': _arr(e.get('geo'), 6), 'kind': e.get('kind', ''),
                'still': e.get('still', '')}

    def _name(self, name, ext):
        name = os.path.basename(str(name).strip())
        if ext and not name.lower().endswith(ext):
            name += ext
        if not name or not SAFE_NAME.match(name) or name.startswith('.'):
            raise ValueError('Use a plain file name (letters, digits, - _ . space).')
        return name

    def _path(self, name):
        return os.path.join(self.cfg.out_dir, name)


ROUTES = {'/api/config': 'config', '/api/optimize': 'optimize', '/api/autosky': 'autosky', '/api/map': 'map',
          '/api/resample': 'resample', '/api/elevation': 'elevation', '/api/files': 'files',
          '/api/save': 'save', '/api/load': 'load', '/api/export': 'export'}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not self.path.startswith(('/still/', '/static/')):
                super().log_message(fmt, *args)

        def _send(self, code, body, ctype='application/json', cache='no-store'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
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
            path = unquote(urlparse(self.path).path)
            if path in ('/', '/index.html'):
                return self._file(os.path.join(WEB_DIR, 'index.html'))
            if path.startswith('/static/'):
                name = os.path.basename(path)
                return self._file(os.path.join(WEB_DIR, name))
            if path.startswith('/still/'):
                name = os.path.basename(path)
                if name not in app.bars:
                    return self._send(404, {'error': 'unknown still'})
                return self._file(os.path.join(app.cfg.stills_dir, name), cache='max-age=3600')
            if path in ('/api/config', '/api/files'):
                return self._api(path, {})
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
                with app.lock if path in ('/api/save', '/api/export') else contextlib.nullcontext():
                    self._send(200, getattr(app, ROUTES[path])(req))
            except (ValueError, KeyError, OSError, TypeError) as err:
                self._send(400, {'error': '%s: %s' % (type(err).__name__, err)})
    return Handler


def serve(cfg: Config, port=8765, open_browser=True, host='127.0.0.1'):
    print('Loading DEM and computing skyline...')
    app = App(cfg)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = 'http://%s:%d/' % ('localhost' if host in ('127.0.0.1', '0.0.0.0') else host, port)
    print('Serving %s  (files in %s; Ctrl-C to stop)' % (url, cfg.out_dir))
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        httpd.server_close()
