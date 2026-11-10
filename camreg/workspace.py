"""Fire workspaces: the cameras of one fire with their stills, panoramas, calibrations and drawn edges.

A workspace is a folder:

    workspace.json   settings and, per camera: site, stills folder, bar details, panoramas, calibration
                     (mount correction and where it came from), landmarks, skyline trace, fire edges
    panoramas/       uploaded panorama images
    reference/       reference points around each camera (landmark candidates; reference.py), <camera>.json
    calibration/     panorama calibration results (<camera>_panorama.mat and its overlay image)
    exports/         CSV/KML exports and exported .mat calibrations

Stills are not copied: each camera points at its stills folder, where camstills keeps stills_index.csv.
Paths inside the project folder are stored relative to it, so a workspace still opens after the project
folder is moved or opened from another machine.
"""
import json
import os
import re
import time

SCHEMA = 1
SAFE_FILE = re.compile(r'^[\w.\- ()]+$')
DEFAULT_DEM = {'source': 'srtm', 'radius_km': 25.0}   # 'srtm' (30 m tiles) or '3dep10' (USGS 3DEP 10 m near the cameras, SRTM beyond)
DEFAULT_HORIZON_KM = 90.0    # how far out a camera's skyline and rays follow the terrain (calibration step)
DEFAULT_HFOV = 60.0          # starting Z1 field of view (the cameras' nominal 60 deg); the panorama fit or Optimize refines it


def slug(name):
    """'Hawk Fire 2026' -> 'hawk-fire-2026'."""
    return re.sub(r'[^a-z0-9]+', '-', str(name).lower()).strip('-') or 'item'


def safe_file(name):
    """A bare file name (letters, digits, - _ . space and parentheses), or ValueError."""
    name = os.path.basename(str(name).strip())
    if not name or name.startswith('.') or not SAFE_FILE.match(name):
        raise ValueError('Use a plain file name (letters, digits, - _ . space): %r' % name)
    return name


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def new_camera(cid, name, lat, lon, elev_m, stills_dir):
    return {'id': cid, 'name': name, 'lat': float(lat), 'lon': float(lon), 'elev_m': float(elev_m),
            'stills_dir': stills_dir, 'glyphs': '', 'bar_px': None, 'image_size': None, 'hfov0': DEFAULT_HFOV,
            'panoramas': [], 'index': None, 'tiles': [], 'tiles_missing': [], 'horizon_km': DEFAULT_HORIZON_KM,
            'calibration': {'c': None, 'source': 'none', 'note': '', 'report': None},
            'landmarks': {'gcp': [], 'stills': [], 'names': [], 'info': []}, 'skyline': {'uv': [], 'still': ''},
            'edges': [], 'ui': {}}


class Workspace:
    def __init__(self, root, project_root, data):
        self.root = root                      # folder of all workspaces
        self.project_root = project_root      # relative paths resolve against this folder
        self.data = data

    # ----------------------------------------------------------- files --
    @property
    def slug(self):
        return self.data['slug']

    @property
    def dem(self):
        """DEM settings (DEFAULT_DEM for workspaces made before they existed)."""
        return dict(DEFAULT_DEM, **(self.data.get('dem') or {}))

    @property
    def dir(self):
        return os.path.join(self.root, self.slug)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def resolve(self, p):
        """Absolute path of a stored path (relative ones are relative to the project folder)."""
        p = os.path.expanduser(str(p))
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(self.project_root, p))

    def store(self, p):
        """Path to store: relative to the project folder when inside it."""
        p = os.path.abspath(os.path.expanduser(str(p)))
        rel = os.path.relpath(p, self.project_root)
        return p if rel.startswith('..') else rel

    @classmethod
    def create(cls, root, project_root, name, utc_offset='', dem_dir='.'):
        name = str(name).strip()
        if not name:
            raise ValueError('Give the workspace a name.')
        s, k = slug(name), 1
        while os.path.exists(os.path.join(root, s if k == 1 else '%s-%d' % (s, k))):
            k += 1
        s = s if k == 1 else '%s-%d' % (s, k)
        ws = cls(root, project_root, {'schema': SCHEMA, 'slug': s, 'name': name, 'created': now(), 'modified': now(),
                                      'utc_offset': utc_offset.strip(), 'dem_dir': dem_dir, 'export_name': s,
                                      'space': 'calibration', 'active_camera': '',
                                      'combine': {'centre': None, 'max_age_min': None, 'close': True},
                                      'dem': dict(DEFAULT_DEM), 'cameras': []})
        for d in ('panoramas', 'calibration', 'exports'):
            os.makedirs(ws.path(d), exist_ok=True)
        ws.save()
        return ws

    @classmethod
    def load(cls, root, project_root, s):
        path = os.path.join(root, safe_file(s), 'workspace.json')
        with open(path) as f:
            data = json.load(f)
        data['slug'] = s
        ws = cls(root, project_root, data)
        for d in ('panoramas', 'calibration', 'exports'):
            os.makedirs(ws.path(d), exist_ok=True)
        return ws

    def save(self):
        self.data['modified'] = now()
        tmp = self.path('workspace.json.tmp')
        os.makedirs(self.dir, exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path('workspace.json'))

    # --------------------------------------------------------- cameras --
    @property
    def cameras(self):
        return self.data['cameras']

    def camera(self, cid):
        for c in self.cameras:
            if c['id'] == cid:
                return c
        raise ValueError('No camera %r in this workspace' % cid)

    def add_camera(self, name, lat, lon, elev_m, stills_dir):
        name = str(name).strip()
        if not name:
            raise ValueError('Give the camera a name.')
        base, k = slug(name), 1
        ids = {c['id'] for c in self.cameras}
        while (base if k == 1 else '%s-%d' % (base, k)) in ids:
            k += 1
        cam = new_camera(base if k == 1 else '%s-%d' % (base, k), name, lat, lon, elev_m, self.store(stills_dir))
        self.cameras.append(cam)
        return cam


def list_workspaces(root):
    """Summaries of the workspaces in root, newest first."""
    out = []
    if not os.path.isdir(root):
        return out
    for s in sorted(os.listdir(root)):
        p = os.path.join(root, s, 'workspace.json')
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        out.append({'slug': s, 'name': d.get('name', s), 'modified': d.get('modified', ''),
                    'cameras': [c.get('name', c.get('id')) for c in d.get('cameras', [])],
                    'edges': sum(len(c.get('edges', [])) for c in d.get('cameras', []))})
    return sorted(out, key=lambda w: w['modified'], reverse=True)
