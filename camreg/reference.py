"""Reference points with known coordinates around a camera: landmark candidates and a check of its site.

Sources (all public, no account or key):
  * NOAA NGS survey marks (Data Explorer API, https://geodesy.noaa.gov/api/nde/bounds): triangulation
    stations and intersection stations (lookouts, towers, peaks, beacons) with adjusted horizontal positions,
    a few metres or better. Bench marks along roads (heights only, mostly not visible) are left out.
  * OpenStreetMap (Overpass API): peaks, towers and masts. Positions are mapped, not surveyed (tens of
    metres); a peak's elevation is its DEM summit.
  * FAA Digital Obstacle File (DAILY_DOF_CSV.ZIP): towers and antennas with their top height above sea level
    and a horizontal accuracy class (6 m to 150 m).

    pts = fetch(box, cache_dir, log)                  # [{id, source, name, kind, lat, lon, elev, ...}, ...]
    cands = candidates(model, pts)                    # visible from the camera, with az/el, range, accuracy

A point's angular accuracy as seen from the camera is atan(acc_m / range); edgefit weighs landmarks by it.
Nothing here is specific to one camera or fire.
"""
import csv
import io
import json
import math
import os
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

from . import core

NGS_BOUNDS = 'https://geodesy.noaa.gov/api/nde/bounds'
NGS_RADIAL = 'https://geodesy.noaa.gov/api/nde/radial'
NGS_LIMIT = 500                     # the API returns at most this many marks; a full answer means "split the box"
OVERPASS = 'https://overpass-api.de/api/interpreter'
FAA_DOF = 'https://aeronav.faa.gov/Obst_Data/DAILY_DOF_CSV.ZIP'
FAA_MAX_AGE_DAYS = 30
TIMEOUT_S = 120
UA = {'User-Agent': 'fire-camera/1 (camera calibration landmarks)'}
K_M_PER_DEG = 111195.0
FT = 0.3048

MIN_RANGE_M, MAX_RANGE_M = 500.0, 40000.0
VISIBLE_TOL_DEG = 0.02              # terrain in front may rise this far above the point and it still counts as visible
SKYLINE_TOL_DEG = 0.05              # a point this close below the skyline counts as on it
NEAR_RADIUS_M = 300.0              # near_terrain looks this far around the camera
NEAR_BLOCK_DEG = -3.0              # ...and counts a direction as blocked when the ground rises above this angle
SUMMIT_RADIUS_M = 100.0             # an OSM peak is moved to the highest DEM point this close
OSM_PEAK_ACC_M, OSM_TOWER_ACC_M = 30.0, 15.0
FAA_ACC_M = {'1': 6.0, '2': 15.0, '3': 30.0, '4': 76.0, '5': 152.0}   # horizontal accuracy codes (+-20 ft .. +-500 ft)
FAA_TYPES = {'TOWER', 'ANTENNA', 'BLDG-TWR', 'NAVAID', 'WINDMILL', 'CTRL TWR', 'STACK', 'TANK', 'LGTHOUSE'}
NGS_GOOD_POS = {'ADJUSTED': 2.0, 'HD_HELD1': 3.0, 'HD_HELD2': 5.0}     # horizontal source -> about how good (m)
NGS_GOOD_HT = {'ADJUSTED', 'VERTCON3', 'VERTCON', 'GPS OBS', 'RESET', 'LEVELING', 'ADJ UNCH', 'POSTED'}
NGS_KIND = (('LOOKOUT', 'lookout'), ('TOWER', 'tower'), ('MAST', 'tower'), ('ANTENNA', 'tower'),
            ('BEACON', 'tower'), ('TANK', 'tank'), ('PEAK', 'peak'), ('CAIRN', 'peak'), ('TRIANGULATION', 'survey mark'))


# ------------------------------------------------------------------ fetching --

def _get(url, data=None, tries=3, log=print):
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return r.read()
        except Exception as e:
            err = e
            if k + 1 < tries:
                log('  %s: %s; retrying in %d s' % (urllib.parse.urlsplit(url).netloc, e, 4 * (k + 1)))
                time.sleep(4 * (k + 1))
    raise IOError('%s: %s' % (urllib.parse.urlsplit(url).netloc, err))


def _ngs_json(url, log):
    for k in range(3):                             # the service sometimes answers a quick repeat with an empty page
        body = _get(url, log=log)
        try:
            return json.loads(body)
        except ValueError:
            time.sleep(3 * (k + 1))
    raise IOError('NGS: no JSON answer from %s' % url)


def ngs_marks(box, log=print, depth=0):
    """All NGS marks in box (south, west, north, east), splitting the box while an answer is full."""
    s, w, n, e = box
    url = NGS_BOUNDS + '?' + urllib.parse.urlencode(dict(minlat='%.6f' % s, maxlat='%.6f' % n,
                                                         minlon='%.6f' % w, maxlon='%.6f' % e))
    marks = _ngs_json(url, log)
    if len(marks) < NGS_LIMIT or depth >= 6:
        return marks
    ml, mo = (s + n) / 2, (w + e) / 2
    out = {}
    for b in ((s, w, ml, mo), (s, mo, ml, e), (ml, w, n, mo), (ml, mo, n, e)):
        for m in ngs_marks(b, log, depth + 1):
            out[m['pid']] = m
    return list(out.values())


def ngs_near(lat, lon, radius_m, log=print):
    """Every NGS mark within radius_m (all kinds; for the site check)."""
    url = NGS_RADIAL + '?' + urllib.parse.urlencode(dict(lat='%.7f' % lat, lon='%.7f' % lon,
                                                         radius='%.0f' % radius_m, units='METER'))
    return _ngs_json(url, log)


def _f(x):
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def _ngs_point(m):
    pos = m.get('posSource', '').strip()
    typ = m.get('monumentType', '').strip()
    lat, lon = _f(m.get('lat')), _f(m.get('lon'))
    if pos not in NGS_GOOD_POS or lat is None or lon is None:
        return None
    code = typ.split('=')[0].strip()
    intersection = code[:1].isdigit()              # numeric codes: an object (lookout, tower, peak...) was observed
    if not (intersection or code == 'DS' or 'TRIANGULATION' in typ.upper() or 'CAIRN' in typ.upper()):
        return None
    label = typ.split('=', 1)[-1].strip().upper() + ' ' + m.get('name', '').upper()
    kind = next((k for word, k in NGS_KIND if word in label), 'survey mark')
    ht = _f(m.get('orthoHt')) if m.get('vertSource', '').strip() in NGS_GOOD_HT else None
    acc = _f(m.get('netAccHz')) or NGS_GOOD_POS[pos]
    return dict(id='ngs:' + m['pid'], source='NGS', ref=m['pid'], name=m.get('name', '').strip(), kind=kind,
                lat=lat, lon=lon, elev=ht, elev_kind='mark' if ht is not None else 'dem', acc_m=float(acc),
                note='%s; position %s; height %s; last recovered %s'
                     % (typ.split('=', 1)[-1].strip().lower() or 'survey mark', pos.lower(),
                        m.get('vertSource', '').strip().lower() or 'none', m.get('lastRecovered', '').strip() or '?'))


def osm_points(box, log=print):
    s, w, n, e = box
    bb = '%.5f,%.5f,%.5f,%.5f' % (s, w, n, e)
    q = ('[out:json][timeout:90];(node["natural"="peak"](%s);node["man_made"~"^(tower|mast|communications_tower)$"](%s););'
         'out body;' % (bb, bb))
    data = json.loads(_get(OVERPASS, urllib.parse.urlencode({'data': q}).encode(), log=log))
    out = []
    for el in data.get('elements', []):
        t = el.get('tags', {})
        peak = t.get('natural') == 'peak'
        if not peak and t.get('tower:type') in ('cooling', 'bell_tower', 'minaret'):
            continue
        h = _f(str(t.get('height', '')).replace('m', ''))
        kind = 'peak' if peak else ('lookout' if t.get('tower:type') in ('observation', 'lookout', 'fire_lookout')
                                    or 'lookout' in t.get('name', '').lower() else 'tower')
        out.append(dict(id='osm:%d' % el['id'], source='OSM', ref=str(el['id']),
                        name=t.get('name') or ('%s (unnamed)' % kind), kind=kind, lat=float(el['lat']),
                        lon=float(el['lon']), elev=None, elev_kind='summit' if peak else 'dem',
                        height_m=h if not peak else None, acc_m=OSM_PEAK_ACC_M if peak else OSM_TOWER_ACC_M,
                        note='OpenStreetMap %s%s' % (kind, ', ele tag %s m' % t['ele'] if t.get('ele') else '')))
    return out


def faa_points(box, cache_dir, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'DAILY_DOF_CSV.ZIP')
    if not os.path.isfile(path) or time.time() - os.path.getmtime(path) > FAA_MAX_AGE_DAYS * 86400:
        log('  downloading the FAA obstacle file (about 20 MB)...')
        body = _get(FAA_DOF, log=log)
        with open(path + '.part', 'wb') as f:
            f.write(body)
        os.replace(path + '.part', path)
    s, w, n, e = box
    out = []
    with zipfile.ZipFile(path) as z:
        name = next(x for x in z.namelist() if x.upper().endswith('.CSV'))
        for d in csv.DictReader(io.TextIOWrapper(z.open(name), encoding='latin-1')):
            lat, lon = _f(d.get('LATDEC')), _f(d.get('LONDEC'))
            if lat is None or lon is None or not (s <= lat <= n and w <= lon <= e):
                continue
            typ = d.get('TYPE', '').strip()
            acc = FAA_ACC_M.get(d.get('ACCURACY', '').strip()[:1])
            if typ not in FAA_TYPES or acc is None:
                continue
            amsl, agl = _f(d.get('AMSL')), _f(d.get('AGL'))
            out.append(dict(id='faa:' + d['OAS'].strip(), source='FAA', ref=d['OAS'].strip(),
                            name='%s %s (%s)' % (d.get('CITY', '').strip().title(), typ.lower(), d['OAS'].strip()),
                            kind='tower', lat=lat, lon=lon, elev=amsl * FT if amsl is not None else None,
                            elev_kind='top', height_m=agl * FT if agl is not None else None, acc_m=acc,
                            note='FAA obstacle %s, %s ft above ground, top %s ft; %s'
                                 % (typ.lower(), int(agl or 0), int(amsl or 0),
                                    'verified' if d.get('VERIFIED STATUS', '').strip() == 'O' else 'unverified')))
    return out


def _dedupe(points, radius_m=30.0):
    """Drop a point when a more accurate one of the same kind lies within radius_m."""
    keep = []
    for p in sorted(points, key=lambda p: p['acc_m']):
        c = math.cos(math.radians(p['lat']))
        if any(q['kind'] == p['kind'] and math.hypot((q['lat'] - p['lat']) * K_M_PER_DEG,
                                                     (q['lon'] - p['lon']) * K_M_PER_DEG * c) < radius_m
               for q in keep):
            continue
        keep.append(p)
    return keep


def fetch(box, cache_dir, log=print):
    """All sources for box (south, west, north, east). A source that fails is reported, not fatal.
    Returns {'box', 'fetched', 'points', 'sources': {name: {'n', 'error'}}}."""
    pts, sources = [], {}
    for name, f in (('NGS survey marks', lambda: [p for p in map(_ngs_point, ngs_marks(box, log)) if p]),
                    ('OpenStreetMap', lambda: osm_points(box, log)),
                    ('FAA obstacles', lambda: faa_points(box, cache_dir, log))):
        log('Reference points: %s...' % name)
        try:
            got = f()
            sources[name] = {'n': len(got), 'error': ''}
            pts += got
            log('  %d points' % len(got))
        except Exception as e:
            sources[name] = {'n': 0, 'error': str(e)}
            log('WARNING: %s not available (%s)' % (name, e))
    return {'box': list(box), 'fetched': time.strftime('%Y-%m-%d %H:%M:%S'), 'points': _dedupe(pts),
            'sources': sources}


def box_around(lat, lon, km=MAX_RANGE_M / 1000):
    d = km * 1000 / K_M_PER_DEG
    return (lat - d, lon - d / max(0.05, math.cos(math.radians(lat))), lat + d,
            lon + d / max(0.05, math.cos(math.radians(lat))))


# ---------------------------------------------------------------- geometry --

def summit(dem, lat, lon, radius_m=SUMMIT_RADIUS_M, step_m=10.0):
    """(lat, lon, elev, moved_m) of the highest DEM point within radius_m."""
    n = np.arange(-radius_m, radius_m + step_m / 2, step_m)
    N, E = np.meshgrid(n, n, indexing='ij')
    inside = np.hypot(N, E) <= radius_m
    la, lo = lat + N / K_M_PER_DEG, lon + E / (K_M_PER_DEG * math.cos(math.radians(lat)))
    z = np.where(inside, dem(la, lo), -np.inf)
    i = np.unravel_index(np.argmax(z), z.shape)
    if not np.isfinite(z[i]):
        raise ValueError('No DEM there.')
    return float(la[i]), float(lo[i]), float(z[i]), float(np.hypot(N[i], E[i]))


def resolve_elevation(p, dem):
    """The point's elevation (m) for projection, from its source or the DEM; may move an OSM peak to its summit."""
    p = dict(p)
    if p.get('elev_kind') == 'summit':
        p['lat'], p['lon'], p['elev'], p['moved_m'] = summit(dem, p['lat'], p['lon'])
    elif p.get('elev') is None:
        g = float(dem(p['lat'], p['lon']))
        p['elev'] = g + (p.get('height_m') or 0.0) if np.isfinite(g) else None
        p['elev_kind'] = 'dem + height' if p.get('height_m') else 'dem'
    return p


def sight(model, p, step_m=None):
    """Azimuth, elevation angle, range, visibility and skyline flag of point p (with elev) from the camera."""
    lat_c, lon_c, z_c = model.cam
    coslat = math.cos(math.radians(lat_c))
    north = math.radians(p['lat'] - lat_c) * core.R_EARTH
    east = math.radians(p['lon'] - lon_c) * core.R_EARTH * coslat
    dist = math.hypot(east, north)
    az = math.degrees(math.atan2(east, north)) % 360
    el = math.degrees(math.atan2(p['elev'] - z_c - dist ** 2 / (2 * core.R_EARTH), dist))
    d = core.sample_distances(model.dem, model.cam, step_m or 50.0, model.max_range_m, 50.0)
    lat = lat_c + np.degrees(d * math.cos(math.radians(az)) / core.R_EARTH)
    lon = lon_c + np.degrees(d * math.sin(math.radians(az)) / (core.R_EARTH * coslat))
    t = np.degrees(np.arctan2(model.dem(lat, lon) - z_c - d ** 2 / (2 * core.R_EARTH), d))
    t[~np.isfinite(t)] = -np.inf
    front = d < dist - max(30.0, p.get('acc_m', 0))          # terrain at the point itself doesn't hide it
    block = float(t[front].max()) if front.any() else -np.inf
    sky = float(t.max())
    return dict(az=az, el=el, range_m=dist, visible=bool(block <= el + VISIBLE_TOL_DEG),
                on_skyline=bool(el >= sky - SKYLINE_TOL_DEG), hidden_by_deg=max(0.0, block - el))


def candidates(model, points, min_range_m=MIN_RANGE_M, max_range_m=MAX_RANGE_M, visible_only=True):
    """Reference points as landmark candidates for this camera, nearest first: elevation resolved,
    sight line checked, angular accuracy sigma_deg = atan(acc_m / range)."""
    lat_c, lon_c, _ = model.cam
    out = []
    for p in points:
        c = math.cos(math.radians(lat_c))
        r = math.hypot((p['lat'] - lat_c) * K_M_PER_DEG, (p['lon'] - lon_c) * K_M_PER_DEG * c)
        if not min_range_m <= r <= max_range_m:
            continue
        try:
            q = resolve_elevation(p, model.dem)
        except ValueError:
            continue
        if q.get('elev') is None:
            continue
        q.update(sight(model, q))
        if visible_only and not q['visible']:
            continue
        q['sigma_deg'] = math.degrees(math.atan2(q['acc_m'], q['range_m']))
        out.append(q)
    return sorted(out, key=lambda q: (not q['on_skyline'], q['range_m']))


def near_terrain(dem, lat, lon, elev_m, radius_m=NEAR_RADIUS_M, step_m=5.0):
    """The terrain right around a camera: per azimuth (1 deg) the highest elevation angle of the ground within
    radius_m. A camera on a summit or mast looks down on all of it; terrain at a few degrees below horizontal or
    higher hides the view in that direction (with a 10 m DEM a site or height a few metres off does that).
    Returns {max_el_deg, az_deg, dist_m} of the worst direction and the share of azimuths blocked above
    NEAR_BLOCK_DEG, plus the highest DEM point within radius_m (summit)."""
    az = np.radians(np.arange(0, 360, 1.0))
    d = np.arange(step_m, radius_m + step_m / 2, step_m)
    A, D = np.meshgrid(az, d, indexing='ij')
    la = lat + np.degrees(D * np.cos(A) / core.R_EARTH)
    lo = lon + np.degrees(D * np.sin(A) / (core.R_EARTH * math.cos(math.radians(lat))))
    el = np.degrees(np.arctan2(dem(la, lo) - elev_m, D))
    el[~np.isfinite(el)] = -90.0
    worst = el.max(axis=1)
    i = int(np.argmax(worst))
    j = int(np.argmax(el[i]))
    s_lat, s_lon, s_z, s_d = summit(dem, lat, lon, radius_m=min(radius_m, 150.0), step_m=step_m)
    return dict(max_el_deg=float(worst[i]), az_deg=float(np.degrees(az[i])), dist_m=float(d[j]),
                blocked_share=float(np.mean(worst > NEAR_BLOCK_DEG)), by_az=np.round(worst, 2).tolist(),
                summit=dict(lat=s_lat, lon=s_lon, elev=s_z, distance_m=s_d, camera_above_m=elev_m - s_z))


def site_check(lat, lon, dem, elev_m=None, radius_m=1500.0, log=print):
    """NGS marks near an entered camera site (all kinds), nearest first, with the DEM ground there.
    elev_m: the camera height used, to report how far above the DEM ground it is at each mark."""
    marks = ngs_near(lat, lon, radius_m, log)
    c = math.cos(math.radians(lat))
    out = []
    for m in marks:
        la, lo = _f(m.get('lat')), _f(m.get('lon'))
        if la is None or lo is None:
            continue
        g = float(dem(la, lo))
        out.append(dict(pid=m['pid'], name=m.get('name', '').strip(), lat=la, lon=lo,
                        type=m.get('monumentType', '').split('=', 1)[-1].strip().lower(),
                        position=m.get('posSource', '').strip().lower(), height=_f(m.get('orthoHt')),
                        height_source=m.get('vertSource', '').strip().lower(),
                        distance_m=math.hypot((la - lat) * K_M_PER_DEG, (lo - lon) * K_M_PER_DEG * c),
                        ground_m=g if np.isfinite(g) else None,
                        camera_above_ground_m=(elev_m - g) if elev_m is not None and np.isfinite(g) else None,
                        last_recovered=m.get('lastRecovered', '').strip()))
    return sorted(out, key=lambda r: r['distance_m'])
