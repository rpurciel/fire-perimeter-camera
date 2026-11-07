"""Combine fire lines drawn on several cameras into one perimeter per time.

Each camera's fire edges are mapped to the terrain with that camera's own model (core.CameraModel), and the
edges drawn on one still form that camera's line for that time (core.fire_lines). Cameras see different
parts of the fire, at different times. The composite is recomputed at every time that some camera has a
line (an "event"), from the latest line of each camera at that time:

  * Around a fixed centre (by default the centroid of the convex hull of all lines, all times), the bearing
    circle is cut into BIN_DEG bins. Each drawn stretch of a line covers the bins between its end bearings.
  * Each bin is owned by the newest line covering it: newest wins, as the fire has grown since the older
    line. For lines less than TIE_S apart, the line nearer its camera there wins (better geolocation).
  * The stretches a line owns, joined where they follow each other, are runs: the observed parts.
  * Closed perimeter (optional): the runs in bearing order, joined end to start by straight connectors
    across the unseen parts and closed back to the first. Connectors are flagged as such, so they are never
    passed off as observed. Area and perimeter length come from that ring.
  * Overlap check: where two cameras' lines less than OVERLAP_DT_S apart cover the same bearings, the
    distance between them (along the bearing) shows how well the two calibrations agree.

Ownership by bearing assumes a perimeter that is roughly star-shaped around the centre. Deep fingers can be
cut oddly; moving the centre fixes most cases.
"""
import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from . import core

BIN_DEG = 0.5            # bearing bin width
TIE_S = 60.0             # lines closer in time than this count as simultaneous (nearer camera wins)
OVERLAP_DT_S = 600.0     # lines closer in time than this are compared in the overlap check
M2_PER_ACRE = 4046.8564224


@dataclass
class Line:
    camera: str              # camera id
    still: str               # still it was drawn on
    t: float                 # time: seconds since 1970, UTC (camera clock if no time zone is known)
    pts: np.ndarray          # N x 4: latitude, longitude, elevation_m, range_m
    real: np.ndarray         # N-1 bools: the stretch from point k to k+1 was drawn (False = a join across a gap)


# ------------------------------------------------------------------- times ---

def parse_offset(utc_offset):
    """'-07:00' -> -25200 (seconds); '' -> 0."""
    if not utc_offset:
        return 0
    m = re.fullmatch(r'([+-])(\d{1,2}):?(\d{2})', str(utc_offset).strip())
    if not m:
        raise ValueError('A UTC offset looks like -07:00, not %r' % utc_offset)
    s = int(m.group(2)) * 3600 + int(m.group(3)) * 60
    return -s if m.group(1) == '-' else s


def bar_time(stamp, utc_offset=''):
    """'YYYY-MM-DD HH:MM:SS[.ss]' as printed on the bar (camera clock) -> seconds since 1970 UTC.

    Without a UTC offset the camera clock is taken as is (fine for ordering cameras on the same clock)."""
    date, clock = str(stamp).strip().split()
    hms, _, frac = clock.partition('.')
    t = calendar.timegm(datetime.strptime(date + ' ' + hms, '%Y-%m-%d %H:%M:%S').timetuple())
    return t + (float('0.' + frac) if frac else 0.0) - parse_offset(utc_offset)


def iso_time(t, utc_offset=''):
    """UTC ISO time 'YYYY-MM-DDTHH:MM:SS.ssZ'; without a known time zone, the camera clock without 'Z'."""
    d = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=round(t, 2))
    return d.strftime('%Y-%m-%dT%H:%M:%S') + '.%02d' % (d.microsecond // 10000) + ('Z' if utc_offset else '')


def clock_time(t, utc_offset=''):
    """'YYYY-MM-DD HH:MM:SS' on the camera clock (local time), for names."""
    d = datetime(1970, 1, 1) + timedelta(seconds=math.floor(t + parse_offset(utc_offset)))   # as the bar shows it
    return d.strftime('%Y-%m-%d %H:%M:%S')


# ------------------------------------------------------------------- lines ---

def camera_lines(camera, geos, stills, times):
    """One Line per still of one camera, from its edges.

    geos: per edge, rows [u, v, lat, lon, elev, range] (NaN lat = no terrain hit); stills: per edge, the
    still it was drawn on; times: {still: time}. Edges of one still are joined as core.fire_lines does."""
    geos = [np.asarray(g, float).reshape(-1, 6) for g in geos]
    out = []
    for still, chain in core.fire_lines(geos, list(stills)).items():
        if len(chain) < 2 or still not in times:
            continue
        pts = np.array([geos[i][k, 2:6] for i, k in chain])
        real = np.array([a[0] == b[0] and abs(int(a[1]) - int(b[1])) == 1 for a, b in zip(chain[:-1], chain[1:])])
        out.append(Line(camera, still, float(times[still]), pts, real))
    return out


class _Plane:
    """Local east/north metres around (lat0, lon0), equirectangular like core."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.k = np.pi / 180 * core.R_EARTH
        self.c = np.cos(np.radians(self.lat0))

    def xy(self, lat, lon):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.k * self.c, (np.asarray(lat) - self.lat0) * self.k])

    def ll(self, x, y):
        return self.lat0 + y / self.k, self.lon0 + x / (self.k * self.c)


def _shoelace(xy):
    """(signed area, centroid x, centroid y) of a closed polygon given without its repeated first point."""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return 0.0, float(x.mean()), float(y.mean())
    return float(a), float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def default_centre(lines):
    """(lat, lon): area centroid of the convex hull of all the lines' points; their mean if they are collinear."""
    pts = np.vstack([l.pts[:, :2] for l in lines])
    plane = _Plane(*pts.mean(0))
    xy = plane.xy(pts[:, 0], pts[:, 1])
    try:
        from scipy.spatial import ConvexHull
        _, cx, cy = _shoelace(xy[ConvexHull(xy).vertices])
    except Exception:                      # fewer than 3 points, or all on one line (Qhull refuses)
        cx, cy = xy.mean(0)
    lat, lon = plane.ll(cx, cy)
    return float(lat), float(lon)


# --------------------------------------------------------------- composite ---

def composite(lines, t, centre, max_age_s=None, close=False, bin_deg=BIN_DEG):
    """Composite perimeter at time t, from the latest line of each camera at or before t (None if none).

    Returns a dict: t, centre, sources (per camera: still, time, age, bearings owned), runs (observed parts:
    camera, still, t, pts N x 4), ring (closed mode: rows of lat, lon, elev, range, source index, and
    whether the stretch to the next point is observed), area_m2/perimeter_m (closed mode), observed_m,
    overlap (the overlap check)."""
    latest = {}
    for l in lines:
        if l.t <= t + 1e-6 and (max_age_s is None or t - l.t <= max_age_s):
            if l.camera not in latest or l.t > latest[l.camera].t:
                latest[l.camera] = l
    use = sorted(latest.values(), key=lambda l: l.camera)
    if not use:
        return None
    plane = _Plane(*centre)
    n_bins = int(round(360 / bin_deg))
    geom = []
    for l in use:
        xy = plane.xy(l.pts[:, 0], l.pts[:, 1])
        th = np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360          # bearing from the centre
        rad = np.hypot(xy[:, 0], xy[:, 1])
        d = (np.diff(th) + 180) % 360 - 180                            # signed bearing step of each stretch
        mid_bin = np.floor(((th[:-1] + d / 2) % 360) / bin_deg).astype(int) % n_bins
        cover = {}                                                      # bin -> [(camera range, radius)]
        for k in np.flatnonzero(l.real):
            m = int(abs(d[k]) / bin_deg) + 2
            bins = np.unique(np.floor(((th[k] + np.linspace(0, d[k], m)) % 360) / bin_deg).astype(int) % n_bins)
            val = (0.5 * (l.pts[k, 3] + l.pts[k + 1, 3]), 0.5 * (rad[k] + rad[k + 1]))
            for b in bins:
                cover.setdefault(int(b), []).append(val)
        cover = {b: (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
                 for b, vals in cover.items()}
        geom.append(dict(line=l, xy=xy, th=th, d=d, mid_bin=mid_bin, cover=cover))

    owner = {}
    for b in set().union(*(g['cover'] for g in geom)):
        cands = [j for j, g in enumerate(geom) if b in g['cover']]
        newest = max(geom[j]['line'].t for j in cands)
        near = [j for j in cands if geom[j]['line'].t >= newest - TIE_S]
        owner[b] = min(near, key=lambda j: geom[j]['cover'][b][0])

    runs = []
    for j, g in enumerate(geom):
        keep = g['line'].real & np.array([owner.get(int(b)) == j for b in g['mid_bin']], bool)
        k = 0
        while k < len(keep):
            if not keep[k]:
                k += 1
                continue
            a = k
            while k < len(keep) and keep[k]:
                k += 1
            idx = np.arange(a, k + 1)                                   # points a..k: stretches a..k-1
            if g['d'][a:k].sum() < 0:                                   # run each part with increasing bearing
                idx = idx[::-1]
            runs.append(dict(src=j, idx=idx, start=float(g['th'][idx[0]])))
    runs.sort(key=lambda r: r['start'])

    def length(xy):
        return float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0

    out_runs, observed_m = [], 0.0
    for r in runs:
        g = geom[r['src']]
        observed_m += length(g['xy'][r['idx']])
        out_runs.append(dict(camera=g['line'].camera, still=g['line'].still, t=g['line'].t, src=r['src'],
                             pts=g['line'].pts[r['idx']]))
    res = dict(t=float(t), centre=[float(centre[0]), float(centre[1])], runs=out_runs, observed_m=observed_m,
               ring=None, area_m2=None, perimeter_m=None,
               sources=[dict(camera=g['line'].camera, still=g['line'].still, t=g['line'].t,
                             age_s=float(t - g['line'].t), owned_deg=sum(o == j for o in owner.values()) * bin_deg)
                        for j, g in enumerate(geom)])
    if close and runs:
        rows, xy = [], []
        for r in runs:
            g = geom[r['src']]
            for q, k in enumerate(r['idx']):
                rows.append((*g['line'].pts[k], r['src'], q < len(r['idx']) - 1))
                xy.append(g['xy'][k])
        xy = np.array(xy)
        area, _, _ = _shoelace(xy)
        res.update(ring=rows, area_m2=abs(area), perimeter_m=length(np.vstack([xy, xy[:1]])))

    res['overlap'] = []
    for a in range(len(geom)):
        for b in range(a + 1, len(geom)):
            la, lb = geom[a]['line'], geom[b]['line']
            common = sorted(set(geom[a]['cover']) & set(geom[b]['cover']))
            if abs(la.t - lb.t) > OVERLAP_DT_S or not common:
                continue
            diff = np.array([abs(geom[a]['cover'][k][1] - geom[b]['cover'][k][1]) for k in common])
            res['overlap'].append(dict(cameras=[la.camera, lb.camera], stills=[la.still, lb.still],
                                       dt_s=float(lb.t - la.t), degrees=len(common) * bin_deg,
                                       median_m=float(np.median(diff)), p90_m=float(np.percentile(diff, 90))))
    return res


def series(lines, centre=None, max_age_s=None, close=False):
    """(composites at every line time, centre used). Each composite is valid until the next one ('until')."""
    lines = [l for l in lines if len(l.pts) >= 2]
    if not lines:
        return [], None
    centre = tuple(centre) if centre else default_centre(lines)
    times = sorted({l.t for l in lines})
    out = []
    for k, t in enumerate(times):
        c = composite(lines, t, centre, max_age_s, close)
        if c is not None:
            c['until'] = times[k + 1] if k + 1 < len(times) else None
            out.append(c)
    return out, centre


# ------------------------------------------------------------------ export ---

def write_combined(base, comps, names, utc_offset='', close=False, title='Combined fire perimeter'):
    """Write <base>.csv and <base>.kml from series() output. names: {camera id: display name}.

    CSV: one row per vertex. 'part' numbers the observed runs of a composite (closed mode: 1, the ring, with
    its first vertex repeated at the end); 'kind' is the stretch from this vertex to the next: observed,
    connector (closed mode, across an unseen part) or empty (end of a part)."""
    def num(x):
        return 'NaN' if x is None or not np.isfinite(x) else '%.15g' % x
    with open(base + '.csv', 'w') as f:
        f.write('composite_id,time_utc,valid_until_utc,part,vertex_order,latitude,longitude,elevation_m,kind,'
                'camera,source_still,source_time_utc,age_min,range_m\n')
        for ci, c in enumerate(comps, start=1):
            head = '%d,%s,%s' % (ci, iso_time(c['t'], utc_offset), iso_time(c['until'], utc_offset) if c['until'] else '')

            def row(part, order, p, kind, src):
                s = c['sources'][src]
                f.write('%s,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                    head, part, order, num(p[0]), num(p[1]), num(p[2]), kind, s['camera'], s['still'],
                    iso_time(s['t'], utc_offset) + ',%.2f' % ((c['t'] - s['t']) / 60), num(p[3])))
            if close and c['ring']:
                ring = c['ring']
                for k, r in enumerate(ring + ring[:1], start=1):
                    kind = '' if k > len(ring) else ('observed' if r[5] else 'connector')
                    row(1, k, r[:4], kind, r[4])
            else:
                for part, run in enumerate(c['runs'], start=1):
                    for k, p in enumerate(run['pts'], start=1):
                        row(part, k, p, 'observed' if k < len(run['pts']) else '', run['src'])

    def coords(pts):
        return '\n'.join('%.8f,%.8f,0' % (p[1], p[0]) for p in pts)

    def line_strings(parts):
        return ''.join('<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>'
                       '<coordinates>\n%s\n</coordinates></LineString>' % coords(p) for p in parts if len(p) > 1)

    def span(c):
        z = 'Z' if utc_offset else ''
        b = iso_time(c['t'], utc_offset)[:19] + z
        e = '<end>%s</end>' % (iso_time(c['until'], utc_offset)[:19] + z) if c['until'] else ''
        return '<TimeSpan><begin>%s</begin>%s</TimeSpan>' % (b, e)

    with open(base + '.kml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        f.write('<name>%s</name>\n' % title)
        f.write('<Style id="observed"><LineStyle><color>ff1f3bff</color><width>4</width></LineStyle></Style>\n')
        f.write('<Style id="connector"><LineStyle><color>9900d7ff</color><width>2</width></LineStyle></Style>\n')
        f.write('<Style id="area"><LineStyle><color>00000000</color><width>0</width></LineStyle>'
                '<PolyStyle><color>401f3bff</color></PolyStyle></Style>\n')
        for c in comps:
            when = clock_time(c['t'], utc_offset)[11:]
            src = '; '.join('%s %s (%s)' % (names.get(s['camera'], s['camera']), clock_time(s['t'], utc_offset)[11:],
                                            'now' if s['age_s'] < 1 else '%.0f min old' % (s['age_s'] / 60))
                            for s in c['sources'])
            desc = 'From: %s. Observed %.2f km.' % (src, c['observed_m'] / 1000)
            if close and c['ring']:
                desc += ' Closed ring: perimeter %.2f km (%.0f%% observed), area %.1f ha (%.0f acres).' % (
                    c['perimeter_m'] / 1000, 100 * c['observed_m'] / max(c['perimeter_m'], 1e-9),
                    c['area_m2'] / 1e4, c['area_m2'] / M2_PER_ACRE)
            f.write('<Folder><name>%s</name>\n' % when)
            f.write('<Placemark><name>%s observed</name><description>%s</description>%s<styleUrl>#observed</styleUrl>'
                    '<MultiGeometry>%s</MultiGeometry></Placemark>\n'
                    % (when, desc, span(c), line_strings([r['pts'] for r in c['runs']])))
            if close and c['ring']:
                ring = c['ring']
                f.write('<Placemark><name>%s area</name><description>%s</description>%s<styleUrl>#area</styleUrl>'
                        '<Polygon><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><outerBoundaryIs>'
                        '<LinearRing><coordinates>\n%s\n</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n'
                        % (when, desc, span(c), coords([r[:2] for r in ring + ring[:1]])))
                gaps = [[ring[k][:2], ring[(k + 1) % len(ring)][:2]] for k in range(len(ring)) if not ring[k][5]]
                f.write('<Placemark><name>%s unseen (connectors)</name>%s<styleUrl>#connector</styleUrl>'
                        '<MultiGeometry>%s</MultiGeometry></Placemark>\n' % (when, span(c), line_strings(gaps)))
            f.write('</Folder>\n')
        f.write('</Document></kml>\n')
