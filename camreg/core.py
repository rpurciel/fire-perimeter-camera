"""Camera registration and fire-edge mapping: numerical core.

Python port of the non-GUI parts of verdi_combined_v3_fixed.m. Nothing here is
specific to one camera or fire; camera location, DEM tiles and starting pose
come from the caller (see verdi_register.py).

The camera's mount correction (see pose_from_bar) adds a pan-axis lean that
the MATLAB script did not have; skyline.py and panorama.py fit it automatically
from a camera's panorama.

Conventions kept from the MATLAB script so results and saved calibrations are
interchangeable:
  * pixel coordinates are 1-based (u = column, v = row; pixel centres at 1..nx),
  * pose q = [azimuth, tilt, roll, hfov] in degrees,
  * local frame is east/north/up from the camera, spherical Earth R = 6371 km,
    with the same d^2/(2R) curvature drop everywhere.
"""
import os
import re

import numpy as np
from scipy.optimize import minimize

R_EARTH = 6371000.0


# --------------------------------------------------------------------- DEM ---

def read_hgt(path):
    """Read one SRTM .hgt tile (big-endian int16, north row first); voids -> NaN."""
    raw = np.fromfile(path, dtype='>i2')
    n = int(round(np.sqrt(raw.size)))
    if n * n != raw.size:
        raise ValueError('Invalid HGT grid: %s' % path)
    tile = raw.reshape(n, n).astype(float)
    tile[tile == -32768] = np.nan
    return tile


def _hgt_corner(path):
    m = re.search(r'([NS])(\d{2})([EW])(\d{3})', os.path.basename(path).upper())
    if not m:
        raise ValueError('Cannot read tile corner from file name: %s' % path)
    lat = int(m.group(2)) * (1 if m.group(1) == 'N' else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == 'E' else -1)
    return lat, lon


class DEM:
    """Mosaic of 1-degree HGT tiles with bilinear lookup.

    Equivalent to griddedInterpolant(..., 'linear', 'none'): NaN outside the
    mosaic and wherever a contributing node is a void. Where tiles share an
    edge row/column the western (then southern) tile's value is kept, as in
    the MATLAB [W, E(:,2:end)] concatenation.
    """

    def __init__(self, paths):
        corners = [_hgt_corner(p) for p in paths]
        tiles = [read_hgt(p) for p in paths]
        n = tiles[0].shape[0]
        if any(t.shape[0] != n for t in tiles):
            raise ValueError('All tiles must have the same resolution')
        self.step = 1.0 / (n - 1)
        self.lat0 = min(c[0] for c in corners)
        self.lon0 = min(c[1] for c in corners)
        nlat = (max(c[0] for c in corners) + 1 - self.lat0) * (n - 1) + 1
        nlon = (max(c[1] for c in corners) + 1 - self.lon0) * (n - 1) + 1
        z = np.full((nlat, nlon), np.nan)          # row 0 = southern edge
        order = sorted(range(len(paths)), key=lambda i: (-corners[i][1], -corners[i][0]))
        for i in order:
            r0 = (corners[i][0] - self.lat0) * (n - 1)
            c0 = (corners[i][1] - self.lon0) * (n - 1)
            z[r0:r0 + n, c0:c0 + n] = tiles[i][::-1]
        self.z = z

    def __call__(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        nr, nc = self.z.shape
        fr = (lat - self.lat0) / self.step
        fc = (lon - self.lon0) / self.step
        inside = (fr >= 0) & (fr <= nr - 1) & (fc >= 0) & (fc <= nc - 1)
        fr = np.where(inside, fr, 0.0)
        fc = np.where(inside, fc, 0.0)
        r = np.minimum(np.floor(fr).astype(int), nr - 2)
        c = np.minimum(np.floor(fc).astype(int), nc - 2)
        tr = fr - r
        tc = fc - c
        z = self.z
        out = ((1 - tr) * ((1 - tc) * z[r, c] + tc * z[r, c + 1])
               + tr * ((1 - tc) * z[r + 1, c] + tc * z[r + 1, c + 1]))
        return np.where(inside, out, np.nan)


# ---------------------------------------------------------------- geometry ---

def huber(r, c):
    r = np.asarray(r, float)
    return 0.5 * np.minimum(r, c) ** 2 + c * np.maximum(r - c, 0)


def azel_rays(az, el):
    az = np.radians(az)
    el = np.radians(el)
    return np.column_stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def _camera_axes(q):
    hd, ti, ro = np.radians(q[:3])
    forward = np.array([np.cos(ti) * np.sin(hd), np.cos(ti) * np.cos(hd), np.sin(ti)])
    right0 = np.array([np.cos(hd), -np.sin(hd), 0.0])
    up0 = np.cross(right0, forward)
    right = np.cos(ro) * right0 + np.sin(ro) * up0
    up = -np.sin(ro) * right0 + np.cos(ro) * up0
    return forward, right, up


class CameraModel:
    """Pinhole camera at a fixed site looking at a DEM.

    cam = (lat, lon, elevation_m); image is nx x ny pixels after the info bar
    is cropped.
    """

    def __init__(self, cam, dem, nx, ny, step_m=50.0, max_range_m=90000.0):
        self.cam = np.asarray(cam, float)
        self.dem = dem
        self.nx = nx
        self.ny = ny
        self.step_m = step_m
        self.max_range_m = max_range_m

    # -- horizon -------------------------------------------------------------
    def horizon_rays(self, az_lo=0.0, az_hi=105.0, az_step=0.1,
                     d=np.arange(100, 90000 + 1, 120)):
        """Rays to the DEM skyline (max elevation angle) for each azimuth."""
        lat_c, lon_c, z_c = self.cam
        az = np.arange(round((az_hi - az_lo) / az_step) + 1) * az_step + az_lo
        A, D = np.meshgrid(az, d, indexing='ij')
        lat = lat_c + np.degrees(D * np.cos(np.radians(A)) / R_EARTH)
        lon = lon_c + np.degrees(D * np.sin(np.radians(A)) / (R_EARTH * np.cos(np.radians(lat_c))))
        zq = self.dem(lat, lon)
        el = np.degrees(np.arctan2(zq - z_c - D ** 2 / (2 * R_EARTH), D))
        el[~np.isfinite(el)] = -np.inf
        sky = el.max(axis=1)
        ok = np.isfinite(sky)
        return azel_rays(az[ok], sky[ok])

    # -- projection ----------------------------------------------------------
    def ground_rays(self, points):
        """Unit rays from the camera to (lat, lon, elevation_m) rows."""
        points = np.atleast_2d(np.asarray(points, float))
        lat_c, lon_c, z_c = self.cam
        north = np.radians(points[:, 0] - lat_c) * R_EARTH
        east = np.radians(points[:, 1] - lon_c) * R_EARTH * np.cos(np.radians(lat_c))
        dist = np.hypot(east, north)
        up = points[:, 2] - z_c - dist ** 2 / (2 * R_EARTH)
        rays = np.column_stack([east, north, up])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, rays, q):
        """Rays -> (u, v, in_front). u, v are 1-based pixel coordinates."""
        forward, right, up = _camera_axes(q)
        depth = rays @ forward
        f = (self.nx / 2) / np.tan(np.radians(q[3] / 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            u = (self.nx + 1) / 2 + f * (rays @ right) / depth
            v = (self.ny + 1) / 2 - f * (rays @ up) / depth
        return u, v, depth > 0

    def pixel_rays(self, uv, q):
        uv = np.atleast_2d(np.asarray(uv, float))
        forward, right, up = _camera_axes(q)
        focal = (self.nx / 2) / np.tan(np.radians(q[3] / 2))
        rays = (forward
                + ((uv[:, :1] - (self.nx + 1) / 2) / focal) * right
                - ((uv[:, 1:2] - (self.ny + 1) / 2) / focal) * up)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    # -- terrain intersection ------------------------------------------------
    def map_pixels(self, uv, q, batch=256):
        """Columns: pixel_u, pixel_v, latitude, longitude, elevation_m, range_m.

        Range is distance along the ray; the first DEM crossing is used. Rows
        with no crossing within max_range_m keep NaN geolocation.
        """
        uv = np.atleast_2d(np.asarray(uv, float))
        n = len(uv)
        out = np.full((n, 6), np.nan)
        out[:, :2] = uv
        if n == 0:
            return out
        lat_c, lon_c, z_c = self.cam
        coslat = np.cos(np.radians(lat_c))
        rays = self.pixel_rays(uv, q)
        ds = np.arange(0, self.max_range_m + self.step_m / 2, self.step_m)
        for k in range(0, n, batch):
            rr = rays[k:k + batch]
            east = rr[:, :1] * ds
            north = rr[:, 1:2] * ds
            h = self.dem(lat_c + np.degrees(north / R_EARTH),
                         lon_c + np.degrees(east / (R_EARTH * coslat)))
            terrain = h - (east ** 2 + north ** 2) / (2 * R_EARTH)
            residual = z_c + rr[:, 2:3] * ds - terrain
            valid = np.isfinite(residual)
            cross = (valid[:, :-1] & valid[:, 1:]
                     & (residual[:, :-1] > 0) & (residual[:, 1:] <= 0))
            has = cross.any(axis=1)
            j = np.flatnonzero(has)
            first = cross[j].argmax(axis=1)
            r0 = residual[j, first]
            r1 = residual[j, first + 1]
            rng = ds[first] + r0 / (r0 - r1) * self.step_m
            e = rr[j, 0] * rng
            nn = rr[j, 1] * rng
            lt = lat_c + np.degrees(nn / R_EARTH)
            ln = lon_c + np.degrees(e / (R_EARTH * coslat))
            out[k + j, 2:] = np.column_stack([lt, ln, self.dem(lt, ln), rng])
        return out


# ------------------------------------------------------------ optimization ---
#
# The camera pose is not free: each frame's info bar gives pan X, tilt Y and
# zoom Z. The model pose is that reading plus one fixed mount correction per
# camera, c = [d_azimuth, d_tilt, roll, hfov_at_Z1, lean_east, lean_north]
# (degrees). The first four act in the pan/tilt head's own frame; the lean
# tips the head's pan axis away from vertical (toward east and toward north),
# so its effect on tilt and roll changes with pan. Optimize fits the
# correction. c = [0, 0, 0, hfov, 0, 0] means "use the bar values as they are".
# A 4-value correction (older calibrations) means no lean.

N_CORRECTION = 6
LEAN_LIM = ((-10.0, 10.0), (-10.0, 10.0))


def full_correction(c):
    """Correction padded to all 6 values (missing lean = 0)."""
    c = np.asarray(c, float).ravel()
    return np.concatenate([c, np.zeros(N_CORRECTION - len(c))]) if len(c) < N_CORRECTION else c[:N_CORRECTION]


def full_limits(lim):
    """Correction limits padded to all 6 rows (default lean limits)."""
    lim = np.asarray(lim, float).reshape(-1, 2)
    return np.vstack([lim, LEAN_LIM[:N_CORRECTION - len(lim)]]) if len(lim) < N_CORRECTION else lim


def _lean_matrix(lean_east, lean_north):
    """Rotation that tips the vertical (pan axis) lean_east toward east and lean_north toward north."""
    a, b = np.radians(lean_east), np.radians(-lean_north)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return ry @ rx


def _axes_to_angles(forward, right):
    """[azimuth, tilt, roll] (deg) of a camera with these forward/right axes (inverse of _camera_axes)."""
    hd = np.arctan2(forward[0], forward[1])
    ti = np.arcsin(np.clip(forward[2], -1, 1))
    right0 = np.array([np.cos(hd), -np.sin(hd), 0.0])
    up0 = np.cross(right0, forward)
    ro = np.arctan2(right @ up0, right @ right0)
    return np.degrees([hd, ti, ro])


def lean_description(c):
    """Lean of the pan axis as (magnitude, azimuth it leans toward), degrees."""
    c = full_correction(c)
    return float(np.hypot(c[4], c[5])), float(np.degrees(np.arctan2(c[4], c[5])) % 360)


def pose_from_bar(bar, c):
    """Pose [azimuth, tilt, roll, hfov] for bar = (X, Y, Z) and mount correction c (4 or 6 values).

    Zoom is treated as optical magnification, tan(hfov/2) = tan(hfov_Z1/2) / Z
    (an assumption; not yet checked against zoomed stills).
    """
    X, Y, Z = bar
    c = full_correction(c)
    hfov = 2 * np.degrees(np.arctan(np.tan(np.radians(c[3] / 2)) / Z))
    q = np.array([np.mod(X + c[0], 360), Y + c[1], c[2], hfov])
    if c[4] or c[5]:
        forward, right, _ = _camera_axes(q)
        m = _lean_matrix(c[4], c[5])
        q[:3] = _axes_to_angles(m @ forward, m @ right)
        q[0] %= 360
    return q


def correction_from_pose(p, bar, lean=(0.0, 0.0)):
    """Inverse of pose_from_bar: the correction (6 values) that gives pose p on this frame, for a given lean."""
    X, Y, Z = bar
    p = np.asarray(p, float)
    a = p[:3]
    if lean[0] or lean[1]:
        forward, right, _ = _camera_axes(p)
        m = _lean_matrix(*lean).T
        a = _axes_to_angles(m @ forward, m @ right)
    d_az = (a[0] - X + 180) % 360 - 180
    return np.array([d_az, a[1] - Y, a[2], 2 * np.degrees(np.arctan(np.tan(np.radians(p[3] / 2)) * Z)),
                     lean[0], lean[1]])


def huber_score(q, model, sky_rays, gcp=None, sky_obs=None, n_landmarks=0):
    """Huber-weighted landmark reprojection + skyline distance in pixels, for pose q.

    n_landmarks is the total landmark count used to weight the skyline.
    """
    score = 0.0
    if gcp is not None and len(gcp):
        u, v, front = model.project(model.ground_rays(gcp[:, 2:5]), q)
        residual = np.hypot(u - gcp[:, 0], v - gcp[:, 1])
        residual[~np.isfinite(residual) | ~front] = 1000
        # Huber loss; 8 pixel transition to limit outlier domination.
        score += huber(residual, 8).sum()
    if sky_obs is not None and len(sky_obs):
        u, v, front = model.project(sky_rays, q)
        good = front & np.isfinite(u) & np.isfinite(v) & (u >= -model.nx) & (u <= 2 * model.nx)
        u = u[good]
        v = v[good]
        if len(u) < 2:
            return score + 1e6
        distances = np.empty(len(sky_obs))
        for i in range(0, len(sky_obs), 80):
            dx = sky_obs[i:i + 80, :1] - u
            dy = sky_obs[i:i + 80, 1:2] - v
            distances[i:i + 80] = np.sqrt((dx ** 2 + dy ** 2).min(axis=1))
        # Skyline gets weight equivalent to roughly 3 landmarks.
        score += 3 * max(1, n_landmarks) * huber(distances, 8).mean()
    return score


def objective(c, model, sky_rays, gcp_groups, sky, lim, c_ref):
    """Score of mount correction c (6 values; lim is 6 x (min, max)).

    gcp_groups: list of (bar, gcp rows [u, v, lat, lon, elev]) per frame.
    sky: (bar, skyline uv) or None.
    """
    c = np.asarray(c, float)
    if np.any(c < lim[:, 0] - 1e-8) or np.any(c > lim[:, 1] + 1e-8):
        return 1e9 + np.sum((c - c_ref) ** 2)
    n = sum(len(g) for _, g in gcp_groups)
    score = sum(huber_score(pose_from_bar(bar, c), model, sky_rays, gcp=g) for bar, g in gcp_groups)
    if sky is not None and len(sky[1]):
        score += huber_score(pose_from_bar(sky[0], c), model, sky_rays, sky_obs=sky[1], n_landmarks=n)
    return score


def optimize_correction(c_start, model, sky_rays, gcp_groups, sky, lim, verbose=True, fit_lean=False):
    """Fit the mount correction (Nelder-Mead with fminsearch's tolerances); returns all 6 values.

    Fits d_azimuth, d_tilt, roll and HFOV; the lean stays as in c_start unless
    fit_lean (it needs observations at several pan directions to be determined).
    The starting simplex steps 0.5 deg in the offsets/roll/lean and 5% in HFOV;
    fminsearch's default (5% of each value) would barely move from zero offsets.
    """
    c_start = full_correction(c_start)
    lim = full_limits(lim)
    free = np.arange(6 if fit_lean else 4)
    steps = np.array([0.5, 0.5, 0.5, 0.05 * c_start[3], 0.5, 0.5])[free]

    def f(x):
        c = c_start.copy()
        c[free] = x
        return objective(c, model, sky_rays, gcp_groups, sky, lim, c_start)

    x0 = c_start[free]
    simplex = np.vstack([x0] + [x0 + np.eye(len(free))[k] * s for k, s in enumerate(steps)])
    res = minimize(f, x0, method='Nelder-Mead',
                   options=dict(maxiter=350 * len(free) // 4, maxfev=1600 * len(free) // 4,
                                xatol=1e-5, fatol=1e-5, initial_simplex=simplex))
    if verbose:
        print('Optimize: %s after %d iterations, %d evaluations; score %.6g'
              % (res.message, res.nit, res.nfev, res.fun))
    c = c_start.copy()
    c[free] = res.x
    return c, res


# ------------------------------------------------------------------ inputs ---

def resample_polyline(pts, spacing, exact_spacing=False):
    """Resample a drawn line by arc length.

    exact_spacing=False: N = max(2, ceil(L/spacing)) evenly spaced samples
    (skyline traces). exact_spacing=True: samples every `spacing` px plus the
    end point (fire edges). Returns None if the line has no length.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return None
    t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
    t, ia = np.unique(t, return_index=True)   # cumulative length is monotone
    pts = pts[ia]
    if t[-1] <= 0:
        return None
    if exact_spacing:
        tt = np.unique(np.concatenate([np.arange(0, t[-1], spacing), [t[-1]]]))
    else:
        tt = np.linspace(0, t[-1], max(2, int(np.ceil(t[-1] / spacing))))
    return np.column_stack([np.interp(tt, t, pts[:, 0]), np.interp(tt, t, pts[:, 1])])


# ----------------------------------------------------------------- outputs ---

def fire_lines(geos, keys):
    """Join the edges of each time into one line.

    geos: per edge, rows [u, v, lat, lon, elev, range] (NaN lat = no terrain hit);
    keys: per edge, what groups them (the still, i.e. the time). Returns
    {key: [(edge index, sample index), ...]}: the terrain points of the key's
    edges in drawing order, joined across gaps and between edges, each next
    edge reversed if that makes the join shorter. Same rule as fireLines() in
    the web app.
    """
    def dist(a, b):
        return np.hypot(a[0] - b[0], (a[1] - b[1]) * np.cos(np.radians(a[0])))
    lines = {}
    for i, (g, key) in enumerate(zip(geos, keys)):
        g = np.asarray(g, float).reshape(-1, 6)
        run = [(i, k) for k in np.flatnonzero(np.isfinite(g[:, 2]) & np.isfinite(g[:, 3]))]
        pts = lines.setdefault(key, [])
        if pts and len(run) > 1:
            ei, ek = pts[-1]
            end = geos[ei][ek][2:4]
            if dist(end, g[run[-1][1], 2:4]) < dist(end, g[run[0][1], 2:4]):
                run.reverse()
        pts.extend(run)
    return lines


def export_edges(edges, model, poses, base, name='Fire edges', times=None, utc_offset='', time_stamps=True):
    """Write <base>.csv and <base>.kml, remapping edge i with pose poses[i].

    All edges drawn on one still (one time) form one line (fire_lines): one KML
    placemark per time, named by the time and, if time_stamps, given a KML
    TimeStamp for Google Earth's time slider. utc_offset (e.g. '-07:00') is the
    bar times' zone; without it the stamps carry no zone. The CSV keeps one
    row per sample, as before, plus the time, the line it belongs to (1 = the
    earliest time) and its position along that line (empty without a terrain hit).
    """
    times = list(times) if times is not None else [e.get('still', '') for e in edges]
    keys = [e.get('still', '') or t for e, t in zip(edges, times)]
    geos = [model.map_pixels(e['uv'], q) for e, q in zip(edges, poses)]
    lines = fire_lines(geos, keys)
    first = {}
    for k, t in zip(keys, times):
        first.setdefault(k, t)
    order = sorted(lines, key=lambda k: first[k])
    line_id = {k: n for n, k in enumerate(order, start=1)}
    position = {}
    for k in order:
        for j, ik in enumerate(lines[k], start=1):
            position[ik] = j
    rows = [np.column_stack([np.full(len(g), i), g]) for i, g in enumerate(geos, start=1)]
    rows = np.vstack(rows) if rows else np.zeros((0, 7))
    num = lambda x: 'NaN' if not np.isfinite(x) else '%.15g' % x
    with open(base + '.csv', 'w') as f:
        f.write('edge_id,pixel_u,pixel_v,latitude,longitude,elevation_m,range_m,time,line_id,line_order\n')
        for i, g in enumerate(geos):
            for k, r in enumerate(g):
                f.write('%d,%s,%s,%d,%s\n' % (i + 1, ','.join(num(x) for x in r), times[i], line_id[keys[i]],
                                              position.get((i, k), '')))
    with open(base + '.kml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
        f.write('<name>%s</name>\n' % name)
        f.write('<Style id="fire"><LineStyle><color>ff0000ff</color><width>4</width></LineStyle></Style>\n')
        for key in order:
            pts = lines[key]
            if len(pts) < 2:
                continue
            t = str(first[key])
            n_edges = len({i for i, _ in pts})
            f.write('<Placemark><name>%s</name><description>%d edge%s, %d points</description>\n'
                    % (t[:19] or key, n_edges, 's' if n_edges > 1 else '', len(pts)))
            if time_stamps and len(t) >= 19:
                f.write('<TimeStamp><when>%sT%s%s</when></TimeStamp>\n' % (t[:10], t[11:19], utc_offset))
            f.write('<styleUrl>#fire</styleUrl>\n')
            f.write('<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><coordinates>\n')
            for i, k in pts:
                f.write('%.8f,%.8f,0\n' % (geos[i][k, 3], geos[i][k, 2]))
            f.write('</coordinates></LineString></Placemark>\n')
        f.write('</Document></kml>\n')
    return rows


def _cellstr(items):
    a = np.empty((len(items), 1), dtype=object)
    for k, x in enumerate(items):
        a[k, 0] = x
    return a


def save_calibration(path, cam, p, gcp, names, sky_obs, edges, image_size, image_file,
                     correction=None, landmark_stills=(), skyline_still='', extra_fields=None):
    """Save in the MATLAB script's .mat layout (struct 'calibration').

    p is the pose on image_file. Extra fields MATLAB ignores: mount_correction
    (d_az, d_tilt, roll, hfov at Z1), mount_lean_east_north, landmark_stills,
    skyline_still, edges.still, and any extra_fields (e.g. panorama details).
    """
    from scipy.io import savemat
    # 'still' (image the edge was drawn on) is an extra field; MATLAB ignores it.
    edge_arr = np.empty((1, len(edges)), dtype=[('uv', object), ('geo', object), ('kind', object),
                                                ('still', object)])
    for k, e in enumerate(edges):
        edge_arr[0, k] = (np.asarray(e['uv'], float), np.asarray(e['geo'], float), e['kind'],
                          e.get('still', ''))
    extra = dict(extra_fields or {})
    if correction is not None:
        correction = full_correction(correction)
        extra.update({'mount_correction_daz_dtilt_roll_hfovz1': correction[None, :4],
                      'mount_lean_east_north': correction[None, 4:],
                      'landmark_stills': _cellstr(list(landmark_stills)),
                      'skyline_still': skyline_still})
    savemat(path, {'calibration': {**extra,
        'camera_lat_lon_z': np.asarray(cam, float)[None, :],
        'parameters_az_tilt_roll_hfov': np.asarray(p, float)[None, :],
        'landmarks_uv_lat_lon_z': np.asarray(gcp, float).reshape(-1, 5),
        'landmark_names': _cellstr(names),
        'skyline_uv': np.asarray(sky_obs, float).reshape(-1, 2),
        'edges': edge_arr,
        'image_size': np.array([[image_size[0], image_size[1]]], float),
        'image_file': image_file}}, long_field_names=True)


def _mat_str(x):
    a = np.asarray(x)
    return '' if a.size == 0 else str(a.squeeze())


def load_calibration(path):
    """Load a calibration saved by this module or by the MATLAB script."""
    from scipy.io import loadmat
    c = loadmat(path, squeeze_me=False, struct_as_record=True)['calibration'][0, 0]

    def arr(name, ncol):
        a = np.asarray(c[name], float)
        return a.reshape(-1, ncol) if a.size else np.zeros((0, ncol))

    names = [_mat_str(x) for x in np.asarray(c['landmark_names']).ravel()]
    edges = []
    if 'edges' in c.dtype.names:
        for e in np.asarray(c['edges']).ravel():
            edges.append({'uv': np.asarray(e['uv'], float).reshape(-1, 2),
                          'geo': np.asarray(e['geo'], float).reshape(-1, 6),
                          'kind': _mat_str(e['kind']),
                          'still': _mat_str(e['still']) if 'still' in e.dtype.names else ''})
    image_file = _mat_str(c['image_file']) if 'image_file' in c.dtype.names else ''
    fields = c.dtype.names
    corr = 'mount_correction_daz_dtilt_roll_hfovz1'
    correction = None
    if corr in fields:
        lean = arr('mount_lean_east_north', 2)[0] if 'mount_lean_east_north' in fields else np.zeros(2)
        correction = np.concatenate([arr(corr, 4)[0], lean])
    return {'p': arr('parameters_az_tilt_roll_hfov', 4)[0],
            'correction': correction,
            'landmark_stills': ([_mat_str(x) for x in np.asarray(c['landmark_stills']).ravel()]
                                if 'landmark_stills' in fields else []),
            'skyline_still': _mat_str(c['skyline_still']) if 'skyline_still' in fields else '',
            'image_file': image_file,
            'gcp': arr('landmarks_uv_lat_lon_z', 5),
            'names': names,
            'sky_obs': arr('skyline_uv', 2),
            'edges': edges}
