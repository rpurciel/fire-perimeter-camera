"""USGS 3DEP 1/3 arc-second (about 10 m) elevation around a fire's cameras, for core.LayeredDEM.

The grid is fetched from the USGS 3DEP ImageServer (no account or key needed):

    https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage

in pieces of at most PIECE x PIECE nodes, on the 1/3 arc-second lattice (node lat/lon are multiples of
1/10800 deg), WGS 84 lat/lon, heights in metres (NAVD 88; the SRTM tiles are EGM96, about a metre apart
here). The service resamples (bilinear) whatever 3DEP has at each place (1 m lidar or the 10 m DEM). It is
bare earth: trees are not in it, unlike SRTM, which partly sees the canopy.

A fetched grid is kept as <dir>/3dep10m_<S>_<W>_<N>_<E>.npy (float32, row 0 = south, NaN = no data) with
a .json next to it (georeference, source, date). A 3DEP GeoTIFF downloaded by hand (e.g. the National Map's
USGS_13_n40w120.tif tiles) can be read with read_geotiff and cached the same way (cache_geotiff).

    box = box_around([(lat, lon), ...], radius_km=25)
    path = fetch(box, out_dir, log=print)             # or find(box, out_dir) for one already there
    fine = load(path)                                 # core.GeoGrid (memory-mapped)
    dem = core.LayeredDEM(fine, core.DEM(srtm_tiles))
"""
import io
import json
import math
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import core

SERVICE = 'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage'
STEP = 1.0 / 10800                 # 1/3 arc-second, deg
PIECE = 2000                       # nodes per side of one request (the service allows 8000)
MIN_PIECE = 250                    # failing pieces are split down to this size
NODATA = -9999.0
DEFAULT_RADIUS_KM = 25.0
MAX_AREA_KM2 = 12000.0             # about 600 MB as float32; larger areas are refused
K_M_PER_DEG = 111195.0
TIMEOUT_S = 180


def box_around(sites, radius_km=DEFAULT_RADIUS_KM):
    """(south, west, north, east) node bounds on the 1/3" lattice covering radius_km around every site."""
    lats = [float(s[0]) for s in sites]
    lons = [float(s[1]) for s in sites]
    r = float(radius_km) * 1000.0
    coslat = math.cos(math.radians(max(abs(min(lats)), abs(max(lats))) + r / K_M_PER_DEG))
    s = min(lats) - r / K_M_PER_DEG
    n = max(lats) + r / K_M_PER_DEG
    w = min(lons) - r / (K_M_PER_DEG * coslat)
    e = max(lons) + r / (K_M_PER_DEG * coslat)
    snap = lambda x, f: f(round(x / STEP, 6)) * STEP
    return snap(s, math.floor), snap(w, math.floor), snap(n, math.ceil), snap(e, math.ceil)


def box_shape(box):
    s, w, n, e = box
    return int(round((n - s) / STEP)) + 1, int(round((e - w) / STEP)) + 1


def box_name(box):
    return '3dep10m_%.4f_%.4f_%.4f_%.4f' % box


def box_area_km2(box):
    s, w, n, e = box
    return (n - s) * K_M_PER_DEG * (e - w) * K_M_PER_DEG * math.cos(math.radians((s + n) / 2)) / 1e6


def find(box, out_dir):
    """Path of a cached grid that covers box, or None."""
    if not os.path.isdir(out_dir):
        return None
    for f in sorted(os.listdir(out_dir)):
        if f.startswith('3dep10m_') and f.endswith('.json'):
            try:
                meta = json.load(open(os.path.join(out_dir, f)))
            except (OSError, ValueError):
                continue
            s, w, n, e = meta['bounds']
            tol = STEP / 2
            npy = os.path.join(out_dir, f[:-5] + '.npy')
            if (s <= box[0] + tol and w <= box[1] + tol and n >= box[2] - tol and e >= box[3] - tol
                    and os.path.isfile(npy)):
                return npy
    return None


def load(path):
    """core.GeoGrid of a cached grid (memory-mapped .npy)."""
    meta = json.load(open(path[:-4] + '.json'))
    z = np.load(path, mmap_mode='r')
    if list(z.shape) != list(meta['shape']):
        raise ValueError('%s: shape %s does not match its .json %s' % (path, z.shape, meta['shape']))
    return core.GeoGrid(z, meta['lat0'], meta['lon0'], meta['dlat'], meta['dlon'])


def meta(path):
    return json.load(open(path[:-4] + '.json'))


def _get(url, tries=4, log=print):
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'fire-camera/1'}),
                                        timeout=TIMEOUT_S) as r:
                body = r.read()
                if not r.headers.get('Content-Type', '').startswith('image/'):
                    raise IOError('the service answered %r instead of an image: %s'
                                  % (r.headers.get('Content-Type'), body[:300].decode('utf8', 'replace')))
                return body
        except Exception as e:                     # network hiccups: retry with backoff
            err = e
            if k + 1 < tries:
                log('  request failed (%s); retrying in %d s' % (e, 5 * (k + 1)))
                time.sleep(5 * (k + 1))
    raise IOError('3DEP download failed: %s' % err)


def _tiff_grid(data):
    """(z float32 north-first, west lon of pixel 0 centre, north lat of row 0 centre, dx, dy) of a GeoTIFF."""
    import tifffile                                # conda-forge tifffile + imagecodecs (compressed tiles)
    with tifffile.TiffFile(io.BytesIO(data) if isinstance(data, bytes) else data) as t:
        page = t.pages[0]
        z = page.asarray().astype(np.float32)
        geo = page.geotiff_tags or {}
    if z.ndim != 2:
        z = z.reshape(z.shape[-2:]) if z.size == z.shape[-1] * z.shape[-2] else z[..., 0]
    if int(geo.get('GTModelTypeGeoKey', 2)) != 2:
        raise ValueError('GeoTIFF is not in geographic lat/lon (model type %s)' % geo.get('GTModelTypeGeoKey'))
    tp, sc = geo.get('ModelTiepoint'), geo.get('ModelPixelScale')
    if not tp or not sc:
        raise ValueError('GeoTIFF has no ModelTiepoint/ModelPixelScale')
    i, j, x, y = tp[0], tp[1], tp[3], tp[4]
    dx, dy = sc[0], sc[1]
    area = int(geo.get('GTRasterTypeGeoKey', 1)) == 1   # pixel-is-area: the tie point is a pixel corner
    w = x - i * dx + (dx / 2 if area else 0.0)
    n = y + j * dy - (dy / 2 if area else 0.0)
    return z, w, n, dx, dy


def fetch(box, out_dir, log=print):
    """Download the grid of box (see box_around) and cache it in out_dir; returns the .npy path."""
    os.makedirs(out_dir, exist_ok=True)
    ny, nx = box_shape(box)
    s, w, n, e = box
    log('3DEP 10 m: %d x %d nodes (%.0f km2, %.0f MB) in %d pieces...'
        % (nx, ny, box_area_km2(box), nx * ny * 4 / 1e6, math.ceil(nx / PIECE) * math.ceil(ny / PIECE)))
    z = np.full((ny, nx), np.nan, np.float32)      # row 0 = south
    pieces = [(r0, min(r0 + PIECE, ny), c0, min(c0 + PIECE, nx)) for r0 in range(0, ny, PIECE) for c0 in range(0, nx, PIECE)]
    t0 = time.time()
    for k, piece in enumerate(pieces):
        _fetch_piece(z, box, *piece, log)
        log('  piece %d/%d done (%.0f s)' % (k + 1, len(pieces), time.time() - t0))
    return _save(z, box, out_dir, dict(source='USGS 3DEP ImageServer (bilinear, 1/3 arc-second lattice)',
                                       service=SERVICE))


def _fetch_piece(z, box, r0, r1, c0, c1, log, tries=3):
    """Fill z[r0:r1, c0:c1]; a piece the service keeps failing on (gateway time-outs) is split in four."""
    s, w = box[0], box[1]
    ps, pn = s + r0 * STEP, s + (r1 - 1) * STEP           # node bounds of the piece
    pw, pe = w + c0 * STEP, w + (c1 - 1) * STEP
    q = dict(bbox='%.10f,%.10f,%.10f,%.10f' % (pw - STEP / 2, ps - STEP / 2, pe + STEP / 2, pn + STEP / 2),
             bboxSR=4326, imageSR=4326, size='%d,%d' % (c1 - c0, r1 - r0), format='tiff', pixelType='F32',
             noData=NODATA, interpolation='RSP_BilinearInterpolation', compression='LZ77', f='image')
    try:
        data = _get(SERVICE + '?' + urllib.parse.urlencode(q), tries=tries, log=log)
    except IOError:
        if max(r1 - r0, c1 - c0) <= MIN_PIECE:
            raise
        log('  splitting a %d x %d piece the service keeps failing on' % (c1 - c0, r1 - r0))
        rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
        for a, b in ((r0, rm), (rm, r1)):
            for c, d in ((c0, cm), (cm, c1)):
                if b > a and d > c:
                    _fetch_piece(z, box, a, b, c, d, log, tries)
        return
    tile, tw, tn, dx, dy = _tiff_grid(data)
    if (tile.shape != (r1 - r0, c1 - c0) or abs(tw - pw) > STEP * 1e-3 or abs(tn - pn) > STEP * 1e-3
            or abs(dx - STEP) > STEP * 1e-6 or abs(dy - STEP) > STEP * 1e-6):
        raise IOError('3DEP piece came back on another grid: %s at (%.8f, %.8f) step %.3g x %.3g; asked %s at '
                      '(%.8f, %.8f) step %.3g' % (tile.shape, tw, tn, dx, dy, (r1 - r0, c1 - c0), pw, pn, STEP))
    tile[(tile <= NODATA + 1) | ~np.isfinite(tile)] = np.nan
    z[r0:r1, c0:c1] = tile[::-1]


def _save(z, box, out_dir, info):
    name = box_name(box)
    path = os.path.join(out_dir, name + '.npy')
    np.save(path + '.part.npy', z)
    os.replace(path + '.part.npy', path)
    voids = float(np.mean(~np.isfinite(z)))
    m = dict(info, bounds=list(box), shape=list(z.shape), lat0=box[0], lon0=box[1], dlat=STEP, dlon=STEP,
             voids=voids, fetched=time.strftime('%Y-%m-%d %H:%M:%S'),
             heights='metres, NAVD 88 (bare earth)', crs='geographic lat/lon (WGS 84 / NAD 83)')
    with open(path[:-4] + '.json', 'w') as f:
        json.dump(m, f, indent=1)
    return path


def read_geotiff(path, box=None):
    """core.GeoGrid of a geographic GeoTIFF (cropped to box if given)."""
    with open(path, 'rb') as f:
        z, w, n, dx, dy = _tiff_grid(f)
    z = z[::-1]                                    # row 0 = south
    s = n - (z.shape[0] - 1) * dy
    z = np.where(z <= NODATA + 1, np.nan, z).astype(np.float32)
    if box is not None:
        r0 = max(0, int(math.floor((box[0] - s) / dy + 1e-6)))
        r1 = min(z.shape[0], int(math.ceil((box[2] - s) / dy - 1e-6)) + 1)
        c0 = max(0, int(math.floor((box[1] - w) / dx + 1e-6)))
        c1 = min(z.shape[1], int(math.ceil((box[3] - w) / dx - 1e-6)) + 1)
        z, s, w = z[r0:r1, c0:c1], s + r0 * dy, w + c0 * dx
    return core.GeoGrid(np.ascontiguousarray(z), s, w, dy, dx)


def cache_geotiff(paths, box, out_dir, log=print):
    """Resample hand-downloaded 3DEP GeoTIFFs onto box's 1/3" lattice and cache them like fetch()."""
    ny, nx = box_shape(box)
    lat = box[0] + np.arange(ny) * STEP
    lon = box[1] + np.arange(nx) * STEP
    z = np.full((ny, nx), np.nan, np.float32)
    for p in paths:
        log('Reading %s...' % os.path.basename(p))
        g = read_geotiff(p, box)
        for r in range(0, ny, 500):                # by rows: a full-size meshgrid would be large
            LA, LO = np.meshgrid(lat[r:r + 500], lon, indexing='ij')
            v = g(LA, LO).astype(np.float32)
            blk = z[r:r + 500]
            blk[np.isnan(blk)] = v[np.isnan(blk)]
    return _save(z, box, out_dir, dict(source='GeoTIFF: ' + ', '.join(os.path.basename(p) for p in paths)))
