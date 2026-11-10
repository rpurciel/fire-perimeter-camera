"""1-degree elevation tiles (.hgt) around a camera: which ones a horizon distance needs, where they are, and
downloading the missing ones.

Tiles are looked for in the DEM folder (the SRTM tiles that came with the project) and in its
downloads/srtm/ folder. Missing ones are fetched from the AWS Terrain Tiles (Mapzen "skadi" tiles, no account):

    https://s3.amazonaws.com/elevation-tiles-prod/skadi/N40/N40W120.hgt.gz

1 arc-second like the project's SRTM tiles; in the US they come from USGS NED (bare earth), so they differ by
metres from SRTM where both exist. Downloaded tiles are kept gzipped (about 10 MB each; core.read_hgt
reads them). A tile the service doesn't have (open sea) is remembered in downloads/srtm/unavailable.txt and not
asked for again.

    names = tile_names(lat, lon, km)
    found, missing = find(names, dem_dir)
    found += fetch(missing, dem_dir, log)          # the ones it could get
"""
import gzip
import math
import os
import time
import urllib.error
import urllib.request

SKADI = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi/%s/%s.hgt.gz'
TIMEOUT_S = 120
K_KM_PER_DEG = 111.2


def tile_names(lat, lon, km):
    """Tile names (e.g. N39W120.hgt) covering km around a site."""
    dlat = km / K_KM_PER_DEG
    dlon = km / (K_KM_PER_DEG * max(0.05, math.cos(math.radians(lat))))
    out = []
    for la in range(math.floor(lat - dlat), math.floor(lat + dlat) + 1):
        for lo in range(math.floor(lon - dlon), math.floor(lon + dlon) + 1):
            out.append(name_of(la, lo))
    return out


def name_of(la, lo):
    return '%s%02d%s%03d.hgt' % ('N' if la >= 0 else 'S', abs(la), 'E' if lo >= 0 else 'W', abs(lo))


def download_dir(dem_dir):
    return os.path.join(dem_dir, 'downloads', 'srtm')


def tile_name(path):
    """'N40W120.hgt' for .../N40W120.hgt or .../N40W120.hgt.gz."""
    b = os.path.basename(path)
    return b[:-3] if b.lower().endswith('.gz') else b


def find(names, dem_dir):
    """(paths found, names missing): tiles (.hgt or .hgt.gz) in dem_dir or its downloads/srtm/ (case-insensitive)."""
    have = {}
    for d in (download_dir(dem_dir), dem_dir):              # the project's own tiles win
        if os.path.isdir(d):
            have.update({tile_name(f).lower(): os.path.join(d, f) for f in os.listdir(d)
                         if f.lower().endswith(('.hgt', '.hgt.gz'))})
    found = [have[n.lower()] for n in names if n.lower() in have]
    return found, [n for n in names if n.lower() not in have]


def unavailable(dem_dir):
    p = os.path.join(download_dir(dem_dir), 'unavailable.txt')
    return set(open(p).read().split()) if os.path.isfile(p) else set()


def fetch(names, dem_dir, log=print, tries=3):
    """Download tiles into downloads/srtm/; returns the paths it got. Tiles the service doesn't have are skipped
    (and remembered); a network failure is logged and the tile skipped."""
    out_dir = download_dir(dem_dir)
    os.makedirs(out_dir, exist_ok=True)
    skip = unavailable(dem_dir)
    got = []
    for n in names:
        if n in skip:
            continue
        stem = n[:-4].upper()
        url = SKADI % (stem[:3], stem)
        for k in range(tries):
            try:
                t0 = time.time()
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'fire-camera/1'}),
                                            timeout=TIMEOUT_S) as r:
                    data = r.read()
                n_px = len(gzip.decompress(data)) // 2                # check it is a whole tile before keeping it
                if int(round(n_px ** 0.5)) ** 2 != n_px:
                    raise IOError('not a tile (%d samples)' % n_px)
                path = os.path.join(out_dir, stem + '.hgt.gz')
                with open(path + '.part', 'wb') as f:
                    f.write(data)
                os.replace(path + '.part', path)
                log('  terrain tile %s downloaded (%.0f MB, %.0f s)' % (stem, len(data) / 1e6, time.time() - t0))
                got.append(path)
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):                     # not in the tile set (sea)
                    with open(os.path.join(out_dir, 'unavailable.txt'), 'a') as f:
                        f.write(n + '\n')
                    break
                err = e
            except Exception as e:                           # network hiccup: retry, then give up on this tile
                err = e
            if k + 1 < tries:
                time.sleep(3 * (k + 1))
            else:
                log('WARNING: terrain tile %s could not be downloaded (%s); the terrain there is not modelled.' % (stem, err))
    return got
