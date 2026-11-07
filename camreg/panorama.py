"""Calibrate a camera's mount correction automatically from its panorama.

A panorama here is one or more camera frames, each with its info bar, either
as separate images or placed side by side in one image (as ALERTCalifornia's
360-degree panoramas are). For every frame the info bar gives pan/tilt/zoom,
the skyline is detected (skyline.py), and one mount correction is fitted so
the DEM skyline lines up with the detected one in all frames at once. With
frames spread around the compass this also fits the lean of the camera's pan
axis (core.pose_from_bar), which a single view cannot separate from tilt and
roll offsets.

Nothing here is specific to one camera: the camera site, DEM tiles and a
starting field of view come from the caller.

    python -m camreg.panorama PANORAMA.jpg --camera verdi --cam 39.472599 -120.039383 2567 \\
        --tiles N39W121.hgt N39W120.hgt --hfov 64.29

writes output/verdi_panorama.mat (loadable in the registration app) and
output/verdi_panorama_overlay.jpg (detected skyline in yellow, fitted DEM
skyline in red, rejected points in orange) for checking.
"""
import argparse
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

from . import core
from .skyline import skyline_points

DEFAULT_GLYPHS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'camstills', 'glyphs', 'alertca.npz')
LOSS_SCALE = 0.1        # deg; residuals beyond this count less (soft L1)
REJECT_DEG = 0.3        # after the first fit, points further than max(this, 4 robust sigma) are dropped
MAX_RESID_DEG = 10.0    # residual used where the DEM skyline does not cover a detected point
LEAN_MIN_SPREAD = 45.0  # deg of pan the frames must span for the lean to be fitted (lean='auto')
# The bar's pan X is the azimuth from north (-180..180), true to a few degrees, so the azimuth offset is only
# searched this far (a wider search lets a short or hazy skyline match some other stretch of horizon).
AZ_RANGE = 10.0
POOR_FIT_PX = 8.0       # median skyline error (px) above which the fit is reported as poor; good fits are 2-4 px


@dataclass
class Frame:
    source: str                  # image file it came from
    index: int                   # frame number within that file (0-based)
    rgb: np.ndarray              # frame without its info bar, uint8 H x W x 3
    full: np.ndarray = None      # frame with its bar (for the overlay)
    bar: tuple = None            # (X, Y, Z) or None if unknown
    time: str = ''
    elev_ft: float = None
    uv: np.ndarray = None        # detected skyline, 1-based pixels
    keep: np.ndarray = None      # points used in the final fit
    note: str = ''               # why it was skipped, if it was
    stats: dict = field(default_factory=dict)

    @property
    def nx(self):
        return self.rgb.shape[1]

    @property
    def ny(self):
        return self.rgb.shape[0]

    @property
    def name(self):
        return '%s#%d' % (os.path.basename(self.source), self.index + 1)


# ------------------------------------------------------------------- frames ---

def split_frames(img, n_frames=None, frame_width=None, aspect=16 / 9):
    """Cut a side-by-side panorama into frames (list of H x w arrays).

    Frame width: frame_width, else width / n_frames, else height * aspect if the
    image width is a multiple of it, else the whole image is one frame.
    """
    h, w = img.shape[:2]
    if frame_width is None and n_frames:
        frame_width = w / n_frames
    if frame_width is None:
        guess = round(h * aspect)
        frame_width = guess if guess and w % guess == 0 else w
    n = int(round(w / frame_width))
    if n < 1 or abs(n * frame_width - w) > n:
        raise ValueError('Image width %d is not a whole number of %g px frames; give --frames or --frame-width'
                         % (w, frame_width))
    edges = np.round(np.arange(n + 1) * w / n).astype(int)
    return [img[:, a:b] for a, b in zip(edges[:-1], edges[1:])]


def load_frames(paths, n_frames=None, frame_width=None, bar_px=None, glyphs=DEFAULT_GLYPHS, bars=None):
    """Frames of all images, with their bar readings. bars: optional list of (X, Y, Z), one per frame,
    used instead of reading the bars (for bars the glyph reader cannot read).

    glyphs: a glyph file, or 'auto' for the stored bar font that reads these frames best. The bar is read in
    the bottom 1/30 of each frame (or bar_px) and, without bar_px, the frame is cropped above its black bar
    (shorter than that strip on ALERTWildfire frames), as the stills are. Where the black bar starts is the
    most common clear reading over the frames of one size (bar_geometry): on one frame alone, a dark scene
    above the bar can look like part of it."""
    from camstills.bar import Glyphs, bar_geometry, default_bar_px, pick_glyphs, read_bar
    parts = [(path, k, part) for path in paths
             for k, part in enumerate(split_frames(np.asarray(Image.open(path).convert('RGB')), n_frames,
                                                   frame_width))]
    grays = [np.asarray(Image.fromarray(part).convert('L')) for _, _, part in parts]
    geometry = {h: bar_geometry([gr for gr in grays if gr.shape[0] == h], bar_px) for h in {gr.shape[0] for gr in grays}}
    g = None
    if bars is None and glyphs == 'auto':                # picked on the frames of the most common size
        heights = [gr.shape[0] for gr in grays]
        h = max(set(heights), key=heights.count)
        _, g = pick_glyphs([gr for gr in grays if gr.shape[0] == h], bar_px, top=geometry[h][0])
        if g is None:
            raise ValueError("None of the stored bar fonts (camstills/glyphs) reads the frames' info bars.")
    elif bars is None:
        g = Glyphs.load(glyphs)
    frames = []
    for (path, k, part), gr in zip(parts, grays):
        nb = bar_px or default_bar_px(part.shape[0])
        top, height = geometry[gr.shape[0]]
        crop = bar_px or height
        fr = Frame(source=path, index=k, rgb=np.ascontiguousarray(part[:-crop]), full=part)
        if bars is None:
            r = read_bar(gr, g, nb, top)
            if r.X is not None and r.Y is not None and r.Z is not None:
                fr.bar, fr.time, fr.elev_ft = (r.X, r.Y, r.Z), r.timestamp, r.elev_ft
            else:
                fr.note = 'info bar not readable (%s)' % '; '.join(r.problems)
        frames.append(fr)
    if bars is not None:
        if len(bars) != len(frames):
            raise ValueError('%d bar readings given for %d frames' % (len(bars), len(frames)))
        for fr, b in zip(frames, bars):
            fr.bar = tuple(float(x) for x in b)
    return frames


def detect(frames, min_columns=0.25, spacing=2, **kw):
    """Detect the skyline in every frame; frames with too little of it are marked skipped."""
    for fr in frames:
        if fr.note:
            continue
        fr.uv = skyline_points(fr.rgb, spacing, **kw)
        if len(fr.uv) < min_columns * fr.nx / spacing:
            fr.note = 'skyline found in only %d%% of columns' % round(100 * len(fr.uv) * spacing / fr.nx)


# ---------------------------------------------------------------------- fit ---

class _Fitter:
    def __init__(self, frames, cam, dem, sky_rays, hfov0):
        self.frames = frames
        self.sky_rays = sky_rays
        self.hfov0 = hfov0
        self.models = {}
        for fr in frames:
            if (fr.nx, fr.ny) not in self.models:
                self.models[fr.nx, fr.ny] = core.CameraModel(cam, dem, fr.nx, fr.ny)

    def residual(self, c, fr, uv=None):
        """Detected minus DEM skyline row at each detected column, in degrees (positive = photo lower).

        Pixels are converted to degrees with the STARTING field of view, a fixed scale: with the fitted
        one, shrinking the HFOV would shrink every residual and the fit could collapse to a tiny HFOV.
        """
        uv = fr.uv if uv is None else uv
        m = self.models[fr.nx, fr.ny]
        q = core.pose_from_bar(fr.bar, c)
        u, v, front = m.project(self.sky_rays, q)
        ok = front & np.isfinite(u) & np.isfinite(v) & (u > -0.2 * fr.nx) & (u < 1.2 * fr.nx)
        out = np.full(len(uv), MAX_RESID_DEG)
        if ok.sum() < 2:
            return out
        o = np.argsort(u[ok])
        mv = np.interp(uv[:, 0], u[ok][o], v[ok][o], left=np.nan, right=np.nan)
        r = (uv[:, 1] - mv) / self.px_per_deg(fr)
        good = np.isfinite(r)
        out[good] = np.clip(r[good], -MAX_RESID_DEG, MAX_RESID_DEG)
        return out

    def px_per_deg(self, fr):
        hfov = core.pose_from_bar(fr.bar, [0, 0, 0, self.hfov0])[3]
        return (fr.nx / 2) / np.tan(np.radians(hfov / 2)) * np.pi / 180

    def all_residuals(self, c, sub=1):
        return np.concatenate([self.residual(c, fr, fr.uv[fr.keep][::sub]) for fr in self.frames])

    def coarse(self, c0, az_range=180.0, n_candidates=5, sub=8):
        """Starting corrections: the best few azimuth offsets within +-az_range, each with its tilt offset.

        While searching azimuth, each frame's own offset (or offset and slope)
        is removed from its residuals, so tilt and roll errors, and a leaning
        pan axis that tilts and rolls each frame differently, do not pull the
        search. Candidates from both variants are kept, because a short
        skyline (one frame) can resemble another stretch of the horizon; the
        full fit decides between them.
        """
        base = core.full_correction(c0)

        def az_score(d_az, slope):
            c = base.copy()
            c[0] = d_az
            s = 0.0
            for fr in self.frames:
                uv = fr.uv[::sub]
                r = self.residual(c, fr, uv)
                a = np.column_stack([np.ones(len(uv)), 2 * uv[:, 0] / fr.nx - 1][:2 if slope else 1])
                trend = a @ np.linalg.lstsq(a, r, rcond=None)[0] if slope else np.median(r)
                s += core.huber(np.abs(r - trend), 0.2).mean()
            return s

        def tilt_score(c, dt):
            c = c.copy()
            c[1] = dt
            return sum(core.huber(np.abs(self.residual(c, fr, fr.uv[::sub])), 0.2).mean() for fr in self.frames)

        span = min(180.0, az_range)
        grid = np.arange(-span, span + 1e-9 if span < 180 else 180, 1.0)
        n = len(grid)
        wrap = span >= 180
        starts = []                                     # (azimuth, slope variant)
        for slope in (True, False):
            scores = np.array([az_score(a, slope) for a in grid])
            is_min = [scores[k] <= scores[(k - 1) % n if wrap else max(k - 1, 0)]
                      and scores[k] <= scores[(k + 1) % n if wrap else min(k + 1, n - 1)] for k in range(n)]
            picked = 0
            for k in sorted(np.flatnonzero(is_min), key=lambda k: scores[k]):
                if all(abs((grid[k] - a + 180) % 360 - 180) >= 5 for a, _ in starts):
                    starts.append((grid[k], slope))
                    picked += 1
                if picked == n_candidates:
                    break
        out = []
        for a0, slope in starts:
            fine = np.arange(a0 - 1.5, a0 + 1.51, 0.1)
            c = base.copy()
            c[0] = fine[np.argmin([az_score(a, slope) for a in fine])]
            tilts = np.arange(-15, 15.01, 0.1)
            c[1] = tilts[np.argmin([tilt_score(c, t) for t in tilts])]
            out.append(c)
        return out

    def solve(self, c0, free, sub=1):
        c = core.full_correction(c0).copy()

        def f(x):
            cc = c.copy()
            cc[free] = x
            return self.all_residuals(cc, sub)
        lo = np.array([-540, -45, -45, 3, -10, -10.])[free]
        hi = np.array([540, 45, 45, 170, 10, 10.])[free]
        # The residual is piecewise linear in the parameters (interpolated DEM skyline), so keep
        # least_squares' default finite-difference step and tight tolerances: with a larger step or
        # looser tolerances it stops early, on the weakly determined azimuth offset in particular.
        res = least_squares(f, np.clip(c[free], lo, hi), bounds=(lo, hi), loss='soft_l1', f_scale=LOSS_SCALE,
                            x_scale=1.0, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=3000)
        c[free] = res.x
        return c, res


def _pan_spread(frames):
    """Largest angular span between the frames' pan directions (deg, 0..360 minus the biggest gap)."""
    pans = np.sort(np.mod([fr.bar[0] for fr in frames], 360))
    if len(pans) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([pans, pans[:1] + 360]))
    return 360 - gaps.max()


def calibrate(frames, cam, tiles, hfov0=60.0, lean='auto', az_range=AZ_RANGE, az_step=0.05, verbose=True, log=print,
              dem=None):
    """Fit the mount correction to the frames' detected skylines.

    lean: 'auto' fits the pan-axis lean when usable frames' pans span >= LEAN_MIN_SPREAD
    deg (each frame's tilt and roll then constrain it); 'fit' always; 'off' never (lean = 0).
    az_range: the azimuth offset (pan zero vs true north) is searched within +-az_range deg (180: anywhere).
    dem: a core.DEM of the tiles already in memory (else it is read from tiles).
    Returns (correction (6 values), report dict). Frames' keep/stats are filled in.
    """
    say = log if verbose else (lambda *a: None)
    use = [fr for fr in frames if not fr.note]
    for fr in frames:
        say('  %-34s %s' % (fr.name, ('X %+8.2f Y %+6.2f Z %4.1f  %4d skyline points'
                                       % (*fr.bar, len(fr.uv))) if not fr.note else 'skipped: ' + fr.note))
    if not use:
        raise ValueError('No frame has a readable bar and a clear skyline.')
    say('Computing the DEM skyline...')
    dem = dem if dem is not None else core.DEM(list(tiles))
    any_model = core.CameraModel(cam, dem, use[0].nx, use[0].ny)
    fit = _Fitter(use, cam, dem, any_model.horizon_rays(0, 360, az_step), hfov0)
    for fr in use:
        fr.keep = np.ones(len(fr.uv), bool)
    spread = _pan_spread(use)
    fit_lean = lean == 'fit' or (lean == 'auto' and len(use) >= 2 and spread >= LEAN_MIN_SPREAD)
    free = np.arange(6 if fit_lean else 4)
    say('Fitting %s to %d frames (pans span %.0f deg)...'
        % ('offsets, HFOV and pan-axis lean' if fit_lean else 'offsets and HFOV (no lean)', len(use), spread))

    # Every candidate on a quarter of the points, then the best one on all of them. Candidates are
    # ranked by median |error|, which ignores up to half the points (trees, roof edges, haze bands).
    tried = []
    for c in fit.coarse([0, 0, 0, hfov0, 0, 0], az_range):
        c, _ = fit.solve(c, free, sub=4)
        tried.append((float(np.median(np.abs(fit.all_residuals(c, 4)))), c))
    tried.sort(key=lambda t: t[0])
    best, c = tried[0]
    rival = next((t for t in tried[1:] if abs((t[1][0] - c[0] + 180) % 360 - 180) > 5), None)
    ambiguous = rival is not None and rival[0] < 1.3 * best
    c, _ = fit.solve(c, free)
    for fr in use:                                   # drop trees, roof edges, clouds...
        r = fit.residual(c, fr)
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        fr.keep = np.abs(r) <= max(REJECT_DEG, 4 * sigma)
    c, res = fit.solve(c, free)
    c[0] = (c[0] + 180) % 360 - 180

    for fr in use:
        r = fit.residual(c, fr) * fit.px_per_deg(fr)
        k = fr.keep
        fr.stats = {'median_abs_px': float(np.median(np.abs(r[k]))) if k.any() else np.nan,
                    'p90_abs_px': float(np.percentile(np.abs(r[k]), 90)) if k.any() else np.nan,
                    'median_px': float(np.median(r[k])) if k.any() else np.nan,
                    'kept': float(k.mean())}
    mag, toward = core.lean_description(c)
    report = {'frames_used': len(use), 'pan_spread_deg': spread, 'lean_fitted': fit_lean, 'ambiguous': ambiguous,
              'lean_deg': mag, 'lean_toward_az_deg': toward, 'cost': float(res.cost),
              'models': fit.models, 'sky_rays': fit.sky_rays, 'fit_residual': fit.residual, 'px_per_deg': fit.px_per_deg}
    say('Correction: d_azimuth %+.3f  d_tilt %+.3f  roll %+.3f  HFOV(Z1) %.3f  lean east %+.3f  north %+.3f'
        % tuple(c))
    if fit_lean:
        say('Pan axis leans %.2f deg toward azimuth %.0f' % (mag, toward))
    if ambiguous:
        say('WARNING: an azimuth offset of %+.1f deg fits almost as well (median error %.3f vs %.3f deg). Check the '
            'overlay; more frames or a smaller --az-range settle it.' % (rival[1][0], rival[0], best))
    if az_range < 180 and abs(c[0]) > 0.9 * az_range:
        say('WARNING: the azimuth offset (%+.1f deg) is at the edge of the +-%g deg search. The bar X should be '
            'true north to a few degrees: check the camera site and the overlay.' % (c[0], az_range))
    for fr in use:
        s = fr.stats
        say('  %-34s median |error| %5.2f px   90%% %6.2f px   median %+6.2f px   %3.0f%% of points used'
            % (fr.name, s['median_abs_px'], s['p90_abs_px'], s['median_px'], 100 * s['kept']))
    overall = float(np.median(np.concatenate([np.abs(fit.residual(c, fr)[fr.keep]) * fit.px_per_deg(fr) for fr in use])))
    report['median_abs_px'] = overall
    if overall > POOR_FIT_PX:
        say('WARNING: the DEM skyline fits poorly (median error %.1f px; good fits are 2-4 px). Common causes: the '
            'camera site is off by some hundred metres (nearby terrain is then wrong), haze or smoke hides the '
            'skyline, or DEM tiles are missing. Check the overlay.' % overall)
    return c, report


# ------------------------------------------------------------------ outputs ---

def overlay(frames, c, report, path, scale=0.5):
    """Image of all frames: detected skyline yellow (orange = not used), fitted DEM skyline red."""
    tiles = []
    for fr in frames:
        im = Image.fromarray(fr.full if fr.full is not None else fr.rgb)
        d = ImageDraw.Draw(im)
        lw = max(1, round(2 / scale / 2))
        if fr.uv is not None and len(fr.uv):
            keep = fr.keep if fr.keep is not None else np.ones(len(fr.uv), bool)
            for (u, v), k in zip(fr.uv, keep):
                d.rectangle((u - 1 - lw, v - 1 - lw, u - 1 + lw, v - 1 + lw),
                            fill=(255, 230, 0) if k else (255, 140, 0))
        if not fr.note:
            m = report['models'][fr.nx, fr.ny]
            u, v, front = m.project(report['sky_rays'], core.pose_from_bar(fr.bar, c))
            ok = front & (u >= 1) & (u <= fr.nx) & (v >= 1) & (v <= fr.ny)
            o = np.argsort(u[ok])
            if ok.sum() > 1:
                d.line(list(zip(u[ok][o] - 1, v[ok][o] - 1)), fill=(255, 40, 40), width=lw + 1)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        label = fr.name + ('  skipped: ' + fr.note if fr.note else
                           '  X %+.2f  median error %.1f px' % (fr.bar[0], fr.stats.get('median_abs_px', np.nan)))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 12 + 6 * len(label), 20), fill=(0, 0, 0))
        d.text((6, 4), label, fill=(255, 255, 255))
        tiles.append(im)
    out = Image.new('RGB', (sum(t.width for t in tiles), max(t.height for t in tiles)))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    out.save(path, quality=90)


def save(path, frames, c, report, cam):
    """Calibration .mat in the registration app's format (no landmarks or skyline trace)."""
    use = [fr for fr in frames if not fr.note]
    ref = use[0]
    sources = sorted({os.path.basename(fr.source) for fr in frames})
    extra = {'panorama_files': core._cellstr(sources),
             'panorama_frames': core._cellstr([fr.name for fr in use]),
             'panorama_frames_X_Y_Z': np.array([fr.bar for fr in use], float),
             'panorama_frame_median_abs_error_px': np.array([[fr.stats['median_abs_px'] for fr in use]]),
             'panorama_lean_fitted': float(report['lean_fitted'])}
    core.save_calibration(path, cam, core.pose_from_bar(ref.bar, c), np.zeros((0, 5)), [], np.zeros((0, 2)), [],
                          (ref.ny, ref.nx), sources[0], correction=c, extra_fields=extra)


def run(paths, cam, tiles, out_dir, camera, hfov0=60.0, lean='auto', n_frames=None, frame_width=None,
        bar_px=None, glyphs=DEFAULT_GLYPHS, bars=None, az_range=AZ_RANGE, verbose=True, log=print, frames_out=None,
        method='edges', dem=None):
    """Whole pipeline; returns (correction, mat path, overlay path). dem: the DEM to use (e.g. a
    core.LayeredDEM already in memory); default the SRTM mosaic of tiles. cam may give lat, lon only:
    the elevation is then taken from the frames' info bars. log receives the progress lines; if
    frames_out is a list, the frames (with their bars, notes and error statistics) are appended to it.

    method 'edges' (default): the skyline fit below is the starting point, then edgefit puts the DEM skyline on
    the frames' silhouette edges (the detected skyline sits a little high, and the skyline fit only measures
    vertical errors). 'skyline': the skyline fit alone."""
    frames = load_frames(paths, n_frames, frame_width, bar_px, glyphs, bars)
    if frames_out is not None:
        frames_out.extend(frames)
    if len(cam) == 2:
        el = [fr.elev_ft for fr in frames if fr.elev_ft]
        if not el:
            raise ValueError('No camera elevation given and none on the info bars.')
        cam = (cam[0], cam[1], float(np.median(el)) * 0.3048)
        if verbose:
            log('Camera elevation from the info bar: %.1f m' % cam[2])
    detect(frames)
    dem = dem if dem is not None else core.DEM(list(tiles))
    c, report = calibrate(frames, cam, tiles, hfov0, lean, az_range, verbose=verbose, log=log, dem=dem)
    if method == 'edges':
        from . import edgefit
        use = [fr for fr in frames if not fr.note]
        efr = [edgefit.EdgeFrame.from_rgb('pano:' + fr.name, fr.name, fr.rgb, fr.bar, 'panorama', fr.note)
               for fr in frames if fr.bar]
        free = range(6) if report['lean_fitted'] else range(4)
        c, _, _ = edgefit.EdgeFit(efr, cam, dem).run(c, free, log=log if verbose else (lambda *a: None))
        for fr in use:                             # the overlay's error statistics, for the new correction
            r = report['fit_residual'](c, fr) * report['px_per_deg'](fr)
            k = fr.keep
            fr.stats.update(median_abs_px=float(np.median(np.abs(r[k]))), median_px=float(np.median(r[k])))
    os.makedirs(out_dir, exist_ok=True)
    mat = os.path.join(out_dir, '%s_panorama.mat' % camera)
    img = os.path.join(out_dir, '%s_panorama_overlay.jpg' % camera)
    save(mat, frames, c, report, cam)
    overlay(frames, c, report, img)
    if verbose:
        log('Wrote %s\n      %s' % (mat, img))
    return c, mat, img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+', help='panorama image(s): frames side by side, or one frame per file')
    ap.add_argument('--camera', required=True, help='camera name, used for the output file names')
    ap.add_argument('--cam', type=float, nargs='+', required=True, metavar='LAT LON [ELEV_M]',
                    help='camera site; without ELEV_M the elevation is read from the info bar')
    ap.add_argument('--tiles', nargs='+', required=True, help='SRTM .hgt tiles covering the view')
    ap.add_argument('--hfov', type=float, default=60.0, help='starting horizontal field of view at Z1 (default 60)')
    ap.add_argument('--lean', choices=('auto', 'fit', 'off'), default='auto',
                    help='fit the pan-axis lean: auto = when the frames span >= %g deg of pan (default)' % LEAN_MIN_SPREAD)
    ap.add_argument('--az-range', type=float, default=AZ_RANGE,
                    help='search the pan-zero offset from north within +-this many degrees (default %(default)g: the '
                         "bar's X is north-referenced to a few degrees; 180 searches all directions)")
    ap.add_argument('--method', choices=('edges', 'skyline'), default='edges',
                    help="'edges' (default): put the DEM skyline on the frames' silhouette edges, starting from the "
                         "skyline fit; 'skyline': the fit to the detected skyline alone (it sits a little high)")
    ap.add_argument('--out-dir', default='output')
    ap.add_argument('--frames', type=int, help='frames per image (default: from a 16:9 frame shape)')
    ap.add_argument('--frame-width', type=float, help='frame width in pixels (instead of --frames)')
    ap.add_argument('--bar-px', type=int, help='info bar height in pixels (default 1/30 of the frame height)')
    ap.add_argument('--glyphs', default='auto',
                    help="bar font glyphs (camstills/glyphs/*.npz), or 'auto' (default): the font that reads the bars")
    ap.add_argument('--bars', help='X,Y,Z for each frame in order, separated by ";", instead of reading the bars; '
                    'write it as --bars="..." when it starts with a minus sign')
    a = ap.parse_args(argv)
    if len(a.cam) not in (2, 3):
        ap.error('--cam takes LAT LON [ELEV_M]')
    bars = [tuple(map(float, b.split(','))) for b in a.bars.split(';')] if a.bars else None
    np.seterr(all='ignore')
    run(a.images, tuple(a.cam), a.tiles, a.out_dir, a.camera, a.hfov, a.lean, a.frames, a.frame_width,
        a.bar_px, a.glyphs, bars, a.az_range, method=a.method)


if __name__ == '__main__':
    main()
