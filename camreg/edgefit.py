"""Calibrate a camera's mount correction by putting the DEM skyline on the silhouette edges of its images.

The panorama fit (panorama.calibrate) compares the model with an automatically detected skyline, and only
vertically. The detector stops at the bright haze band above distant ridges (the fit then sits high), and a
left/right offset is invisible wherever the skyline is flat. Here the model is compared with the images
themselves: along the projected DEM skyline, the image must be bright just above the line (sky) and dark just
below (terrain). Maximising that edge contrast puts the line on the silhouette, in both directions. Landmarks
(a pixel with known lat/lon/elevation) pull the fit toward their projections.

The camera does not return exactly to the pan/tilt its bar reports: each position has its own small error
(Chimney Peak's fire-day pose sat 0.7 deg off in pan while its panorama frames ranged -0.8..+0.8). So:

  1. The shared correction (d_azimuth, d_tilt, roll, HFOV, pan-axis lean) is fitted to the panorama frames, each
     frame with its own small pan offset (a nuisance, re-centred so d_azimuth is their mean). Without a
     panorama the stills are used instead, without offsets.
  2. Each still pose (bar X/Y/Z) then gets a pan/tilt offset from up to a few of its stills, applied when they
     agree (core.pose_from_bar(bar, c, offsets)). Landmarks on a pose's stills always set its offset.

    frames = [EdgeFrame.from_rgb(id, label, rgb_above_bar, (X, Y, Z), kind), ...]   # kind 'panorama' or 'still'
    landmarks = [Landmark(frame_id, u, v, lat, lon, elev, name), ...]
    c, offsets, report = EdgeFit(frames, (lat, lon, elev_m), dem, landmarks).run(c_start, free=range(6))

Edge contrast of a frame: the median, over sampled columns, of (mean grey 1-4 px above the line) - (mean grey
1-4 px below), on the grey image blurred by sigma (widths scaled to a 1920-px-wide frame). It is divided by the
best the frame reaches near the starting pose, so every frame weighs about the same. A landmark's error is its
angular distance from its projection, through a Huber loss. Nothing here is specific to one camera.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares, minimize

from . import core

LANDMARK_WEIGHT = 0.1     # a landmark 0.2 deg off costs about a third of one frame's edge alignment
STILLS_LANDMARK_WEIGHT = 0.3  # without a panorama: fire-day stills (smoke, haze, dusk) have weaker, less reliable edges
LANDMARK_SCALE_DEG = 0.05  # Huber transition of the landmark error
MIN_CONTRAST = 3.0        # frames whose best edge contrast is below this (grey levels) are not used
FULL_CONTRAST = 10.0      # frames weigh less below this edge contrast (weak, hazy or cluttered edges)
SHIFT_X, SHIFT_Y = 40, 50  # local shift search, +- px of a 1920-px-wide frame
RAY_STEP = 0.03           # deg between DEM skyline rays
POSE_MIN_DEG = 0.1        # a pose offset (edges only) is applied if at least this large...
POSE_MAX_SPREAD = 0.15    # ...and its stills agree to within this (median absolute deviation, deg)
POSE_MIN_STILLS = 3       # ...measured on at least this many stills
LM_ROLL_SPAN_DEG = 10.0    # landmarks must span this much azimuth for the landmark start to fit roll...
LM_HFOV_SPAN_DEG = 20.0    # ...and this much (with 4 or more) to fit the HFOV
NAMES = ('d_azimuth', 'd_tilt', 'roll', 'HFOV(Z1)', 'lean east', 'lean north')


@dataclass
class EdgeFrame:
    id: str                    # a still's file name, or 'pano:<panorama file>#<k>'
    label: str
    gray: np.ndarray           # H x W float grey image above the info bar
    bar: tuple                 # (X, Y, Z) as read from the bar
    kind: str = 'still'        # 'still' or 'panorama'
    note: str = ''             # why it is not used, if it isn't
    off: list = field(default_factory=lambda: [0.0, 0.0])   # pan, tilt offset (deg) of its position
    turn: tuple = (0.0, 1.0)   # roll (deg) and scale of its position (a manual adjustment; core.turn_pose)
    blurred: dict = field(default_factory=dict, repr=False)
    s: float = np.nan          # best edge contrast near the starting pose (normalisation)

    @classmethod
    def from_rgb(cls, id, label, rgb, bar, kind='still', note=''):
        gray = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).convert('L'), float)
        return cls(id, label, gray, tuple(float(x) for x in bar), kind, note)

    @property
    def w(self):
        return self.gray.shape[1]

    @property
    def h(self):
        return self.gray.shape[0]

    @property
    def scale(self):
        return self.w / 1920.0

    @property
    def eff_bar(self):
        return (self.bar[0] + self.off[0], self.bar[1] + self.off[1], self.bar[2])

    def image(self, sigma):
        if sigma not in self.blurred:
            self.blurred[sigma] = gaussian_filter(self.gray, sigma * self.scale)
        return self.blurred[sigma]


@dataclass
class Landmark:
    frame: str                 # EdgeFrame.id it was clicked on
    u: float                   # 1-based pixel column, row in that frame
    v: float
    lat: float
    lon: float
    elev: float
    name: str = ''
    sigma_deg: float = 0.0     # angular accuracy of its coordinates seen from the camera (reference.candidates)

    @property
    def scale_deg(self):
        """Huber transition of its error: LANDMARK_SCALE_DEG, or wider for coordinates known less well."""
        return max(LANDMARK_SCALE_DEG, float(self.sigma_deg or 0.0))


class EdgeFit:
    def __init__(self, frames, cam, dem, landmarks=()):
        self.cam = tuple(float(x) for x in cam)
        self.frames = list(frames)
        self.by_id = {fr.id: fr for fr in self.frames}
        self.landmarks = [lm for lm in landmarks if lm.frame in self.by_id]
        self.landmark_weight = LANDMARK_WEIGHT
        self.models = {}
        for fr in self.frames:
            if (fr.w, fr.h) not in self.models:
                self.models[fr.w, fr.h] = core.CameraModel(self.cam, dem, fr.w, fr.h)
        self.rays = next(iter(self.models.values())).horizon_rays(0, 360, RAY_STEP)

    # ------------------------------------------------------------ pieces --
    def model(self, fr):
        return self.models[fr.w, fr.h]

    def pose(self, fr, c):
        return core.turn_pose(core.pose_from_bar(fr.eff_bar, c), *fr.turn)

    def px_per_deg(self, fr, c):
        return (fr.w / 2) / np.tan(np.radians(self.pose(fr, c)[3] / 2)) * np.pi / 180

    def line(self, fr, c, margin=0.05):
        """Sample points (u, v), 1-based, of the projected DEM skyline inside the frame (None if too few)."""
        u, v, front = self.model(fr).project(self.rays, self.pose(fr, c))
        ok = front & np.isfinite(u) & (u >= margin * fr.w) & (u <= (1 - margin) * fr.w) \
            & (v > margin * fr.h) & (v < (1 - margin) * fr.h)
        if ok.sum() < 20:
            return None
        o = np.argsort(u[ok])
        uu = np.arange(np.ceil(u[ok].min()), np.floor(u[ok].max()), max(1.0, fr.w / 960.0))
        if len(uu) < 20:
            return None
        return uu, np.interp(uu, u[ok][o], v[ok][o])

    def contrast(self, fr, c, sigma=1.0, line=None, dx=0.0, dy=0.0):
        """Median grey difference just above minus just below the (shifted) skyline; nan without a line."""
        line = self.line(fr, c) if line is None else line
        if line is None:
            return np.nan
        uu, vv = line
        g = fr.image(sigma)
        k0, k1 = max(1, round(fr.scale)), max(3, round(4 * fr.scale * sigma ** 0.5))
        x = uu - 1 + dx
        above = np.mean([map_coordinates(g, [vv - 1 + dy - k, x], order=1, mode='nearest') for k in range(k0, k1 + 1)], 0)
        below = np.mean([map_coordinates(g, [vv - 1 + dy + k, x], order=1, mode='nearest') for k in range(k0, k1 + 1)], 0)
        return float(np.median(above - below))

    def shift_search(self, fr, c, sigma=1.0, measure=False):
        """(best contrast, dx, dy in px, dx, dy in deg): where the silhouette is relative to the drawn line.

        The fit uses a coarse grid of 4 px across by 2 px down. measure=True (the reports) uses 8 px across by
        1 px down instead: a sharp edge is only a few px tall, so when the line is an odd number of px off it the
        2-px grid can step over it and settle on a weaker edge far to the side (a hand adjustment shows that);
        sideways the contrast changes slowly, and the fine search covers +-4 px."""
        line = self.line(fr, c)
        if line is None:
            return np.nan, 0.0, 0.0, 0.0, 0.0
        s = fr.scale
        step_x, step_y = (8, 1) if measure else (4, 2)
        grid = [(self.contrast(fr, c, sigma, line, dx * s, dy * s), dx, dy)
                for dx in range(-SHIFT_X, SHIFT_X + 1, step_x) for dy in range(-SHIFT_Y, SHIFT_Y + 1, step_y)]
        _, bx, by = max(grid)
        fine = [(self.contrast(fr, c, sigma, line, dx * s, dy * s), dx, dy)
                for dx in np.arange(bx - 4, bx + 4.01, 0.5) for dy in np.arange(by - 2, by + 2.01, 0.5)]
        best, bx, by = max(fine)
        ppd = self.px_per_deg(fr, c)
        return best, bx * s, by * s, bx * s / ppd, by * s / ppd

    def landmark_errors(self, c, landmarks=None):
        """Per landmark: (error deg, error px, predicted u, predicted v)."""
        out = []
        for lm in self.landmarks if landmarks is None else landmarks:
            fr = self.by_id[lm.frame]
            m = self.model(fr)
            u, v, front = m.project(m.ground_rays([[lm.lat, lm.lon, lm.elev]]), self.pose(fr, c))
            if not front[0] or not np.isfinite(u[0]):
                out.append((np.inf, np.inf, np.nan, np.nan))
                continue
            px = float(np.hypot(u[0] - lm.u, v[0] - lm.v))
            out.append((px / self.px_per_deg(fr, c), px, float(u[0]), float(v[0])))
        return out

    def objective(self, c, sigma, frames, landmarks):
        edge, w = [], []
        for fr in frames:
            k = self.contrast(fr, c, sigma)
            edge.append(k / fr.s if np.isfinite(k) else -1.0)
            w.append(min(1.0, fr.s / FULL_CONTRAST))
        lms = self.landmarks if landmarks is None else landmarks
        lm = sum(float(core.huber(min(e, 10.0) / l.scale_deg, 1.0)) for l, (e, *_) in zip(lms, self.landmark_errors(c, lms)))
        return -float(np.dot(edge, w) / np.sum(w)) + self.landmark_weight * lm / float(np.sum(w))   # per full-weight frame

    def _nelder_mead(self, f, x0, steps, maxiter):
        simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[k] * steps[k] for k in range(len(x0))])
        return minimize(f, x0, method='Nelder-Mead',
                        options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-4, maxiter=maxiter)).x

    def landmark_start(self, c, log=print):
        """Starting correction from the landmarks alone (no edges): azimuth and tilt offsets from one landmark,
        roll too from two or more at least LM_ROLL_SPAN_DEG apart in azimuth, and HFOV from four or more
        spanning LM_HFOV_SPAN_DEG. Returns (c, names of the fitted values) or (c, []) without landmarks."""
        lms = [lm for lm in self.landmarks if lm.frame in self.by_id and not self.by_id[lm.frame].note]
        if not lms:
            return c, []
        lat_c, lon_c, _ = self.cam
        az = np.array([np.degrees(np.arctan2(np.radians(lm.lon - lon_c) * np.cos(np.radians(lat_c)),
                                             np.radians(lm.lat - lat_c))) % 360 for lm in lms])
        a = np.sort(az)
        span = 0.0 if len(a) < 2 else 360 - np.max(np.diff(np.r_[a, a[0] + 360]))   # smallest arc holding them all
        free = [0, 1] + ([2] if len(lms) >= 2 and span >= LM_ROLL_SPAN_DEG else []) \
            + ([3] if len(lms) >= 4 and span >= LM_HFOV_SPAN_DEG else [])
        c = core.full_correction(c).astype(float)

        def res(x):
            cc = c.copy()
            cc[free] = x
            out = []
            for lm in lms:
                fr = self.by_id[lm.frame]
                m = self.model(fr)
                u, v, front = m.project(m.ground_rays([[lm.lat, lm.lon, lm.elev]]), self.pose(fr, cc))
                ppd = self.px_per_deg(fr, cc)
                if not front[0] or not np.isfinite(u[0]):
                    out += [30.0, 30.0]
                else:
                    out += [(u[0] - lm.u) / ppd / lm.scale_deg, (v[0] - lm.v) / ppd / lm.scale_deg]
            return np.array(out)

        r = least_squares(res, c[free], loss='soft_l1', f_scale=1.0)
        before = self.landmark_errors(c, lms)
        c[free] = r.x
        after = self.landmark_errors(c, lms)
        log('Landmark start (%d landmarks spanning %.0f deg of azimuth): %s; landmark errors %s -> %s deg'
            % (len(lms), span, ', '.join('%s %+.3f' % (NAMES[i], c[i]) for i in free),
               ' '.join('%.2f' % e for e, *_ in before), ' '.join('%.3f' % e for e, *_ in after)))
        return c, [NAMES[i] for i in free]

    def _grid(self, c, d_az, d_tilt, sigma, frames, landmarks, normalised=True):
        best = (np.inf, c)
        for da in d_az:
            for dt in d_tilt:
                cc = c.copy()
                cc[0] += da
                cc[1] += dt
                j = self.objective(cc, sigma, frames, landmarks) if normalised else \
                    -np.nanmean([self.contrast(fr, cc, sigma) for fr in frames])
                if j < best[0]:
                    best = (j, cc)
        return best[1]

    def _frame_pan(self, fr, c):
        """Pan offset of one panorama frame that best aligns its own edges (and its landmarks)."""
        lms = [lm for lm in self.landmarks if lm.frame == fr.id]
        base = fr.off[0]
        best = (np.inf, base)
        for sigma, span, step in ((2.0, 1.5, 0.05), (1.0, 0.06, 0.01)):
            centre = best[1]
            for d in np.arange(centre - span, centre + span + 1e-9, step):
                fr.off[0] = d
                j = self.objective(c, sigma, [fr], lms)
                if j < best[0]:
                    best = (j, d)
        fr.off[0] = best[1]

    @staticmethod
    def _recentre(c, frames):
        """Move the frames' mean pan offset into d_azimuth (the same poses)."""
        mean = float(np.mean([fr.off[0] for fr in frames]))
        for fr in frames:
            fr.off[0] -= mean
        c = c.copy()
        c[0] += mean
        return c

    # --------------------------------------------------------------- fit --
    def run(self, c_start, free=(0, 1, 2), wide=False, log=print):
        """Returns (correction (6), pose offsets {pose key: [pan, tilt]}, report)."""
        c = core.full_correction(c_start).astype(float)
        free = np.array(sorted(set(int(i) for i in free)))
        pano = [fr for fr in self.frames if fr.kind == 'panorama' and not fr.note]
        stills = [fr for fr in self.frames if fr.kind == 'still' and not fr.note]
        if wide:                                   # far from any calibration: a broad search on blurred images
            log('Broad search: azimuth +-10 deg, tilt +-3 deg...')
            c = self._grid(c, np.arange(-10, 10.01, 0.25), np.arange(-3, 3.01, 0.1), 8.0, pano or stills, [], normalised=False)
        log('Measuring the edges near the starting pose...')
        before = {}
        for fr in self.frames:
            if fr.note:
                continue
            best, _, _, dxd, dyd = self.shift_search(fr, c)
            fr.s = best
            before[fr.id] = dict(contrast=self.contrast(fr, c), best=best, dx_deg=dxd, dy_deg=dyd)
            if not np.isfinite(best) or best < MIN_CONTRAST:
                fr.note = 'no clear skyline edge (contrast %.1f)' % (best if np.isfinite(best) else 0)
        for fr in self.frames:
            b = before.get(fr.id)
            log('  %-40s %s' % (fr.label, ('silhouette %+.2f deg right, %+.2f deg down of the line; edge %.1f'
                                            % (b['dx_deg'], b['dy_deg'], b['best'])) if not fr.note else 'not used: ' + fr.note))
        pano = [fr for fr in pano if not fr.note]
        stills = [fr for fr in stills if not fr.note]
        glob = pano or stills
        if not glob:
            raise ValueError('No frame shows a clear skyline edge near the starting pose.')
        glob_ids = {fr.id for fr in glob}
        glob_lm = [lm for lm in self.landmarks if lm.frame in glob_ids]
        log('Fitting %s to %d %s%s...' % (', '.join(NAMES[i] for i in free), len(glob),
                                          'panorama frames (each with its own pan offset)' if pano else 'stills',
                                          ' and %d landmarks' % len(glob_lm) if glob_lm else ''))
        if pano:                                   # each frame's own pan error first, so it can't bend the shared values
            for fr in pano:
                fr.off[0] = -before[fr.id]['dx_deg']
            c = self._recentre(c, pano)
        else:
            c = self._grid(c, np.arange(-1.5, 1.501, 0.1), np.arange(-1.0, 1.001, 0.05), 4.0, glob, glob_lm)
        steps = np.array([0.1, 0.1, 0.2, 0.01 * c[3], 0.2, 0.2])
        for sigma in (4.0, 2.0, 1.0, 1.0):
            def f(x, sigma=sigma):
                cc = c.copy()
                cc[free] = x
                return self.objective(cc, sigma, glob, glob_lm)
            c[free] = self._nelder_mead(f, c[free], steps[free], 150 * len(free))
            steps = steps / 2
            if pano:                               # then re-measure each frame's pan error with the new values
                for fr in pano:
                    if any(lm.frame == fr.id for lm in glob_lm):
                        self._frame_pan(fr, c)
                    else:
                        fr.off[0] -= self.shift_search(fr, c)[3]
                c = self._recentre(c, pano)
        c[0] = (c[0] + 180) % 360 - 180
        offsets = self._pose_offsets(c, stills if pano else [], log)
        return c, offsets, self.report(core.full_correction(c_start), c, offsets, before, free, bool(pano), log)

    def _pose_offsets(self, c, stills, log):
        """Pan/tilt offset per still pose: from its landmarks if it has any, else from its stills' edges when they
        agree. Applied to the frames (fr.off); returns {pose key: dict}."""
        out = {}
        poses = {}
        for fr in stills:
            poses.setdefault(core.pose_key(fr.bar), []).append(fr)
        for key, frs in poses.items():
            lms = [lm for lm in self.landmarks if lm.frame in {fr.id for fr in frs}]
            shifts = np.array([self.shift_search(fr, c)[3:] for fr in frs])     # silhouette right, down (deg)
            pan, tilt = -np.median(shifts[:, 0]), np.median(shifts[:, 1])
            mad = np.median(np.abs(shifts - np.median(shifts, 0)), 0) if len(frs) > 1 else np.array([np.inf, np.inf])
            if lms:                                # landmarks: absolute; fit the two offsets on edges + landmarks
                for fr in frs:
                    fr.off = [pan, tilt]

                def f(x):
                    for fr in frs:
                        fr.off = [x[0], x[1]]
                    return self.objective(c, 1.0, frs, lms)
                x = self._nelder_mead(f, np.array([pan, tilt]), [0.05, 0.05], 200)
                pan, tilt, how, applied = float(x[0]), float(x[1]), 'landmarks', True
            else:
                applied = bool(len(frs) >= POSE_MIN_STILLS and max(abs(pan), abs(tilt)) >= POSE_MIN_DEG
                               and max(mad) <= POSE_MAX_SPREAD)
                how = 'edges' if applied else 'not applied (%s)' % (
                    'only %d stills' % len(frs) if len(frs) < POSE_MIN_STILLS else
                    'stills disagree by %.2f deg' % max(mad) if max(mad) > POSE_MAX_SPREAD else 'below %.1f deg' % POSE_MIN_DEG)
                pan, tilt = (float(pan), float(tilt)) if applied else (0.0, 0.0)
            for fr in frs:
                fr.off = [pan, tilt]
            out[key] = dict(pan=float(pan), tilt=float(tilt), applied=bool(applied), how=how, n_stills=len(frs),
                            spread_deg=[float(x) for x in mad] if len(frs) > 1 else None, n_landmarks=len(lms))
            log('  pose %-18s offset pan %+.2f, tilt %+.2f deg: %s (%d stills%s)'
                % (key, pan, tilt, how, len(frs), ', %d landmarks' % len(lms) if lms else ''))
        return out

    def report(self, c_old, c, offsets, before, free, by_panorama, log=print):
        frames = []
        for fr in self.frames:
            row = dict(id=fr.id, label=fr.label, kind=fr.kind, used=not fr.note, note=fr.note, bar=list(fr.bar),
                       offset=list(fr.off))
            if fr.id in before:
                best, _, _, dxd, dyd = self.shift_search(fr, c, measure=True)
                b = before[fr.id]
                row.update(contrast_before=b['contrast'], contrast_after=self.contrast(fr, c), best_contrast=best,
                           shift_before_deg=[b['dx_deg'], b['dy_deg']], shift_after_deg=[dxd, dyd])
            frames.append(row)
        lms = []
        for lm, (e, px, u, v) in zip(self.landmarks, self.landmark_errors(c)):
            lms.append(dict(name=lm.name, frame=lm.frame, error_deg=e, error_px=px, predicted=[u, v],
                            sigma_deg=lm.sigma_deg))
        mag, toward = core.lean_description(c)
        log('Correction: ' + '  '.join('%s %+.3f' % (n, x) for n, x in zip(NAMES, c)))
        log('  (start    ' + '  '.join('%s %+.3f' % (n, x) for n, x in zip(NAMES, c_old)) + ')')
        for row in frames:
            if row['used']:
                log('  %-40s remaining offset %+.2f deg right, %+.2f deg down; edge %.1f -> %.1f'
                    % (row['label'], *row['shift_after_deg'], row['contrast_before'], row['contrast_after']))
        for lm in lms:
            log('  landmark %-24s error %.3f deg (%.1f px)' % (lm['name'], lm['error_deg'], lm['error_px']))
        stills = [r for r in frames if r['used'] and r['kind'] == 'still']
        worst = max((max(abs(x) for x in r['shift_after_deg']) for r in stills), default=np.nan)
        return dict(method='edges', by='panorama frames' if by_panorama else 'stills', c_start=list(c_old), c=list(c),
                    free=[NAMES[i] for i in free], pose_offsets=offsets, frames=frames, landmarks=lms,
                    lean_deg=mag, lean_toward_deg=toward, worst_still_offset_deg=worst,
                    n_frames=len([r for r in frames if r['used']]), n_landmarks=len(lms))


# ------------------------------------------------------------------ output ---

def overlay(fit, c_old, offsets_old, c, path, height=360):
    """Frames side by side: old skyline red, new green (each frame with its offsets), landmarks (clicked: green
    circle; predicted: cyan x)."""
    tiles = []
    new_lm = fit.landmark_errors(c)
    for fr in fit.frames:
        im = Image.fromarray(np.clip(fr.gray, 0, 255).astype(np.uint8)).convert('RGB')
        d = ImageDraw.Draw(im)
        lw = max(1, round(2 * fr.scale))
        for q, col in ((core.pose_from_bar(fr.bar, c_old, offsets_old), (255, 40, 40)), (fit.pose(fr, c), (40, 230, 60))):
            u, v, f = fit.model(fr).project(fit.rays, q)
            ok = f & (u >= 1) & (u <= fr.w) & (v >= 1) & (v <= fr.h)
            o = np.argsort(u[ok])
            if ok.sum() > 1:
                d.line(list(zip(u[ok][o] - 1, v[ok][o] - 1)), fill=col, width=lw)
        for lm, (_, _, pu, pv) in zip(fit.landmarks, new_lm):
            if lm.frame != fr.id:
                continue
            r = 8 * fr.scale
            d.ellipse((lm.u - 1 - r, lm.v - 1 - r, lm.u - 1 + r, lm.v - 1 + r), outline=(40, 230, 60), width=lw)
            if np.isfinite(pu):
                d.line((pu - 1 - r, pv - 1 - r, pu - 1 + r, pv - 1 + r), fill=(0, 230, 255), width=lw)
                d.line((pu - 1 - r, pv - 1 + r, pu - 1 + r, pv - 1 - r), fill=(0, 230, 255), width=lw)
        im = im.resize((max(1, round(im.width * height / im.height)), height))
        label = fr.label + ('  (not used: %s)' % fr.note if fr.note else '')
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 12 + 6 * len(label), 16), fill=(0, 0, 0))
        d.text((6, 3), label, fill=(255, 255, 255))
        tiles.append(im)
    out = Image.new('RGB', (sum(t.width for t in tiles), height))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width
    out.save(path, quality=88)


def representative(stills, is_bright, n_poses=2, per_pose=2):
    """A few stills to calibrate on and click landmarks on: for each of the n_poses most common zoom-1 poses,
    up to per_pose stills for which is_bright(still) is true, spread over the pose's time (first and last
    always). stills: dicts with X, Y, Z, t."""
    groups = {}
    for st in stills:
        if st['Z'] == 1:
            groups.setdefault((st['X'], st['Y'], st['Z']), []).append(st)
    picks = []
    for g in sorted(groups.values(), key=len, reverse=True)[:n_poses]:
        g = sorted(g, key=lambda st: st['t'])
        want = [round(k) for k in np.linspace(0, len(g) - 1, per_pose)] if per_pose > 1 else [0]
        chosen = []
        for i in want:                             # nearest bright still to each target, searching outward
            for j in sorted(range(len(g)), key=lambda j: abs(j - i)):
                if g[j] not in chosen and is_bright(g[j]):
                    chosen.append(g[j])
                    break
        picks += [st for st in chosen if st not in picks]
    return picks
