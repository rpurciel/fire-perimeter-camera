"""Read the camera info bar (pan/tilt/zoom, elevation, timestamp) by glyph-template matching.

The bar is the black strip at the bottom of each frame, with white text such as

    axis-verdipeaklookout2 elev:8422' X:+31.96 Y:-0.16 Z:1.0 .NV-SFC.rgaines.08-22T16:00 © ALERTCalifornia 2026/08/22 17:08:34.51

It is drawn in one fixed font, so no OCR engine is needed: the bar is cut into glyphs at the dark columns
between characters (or, for a monospace font such as the ALERTWildfire bar's, into its fixed-width character
cells), and each glyph is matched against stored examples of that font (see train.py). Only the
fields with a fixed form are parsed (X, Y, Z, elev and the date/time); the free-text label in the middle is
not needed and is never relied on.

Bars from frames of other sizes are rescaled to the reference height (24 px of a 720 px frame) first: the text
size follows the frame height. The black bar itself can be shorter than that window (ALERTWildfire bars are
20 px of 720), so the rows of scene above it are found (bar_top) and blanked before reading.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image

REF_BAR_PX = 24            # bar height in the reference 1280 x 720 frames
REF_FRAME_H = 720
DARK = 40                  # bar pixels are darker than this (0-255)
INK = 130                  # a column holds text if any pixel is brighter than this (0-255)
MAX_GAP_MERGE = 2          # pieces closer than this many columns may be one glyph
SHIFTS = (-1, 0, 1)        # sub-pixel placement varies; compare at +-1 column
WIDTH_COST = 0.005         # per column of width difference
PAD = 2                    # empty columns left of each glyph on its canvas
CELL_INK = 105             # a monospace cell holds a character if any pixel is brighter than this (0-255)


def default_bar_px(frame_h):
    return max(8, round(frame_h * REF_BAR_PX / REF_FRAME_H))


def bar_strip(gray, bar_px=None):
    """Bottom bar of a grayscale frame (H x W, 0-255), rescaled to REF_BAR_PX rows, as float."""
    n = bar_px or default_bar_px(gray.shape[0])
    b = np.asarray(gray[-n:], dtype=np.uint8)
    if n != REF_BAR_PX:
        w = max(1, round(b.shape[1] * REF_BAR_PX / n))
        b = np.asarray(Image.fromarray(b).resize((w, REF_BAR_PX), Image.BILINEAR))
    return b.astype(float)


def dark_rows(b):
    """Per row of the bar strip, the fraction of near-black pixels in the left quarter (always inside the bar)."""
    return (b[:, :max(1, b.shape[1] // 4)] < DARK).mean(1)


def bar_top(b):
    """(first row of the black bar in the strip, whether that is certain).

    Scanning up from the bottom, text rows are at least ~40 % black and scene rows usually much less. If the
    black reaches the top of the strip, the bar fills it (or the scene above is as dark as the bar, as at
    night): (0, False). Otherwise the top is the first fully black row below the scene: (top, True)."""
    d = dark_rows(b)
    above = np.flatnonzero(d < 0.35)
    if not len(above):
        return 0, False
    k = above[-1] + 1
    while k < len(d) - 1 and d[k] < 0.97:                 # skip the blurred scene/bar boundary rows
        k += 1
    if k > len(d) // 2:                                    # no room left for the text: not a bar top
        return 0, False
    return int(k), True


def common_top(strips):
    """Bar top for a recording: the most common certain bar_top of some of its bar strips, else 0."""
    tops = [t for t, sure in map(bar_top, strips) if sure]
    return max(set(tops), key=tops.count) if tops else 0


def blank_above(b, top):
    """The strip with the scene rows above the bar set to black."""
    if top:
        b = b.copy()
        b[:top] = 0
    return b


def bar_end(b, top=0):
    """Column where the black bar stops (the bar is only as wide as its text).

    Uses the empty rows just below the bar's top: rows 1-4 of a bar that fills the strip (row 0 picks up
    compression bleed from the scene above, and glyphs start below row 4); below a shorter bar's top, the
    fully black ones of the next 4 rows, as its glyphs can start sooner."""
    rows = np.arange(top + 1, min(top + 5, len(b)))
    if top:
        clean = rows[dark_rows(b)[rows] >= 0.99]
        rows = clean if len(clean) else rows
    lit = b[rows].max(0) > 30                             # the bar itself is near-black there
    run = np.convolve(lit, np.ones(4), 'valid') >= 4      # 4 lit columns in a row = scene, not text
    i = np.flatnonzero(run)
    return int(i[0]) if len(i) else b.shape[1]


def ink_runs(b, end=None):
    """(start, stop) column ranges that contain text, left to right."""
    ink = b.max(0) > INK
    if end is not None:
        ink[end:] = False
    d = np.diff(np.r_[0, ink.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def cell_grid(b, pitch, end=None, spread=0.01, step=None):
    """(pitch, phase) of the character cells of a monospace bar: cell k spans phase + k*pitch to the next.

    The cell boundaries fall between characters, where the bar is darkest; the pitch is refined within
    +-`spread` of the given one (the rescaling to the reference height can leave it slightly off), by default
    in steps that move the last cell by 0.2 columns."""
    col = b.max(0)[:end]
    step = step or 0.2 * pitch / len(col)
    best = (np.inf, pitch, 0.0)
    for p in np.linspace(pitch * (1 - spread), pitch * (1 + spread), int(round(2 * spread * pitch / step)) + 1):
        phases = np.arange(0, p, 0.1)
        k = np.arange(int(len(col) / p) + 1)
        x = np.round(phases[:, None] + k[None, :] * p).astype(int)
        inside = x < len(col)
        cost = np.where(inside, col[np.minimum(x, len(col) - 1)], 0).sum(1) / inside.sum(1)
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (cost[i], p, phases[i])
    return best[1], best[2]


def char_cells(b, pitch, end=None):
    """(start, stop) column ranges of the monospace cells that hold a character (empty cells are spaces)."""
    end = b.shape[1] if end is None else end
    p, ph = cell_grid(b, pitch, end)
    col = b.max(0)
    edges = np.round(ph + np.arange(int((end - ph) / p) + 1) * p).astype(int)
    return [(int(s), int(e)) for s, e in zip(edges[:-1], edges[1:]) if e <= end and col[s:e].max() > CELL_INK]


class Glyphs:
    """Stored examples of each character of the bar font."""

    def __init__(self, chars, widths, canvases, reject, pitch=0):
        self.chars = np.asarray(chars)           # (n,) one character per example
        self.widths = np.asarray(widths, int)    # (n,) glyph width in columns
        self.canvases = np.asarray(canvases)     # (n, REF_BAR_PX, cw) glyph at column PAD, values 0-1
        self.reject = float(reject)              # distance above which a glyph is unknown ('?')
        self.pitch = float(pitch)                # character pitch of a monospace font in strip columns, or 0
        self.cw = self.canvases.shape[2]
        self.max_w = self.cw - 2 * PAD
        # pre-shifted copies for the +-1 column comparison
        self._shifted = np.stack([np.roll(self.canvases, s, axis=2) for s in SHIFTS], 1)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['chars'], z['widths'], z['canvases'], z['reject'], z['pitch'] if 'pitch' in z else 0)

    def save(self, path):
        np.savez_compressed(path, chars=self.chars, widths=self.widths, canvases=self.canvases,
                            reject=self.reject, pitch=self.pitch)

    def canvas(self, b, s, e):
        c = np.zeros((REF_BAR_PX, self.cw))
        c[:, PAD:PAD + e - s] = b[:, s:e] / 255.0
        return c

    def match(self, b, s, e):
        """(char, distance) of the best example for columns s:e of the bar."""
        w = e - s
        if w > self.max_w:
            return '?', np.inf
        near = np.flatnonzero(np.abs(self.widths - w) <= 2)
        if not len(near):
            return '?', np.inf
        c = self.canvas(b, s, e)
        # monospace cells are 9 or 10 columns only by where their edges round to: no width cost
        cost = 0 if self.pitch else WIDTH_COST
        d = np.abs(self._shifted[near] - c).mean((2, 3)).min(1) + cost * np.abs(self.widths[near] - w)
        k = int(np.argmin(d))
        return str(self.chars[near[k]]), float(d[k])



def decode(b, glyphs, end=None):
    """Glyph sequence of the bar: list of (char, distance, start, stop), '?' for unknown glyphs.

    A monospace font: each character cell is one glyph. Otherwise each ink run is one glyph when it matches one. A run that matches nothing may be part of a glyph that
    compression split (joined with the next run when the two together match) or two glyphs that touch
    (split at the column that gives the best pair)."""
    if glyphs.pitch:
        out = []
        for s, e in char_cells(b, glyphs.pitch, end):
            ch, d = glyphs.match(b, s, e)
            out.append((ch if d <= glyphs.reject else '?', d, s, e))
        return out
    runs = ink_runs(b, end)
    out, i = [], 0
    while i < len(runs):
        s, e = runs[i]
        ch, d = glyphs.match(b, s, e)
        if d > glyphs.reject and i + 1 < len(runs) and runs[i + 1][0] - e <= MAX_GAP_MERGE:
            c2, d2 = glyphs.match(b, s, runs[i + 1][1])
            if d2 <= glyphs.reject:
                out.append((c2, d2, s, runs[i + 1][1]))
                i += 2
                continue
        if d > glyphs.reject and e - s >= 6:
            pairs = [(glyphs.match(b, s, k), glyphs.match(b, k, e), k) for k in range(s + 2, e - 1)]
            (c1, d1), (c2, d2), k = min(pairs, key=lambda p: max(p[0][1], p[1][1]))
            if max(d1, d2) <= glyphs.reject:
                out += [(c1, d1, s, k), (c2, d2, k, e)]
                i += 1
                continue
        out.append((ch if d <= glyphs.reject else '?', d, s, e))
        i += 1
    return out


# A colon whose upper dot is faint reads as '.', so the fixed separators accept either.
NUM = r'([+-]?\d{1,3}\.\d\d)'
FIELDS = {'X': re.compile(r'X[:.]' + NUM), 'Y': re.compile(r'Y[:.]' + NUM),
          'Z': re.compile(r'Z[:.](\d{1,2}\.\d)'), 'elev': re.compile(r"elev[:.](\d{2,5})")}
TIME = re.compile(r'(\d{4})/(\d\d)/(\d\d)(\d\d)[:.](\d\d)[:.](\d\d)(?:[.:](\d\d))?')
LIMITS = {'X': (-360, 360), 'Y': (-90, 90), 'Z': (1, 99)}


@dataclass
class BarReading:
    text: str                       # decoded bar text, spaces at word gaps ('?' = unknown glyph)
    X: float = None                 # pan, degrees
    Y: float = None                 # tilt, degrees
    Z: float = None                 # zoom factor
    elev_ft: float = None           # camera elevation from the bar, feet
    time: datetime = None
    timestamp: str = ''             # 'YYYY-MM-DD HH:MM:SS.ss', as printed on the bar
    score: float = np.inf           # worst glyph distance among the parsed fields (lower = surer)
    problems: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.problems


def read_bar(gray, glyphs, bar_px=None, top=None):
    """Parse one frame's info bar. `gray` is the frame (or at least its bottom) as H x W 0-255.

    `top` is the bar's first row in the rescaled strip (common_top of the recording); by default it is found
    on this frame alone."""
    b = bar_strip(gray, bar_px)
    if top is None:
        top = bar_top(b)[0]
    b = blank_above(b, top)
    seq = decode(b, glyphs, bar_end(b, top))
    chars = ''.join(c for c, *_ in seq)
    # text with spaces at word gaps, for people to read
    text = ''.join((' ' if k and s - seq[k - 1][3] >= 7 else '') + c for k, (c, d, s, e) in enumerate(seq))
    r = BarReading(text=text)
    worst = []
    for name, rx in FIELDS.items():
        m = rx.search(chars)
        if not m:
            if name != 'elev':
                r.problems.append(f'no {name} field')
            continue
        v = float(m.group(1))
        lo, hi = LIMITS.get(name, (-np.inf, np.inf))
        if not lo <= v <= hi:
            r.problems.append(f'{name}={v} out of range')
            continue
        setattr(r, 'elev_ft' if name == 'elev' else name, v)
        worst += [seq[k][1] for k in range(m.start(1), m.end(1))]
    m = None
    for m in TIME.finditer(chars):
        pass                         # the timestamp is the last date/time on the bar
    if m:
        g = m.groups()
        frac = (g[6] or '0')
        try:
            r.time = datetime(*map(int, g[:6]), int(frac.ljust(6, '0')[:6]))
            r.timestamp = '%s-%s-%s %s:%s:%s' % g[:6] + (('.' + g[6]) if g[6] else '')
            worst += [seq[k][1] for k in range(m.start(), m.end())]
            if not g[6] and chars[m.end():m.end() + 1] in ('.', ':'):
                r.problems.append('hundredths of the time not readable')   # don't pass it off as .00
        except ValueError:
            r.problems.append('impossible date/time ' + m.group(0))
    else:
        r.problems.append('no date/time')
    r.score = max(worst) if worst else np.inf
    return r
