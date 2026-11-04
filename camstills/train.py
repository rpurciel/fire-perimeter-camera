"""Learn the bar font's glyphs from stills whose bar values are already known.

    python -m camstills.train --index stills/stills_index.csv --prefix "axis-verdipeaklookout2 elev:8422'"

Needs an index CSV with file, timestamp, pan_X_deg, tilt_Y_deg, zoom_Z (the format extract.py writes), and
the bar text that comes before "X:" on those stills (the camera name and elevation), and --suffix if the bar
has text after the time (ALERTWildfire bars end in "ALERTWildfire.org"). Each still's glyphs are labelled from
both ends of the bar: the start (prefix + X/Y/Z) and the end (date and time + suffix). The free-text label in
the middle is skipped. Stills whose glyph widths do not line up with the labels are dropped. For a monospace
font (--monospace; the ALERTWildfire bar) the glyphs are its fixed-width character cells, and the cell pitch
is measured from the stills and stored with the glyphs.

Only needed when the shipped glyphs (glyphs/alertca.npz) do not read a new camera's bar, for example a
different font. A fraction of the stills is held out and read back to check the result.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from .bar import (PAD, REF_BAR_PX, Glyphs, bar_end, bar_strip, blank_above, cell_grid, char_cells, common_top,
                  ink_runs, read_bar)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLYPHS = os.path.join(HERE, 'glyphs', 'alertca.npz')


def labels(row, prefix, suffix='', spaces=False):
    """(text at the start of the bar, text at the end), spaces removed unless `spaces`."""
    head = prefix + ' X:%+.2f Y:%+.2f Z:%.1f' % (
        float(row['pan_X_deg']), float(row['tilt_Y_deg']), float(row['zoom_Z']))
    date, time = row['timestamp'].split()
    tail = date.replace('-', '/') + ' ' + time + (' ' + suffix if suffix else '')
    return (head, tail) if spaces else (head.replace(' ', ''), tail.replace(' ', ''))


def load_bar(path, bar_px=None):
    return bar_strip(np.asarray(Image.open(path).convert('L')), bar_px)


def measure_pitch(bars):
    """Character pitch of a monospace bar font, in strip columns (assumed 6-14), from (bar, end) pairs."""
    coarse = np.median([cell_grid(b, 10, end, spread=0.4, step=0.02)[0] for b, end in bars])
    return float(np.median([cell_grid(b, coarse, end)[0] for b, end in bars]))


def collect_cells(rows, stills_dir, prefix, bar_px, top, suffix, pitch):
    """collect() for a monospace font: a label is placed where its spaces fall on empty cells."""
    kept, total = [], 0
    for r in rows:
        b = blank_above(load_bar(os.path.join(stills_dir, r['file']), bar_px), top)
        end = bar_end(b, top)
        p, ph = cell_grid(b, pitch, end)
        cells = char_cells(b, pitch, end)
        pos = [round((s - ph) / p) for s, e in cells]            # cell number of each non-empty cell
        for k, text in enumerate(labels(r, prefix, suffix, spaces=True)):
            total += 1
            want = [i for i, c in enumerate(text) if c != ' ']    # cell numbers of the label's characters
            chars = text.replace(' ', '')
            # the head starts at the first cell; the tail may be followed by a few cells of stray scene
            starts = [0] if k == 0 else [len(cells) - len(chars) - j for j in range(4)]
            for i0 in starts:
                got = pos[i0:i0 + len(chars)] if i0 >= 0 else []
                if len(got) == len(chars) and [g - got[0] for g in got] == want:
                    kept.append([(c, b, s, e) for c, (s, e) in zip(chars, cells[i0:i0 + len(chars)])])
                    break
    return [g for grp in kept for g in grp], len(kept), total


def collect(rows, stills_dir, prefix, bar_px, top, suffix='', pitch=0):
    """Labelled glyph crops: list of (char, bar, start, stop)."""
    if pitch:
        return collect_cells(rows, stills_dir, prefix, bar_px, top, suffix, pitch)
    heads, tails = [], []                      # per still: labelled runs at the start; candidate ends
    for r in rows:
        b = blank_above(load_bar(os.path.join(stills_dir, r['file']), bar_px), top)
        runs = ink_runs(b, bar_end(b, top))
        head, tail = labels(r, prefix, suffix)
        n = len(runs)
        if n < len(head) + len(tail) + 3:
            continue
        heads.append([(c, b, s, e) for c, (s, e) in zip(head, runs[:len(head)])])
        # stray scene pixels can leave a few runs after the timestamp: try ending 0-3 runs early
        tails.append([[(c, b, s, e) for c, (s, e) in zip(tail, runs[n - len(tail) - k:n - k])] for k in range(4)])
    # typical width of each character, from the majority alignment
    med = defaultdict(list)
    for grp in heads + [t[0] for t in tails]:
        for c, b, s, e in grp:
            med[c].append(e - s)
    med = {c: np.median(w) for c, w in med.items()}
    misfit = lambda grp: sum(abs((e - s) - med[c]) for c, b, s, e in grp)
    # a split or merged glyph shifts every label after it: drop any group whose widths don't fit
    raw = heads + [min(t, key=misfit) for t in tails]
    kept = [grp for grp in raw if all(abs((e - s) - med[c]) <= 1 for c, b, s, e in grp)]
    return [g for grp in kept for g in grp], len(kept), len(raw)


def build(samples, per_char=30, reject=np.inf, seed=0, pitch=0):
    """Glyph set from labelled crops: up to `per_char` randomly chosen examples of each character."""
    rng = np.random.default_rng(seed)
    max_w = max(e - s for _, _, s, e in samples) + 3
    cw = max_w + 2 * PAD
    by_char = defaultdict(list)
    for c, b, s, e in samples:
        x = np.zeros((REF_BAR_PX, cw))
        x[:, PAD:PAD + e - s] = b[:, s:e] / 255.0
        by_char[c].append((x, e - s))
    chars, widths, canv = [], [], []
    for c, items in sorted(by_char.items()):
        for k in rng.choice(len(items), min(per_char, len(items)), replace=False):
            chars.append(c); widths.append(items[k][1]); canv.append(items[k][0])
    return Glyphs(chars, widths, canv, reject, pitch)


def check(glyphs, rows, stills_dir, bar_px, top):
    """Read each still's bar and compare with its index row. Returns (n ok, mismatches, glyph distances)."""
    ok, bad = 0, []
    for r in rows:
        g = np.asarray(Image.open(os.path.join(stills_dir, r['file'])).convert('L'))
        got = read_bar(g, glyphs, bar_px, top)
        want = (float(r['pan_X_deg']), float(r['tilt_Y_deg']), float(r['zoom_Z']), r['timestamp'])
        if (got.X, got.Y, got.Z, got.timestamp) == want:
            ok += 1
        else:
            bad.append((r['file'], want, (got.X, got.Y, got.Z, got.timestamp), got.problems, got.text))
    return ok, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True, help='index CSV of stills with known bar values')
    ap.add_argument('--stills', help='folder of the stills (default: the index file\'s folder)')
    ap.add_argument('--prefix', required=True, help='bar text before "X:", e.g. "axis-verdipeaklookout2 elev:8422\'"')
    ap.add_argument('--suffix', default='', help='bar text after the time, e.g. "ALERTWildfire.org"')
    ap.add_argument('--monospace', action='store_true', help='the bar font is monospace (ALERTWildfire bars)')
    ap.add_argument('--out', default=DEFAULT_GLYPHS, help='glyph file to write (default: %(default)s)')
    ap.add_argument('--per-char', type=int, default=30, help='examples kept per character')
    ap.add_argument('--holdout', type=float, default=0.3, help='fraction of stills held out for the check')
    ap.add_argument('--bar-px', type=int, help='bar height in pixels (default: frame height / 30)')
    a = ap.parse_args(argv)
    stills_dir = a.stills or os.path.dirname(os.path.abspath(a.index))
    with open(a.index) as f:
        rows = [r for r in csv.DictReader(f) if r.get('pan_X_deg') not in (None, '')]
    rng = np.random.default_rng(1)
    test = set(rng.choice(len(rows), int(len(rows) * a.holdout), replace=False)) if a.holdout > 0 else set()
    train_rows = [r for k, r in enumerate(rows) if k not in test]
    test_rows = [r for k, r in enumerate(rows) if k in test]

    # where the black bar starts, from all the stills (as extract.py does from a recording)
    top = common_top([load_bar(os.path.join(stills_dir, r['file']), a.bar_px) for r in rows])
    pitch = 0
    if a.monospace:
        bars = [blank_above(load_bar(os.path.join(stills_dir, r['file']), a.bar_px), top) for r in train_rows]
        pitch = measure_pitch([(b, bar_end(b, top)) for b in bars])
        print(f'monospace character pitch: {pitch:.3f} columns of the {REF_BAR_PX}-row strip')
    samples, kept, total = collect(train_rows, stills_dir, a.prefix, a.bar_px, top, a.suffix, pitch)
    glyphs = build(samples, a.per_char, pitch=pitch)
    print(f'training: {len(train_rows)} stills, {kept}/{total} bar ends lined up, '
          f'{len(samples)} glyphs, {len(set(glyphs.chars))} characters: {"".join(sorted(set(glyphs.chars)))}')
    # unknown-glyph threshold: well above the worst distance of a correctly read training glyph
    dist = [glyphs.match(b, s, e)[1] for c, b, s, e in samples[::7]]
    glyphs.reject = 3 * float(np.percentile(dist, 99))
    print(f'unknown-glyph threshold {glyphs.reject:.3f}')
    if test_rows:
        ok, bad = check(glyphs, test_rows, stills_dir, a.bar_px, top)
        print(f'held-out check: {ok}/{len(test_rows)} stills read exactly')
        for b in bad[:20]:
            print('  ', b[0], 'index', b[1], 'read', b[2], b[3])
            print('     bar:', b[4])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    glyphs.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
