"""Read the info bar of every still image in a folder: stills that are already images, not a recording.

    python -m camstills.index FOLDER                       # bar font picked automatically
    python -m camstills.index FOLDER --glyphs camstills/glyphs/alertwildfire.npz

Writes into FOLDER, in the same format as camstills.extract (the video columns are left empty):
    stills_index.csv    one row per still: file, timestamp, pan_X_deg, tilt_Y_deg, zoom_Z, pose_id, ...
    poses.csv           one row per distinct pan/tilt/zoom
    review.csv          images that need a person (unreadable bar, a second image of the same bar time, an
                        image of another size)

The images keep their names; nothing is copied or deleted. A bar that can't be read is never guessed.
Where the black bar starts is found on a dozen images spread over the folder (as extract does for a
recording), and the bar font is the stored glyph set that reads a few sample images best.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from .bar import Glyphs, bar_geometry, default_bar_px, pick_glyphs, read_bar
from .extract import gray, write_index_files

IMAGE_EXT = ('.png', '.jpg', '.jpeg')
N_SAMPLE_TOP = 12          # images used to find where the black bar starts
N_SAMPLE_FONT = 5          # images used to pick the bar font


def list_images(folder):
    """Image files directly in folder, sorted by name."""
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXT) and not f.startswith('.') and os.path.isfile(os.path.join(folder, f)))


def bottom_gray(path, bar_px=None):
    """(grayscale bottom strip of the image, (width, height)): the same grey levels as extract.gray."""
    im = Image.open(path)
    w, h = im.size
    px = bar_px or default_bar_px(h)
    rgb = np.asarray(im.convert('RGB').crop((0, h - px, w, h)))
    return gray(rgb), (w, h)


def spread(items, n):
    """Up to n items spread evenly over the list."""
    if len(items) <= n:
        return list(items)
    return [items[int(round(k))] for k in np.linspace(0, len(items) - 1, n)]


_GLYPHS = None


def _init_worker(glyph_path):
    global _GLYPHS
    _GLYPHS = Glyphs.load(glyph_path)


def _read_one(job):
    """Bar reading of one image (runs in a worker process)."""
    path, bar_px, top = job
    try:
        g, size = bottom_gray(path, bar_px)
    except OSError as err:
        return dict(size=None, error='image not readable: %s' % err)
    r = read_bar(g, _GLYPHS, bar_px or default_bar_px(size[1]), top)
    return dict(size=size, ts=r.timestamp or None, pose=(r.X, r.Y, r.Z) if None not in (r.X, r.Y, r.Z) else None,
                elev=r.elev_ft, score=r.score, text=r.text, problems=r.problems)


def index_folder(folder, glyphs='auto', bar_px=None, workers=None, progress=None, log=print):
    """Read every image's bar and write the index files. Returns a summary dict.

    glyphs: a glyph file, or 'auto' to pick the stored font that reads sample images best.
    progress(done, total) is called as images are read. workers: processes (default: CPU count, max 4).
    """
    files = list_images(folder)
    if not files:
        raise ValueError('No images (%s) in %s' % ('/'.join(IMAGE_EXT), folder))
    t0 = time.time()
    samples = [bottom_gray(os.path.join(folder, f), bar_px) for f in spread(files, N_SAMPLE_TOP)]
    size0 = samples[0][1]
    px = bar_px or default_bar_px(size0[1])         # the samples are only the bottom px rows
    top, height = bar_geometry([g for g, s in samples if s == size0], px)
    if glyphs == 'auto':
        glyphs, _ = pick_glyphs([g for g, s in spread(samples, N_SAMPLE_FONT) if s == size0], px, top=top)
        if glyphs is None:
            raise ValueError('None of the stored bar fonts (camstills/glyphs) reads these images. Train one '
                             'with camstills.train and pass it with --glyphs.')
    log('%d images of %dx%d; black bar is the bottom %d px%s; bar font %s'
        % (len(files), size0[0], size0[1], height, ' (scene above it ignored)' if top else '',
           os.path.basename(glyphs)))

    jobs = [(os.path.join(folder, f), bar_px, top) for f in files]
    readings = [None] * len(files)
    # The glyph matching is limited by memory bandwidth more than by CPU: more than ~4 processes don't help.
    n_workers = max(1, workers or min(4, os.cpu_count() or 1))
    done = 0
    try:
        with ProcessPoolExecutor(n_workers, initializer=_init_worker, initargs=(glyphs,)) as pool:
            for k, r in enumerate(pool.map(_read_one, jobs, chunksize=4)):
                readings[k] = r
                done += 1
                if progress:
                    progress(done, len(files))
    except (OSError, RuntimeError) as err:            # no worker processes here: read in this process
        log('  (reading in one process: %s)' % err)
        _init_worker(glyphs)
        for k in range(done, len(files)):
            readings[k] = _read_one(jobs[k])
            if progress:
                progress(k + 1, len(files))

    stills, review = {}, []
    for f, r in zip(files, readings):
        if r.get('error'):
            review.append(dict(image=f, reason=r['error']))
        elif r['size'] != size0:
            review.append(dict(image=f, reason='image is %dx%d, the folder\'s stills are %dx%d; not used'
                               % (*r['size'], *size0)))
        elif not r['ts']:
            review.append(dict(image=f, reason='bar not readable: ' + '; '.join(r['problems']), bar_text=r['text']))
        elif not r['pose']:
            review.append(dict(image=f, reason='time read but not pan/tilt/zoom', bar_text=r['text'],
                               timestamp=r['ts']))
        elif r['ts'] in stills:
            review.append(dict(image=f, reason='same bar time as %s (that one is used)' % stills[r['ts']]['file'],
                               bar_text=r['text'], timestamp=r['ts'], pan_X_deg=r['pose'][0],
                               tilt_Y_deg=r['pose'][1], zoom_Z=r['pose'][2]))
        else:
            stills[r['ts']] = dict(file=f, **r)
    rows = sorted(stills.values(), key=lambda s: s['ts'])
    poses = {}
    index_rows = []
    for s in rows:
        pid = poses.setdefault(s['pose'], 'P%02d' % (len(poses) + 1))
        index_rows.append([s['file'], s['ts'], *s['pose'], pid, '%g' % s['elev'] if s['elev'] else '',
                           '', '', '', 1, 0, 'bar', '1/1', '%.4f' % s['score']])
    write_index_files(folder, index_rows, poses, review)
    elev = sorted(s['elev'] for s in rows if s['elev'])
    summary = dict(n_images=len(files), n_stills=len(rows), n_poses=len(poses), n_review=len(review),
                   glyphs=os.path.basename(glyphs), bar_px=height, image_size=list(size0),
                   first=rows[0]['ts'] if rows else '', last=rows[-1]['ts'] if rows else '',
                   elev_ft=elev[len(elev) // 2] if elev else None, seconds=round(time.time() - t0, 1),
                   review=review[:20])
    log('%d stills, %d poses, %d for review%s (%.0f s)'
        % (len(rows), len(poses), len(review), ', %s to %s' % (summary['first'], summary['last']) if rows else '',
           summary['seconds']))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='folder of still images of one camera')
    ap.add_argument('--glyphs', default='auto', help="bar font glyph file, or 'auto' (default)")
    ap.add_argument('--bar-px', type=int, help='bar strip height in pixels (default: image height / 30)')
    ap.add_argument('--workers', type=int, help='processes reading images (default: CPU count, max 4)')
    a = ap.parse_args(argv)
    index_folder(a.folder, a.glyphs, a.bar_px, a.workers)


if __name__ == '__main__':
    main()
