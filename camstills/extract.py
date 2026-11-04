"""Turn camera recordings into one image per still, with each still's pan/tilt/zoom and time from its info bar.

    python -m camstills.extract recording.mp4 --camera verdi --out stills/verdi
    python -m camstills.extract --manifest recordings.csv          # columns: video, camera[, out]

A time-lapse recording shows each camera image for one or more video frames. Where the black bar starts is
found on a dozen frames spread over the recording; then every frame's info bar is read (bar.py). Consecutive
frames with the same bar timestamp are one still, saved once as
<camera>_<YYYYMMDD>_<HHMMSS>_<hundredths>.png (full frame, bar included). A frame whose bar can't be read is
saved under review/ for a person; it is never guessed.

Writes to the output folder:
    <stills>.png        one image per still
    stills_index.csv    one row per still: file, timestamp, pan_X_deg, tilt_Y_deg, zoom_Z, pose_id, ...
                        (the format the registration app reads)
    poses.csv           one row per distinct pan/tilt/zoom
    review.csv          frames that need a person (unreadable bar, frames of one still that disagree)
    review/             images of unreadable frames

To fix a frame, copy its row from review.csv into an overrides CSV, fill in timestamp, pan_X_deg, tilt_Y_deg
and zoom_Z, and run again with --overrides. Running again rewrites the index files and images; it deletes nothing.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

from .bar import REF_BAR_PX, Glyphs, bar_strip, common_top, default_bar_px, read_bar
from .train import DEFAULT_GLYPHS

SAME_PICTURE = 1.0         # mean grey-level change (0-255) below which review.csv notes a likely repeat
REVIEW_COLS = ['video', 'video_frame', 'image', 'reason', 'bar_text',
               'timestamp', 'pan_X_deg', 'tilt_Y_deg', 'zoom_Z']


def frames(path):
    """(frame index, H x W x 3 uint8 array) for every frame of a video."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24')
    meta = next(reader)
    w, h = meta['size']
    for i, buf in enumerate(reader):
        yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def sample_frames(path, n=12):
    """n frames spread evenly over a video (seeking, so the whole video is not decoded)."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24')
    duration = next(reader).get('duration') or 0
    reader.close()
    for k in range(n):
        reader = imageio_ffmpeg.read_frames(path, pix_fmt='rgb24',
                                            input_params=['-ss', '%.3f' % (duration * (k + 0.5) / n)])
        w, h = next(reader)['size']
        buf = next(reader, None)
        reader.close()
        if buf is not None:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def find_bar_top(video, bar_px=None):
    """(bar top row in the rescaled bar strip, bar height in frame pixels) of a recording."""
    strips = []
    for rgb in sample_frames(video):
        px = bar_px or default_bar_px(rgb.shape[0])
        strips.append(bar_strip(gray(rgb[-px:]), px))
    if not strips:
        return 0, None
    top = common_top(strips)
    return top, round((REF_BAR_PX - top) * px / REF_BAR_PX)


def gray(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)


def signature(rgb, bar_px):
    """Small grey thumbnail of the picture above the bar, to tell whether two frames show the same image."""
    return np.asarray(Image.fromarray(rgb[:-bar_px]).convert('L').resize((64, 36), Image.BOX), float)


def still_name(camera, timestamp):
    """'verdi', '2026-08-22 17:08:34.51' -> 'verdi_20260822_170834_51.png'."""
    date, clock = timestamp.split()
    hms, _, frac = clock.partition('.')
    return f"{camera}_{date.replace('-', '')}_{hms.replace(':', '')}_{(frac or '00')[:2].ljust(2, '0')}.png"


def load_overrides(path):
    """{(video basename or '', frame): (timestamp, X, Y, Z)} from a CSV in review.csv's format."""
    out = {}
    if not path:
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            vals = [r.get(k, '').strip() for k in ('timestamp', 'pan_X_deg', 'tilt_Y_deg', 'zoom_Z')]
            if not all(vals):
                continue
            key = (os.path.basename(r.get('video', '') or ''), int(r['video_frame']))
            out[key] = (vals[0], float(vals[1]), float(vals[2]), float(vals[3]))
    return out


def extract(videos, camera, out, glyphs, bar_px=None, overrides=None, log=print):
    overrides = overrides or {}
    os.makedirs(os.path.join(out, 'review'), exist_ok=True)
    stills = {}                 # timestamp -> still record
    review = []                 # rows for review.csv
    for video in videos:
        vname = os.path.basename(video)
        prev = None             # (still record or None, signature) of the previous frame
        t0, n_read, n_frames = time.time(), 0, 0
        top, height = find_bar_top(video, bar_px)
        if height:
            log(f'  {vname}: black bar is the bottom {height} px'
                + (' (scene above it ignored)' if top else ''))
        for i, rgb in frames(video):
            n_frames += 1
            px = bar_px or default_bar_px(rgb.shape[0])
            sig = signature(rgb, px)
            ov = overrides.get((vname, i)) or overrides.get(('', i))
            if ov:
                ts, X, Y, Z = ov
                reading = dict(ts=ts, pose=(X, Y, Z), elev=None, score=0.0, how='override', text='')
            else:
                r = read_bar(gray(rgb[-px:]), glyphs, px, top)
                pose = (r.X, r.Y, r.Z) if None not in (r.X, r.Y, r.Z) else None
                reading = dict(ts=r.timestamp or None, pose=pose, elev=r.elev_ft, score=r.score,
                               how='bar', text=r.text, problems=r.problems)
                n_read += r.ok
            s = stills.get(reading['ts']) if reading['ts'] else None
            if reading['ts'] and s is None:
                # a new still: save its first frame
                s = stills[reading['ts']] = dict(ts=reading['ts'], file=still_name(camera, reading['ts']),
                                                 frames=[], votes=Counter(), elev=Counter(), how=Counter(),
                                                 score=np.inf, text=reading['text'], order=len(stills))
                Image.fromarray(rgb).save(os.path.join(out, s['file']))
            if s is None:
                # unreadable bar: a person must look. Separate stills can look almost identical (night,
                # static scenes), so the frame is never joined to a neighbour automatically.
                img = f'review/{camera}_{os.path.splitext(vname)[0]}_f{i:06d}.png'
                Image.fromarray(rgb).save(os.path.join(out, img))
                reason = 'bar not readable: ' + '; '.join(reading.get('problems', []))
                if prev and prev[0] is not None and np.abs(sig - prev[1]).mean() < SAME_PICTURE:
                    reason += f"; picture looks like the previous still {prev[0]['file']} (same still?)"
                review.append(dict(video=vname, video_frame=i, image=img, reason=reason,
                                   bar_text=reading['text']))
                prev = (None, sig)
                continue
            s['frames'].append((vname, i))
            if reading['pose']:
                s['votes'][reading['pose']] += 1
                s['how'][reading['how']] += 1
                s['score'] = min(s['score'], reading['score'])
            if reading['elev']:
                s['elev'][reading['elev']] += 1
            prev = (s, sig)
            if i % 200 == 0:
                log(f'  {vname}: frame {i}, {len(stills)} stills so far ({time.time() - t0:.0f} s)')
        log(f'  {vname}: done, bar read cleanly on {n_read} of {n_frames} frames')

    # one pose per still: the reading most of its frames agree on
    rows = []
    for s in stills.values():
        first = s['frames'][0]
        if not s['votes']:
            review.append(dict(video=first[0], video_frame=first[1], image=s['file'],
                               reason='time read but not pan/tilt/zoom', bar_text=s['text'], timestamp=s['ts']))
            continue
        (pose, n), = s['votes'].most_common(1)
        total = sum(s['votes'].values())
        if n < total:
            review.append(dict(video=first[0], video_frame=first[1], image=s['file'],
                               reason=f'frames disagree ({n}/{total} agree); kept the majority',
                               bar_text=s['text'], timestamp=s['ts'], pan_X_deg=pose[0], tilt_Y_deg=pose[1],
                               zoom_Z=pose[2]))
        rows.append(dict(s=s, pose=pose, agree=f'{n}/{total}'))
    rows.sort(key=lambda r: r['s']['ts'])

    # pose ids in time order; out-of-order = shown in the video after a later time
    poses = {}
    for r in rows:
        r['pose_id'] = poses.setdefault(r['pose'], 'P%02d' % (len(poses) + 1))
    latest = ''
    for r in sorted(rows, key=lambda r: r['s']['order']):
        r['ooo'] = int(r['s']['ts'] < latest)
        latest = max(latest, r['s']['ts'])

    # images left by an earlier run are not deleted, only reported
    written = {s['file'] for s in stills.values()} | {r['image'] for r in review}
    stale = [f for d in ('', 'review') for f in os.listdir(os.path.join(out, d))
             if f.lower().endswith('.png') and (f if not d else d + '/' + f) not in written]
    if stale:
        log(f'  note: {len(stale)} older images in {out} are not from this run (e.g. {stale[0]})')

    with open(os.path.join(out, 'stills_index.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'timestamp', 'pan_X_deg', 'tilt_Y_deg', 'zoom_Z', 'pose_id', 'elev_ft', 'video',
                    'video_frame_first', 'video_frame_last', 'n_frames', 'out_of_order_in_video', 'read',
                    'frames_agree', 'glyph_score'])
        for r in rows:
            s = r['s']
            elev = '%g' % s['elev'].most_common(1)[0][0] if s['elev'] else ''
            w.writerow([s['file'], s['ts'], *r['pose'], r['pose_id'], elev, s['frames'][0][0],
                        s['frames'][0][1], s['frames'][-1][1], len(s['frames']), r['ooo'],
                        '+'.join(sorted(s['how'])), r['agree'], '%.4f' % s['score']])
    with open(os.path.join(out, 'poses.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pose_id', 'pan_X_deg', 'tilt_Y_deg', 'zoom_Z', 'n_stills', 'first_time', 'last_time'])
        for pose, pid in poses.items():
            ts = [r['s']['ts'] for r in rows if r['pose_id'] == pid]
            w.writerow([pid, *pose, len(ts), ts[0], ts[-1]])
    with open(os.path.join(out, 'review.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, REVIEW_COLS)
        w.writeheader()
        for r in review:
            w.writerow({k: r.get(k, '') for k in REVIEW_COLS})

    label = Counter(s['text'].split(' elev')[0] for s in stills.values() if ' elev' in s['text'])
    log(f'{camera}: {len(rows)} stills, {len(poses)} poses, {len(review)} for review'
        + (f", {rows[0]['s']['ts']} to {rows[-1]['s']['ts']}" if rows else '')
        + (f"; bar camera name reads '{label.most_common(1)[0][0]}'" if label else ''))
    return rows, review


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='*', help='recording(s) of one camera, in time order')
    ap.add_argument('--camera', help='short camera name, used in file names (e.g. verdi)')
    ap.add_argument('--out', help='output folder (default: stills/<camera>)')
    ap.add_argument('--manifest', help='CSV with columns video, camera and optionally out, to run many cameras')
    ap.add_argument('--overrides', help='CSV of hand-read frames (review.csv format) to use instead of the bar')
    ap.add_argument('--glyphs', default=DEFAULT_GLYPHS, help='bar font glyphs (default: %(default)s)')
    ap.add_argument('--bar-px', type=int, help='bar height in pixels (default: frame height / 30)')
    a = ap.parse_args(argv)

    jobs = {}                   # (camera, out) -> [videos]
    if a.manifest:
        base = os.path.dirname(os.path.abspath(a.manifest))
        with open(a.manifest) as f:
            for r in csv.DictReader(f):
                cam = r['camera'].strip()
                out = (r.get('out') or '').strip() or os.path.join('stills', cam)
                jobs.setdefault((cam, out), []).append(os.path.join(base, r['video'].strip()))
    if a.videos:
        if not a.camera:
            ap.error('--camera is required with videos')
        jobs.setdefault((a.camera, a.out or os.path.join('stills', a.camera)), []).extend(a.videos)
    if not jobs:
        ap.error('give video files with --camera, or --manifest')

    glyphs = Glyphs.load(a.glyphs)
    overrides = load_overrides(a.overrides)
    for (cam, out), videos in jobs.items():
        print(f'{cam}: {len(videos)} recording(s) -> {out}')
        for v in videos:
            if not os.path.exists(v):
                sys.exit(f'not found: {v}')
        extract(videos, cam, out, glyphs, a.bar_px, overrides)


if __name__ == '__main__':
    main()
