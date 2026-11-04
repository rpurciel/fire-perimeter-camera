#!/usr/bin/env python3
"""Verdi Peak camera: registration + fire-edge mapping web app on the Hawk Fire stills.

Python/HTML port of verdi_combined_v3_fixed.m. Camera site, DEM tiles and the
Z1 field of view are the MATLAB script's values; the camera pose on each still
comes from its info bar. Starts a local server and opens the app in the browser.

    python verdi_register.py                  # opens on the still nearest 17:08:45 (the script's 5.08.45 PM screenshot)
    python verdi_register.py --time 15:30     # start on another still (any still can be picked in the app)
    python verdi_register.py --demo           # also load the demo calibration (run verdi_demo.py once first)
    python verdi_register.py --panorama Verdi_Peak_Lookout_2_1793210321_7176_p.jpg
                                              # calibrate from the camera's panorama, then open with it loaded
    python verdi_register.py --calibration verdi_panorama.mat    # open with any saved calibration
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np

from camreg import panorama
from camreg.server import Config, serve

ROOT = os.path.dirname(os.path.abspath(__file__))
STILLS = os.path.join(ROOT, 'stills')
OUT = os.path.join(ROOT, 'output')
CAM = (39.472599, -120.039383, 8422 * 0.3048)          # lat, lon, elevation m
# Pose = each still's bar pan/tilt/zoom + this mount correction (d_azimuth, d_tilt, roll, HFOV at
# Z1, pan-axis lean east and north). Starts as the raw bar values, with the MATLAB script's 64.29 deg
# HFOV; Optimize or a panorama calibration (--panorama) fits it.
HFOV0 = 64.29
CORRECTION0 = (0, 0, 0, HFOV0, 0, 0)
CORRECTION_LIM = ((-20, 20), (-20, 20), (-12, 12), (15, 110), (-10, 10), (-10, 10))
TILES = ('N39W121.hgt', 'N39W120.hgt')
BAR_PX = 24      # info bar height on the 1280x720 stills (MATLAB crops 42 px from its screenshot)
DEMO_CALIBRATION = 'demo_registration.mat'
UTC_OFFSET = '-07:00'                                  # bar times are Pacific Daylight Time (Aug 2026)
CAMERA = 'verdi'                                       # output names: output/verdi_panorama.mat, ..._overlay.jpg


def find_still(hhmmss):
    """Row of stills_index.csv nearest to a time of day on 2026-08-22."""
    with open(os.path.join(STILLS, 'stills_index.csv')) as f:
        rows = list(csv.DictReader(f))
    parts = (hhmmss.split(':') + ['0', '0'])[:3]
    target = datetime(2026, 8, 22, int(parts[0]), int(parts[1]), 0).timestamp() + float(parts[2])
    stamp = lambda r: datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S.%f').timestamp()
    return min(rows, key=lambda r: abs(stamp(r) - target))


def config(still='', initial=''):
    return Config(cam=CAM, correction0=CORRECTION0, correction_lim=CORRECTION_LIM,
                  tiles=tuple(os.path.join(ROOT, t) for t in TILES),
                  stills_dir=STILLS, stills_index=os.path.join(STILLS, 'stills_index.csv'),
                  out_dir=OUT, bar_px=BAR_PX, default_still=still, initial_calibration=initial,
                  title='Verdi Peak — registration and fire-edge mapping', kml_name='Verdi fire edges',
                  calib_name='verdi_registration_v2.mat', export_name='verdi_fire_edges',
                  utc_offset=UTC_OFFSET)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--time', default='17:08:45', help='time of day of the starting still (default %(default)s)')
    ap.add_argument('--demo', action='store_true', help='load output/%s at start' % DEMO_CALIBRATION)
    ap.add_argument('--calibration', metavar='NAME', help='load this calibration from output/ at start')
    ap.add_argument('--panorama', nargs='+', metavar='IMAGE',
                    help="calibrate from the camera's panorama image(s) first (writes output/%s_panorama.mat "
                         "and an overlay image to check), then open with it loaded" % CAMERA)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true', help="don't open a browser tab")
    ap.add_argument('--host', default='127.0.0.1',
                    help='bind address (use 0.0.0.0 only inside Docker with the port published)')
    args = ap.parse_args()

    row = find_still(args.time)
    initial = args.calibration or (DEMO_CALIBRATION if args.demo else '')
    if args.panorama:
        _, mat, _ = panorama.run(args.panorama, CAM, [os.path.join(ROOT, t) for t in TILES], OUT, CAMERA, HFOV0)
        initial = os.path.basename(mat)
    if initial and not os.path.isfile(os.path.join(OUT, initial)):
        ap.error('output/%s not found%s' % (initial, '; run verdi_demo.py first' if args.demo else ''))
    serve(config(row['file'], initial), port=args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
