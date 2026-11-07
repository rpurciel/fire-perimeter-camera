#!/usr/bin/env python3
"""Fire camera workspaces: calibrate wildfire cameras and map fire perimeters from their stills.

Starts a local server and opens the portal in the browser. There you create a workspace for a fire (or open
a saved one), add its cameras (name, location, height, panorama, stills folder) and load them: the stills'
info bars are read if they haven't been, and each camera is calibrated from its panorama. The calibration
space shows and fixes each camera's calibration; the analysis space is for drawing the fire edges on each
camera and exporting them, per camera and combined into one perimeter.

    python fire_camera.py                          # http://127.0.0.1:8790
    python fire_camera.py --port 8800 --no-browser
    python fire_camera.py --workspaces ~/fires --dem-dir ~/srtm
"""
import argparse
import os

import numpy as np

from camreg.server import serve

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspaces', default=os.path.join(ROOT, 'workspaces'),
                    help='folder holding the workspaces (default: %(default)s)')
    ap.add_argument('--dem-dir', default=ROOT, help='folder with the SRTM .hgt tiles for new workspaces '
                                                    '(default: %(default)s)')
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--no-browser', action='store_true', help="don't open a browser tab")
    ap.add_argument('--host', default='127.0.0.1',
                    help='bind address (use 0.0.0.0 only inside Docker with the port published)')
    args = ap.parse_args()
    serve(ROOT, args.workspaces, args.dem_dir, port=args.port, open_browser=not args.no_browser, host=args.host)


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
