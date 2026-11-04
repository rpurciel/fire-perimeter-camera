#!/usr/bin/env python3
"""Build the demo registration on the 17:08:34 still (no browser needed).

Does what a user would do in the app: add a landmark, trace the skyline and
optimize the camera's mount correction. The pose on every still is that
still's info-bar pan/tilt/zoom plus this correction. Then
`python verdi_register.py --demo` opens the app with this state loaded.

No fire edge is drawn; fire edges are for the user to trace. The skyline and
landmark pixels were read off the still by eye.
"""
import os

import numpy as np

import verdi_register as vr
from camreg import core
from camreg.server import App

STILL_TIME = '17:08:45'      # the MATLAB script's 'Screenshot ... at 5.08.45 PM'
# Visible far skyline right of the smoke (x > 1000), where the DEM covers the
# skyline ridges (~57 km away). Left of the smoke the visible horizon lies
# north of 40 N, outside the two DEM tiles, so it is not traced.
SKYLINE = [(1000, 397), (1040, 397), (1070, 394), (1110, 391), (1160, 390),
           (1200, 390), (1240, 389), (1278, 389)]
# One landmark: summit of Peterson Mountain, the flat-topped mountain left of
# centre (photo summit ~x199, y360), taken from the DEM.
LANDMARK = ('Peterson Mtn (DEM summit)', 199, 360.5, 39.7958, -119.9761, 2386.0)


def main():
    still = vr.find_still(STILL_TIME)['file']
    app = App(vr.config(still))
    m, bar = app.model, app.bars[still]
    print('Still: %s  bar X %.2f Y %.2f Z %.1f' % (still, *bar))

    gcp = np.array([LANDMARK[1:]], float)
    sky_obs = core.resample_polyline(SKYLINE, 8)
    c0 = np.array(vr.CORRECTION0, float)
    for label, c in (('raw bar values', c0), (None, None)):
        if c is None:
            c, res = core.optimize_correction(c0, m, app.sky_rays, [(bar, gcp)], (bar, sky_obs),
                                              app.lim, verbose=False)
            label = 'fitted correction [%s]' % ', '.join('%.3f' % x for x in c)
        p = core.pose_from_bar(bar, c)
        u, v, _ = m.project(m.ground_rays(gcp[:, 2:]), p)
        print('%-48s pose [%s]  landmark error %.1f px' % (
            label, ', '.join('%.3f' % x for x in p), np.hypot(u - gcp[:, 0], v - gcp[:, 1])[0]))

    path = os.path.join(vr.OUT, vr.DEMO_CALIBRATION)
    core.save_calibration(path, vr.CAM, p, gcp, [LANDMARK[0]], sky_obs, [], (m.ny, m.nx), still,
                          correction=c, landmark_stills=[still], skyline_still=still)
    print('Wrote', path)


if __name__ == '__main__':
    main()
