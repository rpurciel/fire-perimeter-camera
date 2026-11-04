"""Automatic skyline detection in a camera image (any camera, any resolution).

The sky is the smooth, bright, bluish region at the top of the frame. It is
grown from sky-coloured pixels in the top rows across pixels whose colour
barely changes from their neighbours, so it stops at the terrain silhouette,
at structures (roof eaves, poles) and at cloud and smoke edges. Small isolated
patches are dropped. In each column the skyline is the lowest sky pixel, so
sky seen below an overhanging roof edge still counts.

Works best on clear days. Haze or smoke can hide distant ridges: the detected
line is then the nearest ridge that is still visible, below the true skyline,
and cloud or plume bottoms can be picked up as well. Check the result before
fitting to it (the overlay and the app's yellow points are for that).
"""
import numpy as np
from scipy import ndimage as ndi


def detect_skyline(rgb, step=2.2, blue_margin=3.0, min_blue=100.0, min_region=0.005, smooth_px=None):
    """Row of the sky/terrain boundary in each column of an RGB image (H x W x 3, 0-255).

    Returns (v, sky): v is the boundary in 1-based pixel coordinates (pixel
    centres at 1..H; the boundary below 0-based row y is at y + 1.5), NaN where
    a column has no sky; sky is the boolean sky mask.

    step:        largest colour change between neighbouring pixels (after smoothing) inside the sky, 0-255
    blue_margin: seed pixels in the top rows need blue >= red + blue_margin
    min_blue:    seed pixels need blue > min_blue; sky pixels need blue > 0.75 * min_blue
    min_region:  sky regions smaller than this fraction of the image are dropped
    smooth_px:   Gaussian smoothing sigma in pixels (default scales with image height, 1.2 px at 1080 rows)
    """
    im = np.asarray(rgb, np.float32)[..., :3]
    h, w, _ = im.shape
    sigma = smooth_px if smooth_px is not None else max(0.8, 1.2 * h / 1080)
    im = ndi.gaussian_filter(im, (sigma, sigma, 0))
    r, b = im[..., 0], im[..., 2]
    dy = np.abs(np.diff(im, axis=0)).max(axis=2)
    dx = np.abs(np.diff(im, axis=1)).max(axis=2)
    change = np.zeros((h, w), np.float32)          # largest step to any 4-neighbour
    change[:-1] = np.maximum(change[:-1], dy)
    change[1:] = np.maximum(change[1:], dy)
    change[:, :-1] = np.maximum(change[:, :-1], dx)
    change[:, 1:] = np.maximum(change[:, 1:], dx)
    smooth = (change < step) & (b > 0.75 * min_blue)
    top = max(4, round(0.04 * h))
    seed = np.zeros((h, w), bool)
    seed[:top] = (b[:top] >= r[:top] + blue_margin) & (b[:top] > min_blue)
    labels, n = ndi.label(smooth)
    if n == 0:
        return np.full(w, np.nan), np.zeros((h, w), bool)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.unique(labels[seed & smooth])
    keep = keep[(keep > 0) & (sizes[keep] >= min_region * h * w)]
    sky = np.isin(labels, keep)
    sky = ndi.binary_closing(sky, np.ones((5, 5), bool)) | sky
    rows = np.arange(h)[:, None]
    lowest = np.where(sky, rows, -1).max(axis=0)
    v = np.where(lowest >= 0, lowest + 1.5, np.nan)
    return v, sky


def skyline_points(rgb, spacing=1, **kw):
    """Detected skyline as 1-based (u, v) rows, one every `spacing` columns; NaN columns left out."""
    v, _ = detect_skyline(rgb, **kw)
    u = np.arange(1, len(v) + 1, dtype=float)
    pts = np.column_stack([u, v])[::max(1, int(spacing))]
    return pts[np.isfinite(pts[:, 1])]
