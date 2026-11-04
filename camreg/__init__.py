"""Wildfire-camera registration and fire-edge mapping (port of verdi_combined_v3_fixed.m)."""
from .core import (DEM, CameraModel, correction_from_pose, export_edges, full_correction, lean_description,
                   load_calibration, optimize_correction, pose_from_bar, resample_polyline, save_calibration)
from .skyline import detect_skyline, skyline_points
