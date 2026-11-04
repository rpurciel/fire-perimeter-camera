# Python + web port of `verdi_combined_v3_fixed (1).m`

| File | What it is |
|---|---|
| `camreg/core.py` | Numerical core: SRTM mosaic with bilinear lookup, DEM skyline, pinhole projection, pixel → terrain ray march, Huber objective with Nelder-Mead fit, CSV/KML export, `.mat` save/load. Not tied to one camera. |
| `camreg/skyline.py` | Automatic skyline detection in any camera image (sky grown from the top of the frame, stopped at the terrain silhouette). |
| `camreg/panorama.py` | Calibrates a camera from its panorama: splits the frames, reads their info bars, detects the skyline and fits the mount correction, including the lean of the pan axis. Not tied to one camera (`python -m camreg.panorama`). |
| `camreg/server.py` | Local web server (Python standard library). Serves the page and a JSON API for optimize, map pixels, resample lines, DEM elevation, and save/load/export. |
| `camreg/web/` | Front end (`index.html`, `app.js`, `style.css`), no build step. Same controls as the MATLAB figure, plus a still picker and a map. |
| `verdi_register.py` | Launcher with the Verdi Peak values (camera site, DEM tiles, the script's 64.29° HFOV as the starting Z1 field of view). `--panorama` calibrates from the camera's panorama first. |
| `verdi_demo.py` | Builds the demo registration (one landmark, skyline trace, fitted mount correction) into `output/`. No fire edge is drawn. |
| `camstills/extract.py` | Recording → stills, for any camera: reads every frame's info bar, saves one image per still, writes `stills_index.csv`, `poses.csv` and `review.csv`. |
| `camstills/bar.py` | Reads the info bar (pan X, tilt Y, zoom Z, elevation, timestamp) by matching glyphs of the bar font. No OCR engine. |
| `camstills/train.py` | Learns the bar font's glyphs from stills with known values. Only needed if a camera's bar uses another font. |
| `camstills/glyphs/alertca.npz` | ALERTCalifornia bar glyphs (proportional font), learned from the 895 Verdi stills. The default. |
| `camstills/glyphs/alertwildfire.npz` | ALERTWildfire bar glyphs (monospace font, bar ends in "ALERTWildfire.org"), learned from 45 Chimney Peak stills. Use with `--glyphs`. |

## Run (on the Mac)

```
conda env create -f environment.yml            # once; later: conda env update -f environment.yml --prune
conda activate fire-camera
python verdi_demo.py                           # once: writes output/demo_registration.mat
python verdi_register.py --demo                # opens http://localhost:8765 with the demo loaded
python verdi_register.py                       # clean session on the 17:08:34 still
python verdi_register.py --panorama Verdi_Peak_Lookout_2_1793210321_7176_p.jpg
                                               # calibrate from the panorama, then open with it loaded
python verdi_register.py --calibration verdi_panorama.mat   # later: open with that calibration
```

Saved calibrations and exports go to `output/`. The map needs internet for its tiles. Everything else is local.

## Recordings to stills (any camera)

```
python -m camstills.extract recording.mp4 --camera verdi --out stills/verdi
python -m camstills.extract --manifest recordings.csv     # many cameras: columns video, camera[, out]
python -m camstills.extract ssstwitter.com_1793654368832.mp4 --camera chimney --out stills/chimney \
    --glyphs camstills/glyphs/alertwildfire.npz          # an ALERTWildfire camera (Chimney Peak)
```

Run from this folder. Each frame's bar is read; frames with the same bar timestamp are one still, saved as
`<camera>_<YYYYMMDD>_<HHMMSS>_<hh>.png`. The output folder gets `stills_index.csv` (the format the registration
app reads), `poses.csv` and `review.csv`. Several recordings of one camera can be given in one run.

- **Fields read:** pan X, tilt Y, zoom Z, the camera elevation and the date/time. The free-text label in the
  middle of the bar is not used.
- **Where the bar is:** the text size follows the frame height (the bottom 1/30 of the frame is read, rescaled
  to 24 px; `--bar-px` changes it). The black bar itself can be shorter: ALERTCalifornia bars fill that strip
  (24 px at 720p), ALERTWildfire bars are 1/36 of the frame (60 px at 2160p). Where it starts is found on a
  dozen frames spread over each recording (the log says e.g. "black bar is the bottom 60 px") and the scene
  above it is ignored. Frames where scene and bar look alike (night, dusk) use the recording's value.
- **Which glyphs:** ALERTCalifornia bars (proportional font) use the default glyphs; ALERTWildfire bars
  (monospace font, "© NV Seismo Lab ... ALERTWildfire.org") need `--glyphs camstills/glyphs/alertwildfire.npz`.
  With the wrong glyphs nothing reads and every frame goes to `review/`.
- **Frames it can't read** are never guessed. They are saved in `review/` and listed in `review.csv`. To fix
  them, copy the rows into a CSV, fill in `timestamp, pan_X_deg, tilt_Y_deg, zoom_Z`, and run again with
  `--overrides that.csv`. Stills whose frames disagree are listed too; the majority reading is kept.
- **Checked:** on the Verdi recording it reproduces the earlier OCR index exactly (895 stills, 24 poses, every
  value), including frame 500, which OCR could not read. Held-out stills not used for learning the glyphs:
  268/268 exact. Stills scaled to 1080p: 22/23 exact (the one miss went to review).
- **Checked (ALERTWildfire, Chimney Peak, 3840x2160 60 fps Twitter download):** 120 stills, one pose; glyphs
  learned from the first 45 (read by eye), the other 75 (15:56-19:36, into dusk) read exactly as read by eye.
- **Not yet working:** frames smaller than 720p (text too small for these glyphs), and bars in another
  font. For those, label some stills (for example via `--overrides`) and run `python -m camstills.train
  --index <that stills_index.csv> --prefix "<bar text before X:>" --out <new glyphs>`, then pass `--glyphs`.
  Add `--suffix "<text after the time>"` if the bar goes on after the time, and `--monospace` if every
  character takes the same width (the ALERTWildfire glyphs were made with
  `--prefix "axis-chimneypeak elev:8179'" --suffix "ALERTWildfire.org" --monospace`). Always pass `--out`:
  the default overwrites `alertca.npz`.
- **Speed:** about 0.23 s per frame (899 frames in 3.5 min).

`stills/scripts/` holds the one-off OCR scripts that first made `stills/`; `camstills` replaces them.

## Calibrating a camera from its panorama (any camera)

ALERTCalifornia cameras publish a 360° panorama: several frames side by side, each with its own info bar.
`camreg.panorama` reads each frame's pan/tilt/zoom from the bar, finds the skyline automatically and fits one
mount correction so the DEM skyline lines up in every frame at once. No landmarks or tracing are needed.

```
python -m camreg.panorama PANORAMA.jpg --camera NAME --cam LAT LON [ELEV_M] --tiles A.hgt B.hgt [--hfov 60]
```

- **Writes** `output/NAME_panorama.mat` (load it in the app, or start with `--calibration NAME_panorama.mat`)
  and `output/NAME_panorama_overlay.jpg`. **Check the overlay:** yellow = detected skyline, orange = points
  left out (trees, roof edges, haze), red = DEM skyline with the fitted correction.
- **Prints** the correction and each frame's median error in pixels. Frames without a clear skyline
  (e.g. one facing the lookout) are skipped and the reason is shown.
- **The pan-axis lean** is fitted when the frames span at least 45° of pan. Verdi's mount leans 1.67° toward
  azimuth 290°, so the horizon in the photos is ~2° too low looking east and ~1° too high looking west when
  the lean is ignored. A single view can't separate the lean from tilt and roll offsets, so the panorama is
  the way to get it.
- **Options:** without `ELEV_M` the camera elevation comes from the bar (`elev:`). `--frames N` or
  `--frame-width W` if the frames aren't 16:9. Several images can be given (e.g. one file per frame).
  `--bars="X,Y,Z;X,Y,Z;..."` gives the pan/tilt/zoom by hand if the bar can't be read. `--az-range 30` limits
  the search for the pan-zero offset if you know it's roughly north. `--lean off` keeps the pan axis vertical.
- **Single frames** work too (no lean), but a 64°-wide skyline can resemble another stretch of horizon. A
  warning is printed when another direction fits nearly as well. Use more frames, or `--az-range`.
- **Checked on Verdi:** the 6-frame panorama (one frame is all lookout, skipped) fits to 1.1–2.0 px median
  error per 1920 px frame. The same result (within 0.1°) from the frames as separate files, scaled to 720p,
  or from a starting HFOV of 50° or 55°. From only 2–3 of the frames (0+1, 1+2, 4+5, 0+2+5 tried) the
  azimuth offset agrees within 0.05°, the lean is 1.67–1.77° and the HFOV within 0.7°. On clear August stills
  from the same pan positions it is within 0.2–2.5 px. Fitted HFOV 64.66°, close to the script's 64.29°.

## Adding another camera

1. Make its stills: `python -m camstills.extract recording.mp4 --camera NAME --out stills/NAME`.
2. Copy `verdi_register.py` to `NAME_register.py` and change the constants at the top: `CAM` (lat, lon,
   elevation m), `TILES` (SRTM tiles covering ~90 km around it), `HFOV0`, `STILLS`, `BAR_PX`, `CAMERA` and
   the titles. `find_still` and `--time` assume the Hawk Fire date; change them to the new recording's date.
3. Get the camera's panorama and run `python NAME_register.py --panorama THE_PANORAMA.jpg`.

Nothing in `camreg/` or `camstills/` is camera-specific. The bar glyphs (`camstills/glyphs/alertca.npz`) fit
the ALERTCalifornia bar; another network's bar font needs `camstills.train` and `--glyphs`.

## Using the page

- **Image view:**
  - Scroll to zoom.
  - Drag to pan. While a tool is active, use right-drag or Space+drag.
  - **F** fits the image again.
  - The cursor's pixel position is shown bottom-right.
- **Tools:** same buttons as MATLAB.
  - Polylines finish on a double-click or Enter.
  - Backspace removes the last vertex. Esc cancels.
  - **Undo** (Fire edges card, or Ctrl/⌘+Z) removes the last vertex while a line is being drawn; otherwise it undoes the last change to the edges: an edge added, deleted with ✕, Clear edges, or edges replaced by loading a calibration.
  - **Clear edges** removes only the edges drawn on the current still (after a confirmation); edges on other stills stay.
  - Freehand draws while the mouse button is held down.
- **Pose:** no sliders. The model camera on each still takes that frame's info-bar pan (X), tilt (Y) and zoom (Z), plus one fixed **mount correction** for the camera (azimuth offset, tilt offset, roll, HFOV at Z1, and the lean of the pan axis toward east and north).
  - The correction starts at zero, i.e. the raw bar values with the script's 64.29° HFOV, or at the loaded calibration. **Optimize correction** fits it from landmarks and the skyline, which may be clicked on different stills, and it then applies to every frame.
  - The lean is shown under the pose table. Optimize keeps it fixed unless **"also fits the pan-axis lean"** is ticked, which needs landmarks on stills at several pan directions. Normally it comes from the panorama.
  - With a panorama calibration loaded you usually don't need Optimize. Use Auto skyline or landmarks to check it. Optimizing on one still can pull the correction toward that still's errors (e.g. skyline points on a smoke plume).
  - **Reset correction** returns to the raw bar values.
  - For zoomed stills the HFOV is assumed to be the Z1 value divided by Z. This hasn't been verified.
  - The pose table shows bar + correction = model pose.
- **Auto skyline** detects the skyline on the current still and puts it in as yellow points, like a traced skyline (Clear skyline removes it). The log says how far it lies from the modelled (red) skyline. On clear stills it is within a few pixels. Haze and smoke hide distant ridges, so on fire-day stills it often finds a nearer ridge 30+ px lower. The app warns when it is more than 10 px off; don't optimize on that.
- **Landmarks:**
  - After clicking the photo, type lat/lon/elevation, or click the map to fill them from the DEM.
  - Each landmark shows its pixel error, and ✕ removes it.
- **Map:**
  - Shows the camera, the field of view, landmarks, mapped edges and the last terrain click.
  - **One fire line per time:** all edges drawn on one still are shown (and exported) as one line, joined across gaps where a stretch has no terrain hit and between separate strokes. Each next stroke is reversed if that makes the join shorter. The current still's line is bold, other times faint.
  - **Linked hover:** pointing at a drawn edge on the photo marks the same point on the map, with its coordinates, elevation and range. Pointing at a fire line on the map rings the point on the photo; if that edge was drawn on another still, the map label names the still.
  - Clicking the map puts a cyan **+** where that ground point should appear in the photo. This helps find landmarks.
- **Stills:** the slider and ◀ ▶ (or ← →) move through all 895 stills.
  - Pink marks under the slider show which stills have fire edges (hover for the time and count; click to go there). The current still's mark is outlined.
  - The bar's X/Y/Z is shown for each still, with a note on zoomed stills. The modeled skyline covers all 360° of pan.
  - Landmarks, the skyline trace and edges remember their still. On other stills, landmarks show only their predicted position (cyan ×), and edges are drawn faint if the pan/tilt/zoom is the same.

## Differences from the MATLAB script

- **Photo:** chosen in the page instead of a hard-coded screenshot. The bar crop is 24 px instead of 42.
- **Saved files:** dialogs are replaced by in-page controls, and files are read and written only in `output/`.
- **Edge locations:** each edge is mapped with the pose of the still it was drawn on, and recomputed whenever the correction changes. MATLAB stored the draw-time location and only recomputed on export. Export output is the same either way.
- **Extra `.mat` field:** each saved edge also records its still (`still`), and MATLAB ignores the extra field.
- **Pose and optimizer:** the free 4-parameter pose is replaced by bar values + mount correction, and the optimizer fits the correction. The correction adds a pan-axis lean that MATLAB didn't have.
  - Saved `.mat` files add `mount_lean_east_north`; files without it load with no lean. Panorama calibrations also record `panorama_files`, `panorama_frames`, `panorama_frames_X_Y_Z` and each frame's median error.
  - Same Huber objective and tolerances as MATLAB, but the start simplex steps 0.5° (fminsearch's 5% rule would barely move from zero offsets).
  - Saved `.mat` files still contain `parameters_az_tilt_roll_hfov` (the pose on the saved still), plus `mount_correction_daz_dtilt_roll_hfovz1`, `landmark_stills` and `skyline_still`. A MATLAB-made file loads by deriving the correction from its saved pose.
- **Export:** the KML has one placemark per time (named by the still's time, time-stamped for Google Earth's time slider unless **Time-stamp each line** next to Export is unticked; the stamps use the launcher's `UTC_OFFSET`, −07:00 for Verdi), not one per edge segment. The CSV keeps MATLAB's columns and one row per sample, and adds `time`, `line_id` (1 = earliest time) and `line_order` (position along that time's line; empty without a terrain hit).
- **Unchanged:** projection maths, conventions (1-based pixels). Only the Python round trip has been tested (no MATLAB here).
- **Network:** the server only listens on `127.0.0.1`. Running it inside Docker needs `--host 0.0.0.0` and a published port.
