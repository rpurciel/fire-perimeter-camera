# Fire camera workspaces (Python + web port of `verdi_combined_v3_fixed (1).m`)

| File | What it is |
|---|---|
| `fire_camera.py` | **The one launcher.** Starts the local server and opens the portal: create or open a fire workspace, add its cameras, load and calibrate them, then draw and export perimeters. |
| `camreg/core.py` | Numerical core: SRTM mosaic and other lat/lon grids with bilinear lookup (a fine grid over SRTM: `LayeredDEM`), DEM skyline, pinhole projection, pixel → terrain ray march, Huber objective with Nelder-Mead fit, CSV/KML export, `.mat` save/load. Not tied to one camera. |
| `camreg/edgefit.py` | The calibration: puts the DEM skyline on the images' silhouette edges, with optional landmarks (see [How the calibration works](#how-the-calibration-works-edge-fit)). |
| `camreg/tiles.py` | The 1° terrain tiles a camera's horizon distance needs: found in the DEM folder, or downloaded from the AWS Terrain Tiles into `downloads/srtm/` (see [Terrain tiles](#terrain-tiles-and-the-horizon-distance)). |
| `camreg/dem3dep.py` | USGS 3DEP 1/3 arc-second (about 10 m) elevation around a fire's cameras, downloaded from the USGS 3DEP ImageServer (no account) and cached in `downloads/3dep/` (see [Elevation model and reference points](#elevation-model-dem-and-reference-points)). |
| `camreg/reference.py` | Points with known coordinates around a camera: NOAA NGS survey marks, OpenStreetMap peaks and towers, FAA obstacles. Landmark candidates (visibility, skyline, accuracy) and the site check. |
| `camreg/combine.py` | Combines the fire lines of several cameras into one perimeter per time (see [Combining cameras](#combining-cameras)). |
| `camreg/workspace.py` | Fire workspaces: `workspaces/<name>/workspace.json` plus its panoramas, calibrations and exports. |
| `camreg/skyline.py` | Automatic skyline detection in any camera image (sky grown from the top of the frame, stopped at the terrain silhouette). |
| `camreg/panorama.py` | Calibrates a camera from its panorama: splits the frames, reads their info bars, detects the skyline and fits the mount correction, including the lean of the pan axis. Not tied to one camera (`python -m camreg.panorama`). |
| `camreg/server.py` | Local web server (Python standard library): the portal, the workspace page, background jobs (bar reading, panorama calibration, loading cameras) and a JSON API. |
| `camreg/web/` | Front end, no build step: `portal.html`/`portal.js` (workspaces and cameras), `setup.html`/`setup.js` (the calibration step), `index.html`/`app.js` (calibration and analysis spaces), `common.js`, `style.css`. |
| `camstills/index.py` | Stills that are already images → their bar readings (`stills_index.csv`, `poses.csv`, `review.csv` in the stills folder). The portal runs it when a stills folder has no readings yet. |
| `camstills/extract.py` | Recording → stills, for any camera: reads every frame's info bar, saves one image per still, writes `stills_index.csv`, `poses.csv` and `review.csv`. |
| `camstills/bar.py` | Reads the info bar (pan X, tilt Y, zoom Z, elevation, timestamp) by matching glyphs of the bar font. No OCR engine. Also picks the stored font that reads a camera's bars. |
| `camstills/train.py` | Learns the bar font's glyphs from stills with known values. Only needed if a camera's bar uses another font. |
| `camstills/glyphs/alertca.npz` | ALERTCalifornia bar glyphs (proportional font), learned from the 895 Verdi stills. |
| `camstills/glyphs/alertwildfire.npz` | ALERTWildfire bar glyphs (monospace font, bar ends in "ALERTWildfire.org"), learned from 45 Chimney Peak stills (3840x2160). |
| `camstills/glyphs/alertwildfire_1080.npz` | The same font as drawn on 1920x1080 frames (ALERTWildfire panoramas), learned from the 6 frames of the Chimney Peak panorama. The camera draws the text differently at this size, so the 4K glyphs don't read it. |

## Run (on the Mac)

```
conda env create -f environment.yml            # once; later: conda env update -f environment.yml --prune
conda activate fire-camera
python fire_camera.py                          # opens http://127.0.0.1:8790 (the portal)
python fire_camera.py --port 8800 --no-browser
python fire_camera.py --workspaces ~/fires --dem-dir ~/srtm    # other folders for workspaces / DEM tiles
```

The map needs internet for its tiles. Everything else is local. The old single-camera launcher
(`verdi_register.py`, `verdi_demo.py`) is in `unused/`.

## Workflow

1. **Portal: open or create a workspace.** One workspace per fire. A new one asks for a name and, optionally,
   the **UTC offset of the camera clocks** (the time on the info bars, e.g. `-07:00` for PDT). The offset
   time-stamps the exports for Google Earth; without it the times stay on the camera clock. All cameras of a
   workspace are assumed to show the same clock zone.
2. **Add each camera:** a name, its location (lat/lon, or click the map; a pasted "lat, lon" pair in the
   Latitude box fills both, and a typographic minus, degree sign or trailing N/S/E/W is fine), its height
   (camera elevation, ft or m), its panorama (one image with the frames side by side, or several frames; uploaded into the
   workspace) and its stills folder (type a path or Browse; the images stay where they are).
3. **Load cameras.** One job per camera, with progress:
   - **Bar readings:** if the stills folder has no `stills_index.csv` (or it lists none of the folder's
     images), every still's info bar is read (`camstills.index`). The bar font and the height of the black
     bar are found automatically. Unreadable stills are listed in `review.csv`, never guessed.
   - **DEM:** the 1° terrain tiles out to the camera's horizon distance (default 90 km; calibration step) are
     looked up in the DEM folder and its `downloads/srtm/`; missing ones are downloaded once (see
     [Terrain tiles](#terrain-tiles-and-the-horizon-distance)). Tiles within 40 km are used at full resolution,
     the rest at 3 arc-seconds. If the workspace uses the USGS 3DEP 10 m DEM (portal, **Elevation model**
     card), it is used within its radius.
   - **Site:** a warning if the camera height is below the DEM ground at its site, or if the ground right
     around the camera rises above the middle of its main views (the site or height is then off; see **Check
     site**).
   - **Model:** the camera's 360° DEM skyline. Cameras with the same tiles share one DEM in memory.
   When all are done, the **calibration step** opens (or the workspace, if every camera is calibrated).
4. **Calibration step** (`/setup`; also from the Calibration card in the workspace). One tab per camera.
   - **Frames:** every panorama frame and a few fire-day stills: for the camera's two most common positions,
     the first and last daylight still (night images, grey infrared, are skipped). Click a thumbnail to view
     it; scroll zooms, drag pans. Red: the modelled skyline of the calibration in use; green: the proposed
     calibration.
   - **Landmarks (optional):** **Suggest landmarks** lists the peaks, lookouts and towers with known coordinates
     that fall in the frame (fetched once per camera; see [Reference points](#reference-points-landmark-candidates)),
     drawn as ◇ where the calibration puts them. Click one, then click the feature itself in the image, and
     confirm: its coordinates and how well they are known come with it. Or **Add landmark**, click the feature
     in the frame, then give its location: type
     or paste it (a "lat, lon" pair fills both), or click the map (the elevation comes from the DEM).
     **Snap to summit** moves the point to the highest DEM point within 300 m. Landmarks pin the calibration
     where the skyline alone is weak, above all the pan direction; on the **stills** they set that camera
     position's own offset (see below). After a calibration the list shows each one's error.
   - **Horizon distance** (slider, 20–250 km, per camera): how far out the modelled skyline (the red and green
     lines, and what Calibrate fits) and the rays that drape the edges follow the terrain. Longer reaches distant
     peaks on clear days; on hazy or smoky frames a shorter distance matches the horizon you can see. Changing it
     reloads the camera (downloading the terrain tiles it needs, once); calibrate again afterwards. The
     calibration details say which distance a calibration was fitted with.
   - **Calibrate this camera / all cameras:** about a minute per camera. The result is a **proposal**: the
     calibration in use doesn't change until you accept it. Calibration details show the correction (in use,
     proposed), and for each frame where the real silhouette is relative to the modelled skyline before and
     after (0 = on it), plus the overlay image.
   - **Check & accept:** look at the green line on the frames (red: the calibration in use). **Accept
     calibration** puts it in use; **Discard proposal** keeps the calibration in use. The proposal is saved in
     the workspace; Open the workspace warns about proposals not yet accepted. (Hand adjustment of the line was
     removed; calibrations made with it still load, with their positions' roll and stretch.)
5. **Calibration space** (in the workspace, switch at any time, top left): per camera, check and fix.
   - **Camera card:** edit the name, location, height or stills folder (Apply reloads the camera); **Read
     bars again** re-reads the whole stills folder.
   - **Calibration card:** the calibration in use with its results and overlay, **Landmarks & calibration →**
     (it says when a proposal is waiting there), and the camera's panoramas (add or remove).
   - **Camera pose, Registration:** manual tools (landmarks on the timeline's stills, trace/auto skyline,
     Optimize, Reset).
   - **Calibration files:** import a `.mat` calibration of this camera (same site within 500 m) from the
     workspace or the project's `output/` (e.g. `output/verdi_panorama.mat`): its correction, landmarks,
     skyline trace and edges. Export writes the MATLAB layout to the workspace's `exports/`.
   A workspace tab left open while the calibration step changes a camera does not overwrite the new
   calibration when it is saved (it says so; reload it).
6. **Analysis space:** draw fire edges on each camera (camera tabs, or keys 1–9). All cameras share one
   **timeline** (see Using the page): at the selected time each camera shows its most recent still, so
   switching cameras stays at the same time. The map shows every camera, its field of view and its lines, and
   the **combined perimeter at the current time** (observed parts only). **Export** writes, to `exports/`,
   each camera's lines (`<name>_<camera>.csv/.kml`, the format below) and the combined perimeter
   (`<name>_combined.csv/.kml`).
7. **Save workspace** (or Ctrl/⌘+S; "● unsaved" shows unsaved changes, and leaving the page warns). The
   workspace keeps each camera's calibration, landmarks, skyline trace, edges and current still, and opens
   later from the portal where you left it.

A workspace is a folder in `workspaces/`: `workspace.json`, `panoramas/` (uploaded panoramas),
`calibration/` (`<camera>_panorama.mat` and its overlay) and `exports/`. Stills are only referenced; paths
inside the project folder are stored relative to it, so the project folder can move. Edge terrain points
are not stored: they are recomputed from the current calibration when the workspace opens.

## Elevation model (DEM) and reference points

### Which DEM

The camera model needs terrain twice: the skyline the calibration fits and the ground the fire edges are
draped on. Two choices per workspace (portal, **Elevation model** card):

- **SRTM 30 m** (the `.hgt` tiles in the DEM folder; the default). It smooths summits and narrow ridges.
- **USGS 3DEP 10 m** (1/3 arc-second) within a radius of the cameras (default 25 km), SRTM beyond.
  **Download 3DEP** fetches it from the USGS 3DEP ImageServer (no account or key): about 145 MB for the two
  Hawk Fire cameras, a few minutes. The service sometimes answers with gateway errors; requests are retried
  and a failing piece is split. The grid is cached in `downloads/3dep/` next to the DEM tiles (git-ignored) and
  shared by the workspaces that cover the same area. It is bare earth: trees are not in it.
  Changing the DEM unloads the cameras (Load cameras again); a calibration made on the other DEM is marked
  in the portal and the workspace (calibrate the camera again).

**Checked (Hawk Fire, 2026-11-05, panorama + edge fit):** Chimney Peak with 3DEP: every fire-day still within
0.2° of its silhouette, its position offset 0.21° (SRTM: stills up to 1.5° off, offset 0.97°), Peavine Peak
0.26° from its projection (SRTM 0.67°, also with the camera at the same site, so the DEM made the difference).
Verdi: about the same on both.

**With 3DEP the camera's site and height matter to a few metres.** SRTM puts Verdi Peak's summit at 2553 m; 3DEP
at 2571 m, exactly at the NGS lookout mark, so the bar's 2567 m height is 4 m *below* the ground there. Chimney
Peak's 3DEP summit (2492 m) is 32 m from the earlier fitted site: from that site, rays more than about 5° below
horizontal hit the summit 20–40 m away. Use **Check site** (below) and put the camera on its summit or mast.

### Terrain tiles and the horizon distance

Each camera's terrain reaches its **horizon distance** (calibration step slider, 20–250 km, default 90 km):
the 360° skyline, the calibration's fit and the rays that drape the edges stop there. The 1° tiles it needs
(`N39W120.hgt` …) are looked for in the DEM folder (the project's two SRTM tiles) and in
`downloads/srtm/`; missing ones are downloaded once from the AWS Terrain Tiles
(`https://s3.amazonaws.com/elevation-tiles-prod/skadi/N40/N40W120.hgt.gz`, no account), about 10 MB each,
kept gzipped (git-ignored). They are 1 arc-second like SRTM; in the US they are USGS NED (bare earth), so
they differ by metres from SRTM. Tiles within 40 km of the camera are used at full resolution, the others at
3 arc-seconds (about 90 m; enough for distant ridges, and 30 tiles then take about 170 MB of memory instead of
3 GB). A tile the service doesn't have (open sea) is noted and skipped.

Before this, only the two tiles in the project folder were used: everything north of 40° N (about 60 km
north of the Hawk Fire cameras, in their view) and beyond 120°–121° W was missing from the skyline.

The model has no atmospheric refraction: beyond about 100 km it puts distant peaks a little low (about 0.09°
at 150 km).

### Check site

Portal, on each camera: the DEM ground at the entered site and the camera's height above it, the highest DEM
point within 150 m, how high the ground within 300 m rises as seen from the camera, and the NOAA NGS survey
marks within 1.5 km (a lookout or tower mark is often the camera's own site; e.g. KS2006 "VERDI PEAK LOH",
lookout house, adjusted position 39.4724693, -120.0396433). **Use** puts a point's position in the boxes (the
height is raised to its ground + 5 m if below it); check the height (the bar's elevation, or the mast above
the ground) and **Apply**.

### Reference points (landmark candidates)

Fetched per camera within 40 km (`reference/<camera>.json` in the workspace; about a minute):

| Source | What | Position known to |
|---|---|---|
| NOAA NGS survey marks (Data Explorer API) | triangulation stations, and intersection stations: lookouts, towers, beacons, peaks; adjusted positions only | about 2 m |
| OpenStreetMap (Overpass API) | peaks (moved to the highest DEM point within 100 m), towers, masts | about 30 m (peaks), 15 m (towers) |
| FAA Digital Obstacle File | towers and antennas, with their top height | 6–150 m (its accuracy class) |

A candidate is kept if it is 0.5–40 km away and the DEM in front of it does not hide it; "on the skyline" if
nothing behind it is higher (those are the easiest to find). Its accuracy as seen from the camera is
atan(position error / range), shown as ±°. The calibration weighs a landmark by it: a mapped peak at 15 km
(±0.12°) pulls less than a surveyed mark (±0.01°). A source that cannot be reached is reported; the others are
used.

What external data cannot give: the camera's pointing (azimuth and tilt offsets, roll, the lean of the pan
axis, each position's own offset). Those belong to how this unit was mounted; they need something of known
direction seen in its images (the skyline, landmarks). The reference points make that easier and the DEM makes
it more accurate.

**Without a panorama**, landmarks also give the starting point: azimuth and tilt offsets from one landmark,
roll from two or more at least 10° apart in azimuth, the field of view from four or more spanning 20°
(`EdgeFit.landmark_start`). Fire-day stills (smoke, haze, dusk) have weak edges, so without a panorama the
landmarks weigh three times more in the edge fit. Checked on the 1080p Hawk Fire stills: the stills-only fit put
Verdi's azimuth offset at +5.3° (its panorama fit: +0.2°); with Peavine Peak as a landmark +0.4°. The field of
view cannot be fitted from such stills (it went to 58–60°): enter the camera's from its datasheet as the
starting HFOV (e.g. 65.1° for an Axis Q6075 at the wide end).

## Combining cameras

Each camera's edges are mapped to the terrain with that camera's own calibration, and the edges drawn on one
still form that camera's line for that time (as in the per-camera export). The combined perimeter is
recomputed at **every time any camera has a line**, from the **latest line of each camera at that time**
(optionally ignoring lines older than a set number of minutes). It is valid until the next such time (a
KML TimeSpan, so Google Earth's time slider shows one combined perimeter at a time).

- **Where cameras see the same part of the fire, the newest line wins** (the fire has grown since the
  older one). The bearing circle around a **fire centre** is cut into 0.5° bins; each drawn stretch covers
  the bins between its end bearings; each bin goes to the newest line covering it (for lines less than 60 s
  apart, to the one nearer its camera there). The stretches a line wins are the **observed parts**.
- **Fire centre:** by default the centroid of the convex hull of all lines (all cameras and times), fixed
  for the whole series. **Set fire centre on map** overrides it.
- **Export checkbox "Close the combined perimeter":** off = the observed parts only (one KML MultiGeometry
  per time). On = a closed ring: the observed parts in bearing order, joined by straight **connectors**
  across the parts no camera sees, and closed back to the start. Connectors are styled differently in the
  KML (pale yellow, "unseen") and flagged in the CSV; the KML adds the area polygon and the description
  gives the area (ha, acres), the perimeter length and the observed share. The map in the app always shows
  the observed parts only.
- **Overlap check:** where two cameras' lines less than 10 min apart cover the same bearings, the distance
  between them along the bearing (median and 90 %) is shown in the Combined card and the export message. It
  shows how well the two calibrations agree (a large offset points at a calibration, not the fire).
- **Combined CSV columns:** `composite_id, time_utc, valid_until_utc, part, vertex_order, latitude,
  longitude, elevation_m, kind, camera, source_still, source_time_utc, age_min, range_m`. `kind` is the
  stretch from this vertex to the next: `observed`, `connector`, or empty (end of a part; a closed ring
  repeats its first vertex at the end). Times are UTC (`Z`) when the workspace has a UTC offset.
- **Limitation:** ownership by bearing assumes the perimeter is roughly star-shaped around the centre. Deep
  fingers can be cut oddly; moving the centre fixes most cases.
- **A panorama must show a clear skyline.** A fire-day still with haze and smoke is no substitute: the skyline
  detector stops in the haze, and the fit is wrong (tried with a Chimney still: azimuth off by 155°). Without
  a good panorama, trace the skyline or click landmarks and Optimize in the calibration space.

## Recordings to stills (any camera)

(For stills that are already images, the portal reads their bars itself; from the command line:
`python -m camstills.index STILLS_FOLDER`, bar font and bar height picked automatically. Checked: on the
Chimney (120) and Verdi (895) stills folders it reproduces the indexes made from the recordings exactly,
picking the ALERTWildfire font and a 60 px bar, and the ALERTCalifornia font and a 24 px bar.)

```
python -m camstills.extract recording.mp4 --camera verdi --out stills/verdi
python -m camstills.extract --manifest recordings.csv     # many cameras: columns video, camera[, out]
python -m camstills.extract chimney_hawkfire_20260822-23.mp4 --camera chimney --out stills/chimney-1080
                                                         # the bar font is picked automatically
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
- **Which glyphs (bar font):** picked automatically: each stored font in `camstills/glyphs/` is tried on a
  dozen frames of the recording and the one that reads them is used (the log says which). There is one per
  camera network and frame size: `alertca.npz` (ALERTCalifornia), `alertwildfire.npz` (ALERTWildfire, 4K)
  and `alertwildfire_1080.npz` (ALERTWildfire, 1080p). `--glyphs FILE` forces one. If no stored font reads
  the bar, the run stops with a message; so does it if none of the first 300 frames reads (`--keep-going`
  reads on anyway). Progress is printed every 200 frames, readable or not.
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
  the default overwrites `alertca.npz`. The unknown-glyph threshold is set leave-one-out (each training glyph
  against the other stored examples of its character), so a few stills are enough: `alertwildfire_1080.npz`
  came from 6 panorama frames (trained on 4, the other 2 read exactly).
- **Speed:** about 0.23 s per frame (899 frames in 3.5 min).

`stills/scripts/` holds the one-off OCR scripts that first made `stills/`; `camstills` replaces them.

## How the calibration works (edge fit)

`camreg/edgefit.py`. The model camera on a still is its bar pan/tilt/zoom, plus that camera position's own
small pan/tilt offset, plus one mount correction for the camera (azimuth and tilt offsets, roll, field of view
at Z 1, and the lean of the pan axis).

- **Edges, not a detected skyline.** For a candidate correction the DEM skyline is projected into each
  frame, and the image is sampled just above and just below the line: the sky is bright, the terrain dark. The
  correction that maximises that contrast puts the line on the silhouette, left/right as well as up/down.
  (The earlier fit compared the model with an automatically detected skyline, which stops at the bright haze
  band above distant ridges, so it sat about 0.25° high, and it only measured vertical errors, so it missed
  left/right offsets: 0.7° on Chimney.)
- **Each frame weighs about the same:** its contrast is divided by the best it reaches near the start; weak
  or cluttered edges (contrast below 10) weigh less, and frames without a clear edge (towers, night) are left
  out.
- **The shared correction** is fitted to the panorama frames, starting from the skyline fit
  (`panorama.calibrate`). Each frame gets its own small pan offset during the fit: the camera doesn't return
  exactly to the pan its bar reports (Chimney's frames differ by up to ±0.8°).
- **Each still position** (bar X/Y/Z) then gets its own pan/tilt offset, measured on up to 6 of its daylight
  stills, and applied when they agree (offset ≥ 0.1°, spread ≤ 0.15°, at least 3 stills); landmarks on that
  position's stills always set it. Chimney's fire-day position: pan +0.48°, tilt +0.14°.
- **Landmarks** add a robust term (Huber, 0.05°): one landmark 0.2° off weighs about a third of one frame's
  edge alignment, so a misclick cannot dominate.
- **Without a panorama** the stills are used for the shared correction (and no position offsets).
- **Older calibrations** made with hand adjustment (since removed) can carry a roll and a stretch (scale) per
  position, applied after the rest of its pose (`core.turn_pose`); they still load.
- **Remaining offsets in the reports** come from a search of ±40 px across and ±50 px down for the best edge
  contrast, on a grid 1 px tall (a sharp edge is only a few px tall: a coarser grid can step over it when the
  line is an odd number of px off). On a flat skyline the sideways value is weakly determined.
- **Checked (Hawk Fire, 2026-11-05):** with the panorama fit the real silhouette was 0.15–0.34° below the
  modelled skyline on Verdi and 0.6–0.8° left, 0.4–0.5° below on Chimney. After the edge fit, on stills
  not used in the fit: Verdi within 0.02–0.08° vertically; Chimney within 0.02° vertically and 0.12–0.14°
  sideways. Verdi's sideways values on smoky afternoon stills scatter by ±0.4° (a flat skyline in haze).
  Landmarks identified by eye (Peterson Mountain from both cameras, a 2318 m summit from Chimney): 2.5 px,
  2.0 px and 0.6 px from where the calibration puts them (36 px, 52 px and 5.7 px with the panorama fit).

## Calibrating a camera from its panorama (any camera)

ALERTCalifornia cameras publish a 360° panorama: several frames side by side, each with its own info bar.
`camreg.panorama` reads each frame's pan/tilt/zoom from the bar, finds the skyline automatically and fits one
mount correction so the DEM skyline lines up in every frame at once. No landmarks or tracing are needed.

```
python -m camreg.panorama PANORAMA.jpg --camera NAME --cam LAT LON [ELEV_M] --tiles A.hgt B.hgt [--hfov 60]
```

The calibration step runs this as its starting point, then the edge fit (above). From the command line
(`--method edges`, the default, also runs the edge fit on the panorama frames; `--method skyline` the
skyline fit alone):

- **Writes** `output/NAME_panorama.mat` (import it into the camera in the calibration space)
  and `output/NAME_panorama_overlay.jpg`. **Check the overlay:** yellow = detected skyline, orange = points
  left out (trees, roof edges, haze), red = DEM skyline with the fitted correction.
- **Prints** the correction and each frame's median error in pixels. Frames without a clear skyline
  (e.g. one facing the lookout) are skipped and the reason is shown.
- **The pan-axis lean** is fitted when the frames span at least 45° of pan. Verdi's mount leans 1.67° toward
  azimuth 290°, so the horizon in the photos is ~2° too low looking east and ~1° too high looking west when
  the lean is ignored. A single view can't separate the lean from tilt and roll offsets, so the panorama is
  the way to get it.
- **Bar font:** picked automatically (`--glyphs` to choose). The frame is cropped above its black bar, which
  is shorter than the read strip on ALERTWildfire frames. Where the bar starts is the reading most frames
  agree on (one frame with dark forest above the bar can't tell on its own).
- **The camera site must be right to some tens of metres** when nearby terrain is in view. Chimney Peak at
  the approximate site 39.458017, -119.955943 fitted with 4-26 px errors and a 73° HFOV: the DEM there is
  28 m below the camera and rises to the south, so the model put hills in front of the camera that aren't
  there. A search of positions (camera 10 m above the DEM ground) found 39.459816, -119.958273 (280 m NW;
  ground 2486 m, the bar says 2493 m), where the six frames fit to 1.9-6.9 px median with HFOV 66.4°. If a
  panorama fits much worse than ~3 px, or the HFOV comes out far from ~64°, check the site first.
- **Options:** without `ELEV_M` the camera elevation comes from the bar (`elev:`). `--frames N` or
  `--frame-width W` if the frames aren't 16:9. Several images can be given (e.g. one file per frame).
  `--bars="X,Y,Z;X,Y,Z;..."` gives the pan/tilt/zoom by hand if the bar can't be read. `--lean off` keeps the
  pan axis vertical.
- **Azimuth:** the bar's X is the azimuth from north (−180..180), true to a few degrees, so the azimuth offset
  is searched within ±10° (`--az-range`; 180 searches all directions) and Optimize keeps it within ±20°. A
  warning is printed if the fit ends at the edge of the search. Both panoramas fit with small offsets
  (Verdi +0.5°, Chimney −0.2°), and the same as with an all-directions search.
- **Field of view:** the cameras' nominal field of view at Z 1 is 60°. It is the starting value, and the fit
  refines it: the panoramas fit best at 64.4° (Verdi, median error 1.9 px) and 66.4° (Chimney, 4.1 px); held
  at 60° the errors grow to 2.8 and 6.9 px. A pinhole model without lens distortion may account for part of
  the difference. HFOV(Z) = HFOV(Z1)/Z (as tan) is still untested on zoomed stills.
- **Single frames** work too (no lean), but a 64°-wide skyline can resemble another stretch of horizon. A
  warning is printed when another direction fits nearly as well (within the ±10° search this is rare).
- **Checked on Verdi:** the 6-frame panorama (one frame is all lookout, skipped) fits to 1.1–2.0 px median
  error per 1920 px frame. The same result (within 0.1°) from the frames as separate files, scaled to 720p,
  or from a starting HFOV of 50° or 55°. From only 2–3 of the frames (0+1, 1+2, 4+5, 0+2+5 tried) the
  azimuth offset agrees within 0.05°, the lean is 1.67–1.77° and the HFOV within 0.7°. On clear August stills
  from the same pan positions it is within 0.2–2.5 px. Fitted HFOV 64.66°, close to the script's 64.29°.

## Adding another camera

In the portal: **Add camera** (name, location, height, panorama, stills folder), then **Load cameras**, then
the calibration step (landmarks optional, **Calibrate**, check, **Accept**).
If its stills are still a recording, make them first with `camstills.extract`. Nothing in `camreg/` or
`camstills/` is camera-specific; another network's bar font needs `camstills.train` (above).

## Using the page

- **Spaces and cameras:** the header switches between the Calibration and Analysis spaces and between the
  cameras (tabs or keys 1–9). Each camera keeps its own still, view, calibration, landmarks, skyline trace,
  edges and undo history. In the calibration space the edges are drawn faint; in the analysis space the
  landmarks are hidden.
- **Image view:**
  - Scroll to zoom.
  - Drag to pan. While a tool is active, use right-drag or Space+drag.
  - **F** fits the image again.
  - The cursor's pixel position is shown bottom-right.
- **Tools:** same buttons as MATLAB.
  - Polylines finish on a double-click or Enter.
  - Backspace removes the last vertex. Esc cancels.
  - **Undo** (Fire edges card, or Ctrl/⌘+Z) removes the last vertex while a line is being drawn; otherwise it undoes the last change to the camera's edges: an edge added, deleted with ✕, Clear edges, or edges added by importing a calibration.
  - **Clear edges** removes only the edges drawn on the current still (after a confirmation); edges on other stills stay.
  - Freehand draws while the mouse button is held down.
- **Pose:** no sliders. The model camera on each still takes that frame's info-bar pan (X), tilt (Y) and zoom (Z), plus that camera position's own offsets (pan/tilt from the calibration step; older hand-adjusted calibrations may add a roll/stretch), plus one fixed **mount correction** for the camera (azimuth offset, tilt offset, roll, HFOV at Z1, and the lean of the pan axis toward east and north).
  - The correction comes from the panorama calibration. Without one it starts at zero, i.e. the raw bar values with a 60° HFOV. **Optimize correction** fits it from landmarks and the skyline, which may be clicked on different stills, and it then applies to every frame.
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
  - Shows every camera and its field of view (the active one bold), each camera's mapped edges in its own colour, the combined perimeter (analysis space), the active camera's landmarks (calibration space) and the last terrain click. The layer control turns each camera's lines and the combined perimeter on and off.
  - **One fire line per time:** all edges drawn on one still are shown (and exported) as one line, joined across gaps where a stretch has no terrain hit and between separate strokes. Each next stroke is reversed if that makes the join shorter. The current still's line is bold, other times faint.
  - **Linked hover:** pointing at a drawn edge on the photo marks the same point on the map, with its coordinates, elevation and range. Pointing at a fire line on the map rings the point on the photo; if that edge was drawn on another still, the map label names the still.
  - Clicking the map puts a cyan **+** where that ground point should appear in the photo. This helps find landmarks.
- **Timeline (header):** one time axis for all cameras, on the camera clock (the bars' time).
  - **Start** and **End** (`YYYY-MM-DD HH:MM[:SS]`) set the window; **All stills** spans the first to the last
    still of all cameras. **Bin** (minutes) steps the timeline in fixed windows, which helps with recordings of
    thousands of frames; blank steps from one still (of any camera) to the next.
  - At the selected time **each camera shows its most recent still at or before that time**. The row under
    the slider lists each camera's still and how far before the selected time it was taken; a camera with no
    still yet shows its first one, marked "no still yet". Edges are drawn on the still shown.
  - The slider, ◀ ▶ and ← → move one bin (or to the next still of any camera). **Shift+← →** step through the
    active camera's own stills.
  - **Lanes** under the slider, one per camera in its colour: a tick at each still, a tall mark where fire
    edges are drawn, a dot at the still shown, and the selected time across all of them. Click a lane to go
    to that camera's still there; hover for its time and edge count. Clicking an edge or landmark in a list
    goes to its time.
  - The combined perimeter is shown for the selected time. The window, bin and time are saved with the
    workspace.
  - The bar's X/Y/Z of the active camera's still is shown, with a note on zoomed stills. Landmarks, the
    skyline trace and edges remember their still. On other stills, landmarks show only their predicted
    position (cyan ×), and edges are drawn faint if the pan/tilt/zoom is the same.

## Differences from the MATLAB script

- **Photo:** chosen in the page instead of a hard-coded screenshot. The bar crop is 24 px instead of 42.
- **Saved files:** dialogs are replaced by in-page controls and workspaces; files are written only in the workspace folder (and the bar readings in the stills folder).
- **Edge locations:** each edge is mapped with the pose of the still it was drawn on, and recomputed whenever the correction changes. MATLAB stored the draw-time location and only recomputed on export. Export output is the same either way.
- **Extra `.mat` field:** each saved edge also records its still (`still`), and MATLAB ignores the extra field.
- **Pose and optimizer:** the free 4-parameter pose is replaced by bar values + mount correction, and the optimizer fits the correction. The correction adds a pan-axis lean that MATLAB didn't have.
  - Saved `.mat` files add `mount_lean_east_north`; files without it load with no lean. Panorama calibrations also record `panorama_files`, `panorama_frames`, `panorama_frames_X_Y_Z` and each frame's median error.
  - Same Huber objective and tolerances as MATLAB, but the start simplex steps 0.5° (fminsearch's 5% rule would barely move from zero offsets).
  - Saved `.mat` files still contain `parameters_az_tilt_roll_hfov` (the pose on the saved still), plus `mount_correction_daz_dtilt_roll_hfovz1`, `landmark_stills` and `skyline_still`. A MATLAB-made file loads by deriving the correction from its saved pose.
- **Export:** the KML has one placemark per time (named by the still's time, time-stamped for Google Earth's time slider unless **Time-stamp each line** next to Export is unticked; the stamps use the launcher's `UTC_OFFSET`, −07:00 for Verdi), not one per edge segment. The CSV keeps MATLAB's columns and one row per sample, and adds `time`, `line_id` (1 = earliest time) and `line_order` (position along that time's line; empty without a terrain hit).
- **Unchanged:** projection maths, conventions (1-based pixels). Only the Python round trip has been tested (no MATLAB here).
- **Network:** the server only listens on `127.0.0.1`. Running it inside Docker needs `--host 0.0.0.0` and a published port.
