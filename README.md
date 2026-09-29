# Chandra-Align

LunarGrid's Smart India Hackathon prototype for multi-sensor lunar image co-registration.

The interface provides a labeled demonstration mode and an asynchronous registration workspace. The ingestion schema 2 workspace supports independent PDS3 attached labels, PDS4 XML and image-only inputs, bounded raw raster tiles, mask-aware matching, held-out feature checks and provenance-rich exports through FastAPI and Celery. See [input instructions](docs/pds-inputs.md), [API migration and deployment limits](docs/pds-api-migration.md), and the [implementation walkthrough](walkthrough_pds3_pds4.md).

Local manual testing notes are kept in `MANUAL_TEST_PLAN.md` and excluded from Git because they may contain workstation-specific paths. The demo uses simulated matches and metrics; real-image results require the processing service and independent validation.

## Run the interface

```powershell
pnpm dev
```

The workspace opens empty. Select **Load demo data** to populate the illustrative pair, then **Run Registration** to replay the simulation. Select **Clear workspace** to return to the empty state. For real data, load both image files and connect the processing service in Settings.

## Run the processing service

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
$env:CELERY_TASK_ALWAYS_EAGER = 'true' # local development only; use Celery in deployment
.venv\Scripts\uvicorn main:app --app-dir backend --reload --port 8000
```

Open the API documentation at `http://localhost:8000/docs`. Production accuracy claims must use named Chandrayaan/LROC products and independent held-out checkpoints.

## Adaptive matcher routing (Phase 3)

`backend/lunar_core/matching/router.py` selects `sift_flann` when both the
valid-pixel entropy difference is at most 0.8 bits and geometric disparity is at
most 3.0; otherwise it selects `superpoint_lightglue`. Edit the `routing` section
of `backend/config.yaml` to adjust either threshold or geometry weights `w1` and
`w2` (both default to 0.5). SciPy and PyYAML are included in backend requirements.

With `backend/` on the Python import path:

```python
from lunar_core.io.pds4_parser import parse_pds4_label
from lunar_core.matching import route_pair

decision = route_pair(
    source_gray, reference_gray,  # NumPy grayscale arrays scaled to [0, 255]
    parse_pds4_label("source.xml"), parse_pds4_label("reference.xml"),
)
print(decision.model_dump_json(indent=2))
```

Run routing before Wallis conditioning with explicit source/reference validity masks. Valid zero pixels remain valid; all-null tiles are rejected. Adaptive routing requires GSD and incidence on both sides. Otherwise the job pipeline records `insufficient telemetry` and uses SIFT/FLANN, or an explicit engine override. The
geometry formula uses the directional ratio `GSD_src/GSD_ref`; reversing a pair
can change its decision. Incidence differences use degrees. The returned
Pydantic model contains individual entropies, both disparity metrics, geometry
components, the selected engine, rationales and a configuration snapshot.
Phase 8 dispatches this routing decision through `/api/v1/registration/jobs`.

Run the complete deterministic router checks and display calculation logs:

```powershell
python backend/tests/test_phase3_router.py
# Equivalent pytest invocation:
python -m pytest backend/tests/test_phase3_router.py -q --log-cli-level=INFO
```

## Dual-engine feature matching (Phase 4)

`backend/lunar_core/matching/matcher.py` provides `BaseMatcher`, `SIFTMatcher`,
`LightGlueMatcher`, and `MatchResult`. SIFT runs with the base backend requirements.
For pretrained SuperPoint + LightGlue, install the optional runtime and pinned
upstream package in the same environment:

```powershell
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/requirements-matching.txt
python -m pip install --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8"
$env:TORCH_HOME = Join-Path (Get-Location) 'work/torch'
python backend/tests/test_phase4_matching.py
```

The separate `--no-deps` install prevents upstream's `opencv-python` from
colliding with this project's `opencv-python-headless`. Git is required for the
pinned source install. Install a compatible PyTorch/Torchvision CUDA build first
if GPU acceleration is needed. The first neural constructor downloads pretrained
weights; subsequent runs can use the populated Torch cache offline. Keep
`TORCH_HOME` set to the same directory when reusing that cache.

With `backend/` on the Python import path:

```python
from lunar_core.matching import SIFTMatcher, LightGlueMatcher

classical = SIFTMatcher()
neural = LightGlueMatcher(device="auto", max_num_keypoints=2048)  # or 4096
result = neural.match(source_gray, reference_gray)
points = result.tie_points  # float32 (N,4): x_src, y_src, x_ref, y_ref
print(result.engine_name, len(result), result.execution_time_ms)
```

Both engines accept raw or Wallis-conditioned NumPy arrays: grayscale, single
channel, BGR, or BGRA, with finite intensities in `[0,255]`; normalized floating
point `[0,1]` is also supported. Scale sensor DN values into that range first.
Inputs are preserved. `pts_src` and `pts_ref` have shape `(N,2)` in original pixel
coordinates; `confidences` has shape `(N,)`. Insufficient texture returns empty
arrays with those same trailing dimensions. No RANSAC or transform estimation
occurs inside either engine.

SIFT uses 10,000 features, contrast threshold 0.03, FLANN KD-tree with five trees
and 50 checks, and Lowe ratio 0.75. By default both directional ratio checks must
agree; `mutual_check=False` selects one-way matching. Confidence is the smaller
of the two `1 - distance_ratio` scores, or the forward score for one-way matching.
LightGlue returns its own learned confidence scores. These are engine-specific
quality scores, not interchangeable calibrated probabilities.

`device="auto"` selects available CUDA, then MPS, then CPU. An explicitly requested
unavailable accelerator warns and falls back to CPU. CPU inference uses float32
without CUDA synchronization or mixed precision. SuperPoint resizing is disabled
to retain native resolution; very large images should be tiled by the caller.
Timing includes preparation, feature extraction, matching and transfer to NumPy,
but excludes model construction and weight downloads. Reuse engine instances.

The standalone Phase 4 script requires both engines and never skips the neural
gate. It checks at least 50 matches and mean transfer error below 2 pixels over
**all** matches for scale 1.2, rotation 15 degrees about the image centre, and
added translation `(30,-20)`, before RANSAC. It repeats the gate after independent
Wallis preprocessing and checks input formats, empty results, invalid inputs,
dependency isolation, and device selection. Actual CUDA/MPS inference requires
corresponding hardware. `/register` continues to use its existing registration
pipeline; dispatch integration is separate from these matcher interfaces.

Upstream API and pretrained weights: [official LightGlue repository](https://github.com/cvg/LightGlue).

## Geometric rejection and spatial uniformity (Phase 5)

`backend/lunar_core/registration/outlier_rejection.py` provides MAGSAC++ filtering
and an 8x8 confidence-ranked Grid-NMS, followed by bounded Voronoi spatial
uniformity analysis. OpenCV and SciPy are already in `backend/requirements.txt`.
With `backend/` on the Python import path, pass a Phase 4 match result:

```python
from lunar_core.registration import filter_magsac_and_vsui, RegistrationRejected

try:
    filtered = filter_magsac_and_vsui(
        result.pts_src, result.pts_ref, result.confidences,
        reference_gray.shape[:2],  # (height, width), points use (x, y)
        top_k=1, source_shape=source_gray.shape[:2],
    )
except RegistrationRejected as error:
    diagnostics = error.result  # rejected model/masks are for diagnosis only
    print(error)
else:
    transform = filtered.homography  # source -> reference
    tie_points = filtered.tie_points  # (N,4), after Grid-NMS
    print(filtered.inlier_ratio, filtered.vsui)
```

The function requires `cv2.USAC_MAGSAC`, uses the specified 3.5-pixel threshold,
10,000 maximum iterations, and 0.999 confidence. It rejects registrations below
70% geometric consensus, fewer than 15 distinct points after Grid-NMS, degenerate
geometry, or final VSUI below 0.75. The ratio uses the **original input count**,
before spatial thinning. Repeated source/reference endpoints are reduced to
their strongest one-to-one matches; they cannot inflate consensus.

Grid-NMS retains up to `top_k` sites per reference cell (default one, at most 64
overall); confidence ties prefer earlier input indices. Homography estimation
uses the full consensus before NMS. Both masks preserve original input indexing,
and returned points/confidences stay aligned. Inputs are preserved.

SciPy Voronoi cells are clipped to the full reference rectangle `[0,W] x [0,H]`,
including boundary regions. `compute_vsui` uses population area standard deviation
and the formula `clip(1 - std/(mean + 1e-6), 0, 1)`. Duplicate sites count once;
empty input scores zero. Sparse/collinear sites have defined spatial scores but
cannot bypass registration's count and geometry gates. Grid-NMS cannot create
coverage in empty cells, so poor spatial support can still cause rejection.

Run the standalone suite with acceptance logs:

```powershell
python backend/tests/test_phase5_filtering.py
```

The 60%-outlier stress case measures rejection accuracy using the exception's
diagnostic mask: successful filtering leaves about 40% inliers and **must reject
registration** under the 70% rule. Separate valid cases test acceptance at 70%
and 80%. Tests also cover confidence ranking, a 100-point localized cluster,
determinism, analytic bounded cell areas, rectangular canvases, and quality gates.
This phase exposes the reusable filtering API; `/register` still uses its existing
pipeline. See `walkthrough_phase5.md` for measured results.

OpenCV supplies MAGSAC++'s internal noise marginalization; the Python API does
not offer an Epanechnikov-kernel setting. The 3.5-pixel parameter remains relevant
to consensus and termination. References: [OpenCV USAC](https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html)
and [SciPy Voronoi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html).

## Non-rigid TPS warping (Phase 6)

`backend/lunar_core/registration/tps_warper.py` fits a global thin-plate spline
with SciPy's `RBFInterpolator`, `U(r) = r^2 ln(r)` (zero at the origin), an affine
polynomial, and default regularization `smoothing=0.05`. Source and reference
coordinates are in original pixels; no coordinate scaling changes lambda.
The existing backend requirements already include all dependencies.

With `backend/` on the Python import path, pass accepted Phase 5 correspondences:

```python
from lunar_core.registration import solve_tps_warp

warp = solve_tps_warp(filtered.pts_src, filtered.pts_ref)
aligned = warp.warp(source, reference.shape[:2])
predicted_reference_xy = warp.forward(checkpoint_source_xy)
validation_rmse_px = warp.rmse(checkpoint_source_xy, checkpoint_reference_xy)
assert aligned.shape[:2] == reference.shape[:2]
```

Points use `(x,y)`; image shapes use `(height,width)`. Separate forward and
backward splines are fitted. Raster resampling uses backward coordinates with
`cv2.remap`, bicubic interpolation, and constant zero padding. It preserves
uint8, uint16, int16, float32, or float64 values and grayscale/HWC channel layout
(1-4 channels), without intensity normalization.

Canvases with at least 512x512 pixels use an 8-pixel coordinate-grid step and
bilinear `cv2.resize`. A padded, pixel-center-aligned displacement grid avoids
edge and half-pixel shifts, including canvas sizes not divisible by eight.
Smaller canvases use dense evaluation in bounded batches. Pass `grid_step=1`
for dense evaluation or `grid_step=8` to force the fast path. Full-resolution
float32 coordinate maps still require about eight bytes per destination pixel.

Run `python backend/tests/test_phase6_tps.py` to display the validation gates.
The suite fits 64 ties to the requested sinusoidal distortion and evaluates
500 unseen points per seed; measured RMSE is 0.126-0.128 px on a 384x512 canvas,
below 0.45 px. It also checks raster correction, exact target dimensions, zero
padding, cubic interpolation, the explicit regularized block system, affine
edge alignment, dtype/channel preservation, and coarse-grid accuracy/runtime.
See `walkthrough_phase6.md` for the measured results and reproduction command.

Training residuals (`fitting_rmse_px`, `backward_fitting_rmse_px`) are separate
from held-out validation. The backward fit is an approximate inverse for smooth
one-to-one relief; TPS does not prevent folds or guarantee accuracy outside tie
coverage. Lambda's effect depends on pixel spacing. OpenCV requires each source
and target dimension below 32767; larger rasters require caller-side tiling.
This phase exposes the reusable warping API; `/register` integration is separate.

References: [SciPy RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
and [OpenCV geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

## Moon 2000 GeoTIFF and audit export (Phase 7)

`backend/lunar_core/io/geotiff_exporter.py` exports aligned `(H,W)` or `(H,W,C)`
arrays without changing uint8, uint16, or float32 samples. Install the updated
`backend/requirements.txt` for Rasterio 1.4.4 and pyproj 3.7.2.

**CRS correction:** `EPSG:9001` identifies Earth's IGS97 geocentric CRS, not the
Moon. This exporter embeds the historical `IAU2000:30100` Moon 2000 WKT (spherical
radius 1,737,400 m, degrees, east-positive longitude). GDAL may omit the historical
authority on readback; `CRS_IDENTIFIER=IAU2000:30100` preserves the identifier.
Tests compare the reopened CRS semantically with pyproj and check its actual
spheroid and angular units rather than requiring the incorrect EPSG string.

With `backend/` on the Python import path, continue from Phases 5 and 6. Here `reference_grid` must come from a validated reference TIFF and the selected tile: `inspect_image(path, path.name).grid.window(tile)`. An absent grid requires the ungeoreferenced export path; arbitrary job-supplied numbers cannot establish georeferencing:

```python
import numpy as np
from lunar_core.io.geotiff_exporter import export_registration_bundle

residuals = np.linalg.norm(warp.forward(filtered.pts_src) - filtered.pts_ref, axis=1)
artifacts = export_registration_bundle(
    aligned, "outputs/job-007",
    west_lon=reference_grid.west_lon, north_lat=reference_grid.north_lat,
    pixel_size_deg=reference_grid.pixel_size_deg, validated_grid=reference_grid,
    tie_points=filtered.tie_points,
    residuals_px=residuals, confidences=filtered.confidences,
    job_id="job-007", rmse_px=validation_rmse_px, rmse_basis="held_out_checkpoints",
    vsui=filtered.vsui, inlier_ratio=filtered.inlier_ratio,
    job_telemetry={"source": source_metadata, "reference": reference_metadata},
    pipeline_timeline=[{"stage": "matching", "duration_ms": result.execution_time_ms}],
)
print(artifacts.geotiff, artifacts.tiepoints_csv, artifacts.dossier_json)
```

Each job produces `registered_output.tif`, `tiepoints.csv`, and
`registration_dossier.json`. Existing artifact names are refused. The lower-level
`export_geotiff(array, path, west_lon=..., north_lat=..., pixel_size_deg=..., validated_grid=...)`
writes only the TIFF, stages replacements, and checks its reopened spatial header.
`export_tiepoints_csv` is also independently callable.

The origin is the upper-left pixel **corner**, with the specified
`from_origin(west_lon, north_lat, pixel_size_deg, pixel_size_deg)` transform.
Tie-point CSV coordinates use zero-based pixel **centers**, so reference `(0,0)`
maps to `(west + size/2, north - size/2)`. CSV uses the requested nine columns,
CRLF record endings and RFC-4180 quoting. Residuals must be post-registration
radial errors, not raw source/reference displacements. The dossier keeps supplied
RMSE and its basis separate from tie-point fitting RMSE; telemetry, metrics,
the caller's timeline, measured export stages and raster metadata are recorded.

LZW uses predictor 2 for integers and 3 for floats. Optional `scales`, `offsets`,
and `units` sequences preserve per-band calibration metadata without applying it
to samples. Zero is valid unless explicitly set as `nodata`. Continuous longitudes
allow antimeridian crossings; grids cannot exceed one revolution or the poles.
Inputs must already be aligned on a lunar geographic grid: this step assigns
georeferencing, and does not resample another projection or infer calibration.
The schema 2 job pipeline uses `window_exporter.py`: masked float64 interpolated DN, reference-tile coordinates and four additional full-scene pixel columns. It exports a plain TIFF when the reference grid is unknown.

Run `python backend/tests/test_phase7_export.py` for the 512x512 spatial gate,
header/CRS logs, lossless dtype/channel checks, CSV and dossier validation, and
Phase 6 integration. See `walkthrough_phase7.md` for measured results.
References: [GDAL Moon 2000 definition](https://lists.osgeo.org/pipermail/gdal-dev/2015-September/042718.html),
[EPSG:9001 registry entry](https://sis.apache.org/tables/CoordinateReferenceSystems.html),
[GeoTIFF options](https://gdal.org/en/stable/drivers/raster/gtiff.html),
[RFC 4180](https://www.rfc-editor.org/rfc/rfc4180.html).

## Research workstation controls

The four viewer modes share zoom, pan, fit, and pixel inspection. Pixel values are 8-bit preview samples, not original calibrated DN values. Overlay compares source against reference. Tie Points shows both endpoints, with green circles for accepted matches and red crosses for rejected matches; the selector supports keyboard inspection.

The example uses a single real LROC WAC photograph of Tycho crater (NASA/GSFC/Arizona State University), shifted by 18/-12 pixels for the source. OHRC/NAC dataset names and acquisition metadata are illustrative and do not describe that photograph. All 48 demo correspondences and quality figures are simulated. Reports and CSV rows carry an explicit simulation marker.

Image source: https://science.nasa.gov/image-detail/amf-cc06843a-5b68-4cb4-9119-c62e383b54fa/

For real images, load both files and select metadata independently on each card. An optional XML belongs only to that image. Inspect and validate a bounded tile for each side, then confirm corresponding tiles when overlap is unknown. Set the service URL in Settings. Uploading invalidates the simulated results. The hook submits a job, polls actual progress, and displays completion or review-required status. Accepted TPS control points retain original pixel coordinates even when the previews are reduced. Match confidence is an engine-specific score, not a calibrated probability. Configure `LUNARGRID_ALLOWED_ORIGINS` as a comma-separated list when using another UI origin.

TIFF and tie-point CSV exports download directly from the job's artifact URLs. A validated reference TIFF grid can produce a Moon 2000 GeoTIFF; otherwise the download is explicitly an ungeoreferenced TIFF and CSV latitude/longitude fields are blank. Legacy caller-supplied numeric grids are ignored with a warning. PDS4 acquisition metadata alone does not establish a reference grid. JSON reports record job ID, quality status, RMSE basis, VSUI, and georeferencing status.

## Registration API and worker (ingestion schema 2)

The complete contract, compatibility changes, quotas and recovery instructions are in [API migration notes](docs/pds-api-migration.md). The `/api/v1/registration/jobs` transport remains compatible with small legacy self-describing uploads. New clients use schema 2, independent modes and explicit tile windows. Labels must associate with the uploaded image; XML-only telemetry cannot decode raw IMG.

Start Redis with persistence, then launch API and worker with the same absolute private storage directory and Torch cache:

```powershell
$env:CELERY_BROKER_URL = 'redis://localhost:6379/0'
$env:LUNARGRID_DATA_DIR = Join-Path (Get-Location) 'work/jobs'
$env:TORCH_HOME = Join-Path (Get-Location) 'work/torch'
$env:CELERY_TASK_ALWAYS_EAGER = 'false'
$env:LUNARGRID_EAGER_FALLBACK = 'false'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# Separate terminal with the same environment (Linux production worker):
python -m celery -A backend.tasks:celery_app worker --loglevel=INFO --concurrency=1
# Windows local worker: add --pool=solo; hard process time limits require Linux prefork.
```

Broker failure fails closed by default. Explicit `CELERY_TASK_ALWAYS_EAGER=true` enables synchronous local development without Redis; the POST then waits for processing. Keep deployment storage outside cloud-sync folders. Use an authenticated gateway, private local volume, retention policy, memory-limited workers and compatible proxy limits. This repository does not implement a user-account system or certify a hosted deployment.

The browser keeps its 512 MiB file cap. Selected tiles default to at most 1,048,576 pixels; full raw dimensions are checked without decoding the whole scene. Large inputs use [local extraction with original-label provenance](docs/pds-inputs.md#large-scenes-supported-local-extraction). Raw pixels, radiometric scaling, matching stretch and masks remain separate. Unknown geometry forces review and cannot produce invented latitude/longitude.

The API provides inspection, submission, status/results, cancellation and TIFF/CSV/dossier downloads. Cancellation is cooperative at processing boundaries; Linux worker hard timeouts bound native calls. Inputs are removed after terminal processing. Feature rejection produces a dossier without a falsely successful raster. Valid map georeferencing does not constitute independent ground validation.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:TORCH_HOME = Join-Path (Get-Location) 'work/torch'
python -m pytest backend/tests -q
python -m backend.benchmark_tiles --output work/pds-tile-benchmark.json
node --test backend/tests/registration-client.test.mjs
pnpm exec tsc --noEmit
pnpm build
```

Neural tests use the separately installed pinned requirements and cached weights described above. Browser integration instructions and synthetic mixed-standard acceptance coverage are in [input instructions](docs/pds-inputs.md#reproducible-checks). Real Redis delivery and target-host resource measurements remain deployment checks.

## Vercel deployment

`vercel.json` builds the same React workstation with `vite.vercel.config.ts`. The `vercel-entry` wrapper reuses `app/page.tsx`, all components, the existing CSS, and image assets. The registration API remains an external service configured in Settings.

Import this GitHub repository into Vercel with these settings:

- Framework Preset: **Vite** (single frontend project).
- Root Directory: **.** (repository root, where `vercel.json` lives).
- Build Command: **npm run build**.
- Output Directory: **vercel-dist**.
- Install Command: leave the automatic default.

If the import screen detects `frontend` and `backend` as multiple services and says "vercel.json required to deploy projects with multiple services", switch to a single Vite project before deploying. Do not accept the generated multi-service configuration: this repository's Vercel configuration deploys the frontend, while `backend/` contains a separately run Python processing service. The public sample workflow runs without that service; real uploaded-image registration requires its URL in Settings.

Build locally with `npm run build`, then deploy with `vercel --prod`. Configure `LUNARGRID_ALLOWED_ORIGINS` on your registration backend to include the final Vercel URL. The original Sites build remains available through `pnpm build:sites`.
