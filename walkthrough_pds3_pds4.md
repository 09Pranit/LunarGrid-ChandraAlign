# Mixed PDS3/PDS4 lunar registration — implementation walkthrough

Verified on 29 September 2026. This implementation registers bounded, explicitly selected image tiles. It does not claim that the example LROC and TMC-2 observations overlap or that feature residuals establish lunar ground accuracy.

## What changed

Each source/reference card now has an independent **Auto / PDS4 XML / Embedded PDS3 / None — image only** selector, XML attachment, optional local-extraction manifest, server inspection, and tile controls. The workspace starts empty. The dedicated **Load demo data** button remains the only way to populate simulated metadata and results.

Auto gives an explicitly supplied valid XML precedence, otherwise uses a valid attached PDS3 label, otherwise accepts a supported self-describing image. Invalid XML never silently falls back. A deliberate switch to Embedded PDS3 independently validates that label and preserves the XML conflict warning. One image's missing label never removes the other image's metadata.

The backend now separates raw byte layout, acquisition telemetry, camera geometry and map georeferencing. Missing GSD, incidence, emission, sun azimuth, footprint and grid remain unknown. An attached LROC CDR label provides a raster layout; it does not provide a per-line Moon transform.

## Walk through a real pair

1. Install the backend requirements in a virtual environment. For a local demonstration without Redis, set `CELERY_TASK_ALWAYS_EAGER=true` and run `python -m uvicorn backend.main:app --port 8000`. Start the interface with `pnpm dev`. Use the worker configuration in `docs/pds-api-migration.md` for deployment.
2. Open **Settings** and configure the processing service URL. Leave demo mode empty for real data.
3. Load the source `.IMG`. Choose **Embedded PDS3**, or leave Auto without XML. Load the reference image and its own PDS4 XML; choose **PDS4 XML** or Auto. The standards can also be reversed or the same on both sides.
4. Select **Inspect image and label** on each card. Confirm the server-reported state, label location, product ID, processing level, dimensions, signed/unsigned physical type, geometry, missing fields and warnings. Before this operation the browser claims no authoritative metadata.
5. Set zero-based full-image `x`, `y`, `width`, `height`, and `band` for each tile. Use at least 32 pixels per axis and at most 1,048,576 pixels per tile by default. Select **Validate selected tile** to decode and preview that window.
6. If geometry is unknown, select corresponding windows using observation context or an independent map/camera workflow, then confirm that selection. Arbitrary mission examples are not an overlap test. Verified disjoint supported map grids are rejected.
7. Run registration. Inspect accepted/rejected correspondences, feature residual basis and review reasons. Without GSD and incidence on both sides, Auto records **insufficient telemetry** and selects SIFT/FLANN. An explicit neural override requires the separately installed pinned runtime/weights.
8. Download the dossier and, when the feature gates pass, the TIFF and tie-point CSV. Unknown geometry forces `review_required`. Rejected feature geometry returns a dossier without a misleading warped raster. Changing a file, label, manifest, mode, tile, demo state or starting another job invalidates stale results and downloads.

Controls are native keyboard-accessible form controls. Parsing and validation failures are announced through live status/error regions.

## Large scene route

The browser retains its 512 MiB per-file cap. The TMC-2 layout is 1,184,864,000 bytes, so use local extraction before upload. The public LROC layout is 528,929,736 bytes including its label; even though it fits the browser cap, only a bounded tile is decoded and local extraction avoids repeated large uploads.

```powershell
# Example origins only: select actual corresponding locations yourself.
python -m backend.tile_cli M174353756RC.IMG --x 1200 --y 18000 --width 512 --height 512 --output work/lroc-tile.tif
python -m backend.tile_cli observation.img --xml observation.xml --x 900 --y 72000 --width 512 --height 512 --output work/tmc-tile.tif
```

Load each generated TIFF and its matching `.provenance.json` on that image's card. The manifest contains the original label, product identity, full dimensions, original-file SHA-256 and tile origin. Do not attach the raw-file XML directly to the extracted TIFF; its byte layout differs.

The server verifies the uploaded tile's SHA-256, dimensions, physical type and contained label bounds. It explicitly reports the original full-file hash and extraction correspondence as **local-tool claims**, because the original scene was not uploaded. The manifest cannot assert a map grid. Original-file MD5 verification requires the original file; tile SHA verification is separate.

The extractor seeks only selected row segments for decoding. It separately streams the entire original file in 1 MiB blocks to compute SHA-256; this does not decode the scene or retain it in memory. Existing outputs are refused.

## Pixels, masks and coordinate provenance

For `M174353756RC`, the first sample is at byte **5064**. A row is **5064 × 2 = 10128 bytes**, spanning two physical records. `LSB_INTEGER` with 16 bits decodes as little-endian **int16**, whereas PDS4 `UnsignedLSB2` decodes as little-endian **uint16**. Record length never substitutes for row stride.

Special/null/saturation codes, valid ranges and non-finite floats are masked in the raw domain before scale and offset. Zero remains valid unless explicitly designated otherwise. Raw DN, float64 calibrated values and 8-bit display/matching stretch remain separate. Masks propagate through entropy, Wallis conditioning, feature support, refinement, bilinear interpolation and export.

The registered TIFF contains float64 interpolated raw DN, calibration tags and an internal validity mask, on the reference tile. CSV coordinates contain both tile-local and full-scene zero-based pixel centers, including local-extraction offsets. The dossier includes both translation matrices, standards, product IDs, layout, selected windows, geometry source, warnings, missing fields, matching mode and quality basis.

Only a supported validated reference TIFF grid enables lunar georeferencing and CSV ground coordinates. That grid must be Moon 2000 geographic, radius 1,737,400 m, east-positive longitude and a north-up square-pixel affine. Tile offsets shift its pixel-corner origin; CSV positions refer to pixel centers. Wrapped longitudes are handled. Otherwise the output TIFF has no CRS and latitude/longitude fields are empty. Independent ground validation remains **false**, even for a correctly georeferenced map product.

## Implementation map

| Area | Files |
| --- | --- |
| Independent controls, stale-state and demo isolation | `components/image-ingestion-card.tsx`, `hooks/use-lunar-workspace.ts`, `app/page.tsx`, `lib/ingestion.ts`, `lib/registration-api.ts` |
| Versioned common record and checked layout | `backend/lunar_core/io/metadata.py`, `backend/job_models.py` |
| Bounded ODL and secure associated XML descriptors | `backend/lunar_core/io/pds3_parser.py`, `pds4_parser.py`, `pds4_raster.py` |
| Bounded reading and local lineage | `backend/lunar_core/io/ingestion.py`, `tile_manifest.py`, `backend/tile_cli.py` |
| Quotas, staging, job admission/cancellation/cleanup | `backend/main.py`, `job_store.py`, `tasks.py`, `maintenance.py` |
| Mask-aware matching, quality and reference-tile export | `backend/pipeline.py`, `backend/lunar_core/matching/`, `preprocess/wallis.py`, `io/window_exporter.py`, `io/geotiff_exporter.py` |
| Tests and reproducible benchmark | `backend/tests/test_pds_ingestion_v2.py`, `workspace-ui.test.mjs`, `registration-client.test.mjs`, `backend/benchmark_tiles.py` |

## Validation results

| Check | Result |
| --- | --- |
| Complete backend suite | **449 passed, 2 existing expected failures**, 44.97 seconds |
| Client contract + real browser/React integration | **4 passed**, using installed Edge and Playwright 1.62.1 |
| TypeScript | Passed, `tsc --noEmit` |
| Changed frontend files | Oxlint passed |
| Production frontend build | Passed; 2,087 modules, Vite bundle generated |
| Repository-wide lint | Existing failures remain in unrelated UI components/hooks; changed files pass |

The two backend warnings are expected Rasterio notices when the tests reopen deliberately ungeoreferenced outputs. The two pre-existing expected failures concern Phase 2 global-statistics assumptions.

The new suite covers comments, multiline strings containing END, scoped/repeated keywords, units/sequences/radix/sentinels, malformed nesting and 400 fuzz cases; one-based pointers, size/overflow/header limits; signed/unsigned/endian/padded/windowed rasters, special values, scaling and all-null cases; mixed standards, forced modes, bad XML, identity/layout conflicts, optional telemetry, quota/stale-selection/cancellation paths; local manifests; georeferenced and ungeoreferenced exports.

The known-overlap mixed-standard end-to-end fixture is synthetic distributed texture with a known **(-6,-4) pixel translation**. It exercises real API ingestion, SIFT registration and TIFF/CSV/dossier downloads. Separate tests exercise unknown geometry, disjoint map grids, map-header export guards and mask propagation. This is not independent validation on the two mission observations. Browser integration uses mocked service responses while exercising actual React controls/state; backend end-to-end tests exercise real processing.

Reproduce from the repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TORCH_HOME=Join-Path (Get-Location) 'work/torch'
python -m pytest backend/tests -q
python -m backend.benchmark_tiles --output work/pds-tile-benchmark.json
node --test backend/tests/registration-client.test.mjs
pnpm exec tsc --noEmit
pnpm build
# With the interface running and Playwright 1.62.1 installed:
$env:LUNARGRID_BROWSER_CHANNEL='msedge' # optional: installed Edge instead of bundled Chromium
node --test backend/tests/workspace-ui.test.mjs
```

Neural regression tests require the pinned optional requirements/cached weights described in README. Core new PDS/SIFT checks do not download weights. `LUNARGRID_PLAYWRIGHT_MODULE` can specify the module URL of an existing Playwright installation.

## Bounded-read benchmark

| Exact full-scene layout | Full bytes | Last 512×512 tile origin | Row bytes | Pixel bytes read | Row seeks | Peak traced allocation |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| LROC 5064×52224, int16, offset 5064 | 528,929,736 | (4552, 51712) | 10,128 | 524,288 | 512 | 9,745,150 bytes |
| TMC-2 4000×148108, uint16, offset 0 | 1,184,864,000 | (3488, 147596) | 8,000 | 524,288 | 512 | 9,746,741 bytes |

Latest virtual-file timings were approximately 0.037 and 0.050 seconds. The benchmark asserts exact read volume, no access beyond EOF and a 64 MiB traced-allocation ceiling. It does **not** measure real disk throughput, network speed, native-library baseline/RSS or the original-file SHA pass. No large lunar products were downloaded or added to the repository.

## Operational limits and remaining unsupported modes

Default admission limits are 512 MiB/image, about 1 GiB/aggregate request, 1 MiB/XML or ODL header, 2 MiB/local manifest, 1,048,576 pixels/tile, a 1.5 GiB estimated processing budget, 8 GiB storage, 1 GiB free-space reserve, two active slots, 120-second upload deadline and 900-second worker timeout. Compressed full-image decoding/TIFF block size retains the 16,777,216-pixel guard. Reservations account for spool plus staged copies. Cancellation remains available under upload quota pressure.

These are application guards, not a hosted-deployment certification. Configure a memory-capped Linux prefork worker, proxy limits, authentication, private storage/temp volume and retention. Real Redis, process RSS and actual deployment/proxy throughput still need target-host measurements. Windows solo/eager execution cannot hard-interrupt native calls. Offline crash recovery is `python -m backend.maintenance --offline --older-than-hours 24` after stopping all API/workers.

Unsupported inputs fail explicitly: detached PDS3 labels/includes, packed/compressed/complex samples, unknown interleave, VAX/unknown float encodings and radix float bit-pattern special values. Only tested integer/IEEE representations and band-sequential layouts are accepted. XML never fetches external entities, schemas or URLs. ODL is parsed as data, never executed. Filename paths and remote pointers are rejected.

Camera-model/SPICE integration, automatic corresponding-window selection, rotated/projected map transforms, other lunar frames and full-scene registration are not implemented. Users select bounded corresponding tiles. Generic PDS3 geometry outside the supported profile remains unknown. ISIS is an optional external workflow: USGS directs NAC CDR/RDR users to `pds2isis`, followed by appropriate `spiceinit` and `cam2map`. The documentation references ISIS **7.1.0**, but installs/bundles no ISIS or kernels. Record exact external tool, kernel and body-shape versions/checksums and applicable licenses for separately processed products.

## Further documentation

In the repository, see `docs/pds-inputs.md` for user instructions, exact supported syntax/layouts and scientific limits; `docs/pds-api-migration.md` for schema 2, legacy compatibility, endpoints, quotas and deployment recovery. README links both.

Authoritative sources consulted: [NASA ODL](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter12.pdf), [pointers](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter14.pdf), [record formats](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter15.pdf), [physical data types](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter03.pdf), [LROC SIS 1.18](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DOCUMENT/LROCSIS.PDF), [ISIS NAC import guidance](https://isis.astrogeology.usgs.gov/dev/Application/presentation/Tabbed/lronac2isis/lronac2isis.html), [ISIS 7.1.0 cam2map](https://isis.astrogeology.usgs.gov/7.1.0/Application/presentation/PrinterFriendly/cam2map/cam2map.html). The [USGS software inventory](https://github.com/DOI-USGS/ISIS3/blob/dev/code.json) lists ISIS as Public Domain / CC0-1.0; verify notices for any exact external installation and its third-party dependencies.
