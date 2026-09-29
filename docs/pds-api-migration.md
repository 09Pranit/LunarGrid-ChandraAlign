# Ingestion schema 2 / API 0.9 migration

Existing `/api/v1/registration/jobs`, status/results, artifacts, `/metadata` and `/health` paths remain. New requests should explicitly use `params.schema_version=2`. The URL version describes the transport contract; the schema version describes per-image ingestion records. New internal metadata/layout/grid records are server-owned and rejected if supplied as job options.

| Endpoint | Multipart / behavior |
| --- | --- |
| `POST /api/v1/registration/inspect` | `file`, optional `label` (XML), optional `provenance` (local extraction manifest), `mode`, JSON `tile` or `null`; returns common metadata, canonical detected-label state, validated tile/preview when requested, and pixel budget |
| `POST /api/v1/registration/jobs` | `source_file`, `reference_file`, independently optional `source_label` / `reference_label`, independently optional `source_provenance` / `reference_provenance`, JSON `params` |
| `POST /api/v1/registration/jobs/{id}/cancel` | Queued jobs are removed immediately; running jobs stop at stage boundaries and remove inputs/partial artifacts. The terminal status is `failed` with a cancellation explanation, preserving the existing status enum. |
| `GET /api/v1/registration/jobs/{id}/results` | Independent source/reference common records, selection tiles, overlap/matching mode, full-scene transforms, routing rationale, feature-quality metrics, review reasons, artifact URLs |
| `GET /health` | Adds ingestion version, upload/tile limits and local-extraction entry point; remains a liveness endpoint, not worker readiness. |

Example `params` for a selected unknown-geometry pair:

```json
{
  "schema_version": 2,
  "source_mode": "pds3",
  "reference_mode": "auto",
  "source_tile": {"x": 100, "y": 250, "width": 512, "height": 512, "band": 0},
  "reference_tile": {"x": 75, "y": 300, "width": 512, "height": 512, "band": 0},
  "confirm_unknown_overlap": true,
  "engine": "auto",
  "verify_checksum": false
}
```

`source_sha256` and `reference_sha256` may contain the inspection digests; mismatched bytes reject submission. Files, labels, modes and windows are revalidated at submission regardless of client inspection. The UI invalidates inspection after changes and retains no stale result. A file cannot select a server path. XML and provenance manifests are mutually exclusive for each side.

Common metadata version 2 includes source format, actual/retained label location, ingestion route, product/file identity, SHA-256, raw layout, selected mode, acquisition fields, optional geometric values, coordinate conventions, optional validated map grid, field provenance, missing fields, checksum status, warnings and extraction lineage. Unknown telemetry is JSON `null`. Default scale 1/offset 0 describe identity radiometry only when no calibration is declared, not measured acquisition geometry.

## Compatibility changes

- Schema 1 remains the default for existing clients. Small self-describing full-image uploads can omit tiles and are treated as explicitly uploaded image tiles. Raw IMG always requires a validated layout and explicit window. Schema 2 always requires explicit tile selection and unknown-overlap confirmation when applicable.
- XML labels no longer need to be paired. They must actually associate with their image and supply a supported raw/encoded descriptor. Historical telemetry-only XML fixtures are still accepted by `/metadata` for standalone optional telemetry inspection, but cannot establish file association at job submission. Correct the label; do not rely on silent fallback.
- `LunarTelemetryMetadata` now permits missing acquisition fields. Emission is never defaulted to zero. Existing complete telemetry records still parse. Legacy per-image JSON telemetry remains accepted independently, is marked unverified, and does not drive adaptive geometry routing.
- Legacy numeric `reference_grid` is ignored with a warning. It cannot assert Moon coordinates. Supply a supported lunar geographic TIFF reference whose CRS/affine can be validated. Lower-level `export_geotiff` / `export_registration_bundle` now require an internal `LunarGrid` matching their transform; they are not upload endpoints.
- Missing trustworthy GSD/incidence selects documented SIFT/FLANN fallback. `engine="superpoint_lightglue"` explicitly selects the optional learned engine; unavailable dependencies/weights cause a real failure, not substitution.
- `review_required` can contain a dossier and metrics without TIFF/CSV/preview when feature gates reject the pair. Consumers must check each artifact before enabling its download. Unknown overlap forces review even if feature gates pass. Reports never assert independent lunar ground accuracy.
- Raw inputs are removed when a worker finishes. Successful artifacts and dossiers persist until retention cleanup. Reports now include schema 2 provenance and tile/full-scene coordinate conventions; CSV adds four full-scene pixel columns after the original nine columns.
- Broker failure now fails closed by default. Set `CELERY_TASK_ALWAYS_EAGER=true` only for explicit local development, or deliberately enable `LUNARGRID_EAGER_FALLBACK=true`. Eager POST waits for processing; cancellation without a received job ID and hard interruption of native work are unavailable in that local mode.

## Operational limits

| Environment setting | Default | Scope |
| --- | ---: | --- |
| `LUNARGRID_MAX_UPLOAD_BYTES` | 536,870,912 | Each staged image; browser retains the same upper cap |
| `LUNARGRID_MAX_REQUEST_BYTES` | 1,076,887,552 | Entire request before multipart parsing, including chunked bodies; additionally bounded by two file caps plus label overhead |
| `LUNARGRID_MAX_PIXELS` | 16,777,216 | Whole compressed-image decode / TIFF block guard; not raw scene size |
| `LUNARGRID_MAX_TILE_PIXELS` | 1,048,576 | Each selected registration window, before allocation |
| `LUNARGRID_MAX_MEMORY_BYTES` | 1,610,612,736 | Conservative allocation estimate, not an OS RSS limiter |
| `LUNARGRID_MAX_STORAGE_BYTES` | 8,589,934,592 | Storage admission, including existing files and cross-process reservations |
| `LUNARGRID_MIN_FREE_BYTES` | 1,073,741,824 | Free-space reserve; rechecked while copying staged uploads |
| `LUNARGRID_MAX_ACTIVE_JOBS` | 2 | Active jobs and concurrent upload/inspection reservations |
| `LUNARGRID_UPLOAD_TIMEOUT` | 120 seconds | Total body reception deadline |
| `LUNARGRID_JOB_TIMEOUT` | 900 seconds | Celery soft/hard limits and cooperative stage checks |

Admission conservatively reserves **two request bodies** plus export headroom because multipart spool and staged copies can coexist. Set the process temp directory on the same quota-controlled volume as `LUNARGRID_DATA_DIR`. Temporary directories and stored images use generated names, never user filenames. Failed admission/validation cleans staging and releases reservations. Queue claims, cancellation and admission are transactional in SQLite. API endpoints do expensive parsing, hashing and decoding in worker threads, not the event loop.

Use a Linux prefork Celery worker for production hard time limits, with one job per worker process and a container/cgroup memory cap (initially 2 GiB per worker; lower the tile budget if your measured neural working set exceeds it). The estimated-memory guard is not a substitute for that cap. API/multipart spooling needs separate disk/memory headroom. Windows solo workers and local eager execution have cooperative limits only. Validate real Redis delivery, worker memory, proxy buffering and deployment timeouts on the target host; this checkout's tests do not certify a hosted deployment.

Configure the reverse proxy's body limit no higher than the API aggregate limit, request deadline consistently with the 120-second limit, and storage for any proxy buffering. Most serverless web deployments cannot proxy these scene sizes; serve the interface there and use a separately configured processing service or upload locally extracted small tiles. Do not increase every raster/upload limit to accept TMC-2 whole-scene decoding.

Cancellation stops a running job at its next progress boundary; a native matcher may run until its worker hard limit. Interrupted uploads close their multipart temporary files. A process crash can leave disk files or conservative reservations. Stop API/workers before offline recovery:

```powershell
python -m backend.maintenance --offline --older-than-hours 24
```

This clears crash reservations and removes expired job directories/records and abandoned inspection/upload directories. Do not run it against a live service. Provide normal service authentication/authorization and a private storage volume when deploying beyond a trusted research environment; this project does not introduce a user-account system.
