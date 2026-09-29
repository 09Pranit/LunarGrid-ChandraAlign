"""Production job API. Launch with uvicorn main:app --app-dir backend."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import hashlib
import json
import shutil
import asyncio
from time import monotonic
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4
import xml.etree.ElementTree as ET

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException

if __package__:
    from .job_models import JobParams, ValidatedJobParams, JobResult, JobStatus, Settings
    from .job_store import JobStore, TERMINAL
    from .lunar_core.io.pds4_parser import parse_metadata
    from .pipeline import preview_png
    from .lunar_core.io.ingestion import inspect_image, read_tile, overlap_for
    from .lunar_core.io.metadata import Tile
    from .tasks import build_celery, configure_dispatch, enqueue
else:
    from job_models import JobParams, ValidatedJobParams, JobResult, JobStatus, Settings
    from job_store import JobStore, TERMINAL
    from lunar_core.io.pds4_parser import parse_metadata
    from pipeline import preview_png
    from lunar_core.io.ingestion import inspect_image, read_tile, overlap_for
    from lunar_core.io.metadata import Tile
    from tasks import build_celery, configure_dispatch, enqueue

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024
ARTIFACT = re.compile(r"(job_lunar_[0-9a-f]{32})(\.tif|_tiepoints\.csv|_dossier\.json)\Z")
ARTIFACT_FILES = {".tif": ("registered_output.tif", "image/tiff"),
                  "_tiepoints.csv": ("tiepoints.csv", "text/csv"),
                  "_dossier.json": ("registration_dossier.json", "application/json")}


class UploadLimitMiddleware:
    """Bound the entire body before multipart parsing, including chunked uploads."""
    def __init__(self, app, max_bytes: int, settings):
        self.app, self.max_bytes, self.settings = app, max_bytes, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if length < 0:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({"detail": "Upload exceeds request limit"}, 413)(scope, receive, send)
        consumed = 0
        exceeded = False
        timed_out = False
        deadline = monotonic() + self.settings.upload_timeout

        async def limited_receive():
            nonlocal consumed, exceeded, timed_out
            try:
                message = await asyncio.wait_for(receive(), max(0.001, deadline - monotonic()))
                if monotonic() > deadline:
                    raise TimeoutError
            except TimeoutError as exc:
                timed_out = True
                if headers.get(b"content-type", b"").startswith(b"multipart/form-data"):
                    raise MultiPartException("Upload time quota exceeded; use a smaller local tile") from exc
                raise HTTPException(408, "Upload time quota exceeded") from exc
            if message['type'] == 'http.disconnect' and headers.get(b'content-type', b'').startswith(b'multipart/form-data'):
                raise MultiPartException('Upload disconnected')
            consumed += len(message.get("body", b""))
            if consumed > self.max_bytes:
                exceeded = True
                if headers.get(b"content-type", b"").startswith(b"multipart/form-data"):
                    # The parser closes partial spooled files for this exception.
                    raise MultiPartException("Upload exceeds request limit")
                raise HTTPException(413, "Upload exceeds request limit")
            return message

        async def limited_send(message):
            if (exceeded or timed_out) and message["type"] == "http.response.start":
                message = {**message, "status": 408 if timed_out else 413}
            await send(message)

        # Cancellation must remain available when upload slots/disk are exhausted.
        # Only routes accepting files need an upload storage reservation.
        if scope.get("path") not in {"/metadata", "/api/v1/registration/inspect", "/api/v1/registration/jobs"}:
            return await self.app(scope, limited_receive, limited_send)
        store = await run_in_threadpool(JobStore, self.settings.storage_dir)
        try:
            token = await run_in_threadpool(store.reserve, self.settings)
        except ValueError as exc:
            return await JSONResponse({"detail": str(exc)}, 429)(scope, receive, send)
        try:
            await self.app(scope, limited_receive, limited_send)
        finally:
            await run_in_threadpool(store.release, token)


def save_upload(upload: UploadFile, destination: Path, limit: int, settings=None):
    total = 0
    digest = hashlib.sha256()
    with destination.open("xb") as target:
        while block := upload.file.read(CHUNK_BYTES):
            total += len(block)
            if total > limit:
                raise HTTPException(413, "A file exceeds the configured upload limit")
            if settings and shutil.disk_usage(destination.parent).free < settings.min_free_bytes + len(block):
                raise HTTPException(507, "Upload stopped: free-space reserve reached")
            target.write(block)
            digest.update(block)
    if total == 0:
        raise HTTPException(422, "Empty image upload")
    return digest.hexdigest()


def label_bytes(upload: UploadFile, limit=CHUNK_BYTES):
    content = upload.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(413, "Label/provenance file exceeds its configured size limit")
    return content


def read_label(upload: UploadFile):
    return parse_metadata(label_bytes(upload))


def create_app(settings: Settings | None = None, queue=None) -> FastAPI:
    settings = settings or Settings()
    queue = queue if queue is not None else build_celery(settings)

    @asynccontextmanager
    async def lifespan(application):
        await run_in_threadpool(configure_dispatch, queue, settings)
        yield
        queue.close()

    application = FastAPI(title="Chandra-Align Processing API", version="0.9.0", lifespan=lifespan)
    application.state.settings = settings
    application.state.queue = queue
    application.add_middleware(UploadLimitMiddleware, max_bytes=min(settings.max_request_bytes, 2 * settings.max_upload_bytes + 7 * CHUNK_BYTES), settings=settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("LUNARGRID_ALLOWED_ORIGINS", "http://localhost:3000,https://lunargrid-chandra-align.pranitk0905.chatgpt.site").split(","),
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
        expose_headers=["Location", "Retry-After"],
    )

    @application.get("/health")
    def health():
        return {"status": "ok", "engine": "adaptive-sift-lightglue-tps",
                "execution_mode": "eager" if queue.conf.task_always_eager else "celery",
                "ingestion_schema": 2, "max_upload_bytes": settings.max_upload_bytes,
                "max_tile_pixels": settings.max_tile_pixels,
                "large_raster_route": "python -m backend.tile_cli --help"}

    @application.post("/metadata")
    def metadata(label: UploadFile = File(...)):
        try:
            telemetry = read_label(label)
            return {"filename": label.filename, "telemetry": telemetry.model_dump(mode="json")}
        except (ValueError, ET.ParseError) as exc:
            raise HTTPException(422, "Invalid PDS4 telemetry: " + str(exc)) from exc
        finally:
            label.file.close()

    def resolve_tile(record, tile, *, explicit=False):
        if tile is None:
            if explicit or record.layout.width * record.layout.height > settings.max_tile_pixels:
                raise ValueError("Select a bounded tile with its full-image x/y origin before registration")
            tile = Tile(x=0, y=0, width=record.layout.width, height=record.layout.height)
        tile.validate_for(record.layout, settings.max_tile_pixels)
        if tile.width * tile.height * 128 + 64 * 1024**2 > settings.max_memory_bytes:
            raise ValueError("Tile allocation exceeds configured memory budget")
        if min(tile.width, tile.height) < 32:
            raise ValueError("Registration tiles require at least 32 pixels per axis")
        return tile

    @application.post("/api/v1/registration/inspect")
    def inspect(file: UploadFile = File(...), label: UploadFile | None = File(None),
                mode: str = Form("auto"), tile: str = Form("null"), provenance: UploadFile | None = File(None)):
        try:
            if len(tile) > 1024:
                raise ValueError("Tile selection exceeds 1 KiB")
            selected = Tile.model_validate_json(tile) if tile != "null" else None
            store = JobStore(settings.storage_dir)
            with TemporaryDirectory(prefix=".inspect-", dir=store.root) as staging:
                path = Path(staging) / "image.bin"
                digest = save_upload(file, path, settings.max_upload_bytes, settings)
                record = inspect_image(path, file.filename or "image", mode,
                    label_bytes(label) if label else None, max_pixels=settings.max_pixels, digest=digest, provenance=label_bytes(provenance, 2*CHUNK_BYTES) if provenance else None)
                preview = None
                if selected:
                    selected = resolve_tile(record, selected)
                    decoded = read_tile(path, record, selected, max_pixels=settings.max_tile_pixels)
                    if not decoded.valid.any():
                        raise ValueError("Selected tile contains only null/saturated/non-finite samples")
                    preview = preview_png(decoded.display)
                return {"metadata": record.model_dump(mode="json"),
                        "state": {"PDS3": "PDS3 attached label", "PDS4": "PDS4 XML", "IMAGE_ONLY": "No usable label"}[record.source_format],
                        "tile": selected.model_dump() if selected else None, "preview": preview,
                        "validation": "server-validated uploaded bytes", "max_tile_pixels": settings.max_tile_pixels}
        except (ValueError, ET.ParseError) as exc:
            raise HTTPException(422, "Inspection failed: " + str(exc)) from exc
        finally:
            file.file.close()
            if label:
                label.file.close()
            if provenance:
                provenance.file.close()

    @application.post("/api/v1/registration/jobs", status_code=202,
                      response_model=JobStatus, response_model_exclude_none=True)
    def submit_job(source_file: UploadFile = File(...), reference_file: UploadFile = File(...),
                   params: str = Form("{}"), source_label: UploadFile | None = File(None),
                   reference_label: UploadFile | None = File(None),
                   source_provenance: UploadFile | None = File(None), reference_provenance: UploadFile | None = File(None)):
        uploads = [source_file, reference_file, source_label, reference_label, source_provenance, reference_provenance]
        directory = None
        queued = False
        try:
            if len(params) > 16_384:
                raise ValueError("params exceeds 16 KiB")
            options = JobParams.model_validate_json(params)
            store = JobStore(settings.storage_dir)
            job_id = "job_lunar_" + uuid4().hex
            validated = options.model_dump(mode="json")
            with TemporaryDirectory(prefix=".upload-", dir=store.root) as staging:
                for name, upload, label, provenance in (("source", source_file, source_label, source_provenance), ("reference", reference_file, reference_label, reference_provenance)):
                    path = Path(staging) / (name + ".img")
                    digest = save_upload(upload, path, settings.max_upload_bytes, settings)
                    expected = getattr(options, name + "_sha256")
                    if expected and expected != digest:
                        raise ValueError(name + " changed after inspection; inspect the current file again")
                    record = inspect_image(path, upload.filename or name, getattr(options, name + "_mode"),
                        label_bytes(label) if label else None, max_pixels=settings.max_pixels,
                        verify_checksum=options.verify_checksum, digest=digest, provenance=label_bytes(provenance, 2*CHUNK_BYTES) if provenance else None)
                    legacy = getattr(options, name + "_metadata")
                    if legacy:
                        if label or record.source_format != "IMAGE_ONLY":
                            raise ValueError("Supply label metadata or legacy JSON telemetry, not both")
                        for key in ("gsd_meters", "incidence_angle_deg", "emission_angle_deg", "solar_azimuth_deg", "product_id", "instrument"):
                            setattr(record, key, getattr(legacy, key))
                            record.field_provenance[key] = "legacy client-supplied, unverified telemetry"
                        record.warnings.append("Legacy JSON telemetry is unverified and is not used for adaptive geometry routing.")
                    selected = resolve_tile(record, getattr(options, name + "_tile"),
                        explicit=options.schema_version == 2 or record.layout.decoder == "raw")
                    # Bounded validation catches all-null/truncated selected data before enqueue.
                    decoded = read_tile(path, record, selected, max_pixels=settings.max_tile_pixels)
                    if not decoded.valid.any():
                        raise ValueError(name + " tile contains no valid samples")
                    validated[name + "_record"] = record.finish().model_dump(mode="json")
                    validated[name + "_tile"] = selected.model_dump()
                    del decoded
                from_records = ValidatedJobParams.model_validate(validated)
                overlap = overlap_for(from_records.source_record, from_records.reference_record,
                    from_records.source_tile, from_records.reference_tile)
                if overlap == "nonoverlap":
                    raise ValueError("Selected reference/source map tiles do not overlap; choose corresponding windows")
                if overlap == "unknown" and options.schema_version == 2 and not options.confirm_unknown_overlap:
                    raise ValueError("Overlap is unknown; explicitly confirm corresponding user-selected tiles for image-only matching")
                pixels = sum(t["width"] * t["height"] for t in (validated["source_tile"], validated["reference_tile"]))
                if pixels * 512 + 256 * 1024**2 > settings.max_memory_bytes:
                    raise ValueError("Pair exceeds estimated worker memory budget; select smaller tiles")
                validated.update(overlap_status=overlap, max_tile_pixels=settings.max_tile_pixels,
                    max_memory_bytes=settings.max_memory_bytes, worker_timeout=settings.job_timeout)
                if options.reference_grid:
                    for side in ("source", "reference"):
                        validated[side + "_record"]["warnings"].append("Legacy reference_grid is unverified and ignored; supply a validated lunar GeoTIFF reference.")
                validated["reference_grid"] = None
                directory = store.directory(job_id)
                Path(staging).rename(directory)
            store.create(job_id, validated, settings.max_active_jobs)
            try:
                queued = True
                enqueue(queue, settings, job_id)
            except Exception as exc:
                logger.exception("Cannot queue %s", job_id)
                store.finish(job_id, "failed", error="Registration queue unavailable")
                queued = False
                raise HTTPException(503, {"job_id": job_id, "message": "Registration queue unavailable"}) from exc
            return JSONResponse({"job_id": job_id, "status": "queued", "progress_percent": 0}, 202,
                                headers={"Location": f"/api/v1/registration/jobs/{job_id}", "Retry-After": "1"})
        except (ValidationError, ValueError, ET.ParseError) as exc:
            raise HTTPException(422, "Invalid registration input: " + str(exc)) from exc
        finally:
            if directory and directory.exists() and not queued:
                shutil.rmtree(directory)
            for upload in uploads:
                if upload is not None:
                    upload.file.close()

    def lookup(job_id, *, include_result=False):
        try:
            store = JobStore(settings.storage_dir)
            store.directory(job_id)
            store.expire_stalled(job_id, settings.job_timeout + 30)
            return store, store.get(job_id, include_result=include_result)
        except KeyError as exc:
            raise HTTPException(404, "Registration job not found") from exc

    @application.post("/api/v1/registration/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        store, record = lookup(job_id)
        previous = store.cancel(job_id)
        if previous not in TERMINAL:
            # Running workers observe cancellation at stage boundaries and clean up.
            # A queued job can be removed immediately; atomic claim will then fail.
            if previous == "queued":
                directory = store.directory(job_id)
                if directory.exists():
                    shutil.rmtree(directory)
        return {"job_id": job_id, "status": "failed" if record["status"] not in TERMINAL else record["status"]}

    @application.get("/api/v1/registration/jobs/{job_id}", response_model=JobStatus,
                     response_model_exclude_none=True)
    def job_status(job_id: str):
        _, record = lookup(job_id)
        return JSONResponse(record, headers={"Cache-Control": "no-store"})

    @application.get("/api/v1/registration/jobs/{job_id}/results", response_model=JobResult | JobStatus)
    def job_results(job_id: str):
        _, record = lookup(job_id, include_result=True)
        if record["status"] not in TERMINAL:
            payload = {key: record[key] for key in ("job_id", "status", "progress_percent", "stage")}
            return JSONResponse(payload, 202, headers={"Retry-After": "1", "Cache-Control": "no-store"})
        if record["status"] == "failed":
            return JSONResponse({"job_id": job_id, "status": "failed", "progress_percent": 100,
                                 "error": record["error"]}, headers={"Cache-Control": "no-store"})
        return JSONResponse(record["result"], headers={"Cache-Control": "no-store"})

    @application.get("/api/v1/artifacts/{filename}")
    def artifact(filename: str):
        match = ARTIFACT.fullmatch(filename)
        if not match:
            raise HTTPException(404, "Artifact not found")
        job_id, suffix = match.groups()
        store, record = lookup(job_id, include_result=True)
        if record["status"] not in {"complete", "review_required"} or not record["result"].get("artifacts"):
            raise HTTPException(404, "Artifact not available")
        name, media_type = ARTIFACT_FILES[suffix]
        path = store.directory(job_id) / "artifacts" / name
        if not path.is_file() or not path.resolve().is_relative_to(store.directory(job_id)):
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type=media_type, filename=filename,
                            headers={"X-Content-Type-Options": "nosniff"})

    return application


app = create_app()
