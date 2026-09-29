"""Production job API. Launch with uvicorn main:app --app-dir backend."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
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
    from .job_models import JobParams, JobResult, JobStatus, Settings
    from .job_store import JobStore, TERMINAL
    from .lunar_core.io.pds4_parser import parse_metadata
    from .pipeline import validate_image
    from .tasks import build_celery, configure_dispatch, enqueue
else:
    from job_models import JobParams, JobResult, JobStatus, Settings
    from job_store import JobStore, TERMINAL
    from lunar_core.io.pds4_parser import parse_metadata
    from pipeline import validate_image
    from tasks import build_celery, configure_dispatch, enqueue

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024
ARTIFACT = re.compile(r"(job_lunar_[0-9a-f]{32})(\.tif|_tiepoints\.csv|_dossier\.json)\Z")
ARTIFACT_FILES = {".tif": ("registered_output.tif", "image/tiff"),
                  "_tiepoints.csv": ("tiepoints.csv", "text/csv"),
                  "_dossier.json": ("registration_dossier.json", "application/json")}


class UploadLimitMiddleware:
    """Bound the entire body before multipart parsing, including chunked uploads."""
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({"detail": "Upload exceeds request limit"}, 413)(scope, receive, send)
        consumed = 0
        exceeded = False

        async def limited_receive():
            nonlocal consumed, exceeded
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > self.max_bytes:
                exceeded = True
                if headers.get(b"content-type", b"").startswith(b"multipart/form-data"):
                    # The parser closes partial spooled files for this exception.
                    raise MultiPartException("Upload exceeds request limit")
                raise HTTPException(413, "Upload exceeds request limit")
            return message

        async def limited_send(message):
            if exceeded and message["type"] == "http.response.start":
                message = {**message, "status": 413}
            await send(message)

        await self.app(scope, limited_receive, limited_send)


def save_upload(upload: UploadFile, destination: Path, limit: int):
    total = 0
    with destination.open("xb") as target:
        while block := upload.file.read(CHUNK_BYTES):
            total += len(block)
            if total > limit:
                raise HTTPException(413, "A file exceeds the configured upload limit")
            target.write(block)
    if total == 0:
        raise HTTPException(422, "Empty image upload")


def read_label(upload: UploadFile):
    content = upload.file.read(CHUNK_BYTES + 1)
    if len(content) > CHUNK_BYTES:
        raise HTTPException(413, "PDS4 labels must be <= 1 MiB")
    return parse_metadata(content)


def create_app(settings: Settings | None = None, queue=None) -> FastAPI:
    settings = settings or Settings()
    queue = queue if queue is not None else build_celery(settings)

    @asynccontextmanager
    async def lifespan(application):
        await run_in_threadpool(configure_dispatch, queue, settings)
        yield
        queue.close()

    application = FastAPI(title="Chandra-Align Processing API", version="0.8.0", lifespan=lifespan)
    application.state.settings = settings
    application.state.queue = queue
    application.add_middleware(UploadLimitMiddleware, max_bytes=2 * settings.max_upload_bytes + 3 * CHUNK_BYTES)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("LUNARGRID_ALLOWED_ORIGINS", "http://localhost:3000,https://lunargrid-chandra-align.pranitk0905.chatgpt.site").split(","),
        allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
        expose_headers=["Location", "Retry-After"],
    )

    @application.get("/health")
    def health():
        return {"status": "ok", "engine": "adaptive-sift-lightglue-tps",
                "execution_mode": "eager" if queue.conf.task_always_eager else "celery"}

    @application.post("/metadata")
    def metadata(label: UploadFile = File(...)):
        try:
            telemetry = read_label(label)
            return {"filename": label.filename, "telemetry": telemetry.model_dump(mode="json")}
        except (ValueError, ET.ParseError) as exc:
            raise HTTPException(422, "Invalid PDS4 telemetry: " + str(exc)) from exc
        finally:
            label.file.close()

    @application.post("/api/v1/registration/jobs", status_code=202,
                      response_model=JobStatus, response_model_exclude_none=True)
    def submit_job(source_file: UploadFile = File(...), reference_file: UploadFile = File(...),
                   params: str = Form("{}"), source_label: UploadFile | None = File(None),
                   reference_label: UploadFile | None = File(None)):
        uploads = [source_file, reference_file, source_label, reference_label]
        try:
            if len(params) > 16_384:
                raise HTTPException(422, "params exceeds 16 KiB")
            options = JobParams.model_validate_json(params)
            if (source_label is None) != (reference_label is None):
                raise HTTPException(422, "Supply both source_label and reference_label")
            if source_label is not None:
                if options.source_metadata is not None:
                    raise HTTPException(422, "Supply PDS4 labels or JSON telemetry, not both")
                options = JobParams.model_validate({**options.model_dump(),
                    "source_metadata": read_label(source_label), "reference_metadata": read_label(reference_label)})
            store = JobStore(settings.storage_dir)
            job_id = "job_lunar_" + uuid4().hex
            directory = store.directory(job_id)
            # Original filenames never form paths. Only validated inputs are queued.
            with TemporaryDirectory(prefix=".upload-", dir=store.root) as staging:
                for name, upload in (("source", source_file), ("reference", reference_file)):
                    path = Path(staging) / (name + ".img")
                    save_upload(upload, path, settings.max_upload_bytes)
                    validate_image(path, settings.max_pixels)
                Path(staging).rename(directory)
            store.create(job_id, options.model_dump(mode="json"))
            try:
                enqueue(queue, settings, job_id)
            except Exception as exc:
                logger.exception("Cannot queue %s", job_id)
                store.finish(job_id, "failed", error="Registration queue unavailable")
                raise HTTPException(503, {"job_id": job_id, "message": "Registration queue unavailable"}) from exc
            return JSONResponse({"job_id": job_id, "status": "queued", "progress_percent": 0}, 202,
                                headers={"Location": f"/api/v1/registration/jobs/{job_id}", "Retry-After": "1"})
        except (ValidationError, ValueError, ET.ParseError) as exc:
            raise HTTPException(422, "Invalid registration input: " + str(exc)) from exc
        finally:
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
