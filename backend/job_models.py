"""Versioned API options. Missing telemetry/georeferencing is never invented."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
if __package__:
    from .lunar_core.io.pds4_parser import LunarTelemetryMetadata
    from .lunar_core.io.metadata import ImageMetadata, Mode, Tile, DEFAULT_TILE_PIXELS
else:
    from lunar_core.io.pds4_parser import LunarTelemetryMetadata
    from lunar_core.io.metadata import ImageMetadata, Mode, Tile, DEFAULT_TILE_PIXELS


class ReferenceGrid(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    west_lon: float = Field(ge=-360, le=360)
    north_lat: float = Field(ge=-90, le=90)
    pixel_size_deg: float = Field(gt=0, le=360)


class JobParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1, 2] = 1
    source_mode: Mode = "auto"
    reference_mode: Mode = "auto"
    source_tile: Tile | None = None
    reference_tile: Tile | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reference_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    confirm_unknown_overlap: bool = False
    verify_checksum: bool = False
    engine: Literal["auto", "sift_flann", "superpoint_lightglue"] = "auto"
    source_metadata: LunarTelemetryMetadata | None = None
    reference_metadata: LunarTelemetryMetadata | None = None
    reference_grid: ReferenceGrid | None = None

    @model_validator(mode="after")
    def paired_metadata(self):
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class ValidatedJobParams(JobParams):
    source_record: ImageMetadata
    reference_record: ImageMetadata
    max_tile_pixels: int = DEFAULT_TILE_PIXELS
    max_memory_bytes: int = 1536 * 1024**2
    worker_timeout: int = 900
    overlap_status: str = "unknown"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "complete", "review_required", "failed"]
    progress_percent: int = Field(ge=0, le=100)
    stage: str | None = None
    error: str | None = None


class QualityMetrics(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    rmse_px: float | None = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    vsui_score: float = Field(ge=0, le=1)
    active_engine: Literal["sift_flann", "superpoint_lightglue"]
    rmse_basis: str


class ResultArtifacts(BaseModel):
    registered_preview_base64: str | None = None
    geotiff_download_url: str | None = None
    tie_points_csv_url: str | None = None
    dossier_url: str | None = None
    georeferenced: bool = False


class JobResult(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    job_id: str
    status: Literal["complete", "review_required"]
    progress_percent: Literal[100]
    metrics: QualityMetrics
    artifacts: ResultArtifacts

    @model_validator(mode="after")
    def completed_quality(self):
        if self.status == "complete":
            if self.metrics.rmse_px is None or self.metrics.rmse_px > 0.5 or self.metrics.vsui_score < 0.75:
                raise ValueError("Complete results must pass RMSE and VSUI gates")
            if not all((self.artifacts.registered_preview_base64, self.artifacts.geotiff_download_url,
                        self.artifacts.tie_points_csv_url)):
                raise ValueError("Complete results require preview, TIFF, and CSV artifacts")
        return self


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true, false, 1, or 0")
    return value.lower() in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "LUNARGRID_DATA_DIR", str(Path(__file__).resolve().parents[1] / "work" / "jobs"))))
    broker_url: str = field(default_factory=lambda: os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
    always_eager: bool = field(default_factory=lambda: env_bool("CELERY_TASK_ALWAYS_EAGER", False))
    eager_fallback: bool = field(default_factory=lambda: env_bool("LUNARGRID_EAGER_FALLBACK", False))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_UPLOAD_BYTES", 512 * 1024**2)))
    max_pixels: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_PIXELS", 16_777_216)))
    max_tile_pixels: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_TILE_PIXELS", DEFAULT_TILE_PIXELS)))
    max_memory_bytes: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_MEMORY_BYTES", 1536 * 1024**2)))
    max_storage_bytes: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_STORAGE_BYTES", 8 * 1024**3)))
    min_free_bytes: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MIN_FREE_BYTES", 1024**3)))
    max_request_bytes: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_REQUEST_BYTES", 1024**3 + 3 * 1024**2)))
    max_active_jobs: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_MAX_ACTIVE_JOBS", 2)))
    upload_timeout: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_UPLOAD_TIMEOUT", 120)))
    job_timeout: int = field(default_factory=lambda: int(os.getenv("LUNARGRID_JOB_TIMEOUT", 900)))

    def __post_init__(self):
        if min(self.max_upload_bytes, self.max_pixels, self.job_timeout, self.max_tile_pixels, self.max_memory_bytes, self.max_storage_bytes, self.min_free_bytes, self.max_request_bytes, self.max_active_jobs, self.upload_timeout) <= 0:
            raise ValueError("Upload, pixel, and timeout limits must be positive")
