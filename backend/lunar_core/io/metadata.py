"""Version 2 ingestion records. Decoding, telemetry and ground geometry are distinct."""
from __future__ import annotations

from typing import Literal
import math
import re

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_ADDRESS = 2**63 - 1
HEADER_LIMIT = 1024 * 1024
DEFAULT_TILE_PIXELS = 1024 * 1024
Mode = Literal["auto", "pds4", "pds3", "none"]
DTYPES = {"|u1", "|i1", "<i2", ">i2", "<u2", ">u2", "<i4", ">i4", "<u4", ">u4", "<f4", ">f4", "<f8", ">f8"}


def checked(*values: int) -> int:
    result = 1
    for value in values:
        if type(value) is not int or value < 0 or value > MAX_ADDRESS:
            raise ValueError("Invalid or overflowing raster dimension/offset")
        result *= value
        if result > MAX_ADDRESS:
            raise ValueError("Raster byte count exceeds signed 64-bit address space")
    return result


def safe_name(name: str) -> str:
    # Names are compared only, never opened or joined to storage directories.
    if (not name or len(name) > 255 or name in {".", ".."}
            or any(c in name for c in '/\\:\x00') or any(ord(c) < 32 for c in name)):
        raise ValueError("File association requires a plain filename; paths and URLs are forbidden")
    return name


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Tile(Record):
    x: int = Field(ge=0, strict=True)
    y: int = Field(ge=0, strict=True)
    width: int = Field(gt=0, lt=32767, strict=True)
    height: int = Field(gt=0, lt=32767, strict=True)
    band: int = Field(default=0, ge=0, strict=True)

    def validate_for(self, layout: "RasterLayout", max_pixels=DEFAULT_TILE_PIXELS):
        if checked(self.width, self.height) > max_pixels:
            raise ValueError(f"Selected tile exceeds {max_pixels} pixels; select a smaller tile")
        if self.x + self.width > layout.width or self.y + self.height > layout.height or self.band >= layout.bands:
            raise ValueError("Selected tile or band is outside the full-image raster")
        return self


class RasterLayout(Record):
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    bands: int = Field(default=1, gt=0, le=16, strict=True)
    dtype: str
    sample_type: str
    offset: int = Field(default=0, ge=0, strict=True)
    prefix_bytes: int = Field(default=0, ge=0, strict=True)
    suffix_bytes: int = Field(default=0, ge=0, strict=True)
    interleave: Literal["BSQ"] = "BSQ"
    decoder: Literal["raw", "container"] = "raw"
    container_format: Literal["TIFF", "PNG", "JPEG", "WEBP", "BMP"] | None = None
    scale: float = 1.0
    value_offset: float = 0.0
    special_constants: list[float] = Field(default_factory=list, max_length=32)
    valid_min: float | None = None
    valid_max: float | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.dtype not in DTYPES:
            raise ValueError("Unsupported physical sample representation")
        if self.valid_min is not None and self.valid_max is not None and self.valid_min > self.valid_max:
            raise ValueError("Invalid raw DN range")
        info = np.iinfo(self.dtype) if np.dtype(self.dtype).kind in 'iu' else np.finfo(self.dtype)
        for v in self.special_constants + [v for v in (self.valid_min, self.valid_max) if v is not None]:
            if v < info.min or v > info.max or (np.dtype(self.dtype).kind in 'iu' and not v.is_integer()):
                raise ValueError("Special constant/range cannot be represented in declared sample type")
        checked(self.offset + checked(self.row_bytes, self.height, self.bands))
        return self

    @property
    def row_bytes(self):
        return checked(self.width, np.dtype(self.dtype).itemsize) + self.prefix_bytes + self.suffix_bytes

    @property
    def end_byte(self):
        return self.offset + checked(self.row_bytes, self.height, self.bands)

    def validate_size(self, size: int):
        checked(size)
        if self.decoder == "raw" and self.end_byte > size:
            raise ValueError(f"Truncated raster: data object requires {self.end_byte} bytes, file has {size}")


class LunarGrid(Record):
    """Only independently supplied, header-validated Moon geographic TIFF grids."""
    west_lon: float = Field(ge=-360, le=360)
    north_lat: float = Field(ge=-90, le=90)
    pixel_size_deg: float = Field(gt=0, le=360)
    frame: Literal["IAU2000:30100"] = "IAU2000:30100"
    radius_m: Literal[1737400.0] = 1737400.0
    latitude_type: Literal["planetocentric"] = "planetocentric"
    longitude_direction: Literal["positive_east"] = "positive_east"
    longitude_domain: Literal["continuous"] = "continuous"
    projection: Literal["geographic"] = "geographic"
    pixel_convention: Literal["upper_left_corner"] = "upper_left_corner"
    provenance: Literal["validated_geotiff_header"] = "validated_geotiff_header"

    def window(self, tile: Tile):
        values = self.model_dump()
        values.update(west_lon=((self.west_lon + tile.x * self.pixel_size_deg + 180) % 360) - 180,
                      north_lat=self.north_lat - tile.y * self.pixel_size_deg)
        grid = LunarGrid(**values)
        if grid.north_lat - tile.height * grid.pixel_size_deg < -90 or tile.width * grid.pixel_size_deg > 360:
            raise ValueError("Reference tile grid exceeds lunar bounds")
        return grid


class ImageMetadata(Record):
    schema_version: Literal[2] = 2
    source_format: Literal["PDS3", "PDS4", "IMAGE_ONLY"]
    label_location: Literal["attached", "xml_sidecar", "none", "original_attached_in_manifest", "original_xml_in_manifest"]
    ingestion_route: Literal["direct", "local_tile"] = "direct"
    selected_mode: Mode = "auto"
    file_name: str
    file_size: int = Field(ge=0)
    sha256: str | None = None
    label_sha256: str | None = None
    product_id: str | None = None
    data_set_id: str | None = None
    instrument: str | None = None
    processing_level: str | None = None
    target: str | None = None
    frame_id: str | None = None
    acquisition: dict[str, str] = Field(default_factory=dict)
    layout: RasterLayout
    gsd_meters: float | None = Field(default=None, gt=0)
    incidence_angle_deg: float | None = Field(default=None, ge=0, le=90)
    emission_angle_deg: float | None = Field(default=None, ge=0, le=90)
    solar_azimuth_deg: float | None = Field(default=None, ge=0, lt=360)
    footprint: list[list[float]] | None = None
    grid: LunarGrid | None = None
    geometry_status: Literal["map_projected", "camera_model", "unknown"] = "unknown"
    original_coordinate_convention: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    field_provenance: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checksum: dict[str, str] = Field(default_factory=dict)
    # Local extraction claims are explicitly distinguished from server-verified input bytes.
    lineage: dict = Field(default_factory=dict)

    def finish(self):
        label_source = {'PDS3': 'attached PDS3 OBJECT=IMAGE', 'PDS4': 'associated PDS4 descriptor', 'IMAGE_ONLY': 'self-describing image header'}[self.source_format]
        for key in type(self.layout).model_fields:
            self.field_provenance.setdefault('layout.' + key, label_source + ' (allowlisted layout and specified/default identity calibration)')
        if self.grid:
            g = self.grid
            east, south = g.west_lon + self.layout.width * g.pixel_size_deg, g.north_lat - self.layout.height * g.pixel_size_deg
            self.footprint = [[g.west_lon,g.north_lat],[east,g.north_lat],[east,south],[g.west_lon,south],[g.west_lon,g.north_lat]]
            self.geometry_status = "map_projected"
            self.field_provenance["footprint"] = "exact rectangular footprint from validated geographic TIFF grid; continuous longitude"
            self.field_provenance['grid'] = 'validated Moon 2000 geographic TIFF CRS and affine header; independent ground accuracy unverified'
        self.missing_fields = [f for f in ("gsd_meters", "incidence_angle_deg", "emission_angle_deg", "solar_azimuth_deg", "footprint", "grid") if getattr(self, f) is None]
        if self.grid is None:
            self.warnings.append("Overlap unknown: select corresponding tiles. No camera model or validated map grid.")
        self.warnings = list(dict.fromkeys(self.warnings))
        return self


def finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def digest_shape(value: str):
    if not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError("Invalid SHA-256 digest")
    return value.lower()
