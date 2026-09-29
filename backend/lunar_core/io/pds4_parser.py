"""Bounded PDS4 telemetry parser; missing fields remain None.

Known mission tag aliases are read literally, without inferred instrument GSD
or nadir defaults. XML DTDs/entities are forbidden. Acquisition bounding boxes
are coarse summaries, never pixel-to-ground transforms. File association and
raster descriptors are validated in pds4_raster.py; standalone telemetry parsing
does not establish which uploaded image a label describes.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field, ValidationError, model_validator

__all__ = [
    "InvalidTelemetryError",
    "LunarBoundingBox",
    "LunarTelemetryMetadata",
    "MOON_RADIUS_M",
    "compute_bbox_overlap",
    "compute_scale_gap",
    "parse_metadata",
    "parse_pds4_label",
]


# ── Constants ─────────────────────────────────────────────────────────

MOON_RADIUS_M: float = 1_737_400.0
"""Mean lunar radius in metres (IAU 2015)."""


# ── Exceptions ────────────────────────────────────────────────────────

class InvalidTelemetryError(ValueError):
    """Raised when parsed telemetry violates physical constraints.

    Examples: latitude outside [-90, 90], non-positive GSD, angles
    out of range, or required XML elements missing entirely.
    """


# ── Pydantic v2 Models ───────────────────────────────────────────────

class LunarBoundingBox(BaseModel):
    """Spherical bounding box on the lunar surface (decimal degrees).

    Latitude  : [-90, 90]   (planetocentric)
    Longitude : [-360, 360] (positive-East, allows antimeridian wrap)
    """

    model_config = {"frozen": True, "allow_inf_nan": False}

    lat_min: float = Field(..., description="Southern latitude bound (°)")
    lat_max: float = Field(..., description="Northern latitude bound (°)")
    lon_min: float = Field(..., description="Western longitude bound (°)")
    lon_max: float = Field(..., description="Eastern longitude bound (°)")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "LunarBoundingBox":
        for name, val in [("lat_min", self.lat_min), ("lat_max", self.lat_max)]:
            if val < -90.0 or val > 90.0:
                raise InvalidTelemetryError(
                    f"{name}={val}° is outside valid lunar latitude range [-90, 90]"
                )
        if self.lat_min > self.lat_max:
            raise InvalidTelemetryError(
                f"lat_min ({self.lat_min}°) must be ≤ lat_max ({self.lat_max}°)"
            )
        if abs(self.lon_min) > 360 or abs(self.lon_max) > 360:
            raise InvalidTelemetryError("Longitude outside [-360,360]")
        return self

    def spherical_area(self, radius: float = MOON_RADIUS_M) -> float:
        """Area of this spherical rectangle in m².

        .. math::
            A = R^2 \\cdot |\\sin \\varphi_2 - \\sin \\varphi_1| \\cdot |\\lambda_2 - \\lambda_1|

        where φ = latitude (rad), λ = longitude (rad).
        """
        lat1 = math.radians(self.lat_min)
        lat2 = math.radians(self.lat_max)
        dlon = math.radians((self.lon_max - self.lon_min) % 360 if abs(self.lon_max - self.lon_min) < 360 else 360)
        return radius * radius * abs(math.sin(lat2) - math.sin(lat1)) * dlon


class LunarTelemetryMetadata(BaseModel):
    """Standardised telemetry output from PDS4 or GeoTIFF ingestion.

    All angles in degrees, all distances in metres.
    """

    model_config = {"frozen": True, "allow_inf_nan": False}

    product_id: Optional[str] = Field(
        None, description="PDS4 logical identifier or product ID"
    )
    instrument: Optional[str] = Field(
        None, description="Instrument name (OHRC, TMC-2, IIRS, NAC)"
    )
    gsd_meters: float | None = Field(None, gt=0.0, description="Ground Sample Distance (m)")
    incidence_angle_deg: float | None = Field(
        None, ge=0.0, le=90.0, description="Solar incidence angle (°)"
    )
    emission_angle_deg: float | None = Field(
        None, ge=0.0, le=90.0, description="Sensor emission angle (°)"
    )
    solar_azimuth_deg: float | None = Field(
        None, ge=0.0, lt=360.0, description="Solar azimuth angle (°)"
    )
    bounding_box: Optional[LunarBoundingBox] = None
    pixel_resolution_m: Optional[float] = Field(
        None, gt=0.0, description="Pixel resolution (m)"
    )
    source_format: str = Field("PDS4", description="PDS4 or GeoTIFF")


# ── Pure Geometry Functions ───────────────────────────────────────────

def compute_scale_gap(gsd_src: float, gsd_ref: float) -> float:
    r"""GSD scale-gap ratio between two images.

    .. math::
        S_{\text{gap}} = \frac{\max(GSD_\text{src}, GSD_\text{ref})}
                              {\min(GSD_\text{src}, GSD_\text{ref})}

    Returns
    -------
    float
        Scale gap ratio ≥ 1.0.

    Raises
    ------
    InvalidTelemetryError
        If either GSD is non-positive.
    """
    if gsd_src <= 0.0 or gsd_ref <= 0.0:
        raise InvalidTelemetryError(
            f"GSD must be positive: src={gsd_src}, ref={gsd_ref}"
        )
    return max(gsd_src, gsd_ref) / min(gsd_src, gsd_ref)


def compute_bbox_overlap(
    bbox_src: LunarBoundingBox,
    bbox_ref: LunarBoundingBox,
    radius: float = MOON_RADIUS_M,
) -> float:
    r"""Spherical bounding-box overlap ratio.

    .. math::
        \text{Overlap} = \frac{\text{Area}(B_\text{src} \cap B_\text{ref})}
                              {\min(\text{Area}(B_\text{src}),\;
                                    \text{Area}(B_\text{ref}))}

    Returns
    -------
    float
        Ratio in [0.0, 1.0].  0.0 when disjoint.
    """
    lat_lo = max(bbox_src.lat_min, bbox_ref.lat_min)
    lat_hi = min(bbox_src.lat_max, bbox_ref.lat_max)
    if lat_lo >= lat_hi:
        return 0.0
    def intervals(box):
        span = (box.lon_max - box.lon_min) % 360 if abs(box.lon_max - box.lon_min) < 360 else 360
        start = box.lon_min % 360
        end = start + span
        return [(start, min(end,360))] + ([(0,end-360)] if end > 360 else [])
    longitude = sum(max(0., min(b,d)-max(a,c)) for a,b in intervals(bbox_src) for c,d in intervals(bbox_ref))
    area = radius**2 * abs(math.sin(math.radians(lat_hi))-math.sin(math.radians(lat_lo))) * math.radians(longitude)
    denominator = min(bbox_src.spherical_area(radius),bbox_ref.spherical_area(radius))
    return float(np.clip(area/denominator,0,1)) if denominator > 0 else 0.


# ── XML Tag Mapping ──────────────────────────────────────────────────
# Maps PDS4 local-name (lower-case) → canonical field name used
# internally.  First match per canonical field wins.

_TAG_MAP: dict[str, str] = {
    # Ground Sample Distance
    "pixel_resolution":          "gsd",
    "ground_sampling_distance":  "gsd",
    "spatial_resolution":        "gsd",
    # Illumination & viewing angles
    "incidence_angle":           "incidence_angle",
    "solar_incidence_angle":     "incidence_angle",
    "solar_incidence":           "incidence_angle",
    "incidence":                 "incidence_angle",
    "emission_angle":            "emission_angle",
    "emission":                  "emission_angle",
    "solar_azimuth_angle":       "solar_azimuth",
    "sub_solar_azimuth":         "solar_azimuth",
    "sun_azimuth":               "solar_azimuth",
    "solar_azimuth":             "solar_azimuth",
    # Bounding coordinates (PDS4 Cartography namespace)
    "south_bounding_coordinate": "lat_min",
    "minimum_latitude":          "lat_min",
    "southernmost_latitude":     "lat_min",
    "north_bounding_coordinate": "lat_max",
    "maximum_latitude":          "lat_max",
    "northernmost_latitude":     "lat_max",
    "west_bounding_coordinate":  "lon_min",
    "minimum_longitude":         "lon_min",
    "westernmost_longitude":     "lon_min",
    "east_bounding_coordinate":  "lon_max",
    "maximum_longitude":         "lon_max",
    "easternmost_longitude":     "lon_max",
    # Identity
    "logical_identifier":        "product_id",
    "product_id":                "product_id",
    "instrument_name":           "instrument",
    "instrument_id":             "instrument",
}


# ── Internal Helpers ─────────────────────────────────────────────────

def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix: ``{http://…}local`` → ``local``."""
    idx = tag.rfind("}")
    return tag[idx + 1 :] if idx != -1 else tag


def _try_float(text: Optional[str]) -> Optional[float]:
    """Attempt to parse a float, return *None* on failure."""
    if not text:
        return None
    try:
        return float(text.strip())
    except (ValueError, TypeError):
        return None


def _apply_unit(value: float, unit: Optional[str], field: str) -> float:
    """Convert a raw value to canonical units (metres / degrees)."""
    if unit is None:
        return value
    u = unit.lower().strip().replace(' ', '')
    if field == "gsd":
        distances = {'m':1., 'km':1000., 'cm':.01, 'mm':.001}
        base = u.removesuffix('/pixel').removesuffix('/px')
        if base not in distances:
            raise InvalidTelemetryError('Unsupported GSD units; require physical distance per pixel')
        return value * distances[base]
    if field in ("incidence_angle", "emission_angle", "solar_azimuth", "lat_min", "lat_max", "lon_min", "lon_max"):
        if u in {'rad', 'radian', 'radians'}:
            return math.degrees(value)
        if u not in {'deg', 'degree', 'degrees'}:
            raise InvalidTelemetryError('Unsupported angular unit')
    return value


def _get_xml_root(
    source: Union[str, Path, bytes],
    *,
    lazy_load: bool = True,
) -> ET.Element:
    """Parse at most 1 MiB with depth/node limits; lazy_load is a legacy no-op."""
    if isinstance(source, bytes):
        data = source
    else:
        with Path(source).open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise InvalidTelemetryError("PDS4 XML exceeds 1 MiB")
    # UTF-16/32 and entity/DTD declarations are refused before the XML parser.
    # ElementTree does not follow schemaLocation or processing-instruction URLs.
    if b"\x00" in data or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise InvalidTelemetryError("XML DTDs/entities and non-UTF8 encodings are unsupported")
    root = ET.fromstring(data)
    if root.tag != "{http://pds.nasa.gov/pds4/pds/v1}Product_Observational":
        raise InvalidTelemetryError("Expected PDS4 Product_Observational namespace/root")
    stack = [(root, 1)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > 20000 or depth > 48:
            raise InvalidTelemetryError("XML node/depth quota exceeded")
        stack.extend((child, depth + 1) for child in node)
    return root


def _extract_fields(
    root: ET.Element,
) -> tuple[dict[str, tuple[str, Optional[str]]], list[float], list[float]]:
    """Walk every element and collect canonical fields and any corner coordinates."""
    found: dict[str, tuple[str, Optional[str]]] = {}
    corner_lats: list[float] = []
    corner_lons: list[float] = []

    for elem in root.iter():
        local = _strip_ns(elem.tag).lower()
        text = (elem.text or "").strip()
        if not text:
            continue

        # Corner coordinate tracking (ISRO / USGS PDS4 products)
        if "latitude" in local and any(c in local for c in ("upper", "lower", "corner")):
            val = _try_float(text)
            if val is not None:
                corner_lats.append(_apply_unit(val, elem.attrib.get("unit"), "lat_min"))
        elif "longitude" in local and any(c in local for c in ("upper", "lower", "corner")):
            val = _try_float(text)
            if val is not None:
                corner_lons.append(_apply_unit(val, elem.attrib.get("unit"), "lon_min"))

        # Check for Observing_System_Component instrument name
        if local == "type" and text.lower() == "instrument" and "instrument" not in found:
            # Check parent/sibling or previous text
            pass

        # Check for mission/instrument keywords
        if local == "name" and any(inst in text.lower() for inst in ("tmc", "ohrc", "iirs", "nac", "camera", "spectrometer")):
            if "instrument" not in found:
                found["instrument"] = (text.upper(), None)

        canonical = _TAG_MAP.get(local)
        if canonical is not None:
            entry = (text, elem.attrib.get("unit"))
            if canonical in found and found[canonical] != entry:
                if canonical not in {'product_id', 'instrument'}:
                    raise InvalidTelemetryError(f"Conflicting PDS4 telemetry field: {canonical}")
            else:
                found[canonical] = entry

    return found, corner_lats, corner_lons


# ── Main Parser ──────────────────────────────────────────────────────

def parse_metadata(
    source: Union[str, Path, bytes],
    *,
    lazy_load: bool = True,
) -> LunarTelemetryMetadata:
    """Parse a PDS4 XML label into standardised lunar telemetry.

    Walks the XML tree and matches known PDS4 tag names to canonical
    fields (GSD, incidence angle, emission angle, solar azimuth,
    bounding coordinates, product ID, instrument).

    Parameters
    ----------
    source : str | Path | bytes
        File path or raw XML content.
    lazy_load : bool
        When ``True`` (default), uses ``iterparse`` for forward-only
        streaming.

    Returns
    -------
    LunarTelemetryMetadata
        Validated optional telemetry fields; not a validated image association.

    Raises
    ------
    InvalidTelemetryError
        Required fields missing, unparseable, or violating constraints.
    FileNotFoundError
        *source* is a path that does not exist.
    """
    root = _get_xml_root(source, lazy_load=lazy_load)
    fields, corner_lats, corner_lons = _extract_fields(root)

    # ── Required numeric fields ──────────────────────────────────
    def _require_float(canonical: str, display_name: str, default: Optional[float] = None) -> float:
        entry = fields.get(canonical)
        if entry is None:
            if default is not None:
                return default
            return None
        if entry[0].strip().upper() in {"NULL", "UNK", "N/A", "UNKNOWN"}:
            return None
        val = _try_float(entry[0])
        if val is None:
            raise InvalidTelemetryError(
                f"Cannot parse {display_name} as float: '{entry[0]}'"
            )
        return _apply_unit(val, entry[1], canonical)

    gsd = _require_float("gsd", "pixel_resolution / GSD")
    inc = _require_float("incidence_angle", "incidence_angle")
    # Absence is unknown; never manufacture a nadir observation.
    emi = _require_float("emission_angle", "emission_angle")
    azi = _require_float("solar_azimuth", "solar_azimuth_angle")

    # ── Optional text fields ─────────────────────────────────────
    pid_entry = fields.get("product_id")
    product_id = pid_entry[0] if pid_entry else None

    inst_entry = fields.get("instrument")
    instrument = inst_entry[0] if inst_entry else None

    # ── Optional bounding box ────────────────────────────────────
    bbox: Optional[LunarBoundingBox] = None
    coord_keys = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if coord_keys.issubset(fields):
        bbox_vals: dict[str, float] = {}
        for k in coord_keys:
            val = _try_float(fields[k][0])
            if val is None:
                raise InvalidTelemetryError(
                    f"Cannot parse coordinate {k} as float: '{fields[k][0]}'"
                )
            bbox_vals[k] = _apply_unit(val, fields[k][1], k)
        try:
            bbox = LunarBoundingBox(**bbox_vals)
        except (ValidationError, InvalidTelemetryError) as exc:
            raise InvalidTelemetryError(
                f"Invalid bounding box: {exc}"
            ) from exc
    elif corner_lats and corner_lons:
        # Construct bounding box from upper/lower corner coordinates
        try:
            bbox = LunarBoundingBox(
                lat_min=min(corner_lats),
                lat_max=max(corner_lats),
                lon_min=min(corner_lons),
                lon_max=max(corner_lons),
            )
        except (ValidationError, InvalidTelemetryError) as exc:
            raise InvalidTelemetryError(
                f"Invalid bounding box from corner coordinates: {exc}"
            ) from exc

    # ── Assemble & validate final model ──────────────────────────
    try:
        return LunarTelemetryMetadata(
            product_id=product_id,
            instrument=instrument,
            gsd_meters=gsd,
            incidence_angle_deg=inc,
            emission_angle_deg=emi,
            solar_azimuth_deg=azi,
            bounding_box=bbox,
            pixel_resolution_m=gsd,
            source_format="PDS4",
        )
    except (ValidationError, InvalidTelemetryError) as exc:
        raise InvalidTelemetryError(f"Invalid telemetry: {exc}") from exc


# Public alias
parse_pds4_label = parse_metadata
