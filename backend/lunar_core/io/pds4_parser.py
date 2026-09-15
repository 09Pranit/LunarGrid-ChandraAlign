"""PDS4 XML label parser with orbital geometry computations.

Extracts telemetry from NASA/ISRO Planetary Data System v4 labels
and maps them to strongly-typed Pydantic v2 models.

Supported missions
──────────────────
    Chandrayaan-2 : OHRC (0.25 m), TMC-2 (5 m), IIRS (80 m)
    LRO           : LROC NAC

Design decisions
────────────────
    • Namespace-agnostic tag matching via local-name lookup.
    • ``lazy_load=True`` uses ``iterparse`` for forward-only streaming.
    • All ValidationErrors from Pydantic are caught and re-raised as
      ``InvalidTelemetryError`` so callers never see framework internals.
    • 32-bit float precision is enforced at the raster layer (future phases);
      metadata floats remain 64-bit for coordinate precision.

References
──────────
    PDS4 Information Model : https://pds.nasa.gov/datastandards/
    IAU 2015 Lunar Radius  : 1 737 400 m
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

    model_config = {"frozen": True}

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
        if self.lon_min > self.lon_max:
            raise InvalidTelemetryError(
                f"lon_min ({self.lon_min}°) must be ≤ lon_max ({self.lon_max}°)"
            )
        return self

    def spherical_area(self, radius: float = MOON_RADIUS_M) -> float:
        """Area of this spherical rectangle in m².

        .. math::
            A = R^2 \\cdot |\\sin \\varphi_2 - \\sin \\varphi_1| \\cdot |\\lambda_2 - \\lambda_1|

        where φ = latitude (rad), λ = longitude (rad).
        """
        lat1 = math.radians(self.lat_min)
        lat2 = math.radians(self.lat_max)
        dlon = math.radians(abs(self.lon_max - self.lon_min))
        return radius * radius * abs(math.sin(lat2) - math.sin(lat1)) * dlon


class LunarTelemetryMetadata(BaseModel):
    """Standardised telemetry output from PDS4 or GeoTIFF ingestion.

    All angles in degrees, all distances in metres.
    """

    model_config = {"frozen": True}

    product_id: Optional[str] = Field(
        None, description="PDS4 logical identifier or product ID"
    )
    instrument: Optional[str] = Field(
        None, description="Instrument name (OHRC, TMC-2, IIRS, NAC)"
    )
    gsd_meters: float = Field(..., gt=0.0, description="Ground Sample Distance (m)")
    incidence_angle_deg: float = Field(
        ..., ge=0.0, le=90.0, description="Solar incidence angle (°)"
    )
    emission_angle_deg: float = Field(
        ..., ge=0.0, le=90.0, description="Sensor emission angle (°)"
    )
    solar_azimuth_deg: float = Field(
        ..., ge=0.0, lt=360.0, description="Solar azimuth angle (°)"
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
    lon_lo = max(bbox_src.lon_min, bbox_ref.lon_min)
    lon_hi = min(bbox_src.lon_max, bbox_ref.lon_max)

    if lat_lo >= lat_hi or lon_lo >= lon_hi:
        return 0.0

    inter = LunarBoundingBox(
        lat_min=lat_lo, lat_max=lat_hi,
        lon_min=lon_lo, lon_max=lon_hi,
    )
    a_i = inter.spherical_area(radius)
    a_s = bbox_src.spherical_area(radius)
    a_r = bbox_ref.spherical_area(radius)
    denom = min(a_s, a_r)

    if denom <= 0.0:
        return 0.0
    return float(np.clip(a_i / denom, 0.0, 1.0))


# ── XML Tag Mapping ──────────────────────────────────────────────────
# Maps PDS4 local-name (lower-case) → canonical field name used
# internally.  First match per canonical field wins.

_TAG_MAP: dict[str, str] = {
    # Ground Sample Distance
    "pixel_resolution":          "gsd",
    "ground_sampling_distance":  "gsd",
    "map_resolution":            "gsd",
    "map_scale":                 "gsd",
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
    "data_set_id":               "product_id",
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
    u = unit.lower()
    if field == "gsd":
        if "km" in u:
            return value * 1_000.0
        if "cm" in u:
            return value * 0.01
        if "mm" in u:
            return value * 0.001
    elif field in (
        "incidence_angle", "emission_angle", "solar_azimuth",
        "lat_min", "lat_max", "lon_min", "lon_max",
    ):
        if "rad" in u:
            return math.degrees(value)
    return value


def _get_xml_root(
    source: Union[str, Path, bytes],
    *,
    lazy_load: bool = True,
) -> ET.Element:
    """Obtain XML root element from path or raw bytes.

    When *lazy_load* is ``True`` and *source* is a path, ``iterparse``
    is used to stream through the document without holding every
    intermediate node in memory.  For label files (typically < 1 MB)
    the difference is negligible, but the pattern scales to multi-GB
    combined labels in future phases.
    """
    if isinstance(source, bytes):
        return ET.fromstring(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"PDS4 label not found: {path}")

    if lazy_load:
        root: Optional[ET.Element] = None
        for _event, elem in ET.iterparse(str(path), events=("end",)):
            root = elem
        if root is None:
            raise InvalidTelemetryError("Empty XML document")
        return root

    return ET.parse(str(path)).getroot()


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
        if canonical is not None and canonical not in found:
            found[canonical] = (text, elem.attrib.get("unit"))

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
        Fully validated telemetry record.

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
            raise InvalidTelemetryError(f"Missing required field: {display_name}")
        val = _try_float(entry[0])
        if val is None:
            raise InvalidTelemetryError(
                f"Cannot parse {display_name} as float: '{entry[0]}'"
            )
        return _apply_unit(val, entry[1], canonical)

    gsd = _require_float("gsd", "pixel_resolution / GSD")
    inc = _require_float("incidence_angle", "incidence_angle")
    # Emission angle defaults to 0.0 (nadir) if not explicitly present in label
    emi = _require_float("emission_angle", "emission_angle", default=0.0)
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
