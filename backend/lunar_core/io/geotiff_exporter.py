"""Moon 2000 georeferencing and lossless registration artifact export.

Inputs must already be aligned to a north-up lunar longitude/latitude grid.
This module assigns that grid; it does not reproject or radiometrically calibrate
pixels. Arrays use Phase 6's (H, W) or (H, W, C) layout. Optional band scales,
offsets and units describe existing calibration without modifying stored DN.

EPSG:9001 is Earth's IGS97 geocentric CRS, NOT Moon 2000. The historical
IAU2000:30100 definition is embedded as WKT because current PROJ databases need
not resolve that identifier. GDAL may drop its authority on GeoTIFF round-trip;
validate the actual spheroid/units and preserve the identifier in metadata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from numbers import Real
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pyproj
from pyproj import CRS as ProjCRS
import rasterio
from rasterio.transform import Affine, array_bounds, from_origin

logger = logging.getLogger(__name__)

MOON_CRS_ID = "IAU2000:30100"
MOON_2000_WKT = (
    'GEOGCS["Moon 2000",DATUM["D_Moon_2000",'
    'SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
    'AXIS["Longitude",EAST],AXIS["Latitude",NORTH],'
    'AUTHORITY["IAU2000","30100"]]'
)
MOON_CRS = ProjCRS.from_wkt(MOON_2000_WKT)
CSV_COLUMNS = (
    "point_id", "src_x", "src_y", "ref_x", "ref_y", "lat", "lon",
    "residual_px", "confidence",
)


@dataclass(frozen=True)
class ExportArtifacts:
    geotiff: Path
    tiepoints_csv: Path
    dossier_json: Path


def _number(value: float, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if (not np.isfinite(number) or (minimum is not None and number < minimum)
            or (maximum is not None and number > maximum)):
        raise ValueError(f"{name} is non-finite or outside its valid range")
    return number


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json(value, *, indent=None) -> str:
    # Reject NaN/Infinity: they are not valid JSON and conceal broken telemetry.
    return json.dumps(value, default=_json_default, allow_nan=False,
                      ensure_ascii=False, indent=indent)


def _raster_grid(registered, west_lon, north_lat, pixel_size_deg):
    if not isinstance(registered, np.ndarray) or np.ma.isMaskedArray(registered):
        raise ValueError("registered must be an unmasked NumPy array")
    if registered.ndim not in (2, 3) or any(size == 0 for size in registered.shape):
        raise ValueError("registered must have non-empty (H, W) or (H, W, C) shape")
    if registered.dtype.name not in {"uint8", "uint16", "float32"}:
        raise ValueError("registered dtype must be uint8, uint16, or float32")
    west = _number(west_lon, "west_lon", minimum=-360, maximum=360)
    north = _number(north_lat, "north_lat", minimum=-90, maximum=90)
    size = _number(pixel_size_deg, "pixel_size_deg", minimum=0)
    if size == 0:
        raise ValueError("pixel_size_deg must be positive")
    height, width = registered.shape[:2]
    transform = from_origin(west, north, size, size)
    left, bottom, right, top = array_bounds(height, width, transform)
    if not (np.isfinite([left, bottom, right, top]).all()
            and bottom >= -90 and top > bottom and right > left
            and size * width <= 360):
        raise ValueError("grid must have non-empty lunar bounds and span <= 360 degrees")
    # Longitude remains continuous, including grids crossing 180 or 360 degrees.
    bands = registered[np.newaxis] if registered.ndim == 2 else np.moveaxis(registered, -1, 0)
    return bands, transform


def _band_values(values, count, name, default):
    if values is None:
        return (default,) * count
    if len(values) != count:
        raise ValueError(f"{name} must have one value per band")
    return tuple(_number(value, name) for value in values)


def export_geotiff(
    registered: np.ndarray,
    output_path: str | Path,
    *,
    west_lon: float,
    north_lat: float,
    pixel_size_deg: float,
    metadata: Mapping | None = None,
    nodata: float | None = None,
    scales: Sequence[float] | None = None,
    offsets: Sequence[float] | None = None,
    units: Sequence[str] | None = None,
) -> Path:
    """Write and reopen a lossless Moon 2000 TIFF; return its path.

    Origin is the upper-left pixel CORNER, resolution is in degrees, and y
    decreases southward. Zero is valid unless explicitly passed as nodata.
    Integer LZW uses predictor 2; float32 uses predictor 3. Existing output is
    replaced only after a staged file passes spatial/header verification.
    """
    bands, transform = _raster_grid(registered, west_lon, north_lat, pixel_size_deg)
    count, height, width = bands.shape
    band_scales = _band_values(scales, count, "scales", 1.0)
    band_offsets = _band_values(offsets, count, "offsets", 0.0)
    if units is not None and (len(units) != count or any(
            not isinstance(unit, str) or not unit.strip() for unit in units)):
        raise ValueError("units must contain one non-empty string per band")
    if nodata is not None:
        nodata = _number(nodata, "nodata")
        info = np.finfo(bands.dtype) if bands.dtype.kind == "f" else np.iinfo(bands.dtype)
        if not info.min <= nodata <= info.max or (
                bands.dtype.kind != "f" and not nodata.is_integer()):
            raise ValueError("nodata must be representable in the raster dtype")
        if float(np.asarray(nodata, dtype=bands.dtype)) != nodata:
            raise ValueError("nodata must be exactly representable in the raster dtype")
    tags = {
        "AREA_OR_POINT": "Area", "CRS_IDENTIFIER": MOON_CRS_ID,
        "LUNAR_RADIUS_M": "1737400", "COORDINATE_UNITS": "degree",
        "LONGITUDE_DIRECTION": "positive_east", "PIXEL_ORIGIN": "upper_left_corner",
        "METADATA_JSON": _json({} if metadata is None else metadata),
        "SOFTWARE": "Chandra-Align Phase 7",
    }
    predictor = 3 if bands.dtype.kind == "f" else 2
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".geotiff-", dir=path.parent) as staging:
        temporary = Path(staging) / "raster.tif"
        with rasterio.open(
            temporary, "w", driver="GTiff", height=height, width=width,
            count=count, dtype=bands.dtype.name, crs=MOON_2000_WKT,
            transform=transform, compress="LZW", predictor=predictor,
            nodata=nodata, BIGTIFF="IF_SAFER",
        ) as dataset:
            dataset.write(bands)
            dataset.scales = band_scales
            dataset.offsets = band_offsets
            if units is not None:
                for index, unit in enumerate(units, start=1):
                    dataset.set_band_unit(index, unit)
            dataset.update_tags(**tags)
        with rasterio.open(temporary) as dataset:
            _verify_header(dataset, bands, transform, predictor, tags)
            logger.info("GeoTIFF CRS confirmed: %s | %s", MOON_CRS_ID, dataset.crs.to_string())
            logger.info("GeoTIFF header: transform_gdal=%s bounds=%s resolution=%s tags=%s image=%s",
                        dataset.transform.to_gdal(), tuple(dataset.bounds), dataset.res,
                        dataset.tags(), dataset.tags(ns="IMAGE_STRUCTURE"))
        os.replace(temporary, path)
    return path


def _verify_header(dataset, bands, transform, predictor, tags):
    crs = ProjCRS.from_wkt(dataset.crs.to_wkt()) if dataset.crs else None
    bounds = tuple(dataset.bounds)
    image_tags = dataset.tags(ns="IMAGE_STRUCTURE")
    valid = (
        crs is not None and crs.equals(MOON_CRS, ignore_axis_order=True)
        and dataset.transform == transform
        and dataset.shape == bands.shape[1:] and dataset.count == bands.shape[0]
        and dataset.dtypes == (bands.dtype.name,) * bands.shape[0]
        and np.isfinite(bounds).all() and bounds[2] > bounds[0] and bounds[3] > bounds[1]
        and image_tags.get("COMPRESSION") == "LZW"
        and image_tags.get("PREDICTOR") == str(predictor)
        and all(dataset.tags().get(key) == value for key, value in tags.items())
    )
    if not valid:
        raise RuntimeError("GeoTIFF failed spatial/header verification after reopening")


def _tiepoint_rows(tie_points, residuals_px, confidences, transform, point_ids):
    ties = np.asarray(tie_points)
    if ties.ndim != 2 or ties.shape[1] != 4 or ties.dtype.kind not in "uif" or not np.isfinite(ties).all():
        raise ValueError("tie_points must be finite (N, 4): src_x, src_y, ref_x, ref_y")
    count = len(ties)
    residuals = np.asarray(residuals_px)
    scores = np.asarray(confidences)
    for values, name in ((residuals, "residuals_px"), (scores, "confidences")):
        if (values.shape != (count,) or values.dtype.kind not in "uif"
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError(f"{name} must have N finite non-negative values")
    if (scores > 1).any():
        raise ValueError("confidences must be in [0, 1]")
    if not isinstance(transform, Affine) or not np.isfinite(tuple(transform)).all():
        raise ValueError("transform must be a finite Affine in lunar degrees")
    if not (transform.a > 0 and transform.e < 0 and transform.b == transform.d == 0):
        raise ValueError("transform must describe a north-up lunar grid")
    ids = list(range(1, count + 1)) if point_ids is None else list(point_ids)
    if len(ids) != count or any(not isinstance(v, (str, int)) or isinstance(v, bool) for v in ids):
        raise ValueError("point_ids must have N string or integer identifiers")
    ids = [str(value) for value in ids]
    if any(not value for value in ids) or len(set(ids)) != count:
        raise ValueError("point_ids must be non-empty and unique")
    rows = []
    for identifier, tie, residual, score in zip(ids, ties, residuals, scores):
        src_x, src_y, ref_x, ref_y = map(float, tie)
        lon, lat = transform * (ref_x + 0.5, ref_y + 0.5)
        if not -90 <= lat <= 90:
            raise ValueError("tie-point reference latitude is outside [-90, 90]")
        rows.append((identifier, src_x, src_y, ref_x, ref_y, lat, lon,
                     float(residual), float(score)))
    return rows


def export_tiepoints_csv(
    output_path: str | Path, tie_points: np.ndarray, *, residuals_px: np.ndarray,
    confidences: np.ndarray, transform: Affine, point_ids: Sequence[str | int] | None = None,
) -> Path:
    """RFC-4180 CSV with CRLF records, escaped quotes, and a fixed nine-column header.

    Pixel coordinates are zero-based centers, consistent with Phase 4/6. Lat/lon
    come from REFERENCE centers. Residuals must be supplied after registration;
    raw source-reference displacement is not a registration error.
    """
    rows = _tiepoint_rows(tie_points, residuals_px, confidences, transform, point_ids)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, dialect="excel", lineterminator="\r\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    return path


def export_registration_bundle(
    registered: np.ndarray, output_dir: str | Path, *,
    west_lon: float, north_lat: float, pixel_size_deg: float,
    tie_points: np.ndarray, residuals_px: np.ndarray, confidences: np.ndarray,
    job_id: str, rmse_px: float, vsui: float, inlier_ratio: float,
    job_telemetry: Mapping | None = None,
    pipeline_timeline: Sequence[Mapping] = (),
    rmse_basis: str = "unspecified",
    point_ids: Sequence[str | int] | None = None,
    nodata: float | None = None, scales: Sequence[float] | None = None,
    offsets: Sequence[float] | None = None, units: Sequence[str] | None = None,
) -> ExportArtifacts:
    """Stage all three artifacts, then publish them to output_dir.

    Existing artifact names are refused to avoid mixing registration jobs.
    Caller timeline entries need stage (string) and duration_ms (non-negative).
    Supplied job RMSE may be held-out accuracy; tie-point RMSE is recorded
    separately. No metric is inferred from raw image displacement or invented.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    if not isinstance(rmse_basis, str) or not rmse_basis.strip():
        raise ValueError("rmse_basis must be a non-empty string")
    metrics = {
        "rmse_px": _number(rmse_px, "rmse_px", minimum=0),
        "rmse_basis": rmse_basis,
        "vsui": _number(vsui, "vsui", minimum=0, maximum=1),
        "inlier_ratio": _number(inlier_ratio, "inlier_ratio", minimum=0, maximum=1),
    }
    timeline = []
    for event in pipeline_timeline:
        item = dict(event)
        if not isinstance(item.get("stage"), str) or not item["stage"].strip():
            raise ValueError("timeline entries require a non-empty stage")
        item["duration_ms"] = _number(item.get("duration_ms"), "duration_ms", minimum=0)
        timeline.append(item)
    telemetry = {} if job_telemetry is None else job_telemetry
    _json({"telemetry": telemetry, "timeline": timeline})
    _, transform = _raster_grid(registered, west_lon, north_lat, pixel_size_deg)
    rows = _tiepoint_rows(tie_points, residuals_px, confidences, transform, point_ids)
    residuals = np.asarray(residuals_px, dtype=np.float64)
    # Scaled norm avoids overflow when squaring finite, large residuals.
    peak = float(residuals.max()) if len(residuals) else 0.0
    metrics["tiepoint_rmse_px"] = (
        peak * float(np.sqrt(np.mean((residuals / peak) ** 2))) if peak else
        (0.0 if len(residuals) else None)
    )
    directory = Path(output_dir)
    artifacts = ExportArtifacts(directory / "registered_output.tif",
                                directory / "tiepoints.csv",
                                directory / "registration_dossier.json")
    destinations = (artifacts.geotiff, artifacts.tiepoints_csv, artifacts.dossier_json)
    if any(path.exists() for path in destinations):
        raise FileExistsError("output_dir already contains registration artifacts; use a fresh job directory")
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".registration-", dir=directory) as staging:
        stage = Path(staging)
        start = perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        raster_path = export_geotiff(
            registered, stage / artifacts.geotiff.name, west_lon=west_lon,
            north_lat=north_lat, pixel_size_deg=pixel_size_deg,
            metadata={"job_id": job_id, "metrics": metrics, "job_telemetry": telemetry},
            nodata=nodata, scales=scales, offsets=offsets, units=units,
        )
        timeline.append({"stage": "export_geotiff", "started_at_utc": timestamp,
                         "duration_ms": (perf_counter() - start) * 1000})
        start = perf_counter()
        export_tiepoints_csv(stage / artifacts.tiepoints_csv.name, tie_points,
                             residuals_px=residuals_px, confidences=confidences,
                             transform=transform, point_ids=[row[0] for row in rows])
        timeline.append({"stage": "export_tiepoints_csv",
                         "duration_ms": (perf_counter() - start) * 1000})
        with rasterio.open(raster_path) as dataset:
            raster = {
                "width": dataset.width, "height": dataset.height, "bands": dataset.count,
                "dtype": dataset.dtypes[0], "crs_identifier": MOON_CRS_ID,
                "crs_wkt": dataset.crs.to_wkt(), "transform_gdal": dataset.transform.to_gdal(),
                "bounds": dict(zip(("west", "south", "east", "north"), dataset.bounds)),
                "resolution_deg": dataset.res, "nodata": dataset.nodata,
                "scales": dataset.scales, "offsets": dataset.offsets, "units": dataset.units,
                "compression": "LZW", "predictor": int(dataset.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"]),
            }
        dossier = {
            "schema_version": 1, "job_id": job_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_telemetry": telemetry, "metrics": metrics,
            "pipeline_timeline": timeline, "raster": raster,
            "tiepoints": {"count": len(rows), "columns": CSV_COLUMNS,
                          "coordinate_convention": "zero-based pixel centers; lat/lon from reference grid",
                          "residual_convention": "caller-supplied post-registration radial error in pixels"},
            "artifacts": {"geotiff": artifacts.geotiff.name,
                          "tiepoints_csv": artifacts.tiepoints_csv.name,
                          "dossier_json": artifacts.dossier_json.name},
            "software": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                         "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str,
                         "numpy": np.__version__},
        }
        (stage / artifacts.dossier_json.name).write_text(_json(dossier, indent=2) + "\n", encoding="utf-8")
        for destination in destinations:
            os.replace(stage / destination.name, destination)
    return artifacts
