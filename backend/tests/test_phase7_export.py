"""Self-contained Phase 7 acceptance: python backend/tests/test_phase7_export.py.

The prompt's EPSG:9001 equality is intentionally corrected: that identifier is
Earth's IGS97 geocentric CRS. Assert the reopened Moon 2000 spheroid, angular
units, and WKT equivalence instead; never relabel lunar degrees as Earth metres.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
import sys

import numpy as np
from pyproj import CRS
import pytest
import rasterio
from rasterio.transform import from_origin

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.io.geotiff_exporter import (
    CSV_COLUMNS, MOON_2000_WKT, MOON_CRS_ID, export_geotiff,
    export_registration_bundle, export_tiepoints_csv,
)

from lunar_core.io.metadata import LunarGrid

logger = logging.getLogger(__name__)
GRID = {"west_lon": 22.0, "north_lat": 15.0, "pixel_size_deg": 0.001}
# Explicit synthetic reference fixture; API callers cannot supply this internal record.
GRID["validated_grid"] = LunarGrid(**GRID)


def sample_bundle(tmp_path, **overrides):
    args = dict(
        **GRID, tie_points=np.array([[12, 15, 10, 10], [53, 46, 50, 40]], dtype=float),
        residuals_px=np.array([0.3, 0.4]), confidences=np.array([0.9, 0.8]),
        job_id="phase7-test", rmse_px=0.125, rmse_basis="held_out_checkpoints",
        vsui=0.85, inlier_ratio=0.9,
        job_telemetry={"source": {"product_id": "OHRC-test", "gsd_meters": 0.25},
                       "reference": {"product_id": "NAC-test", "gsd_meters": 0.5}},
        pipeline_timeline=[{"stage": "matching", "duration_ms": 12.5},
                           {"stage": "tps_warp", "duration_ms": 3.25}],
    )
    args.update(overrides)
    return export_registration_bundle(
        np.random.default_rng(7).integers(0, 256, (512, 512), dtype=np.uint8),
        tmp_path, **args,
    )


def test_export_requires_validated_reference_grid(tmp_path):
    transform = {k: v for k, v in GRID.items() if k != 'validated_grid'}
    with pytest.raises(ValueError, match='validated'):
        export_geotiff(np.ones((8, 8), np.uint8), tmp_path / 'invented.tif', **transform)
    with pytest.raises(ValueError, match='match'):
        export_geotiff(np.ones((8, 8), np.uint8), tmp_path / 'conflicting.tif', **{**GRID, 'west_lon': 30})
    assert not list(tmp_path.glob('*.tif'))


def assert_lunar_crs(crs):
    assert crs is not None
    actual = CRS.from_wkt(crs.to_wkt())
    assert actual.is_geographic
    assert actual.ellipsoid.semi_major_metre == 1_737_400
    assert actual.ellipsoid.semi_minor_metre == 1_737_400
    assert actual.ellipsoid.inverse_flattening == 0
    assert all(axis.unit_name == "degree" for axis in actual.axis_info)
    assert actual.prime_meridian.longitude == 0
    assert actual.equals(CRS.from_wkt(MOON_2000_WKT), ignore_axis_order=True)
    assert crs.to_string() != "EPSG:9001"
    assert not actual.equals(CRS.from_epsg(9001), ignore_axis_order=True)


def test_required_512_uint8_roundtrip_and_spatial_header(tmp_path):
    data = np.random.default_rng(7).integers(0, 256, (512, 512), dtype=np.uint8)
    original = data.copy()
    path = export_geotiff(data, tmp_path / "registered_output.tif", **GRID,
                          metadata={"job_id": "phase7", "rmse_px": 0.125})
    with rasterio.open(path) as dataset:
        assert_lunar_crs(dataset.crs)
        assert dataset.transform == from_origin(22, 15, 0.001, 0.001)
        np.testing.assert_allclose(dataset.res, (0.001, 0.001), atol=1e-6, rtol=0)
        np.testing.assert_allclose(dataset.bounds, (22, 14.488, 22.512, 15), atol=1e-6, rtol=0)
        assert dataset.bounds.right > dataset.bounds.left
        assert dataset.bounds.top > dataset.bounds.bottom
        assert dataset.shape == (512, 512) and dataset.count == 1
        assert dataset.dtypes == ("uint8",)
        np.testing.assert_array_equal(dataset.read(1), data)
        assert dataset.nodata is None
        assert dataset.read_masks(1).min() == 255  # zero is a valid DN
        tags = dataset.tags()
        assert tags["CRS_IDENTIFIER"] == MOON_CRS_ID
        assert tags["LUNAR_RADIUS_M"] == "1737400"
        assert tags["AREA_OR_POINT"] == "Area"
        assert tags["COORDINATE_UNITS"] == "degree"
        assert json.loads(tags["METADATA_JSON"])["rmse_px"] == 0.125
        assert dataset.tags(ns="IMAGE_STRUCTURE")["COMPRESSION"] == "LZW"
        assert dataset.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == "2"
        logger.info("VERIFIED 512x512 uint8: CRS=%s | readback=%s", MOON_CRS_ID, dataset.crs.to_string())
        logger.info("VERIFIED bounds=%s res=%s header tags=%s", dataset.bounds, dataset.res, tags)
    np.testing.assert_array_equal(data, original)


def test_epsg_9001_is_earth_not_moon():
    incorrect = CRS.from_epsg(9001)
    assert incorrect.name == "IGS97" and incorrect.is_geocentric
    assert incorrect.ellipsoid.semi_major_metre == 6_378_137
    logger.info("CRS correction verified: EPSG:9001 = %s (%s)", incorrect.name, incorrect.type_name)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_dynamic_range_channels_and_calibration_preserved(tmp_path, dtype, channels):
    shape = (29, 37) if channels is None else (29, 37, channels)
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if dtype == np.uint16:
        data.flat[:4] = [0, 256, 32768, 65535]
    elif dtype == np.float32:
        data *= np.float32(0.12345)
        data.flat[:8] = [-123456.25, 1e-20, np.finfo(np.float32).max,
                        np.finfo(np.float32).min, -0.0, np.nan, np.inf, -np.inf]
    before = data.tobytes()
    count = 1 if channels is None else channels
    scales, offsets, units = (0.125,) * count, (-2.5,) * count, ("radiance",) * count
    path = export_geotiff(data, tmp_path / "range.tif", **GRID,
                          scales=scales, offsets=offsets, units=units)
    with rasterio.open(path) as dataset:
        actual = dataset.read(1) if channels is None else np.moveaxis(dataset.read(), 0, -1)
        assert actual.dtype == data.dtype and actual.shape == data.shape
        assert actual.tobytes() == before
        assert dataset.scales == scales and dataset.offsets == offsets and dataset.units == units
        assert dataset.tags(ns="IMAGE_STRUCTURE")["PREDICTOR"] == ("3" if dtype == np.float32 else "2")
    assert data.tobytes() == before


def test_nodata_and_non_contiguous_input(tmp_path):
    data = np.arange(200, dtype=np.uint16).reshape(10, 20)[:, ::2]
    assert not data.flags.c_contiguous
    path = export_geotiff(data, tmp_path / "nodata.tif", **GRID, nodata=0)
    with rasterio.open(path) as dataset:
        assert dataset.nodata == 0
        assert dataset.read_masks(1)[0, 0] == 0
        assert dataset.read_masks(1)[0, 1] == 255
        np.testing.assert_array_equal(dataset.read(1), data)


def test_csv_rfc4180_exact_header_crlf_quoting_and_coordinates(tmp_path):
    identifiers = ['tie,"A"', "tie\r\nB"]
    artifacts = sample_bundle(tmp_path, point_ids=identifiers)
    raw = artifacts.tiepoints_csv.read_bytes()
    assert raw.startswith(b"point_id,src_x,src_y,ref_x,ref_y,lat,lon,residual_px,confidence\r\n")
    assert raw.endswith(b"\r\n") and b"\r\r\n" not in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert b'"tie,""A"""' in raw and b'"tie\r\nB"' in raw
    with artifacts.tiepoints_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert all(len(row) == 9 for row in rows)
    assert [row[0] for row in rows[1:]] == identifiers
    np.testing.assert_allclose(list(map(float, rows[1][1:])),
                               [12, 15, 10, 10, 14.9895, 22.0105, 0.3, 0.9], rtol=0, atol=1e-12)
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\r\n").writerows(rows)
    assert buffer.getvalue().encode("utf-8") == raw
    logger.info("VERIFIED RFC-4180: 9 columns, CRLF, comma/quote/newline escaping, reference pixel centers")


def test_dossier_complete_and_consistent_with_geotiff(tmp_path):
    artifacts = sample_bundle(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "registered_output.tif", "tiepoints.csv", "registration_dossier.json"}
    def reject_non_json(value):
        pytest.fail(f"Non-JSON constant {value}")
    report = json.loads(artifacts.dossier_json.read_text(encoding="utf-8"), parse_constant=reject_non_json)
    assert report["job_id"] == "phase7-test" and report["schema_version"] == 1
    assert report["created_at_utc"].endswith("+00:00")
    assert report["job_telemetry"]["source"]["gsd_meters"] == 0.25
    assert report["metrics"]["rmse_px"] == 0.125
    assert report["metrics"]["rmse_basis"] == "held_out_checkpoints"
    assert report["metrics"]["vsui"] == 0.85
    assert report["metrics"]["inlier_ratio"] == 0.9
    assert report["metrics"]["tiepoint_rmse_px"] == pytest.approx(np.sqrt(0.125))
    assert report["tiepoints"]["count"] == 2
    assert report["tiepoints"]["columns"] == list(CSV_COLUMNS)
    assert [event["stage"] for event in report["pipeline_timeline"]] == [
        "matching", "tps_warp", "export_geotiff", "export_tiepoints_csv"]
    assert all(event["duration_ms"] >= 0 for event in report["pipeline_timeline"])
    with rasterio.open(artifacts.geotiff) as dataset:
        assert report["raster"]["transform_gdal"] == list(dataset.transform.to_gdal())
        assert list(report["raster"]["bounds"].values()) == list(dataset.bounds)
        assert report["raster"]["crs_wkt"] == dataset.crs.to_wkt()
        assert report["raster"]["dtype"] == dataset.dtypes[0]
        assert report["raster"]["width"] == report["raster"]["height"] == 512
        embedded = json.loads(dataset.tags()["METADATA_JSON"])
        assert embedded["metrics"] == report["metrics"]
        assert embedded["job_telemetry"] == report["job_telemetry"]
    logger.info("VERIFIED dossier: metrics=%s timeline=%s", report["metrics"], report["pipeline_timeline"])


def test_phase6_uint16_output_exports_without_conversion(tmp_path):
    from lunar_core.registration import solve_tps_warp
    x, y = np.meshgrid(np.linspace(4, 25, 4), np.linspace(4, 25, 4))
    source_points = np.column_stack((x.ravel(), y.ravel()))
    reference_points = source_points + (2, 1)
    warp = solve_tps_warp(source_points, reference_points)
    source = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40) * 50
    aligned = warp.warp(source, (32, 40))
    residuals = np.linalg.norm(warp.forward(source_points) - reference_points, axis=1)
    artifacts = export_registration_bundle(
        aligned, tmp_path, **GRID, tie_points=np.column_stack((source_points, reference_points)),
        residuals_px=residuals, confidences=np.ones(16), job_id="phase6-to-phase7",
        rmse_px=warp.fitting_rmse_px, rmse_basis="training", vsui=0.8, inlier_ratio=1.0,
    )
    with rasterio.open(artifacts.geotiff) as dataset:
        np.testing.assert_array_equal(dataset.read(1), aligned)
        assert dataset.dtypes == ("uint16",)


def test_empty_tiepoints_produce_header_and_null_fit_rmse(tmp_path):
    artifacts = sample_bundle(tmp_path, tie_points=np.empty((0, 4)),
                              residuals_px=np.empty(0), confidences=np.empty(0))
    assert artifacts.tiepoints_csv.read_bytes() == (",".join(CSV_COLUMNS) + "\r\n").encode()
    report = json.loads(artifacts.dossier_json.read_text())
    assert report["metrics"]["tiepoint_rmse_px"] is None


def test_continuous_antimeridian_and_zero_origin_grid(tmp_path):
    for west, north in [(179.9, 1), (359.9, 1), (0, 0)]:
        path = export_geotiff(np.zeros((2, 4), dtype=np.uint8), tmp_path / f"{west}.tif",
                              west_lon=west, north_lat=north, pixel_size_deg=0.1, validated_grid=LunarGrid(west_lon=west,north_lat=north,pixel_size_deg=.1))
        with rasterio.open(path) as dataset:
            assert dataset.bounds.right == pytest.approx(west + 0.4)
            assert dataset.bounds.bottom == pytest.approx(north - 0.2)


@pytest.mark.parametrize("overrides", [
    {"pixel_size_deg": 0}, {"pixel_size_deg": -0.1}, {"pixel_size_deg": np.nan},
    {"pixel_size_deg": 100}, {"north_lat": 91}, {"north_lat": -89.99},
    {"west_lon": np.inf}, {"west_lon": True}, {"west_lon": "20"},
    {"nodata": -1}, {"nodata": 0.5}, {"nodata": 256},
    {"scales": [1, 2]}, {"offsets": [np.nan]}, {"units": [""]},
    {"metadata": {"rmse": np.nan}},
])
def test_invalid_export_inputs_do_not_create_output(tmp_path, overrides):
    args = dict(GRID)
    args.update(overrides)
    path = tmp_path / "bad.tif"
    with pytest.raises((ValueError, TypeError)):
        export_geotiff(np.zeros((32, 32), dtype=np.uint8), path, **args)
    assert not path.exists()


@pytest.mark.parametrize("data", [np.zeros((2, 3), dtype=np.float64),
                                  np.zeros((2, 3), dtype=np.int16),
                                  np.zeros((0, 3), dtype=np.uint8),
                                  np.zeros((2, 3, 0), dtype=np.uint8),
                                  np.zeros((2, 3, 1, 1), dtype=np.uint8),
                                  np.ma.array([[1]], mask=[[True]], dtype=np.uint8)])
def test_unsupported_rasters_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        export_geotiff(data, tmp_path / "bad.tif", **GRID)


@pytest.mark.parametrize("overrides", [
    {"rmse_px": -1}, {"vsui": 1.1}, {"inlier_ratio": np.nan}, {"job_id": ""},
    {"confidences": np.array([1.1, 0.5])}, {"residuals_px": np.array([-1, 0.2])},
    {"residuals_px": np.array([0.2])}, {"tie_points": np.full((2, 4), np.inf)},
    {"point_ids": ["same", "same"]}, {"job_telemetry": {"angle": np.inf}},
    {"pipeline_timeline": [{"stage": "warp", "duration_ms": -1}]},
])
def test_bad_bundle_leaves_no_published_artifacts(tmp_path, overrides):
    with pytest.raises((ValueError, TypeError)):
        sample_bundle(tmp_path, **overrides)
    assert not list(tmp_path.iterdir())


def test_existing_bundle_is_preserved(tmp_path):
    artifacts = sample_bundle(tmp_path)
    before = {path: path.read_bytes() for path in (
        artifacts.geotiff, artifacts.tiepoints_csv, artifacts.dossier_json)}
    with pytest.raises(FileExistsError):
        sample_bundle(tmp_path, job_id="different-job")
    assert all(path.read_bytes() == data for path, data in before.items())


def test_failed_readback_verification_preserves_existing_tiff(tmp_path, monkeypatch):
    import lunar_core.io.geotiff_exporter as exporter
    path = export_geotiff(np.ones((8, 8), dtype=np.uint8), tmp_path / "output.tif", **GRID)
    before = path.read_bytes()
    def verification_failure(*args):
        raise RuntimeError("simulated readback failure")
    monkeypatch.setattr(exporter, "_verify_header", verification_failure)
    with pytest.raises(RuntimeError, match="readback failure"):
        export_geotiff(np.zeros((8, 8), dtype=np.uint8), path, **GRID)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "--log-cli-level=INFO", *sys.argv[1:],
    ]))
