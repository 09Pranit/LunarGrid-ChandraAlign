"""Real Tycho registration through HTTP, Celery, Phases 1–7, and downloads.

Run: python backend/tests/test_phase8_e2e.py
The neural weights must be cached as documented in the README.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
from rasterio.io import MemoryFile

from job_models import JobResult, Settings
from job_store import JobStore
from main import create_app
from pipeline import quality_status
import tasks


@pytest.fixture(scope="module")
def lunar_pair():
    image = cv2.imread(str(Path(__file__).resolve().parents[2] / "public" / "lunar-tycho.jpg"), 0)
    assert image is not None
    # Known translation (-6,-4), independently cropped from the lunar photograph.
    # PNG preserves the original JPEG crop pixels without a second lossy encoding.
    source, reference = image[300:812, 300:812], image[304:816, 306:818]
    return source, reference


def files_for(pair):
    return {f"{name}_file": (f"{name}.png", cv2.imencode(".png", image)[1].tobytes(), "image/png")
            for name, image in zip(("source", "reference"), pair)}


def label(instrument, gsd, incidence, name):
    return f"""<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
      <instrument_name>{instrument}</instrument_name><pixel_resolution unit="m">{gsd}</pixel_resolution>
      <incidence_angle unit="deg">{incidence}</incidence_angle><emission_angle>2</emission_angle>
      <solar_azimuth_angle>65</solar_azimuth_angle><File_Area_Observational_Supplemental><File><file_name>{name}</file_name></File><Encoded_Image><offset unit="byte">0</offset><encoding_standard_id>PNG</encoding_standard_id></Encoded_Image></File_Area_Observational_Supplemental></Product_Observational>""".encode()


def poll(client, job_id, deadline):
    while time.perf_counter() < deadline:
        response = client.get(f"/api/v1/registration/jobs/{job_id}")
        assert response.status_code == 200
        status = response.json()
        assert 0 <= status["progress_percent"] <= 100
        if status["status"] in {"complete", "review_required", "failed"}:
            response = client.get(f"/api/v1/registration/jobs/{job_id}/results")
            assert response.status_code == 200
            return response.json()
        time.sleep(0.02)
    pytest.fail("Registration exceeded the 15 second end-to-end deadline")


@pytest.fixture
def settings(tmp_path):
    return Settings(storage_dir=tmp_path / "jobs", always_eager=True, eager_fallback=False)


def assert_completed(client, result, elapsed):
    JobResult.model_validate(result)
    assert result["status"] == "review_required", result # unknown lunar geometry remains review-required
    metrics, artifacts = result["metrics"], result["artifacts"]
    assert metrics["active_engine"] == "superpoint_lightglue"
    assert metrics["rmse_basis"] == "withheld_feature_correspondences"
    assert metrics["validation_matches"] >= 8
    assert 0 <= metrics["rmse_px"] < 0.50
    assert 0.75 <= metrics["vsui_score"] <= 1
    assert 0.7 <= metrics["inlier_ratio"] <= 1
    for key in ("accepted_matches", "candidate_matches", "spatial_coverage", "runtime_seconds"):
        assert np.isfinite(metrics[key])
    header, payload = artifacts["registered_preview_base64"].split(",", 1)
    assert header == "data:image/png;base64"
    png = base64.b64decode(payload, validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert cv2.imdecode(np.frombuffer(png, np.uint8), 0).shape == (512, 512)
    tiff = client.get(artifacts["geotiff_download_url"])
    assert tiff.status_code == 200 and tiff.headers["content-type"] == "image/tiff"
    assert tiff.content[:4] in (b"II*\x00", b"MM\x00*")
    csv_response = client.get(artifacts["tie_points_csv_url"])
    assert csv_response.status_code == 200
    ties = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert len(ties) == metrics["accepted_matches"]
    # Verify measurements against the known crop translation, independently of TPS.
    errors = [[float(p["ref_x"]) - float(p["src_x"]) + 6,
               float(p["ref_y"]) - float(p["src_y"]) + 4] for p in ties]
    truth_rmse = float(np.sqrt(np.mean(np.sum(np.square(errors), axis=1))))
    assert truth_rmse < 0.5
    assert elapsed < 15.0
    assert result["image_dimensions"]["reference"] == {"width": 512, "height": 512}
    print(f"\nPhase 8: RMSE={metrics['rmse_px']:.6f}px; known-transform RMSE={truth_rmse:.6f}px; "
          f"VSUI={metrics['vsui_score']:.6f}; inliers={metrics['inlier_ratio']:.4f}; HTTP E2E={elapsed:.3f}s")
    return tiff.content, ties


def test_multisensor_eager_e2e(settings, lunar_pair, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(Path(__file__).resolve().parents[2] / "work" / "torch"))
    uploads = files_for(lunar_pair)
    uploads.update(source_label=("ohrc.xml", label("OHRC", 0.25, 74, "source.png"), "application/xml"),
                   reference_label=("nac.xml", label("LROC NAC", 2.0, 66, "reference.png"), "application/xml"))
    params = {"reference_grid": {"west_lon": 22.0, "north_lat": 15.0, "pixel_size_deg": 0.001}}
    with TestClient(create_app(settings)) as client:
        started = time.perf_counter()
        response = client.post("/api/v1/registration/jobs", files=uploads, data={"params": json.dumps(params)})
        assert response.status_code == 202, response.text
        receipt = response.json()
        assert receipt == {"job_id": receipt["job_id"], "status": "queued", "progress_percent": 0}
        result = poll(client, receipt["job_id"], started + 15)
        tiff, ties = assert_completed(client, result, time.perf_counter() - started)
        with MemoryFile(tiff) as mem, mem.open() as dataset:
            assert dataset.shape == (512, 512)
            assert dataset.crs is None # legacy client grid assertions cannot georeference a file
        assert all(not p["lat"] and not p["lon"] for p in ties)
        assert result["routing"]["delta_g"] > 3
        dossier = client.get(result["artifacts"]["dossier_url"]).json()
        assert dossier["metadata"]["source"]["instrument"] == "OHRC"
    # Recreate API instance: job state/results survive, without a Redis backend.
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/registration/jobs/{receipt['job_id']}/results").json() == result


def test_actual_celery_worker_async_e2e(settings, lunar_pair, monkeypatch):
    from celery.contrib.testing.worker import start_worker
    monkeypatch.setenv("TORCH_HOME", str(Path(__file__).resolve().parents[2] / "work" / "torch"))
    asynchronous = replace(settings, always_eager=False, broker_url="memory://")
    queue = tasks.build_celery(asynchronous)
    with start_worker(queue, pool="solo", concurrency=1, perform_ping_check=False, shutdown_timeout=10):
        with TestClient(create_app(asynchronous, queue)) as client:
            started = time.perf_counter()
            response = client.post("/api/v1/registration/jobs", files=files_for(lunar_pair), data={"params":'{"engine":"superpoint_lightglue"}'})
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            pending = client.get(f"/api/v1/registration/jobs/{job_id}/results")
            assert pending.status_code == 202
            assert pending.json()["status"] in {"queued", "processing"}
            result = poll(client, job_id, started + 15)
            _, ties = assert_completed(client, result, time.perf_counter() - started)
            assert result["artifacts"]["georeferenced"] is False
            assert all(not p["lat"] and not p["lon"] for p in ties)


@pytest.mark.parametrize("rmse,vsui,expected", [(0.5, .75, "complete"), (.50001, 1, "review_required"),
    (.1, .7499, "review_required"), (None, 1, "review_required"), (float("nan"), 1, "review_required")])
def test_quality_threshold_boundaries(rmse, vsui, expected):
    assert quality_status(rmse, vsui) == expected


def test_offline_fallback_and_production_fail_closed(settings, lunar_pair, monkeypatch):
    from kombu.exceptions import OperationalError
    monkeypatch.setattr(tasks, "redis_available", lambda url: False)
    offline = replace(settings, always_eager=False, eager_fallback=True)
    queue = tasks.build_celery(offline)
    with TestClient(create_app(offline, queue)) as client:
        assert client.get("/health").json()["execution_mode"] == "eager"
        assert queue.conf.task_always_eager is True
    strict = replace(settings, always_eager=False, eager_fallback=False)
    queue = tasks.build_celery(strict)
    def unavailable(*args, **kwargs):
        raise OperationalError("offline")
    monkeypatch.setattr(queue.tasks[tasks.TASK_NAME], "apply_async", unavailable)
    with TestClient(create_app(strict, queue)) as client:
        response = client.post("/api/v1/registration/jobs", files=files_for(lunar_pair))
        assert response.status_code == 503
        job_id = response.json()["detail"]["job_id"]
        assert client.get(f"/api/v1/registration/jobs/{job_id}").json()["status"] == "failed"


def test_rejected_geometry_is_review_without_tps(settings, monkeypatch):
    import pipeline
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected Phase 5 matches must never reach TPS")
    monkeypatch.setattr(pipeline, "solve_tps_warp", forbidden)
    pair = [np.full((64, 64), 128, np.uint8)] * 2
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/registration/jobs", files=files_for(pair),
                               data={"params": '{"engine":"sift_flann"}'})
        result = poll(client, response.json()["job_id"], time.perf_counter() + 5)
        assert result["status"] == "review_required"
        assert result["review_reasons"] and not result["artifacts"].get("geotiff_download_url")
        assert client.get(result["artifacts"]["dossier_url"]).status_code == 200


@pytest.mark.parametrize("params", ['{', '{"engine":"fake"}', '{"reference_grid":{"west_lon":NaN}}',
                                      '{"unknown":true}', '{"source_metadata":{"gsd_meters":0}}'])
def test_bad_options_fail_before_queuing(settings, lunar_pair, params):
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/registration/jobs", files=files_for(lunar_pair),
                           data={"params": params}).status_code == 422


def test_missing_corrupt_large_uploads_and_unknown_jobs(settings, lunar_pair):
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/v1/registration/jobs").status_code == 422
        corrupt = files_for(lunar_pair)
        corrupt["source_file"] = ("../../evil.png", b"not an image", "image/png")
        assert client.post("/api/v1/registration/jobs", files=corrupt).status_code == 422
        assert not list(settings.storage_dir.glob("job_lunar_*"))
        for path in ("jobs/job_lunar_" + "0" * 32, "jobs/invalid/results"):
            assert client.get("/api/v1/registration/" + path).status_code == 404
        for name in ("jobs.sqlite3", "job_lunar_" + "0" * 32 + ".tif", "%2e%2e%2fjobs.sqlite3"):
            assert client.get("/api/v1/artifacts/" + name).status_code == 404
    with TestClient(create_app(replace(settings, max_upload_bytes=50))) as client:
        assert client.post("/api/v1/registration/jobs", files=files_for(lunar_pair)).status_code == 413
        assert client.post("/api/v1/registration/jobs", content=b"x", headers={"Content-Length": "9000000"}).status_code == 413


def test_duplicate_delivery_claim_and_worker_failure(settings, monkeypatch):
    import pipeline
    store = JobStore(settings.storage_dir)
    job_id = "job_lunar_" + uuid4().hex
    store.create(job_id, {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: store.claim(job_id), range(16))) == 1
    failed_id = "job_lunar_" + uuid4().hex
    store.create(failed_id, {})
    def broken(*args):
        raise RuntimeError("internal path /secret must not leak")
    monkeypatch.setattr(pipeline, "run_pipeline", broken)
    tasks.process_registration(failed_id, str(settings.storage_dir))
    tasks.process_registration(failed_id, str(settings.storage_dir))
    record = store.get(failed_id)
    assert record["status"] == "failed" and "/secret" not in record["error"]


def test_chunked_request_limit_closes_partial_multipart_files(settings):
    # No Content-Length: reject while reading, including after spool-to-disk.
    def chunks():
        yield b'--boundary\r\nContent-Disposition: form-data; name="source_file"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
        for _ in range(9):
            yield b"x" * 1024**2
        yield b'\r\n--boundary--\r\n'
    with TestClient(create_app(replace(settings, max_upload_bytes=100))) as client:
        response = client.post("/api/v1/registration/jobs", content=chunks(),
                               headers={"Content-Type": "multipart/form-data; boundary=boundary"})
        assert response.status_code == 413


def test_stalled_jobs_expire_and_late_results_cannot_overwrite(settings):
    from contextlib import closing
    store = JobStore(settings.storage_dir)
    job_id = "job_lunar_" + uuid4().hex
    store.create(job_id, {})
    with closing(store.connect()) as db, db:
        db.execute("UPDATE jobs SET updated=0 WHERE job_id=?", (job_id,))
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/api/v1/registration/jobs/{job_id}").json()["status"] == "failed"
        store.finish(job_id, "complete", {"status": "complete"})
        assert client.get(f"/api/v1/registration/jobs/{job_id}/results").json()["status"] == "failed"


def test_broker_failure_after_startup_uses_local_fallback(settings, monkeypatch):
    from kombu.exceptions import OperationalError
    fallback = replace(settings, always_eager=False, eager_fallback=True)
    queue = tasks.build_celery(fallback)
    task = queue.tasks[tasks.TASK_NAME]
    def unavailable(*args, **kwargs):
        raise OperationalError("connection lost during publish")
    monkeypatch.setattr(task, "apply_async", unavailable)
    calls = []
    monkeypatch.setattr(task, "apply", lambda **kwargs: calls.append(kwargs))
    job_id = "job_lunar_" + uuid4().hex
    tasks.enqueue(queue, fallback, job_id)
    assert calls == [{"args": (job_id, str(settings.storage_dir.resolve())), "task_id": job_id}]
    queue.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", "-s", "-W", "error", *sys.argv[1:]]))
