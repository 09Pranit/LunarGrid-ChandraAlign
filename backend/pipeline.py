"""Phase 1–7 orchestration, preserving raw pixels and honest quality metrics."""
from __future__ import annotations

import base64
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__:
    from .job_models import JobParams
    from .lunar_core.io.geotiff_exporter import CSV_COLUMNS, export_registration_bundle
    from .lunar_core.matching import LightGlueMatcher, MatchResult, SIFTMatcher, route_pair
    from .lunar_core.preprocess.wallis import apply_wallis_filter
    from .lunar_core.registration.outlier_rejection import RegistrationRejected, filter_magsac_and_vsui
    from .lunar_core.registration.tps_warper import solve_tps_warp
    from .registration import spatial_coverage
else:
    from job_models import JobParams
    from lunar_core.io.geotiff_exporter import CSV_COLUMNS, export_registration_bundle
    from lunar_core.matching import LightGlueMatcher, MatchResult, SIFTMatcher, route_pair
    from lunar_core.preprocess.wallis import apply_wallis_filter
    from lunar_core.registration.outlier_rejection import RegistrationRejected, filter_magsac_and_vsui
    from lunar_core.registration.tps_warper import solve_tps_warp
    from registration import spatial_coverage

# A worker process owns a cached model. This also protects threaded local eager
# requests and FLANN's state; run multiple Celery processes for parallel jobs.
_inference_lock = Lock()


def validate_image(path: Path, max_pixels: int):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(path)
        with image:
            width, height = image.size
            if image.format not in {"JPEG", "PNG", "TIFF", "WEBP", "BMP"}:
                raise ValueError("Supported images: JPEG, PNG, TIFF, WebP, BMP")
            if min(width, height) < 32 or max(width, height) >= 32767 or width * height > max_pixels:
                raise ValueError(f"Image must be >=32 pixels per axis and <= {max_pixels} total pixels; tile larger rasters")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The uploaded image is invalid or exceeds the decoder limit") from exc


def gray_byte(image: np.ndarray) -> np.ndarray:
    if image is None or not image.size or image.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError("Image must decode to uint8, uint16, or float32 pixels")
    if not np.isfinite(image).all():
        raise ValueError("Non-finite raster pixels must be masked before registration")
    values = image.astype(np.float32)
    if values.ndim == 3:
        values = cv2.cvtColor(values, cv2.COLOR_BGRA2GRAY if values.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        low, high = np.percentile(values, [1, 99])
        values = np.clip((values - low) * (255 / (high - low)), 0, 255) if high > low else np.zeros_like(values)
    return np.rint(values).astype(np.uint8)


def preview_png(image: np.ndarray) -> str:
    display = image if image.dtype == np.uint8 else gray_byte(image)
    height, width = display.shape[:2]
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        display = cv2.resize(display, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", display)
    if not ok:
        raise ValueError("PNG preview encoding failed")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


@lru_cache(maxsize=2)
def matcher_for(engine: str):
    if engine == "sift_flann":
        return SIFTMatcher()
    import torch
    torch.set_num_threads(int(os.getenv("LUNARGRID_TORCH_THREADS", "4")))
    return LightGlueMatcher(device=os.getenv("LUNARGRID_DEVICE", "auto"), max_num_keypoints=1024)


def quality_status(rmse: float | None, vsui: float, *, accepted=True, validated=True) -> str:
    return ("complete" if accepted and validated and rmse is not None
            and np.isfinite(rmse) and rmse <= 0.50 and vsui >= 0.75 else "review_required")


def refine_matches(matches: MatchResult, source: np.ndarray, reference: np.ndarray) -> MatchResult:
    """Refine feature seeds locally; reject inconsistent bidirectional tracks.

    Pyramidal LK operates on Wallis-conditioned radiance and uses no transform or
    ground truth. Different-sized images retain raw feature coordinates because
    OpenCV LK requires equal pyramid sizes. Poor refinement triggers review.
    """
    if len(matches) == 0 or source.shape != reference.shape:
        return matches
    src = matches.pts_src[:, None].copy()
    initial = matches.pts_ref[:, None].copy()
    options = dict(winSize=(21, 21), maxLevel=2,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    refined, forward_ok, _ = cv2.calcOpticalFlowPyrLK(source, reference, src, initial, **options)
    returned, backward_ok, _ = cv2.calcOpticalFlowPyrLK(reference, source, refined, src.copy(), **options)
    ref = refined[:, 0]
    valid = (forward_ok[:, 0].astype(bool) & backward_ok[:, 0].astype(bool)
             & (np.linalg.norm(returned[:, 0] - src[:, 0], axis=1) <= 0.5)
             & (np.linalg.norm(ref - matches.pts_ref, axis=1) <= 3.5)
             & np.isfinite(ref).all(axis=1)
             & (ref[:, 0] >= 0) & (ref[:, 0] < reference.shape[1])
             & (ref[:, 1] >= 0) & (ref[:, 1] < reference.shape[0]))
    return MatchResult(matches.pts_src[valid], ref[valid], matches.confidences[valid],
                       matches.execution_time_ms, matches.engine_name)


def run_pipeline(directory: Path, raw_params: dict, progress) -> dict:
    started = perf_counter()
    params = JobParams.model_validate(raw_params)
    job_id = directory.name
    timeline = []

    def stage(percent, name):
        progress(percent, name)
        timeline.append({"stage": name, "elapsed_seconds": perf_counter() - started})

    stage(10, "ingestion")
    source = cv2.imread(str(directory / "source.img"), cv2.IMREAD_UNCHANGED)
    reference = cv2.imread(str(directory / "reference.img"), cv2.IMREAD_UNCHANGED)
    src_gray, ref_gray = gray_byte(source), gray_byte(reference)
    if params.reference_grid:
        grid = params.reference_grid
        if grid.north_lat - grid.pixel_size_deg * reference.shape[0] < -90 or grid.pixel_size_deg * reference.shape[1] > 360:
            raise ValueError("Reference grid exceeds lunar bounds")
    stage(20, "routing")
    if params.source_metadata is not None:
        decision = route_pair(src_gray, ref_gray, params.source_metadata, params.reference_metadata)
        routing = decision.model_dump(mode="json")
        selected = decision.selected_engine
    else:
        selected = "superpoint_lightglue"
        routing = {"selected_engine": selected, "rationales": ["Telemetry absent; conservatively use learned matching."]}
    if params.engine != "auto":
        selected = params.engine
        routing["explicit_engine_override"] = selected
    stage(30, "conditioning")
    src_conditioned, ref_conditioned = apply_wallis_filter(src_gray), apply_wallis_filter(ref_gray)
    stage(40, "matching")
    with _inference_lock:
        matches = matcher_for(selected).match(src_conditioned, ref_conditioned)
    raw_count = len(matches)
    matches = refine_matches(matches, src_conditioned, ref_conditioned)
    stage(60, "filtering")
    try:
        filtered = filter_magsac_and_vsui(matches.pts_src, matches.pts_ref, matches.confidences,
                                         ref_gray.shape, source_shape=src_gray.shape)
    except RegistrationRejected as exc:
        filtered = exc.result
    metrics = {
        "active_engine": matches.engine_name, "engine": matches.engine_name,
        "candidate_matches": raw_count, "refined_matches": len(matches), "accepted_matches": filtered.retained_count,
        "rejected_matches": raw_count - filtered.retained_count,
        "inlier_ratio": filtered.inlier_count / raw_count if raw_count else 0.0, "vsui_score": filtered.vsui,
        "spatial_coverage": spatial_coverage(filtered.pts_ref, ref_gray.shape[1], ref_gray.shape[0]),
        "rmse_px": None, "registration_method": "TPS (MAGSAC++ / Grid-NMS)",
        "runtime_seconds": perf_counter() - started,
        "validation_note": "Feature residuals are not independent lunar ground-control accuracy.",
    }
    result = {"job_id": job_id, "status": "review_required", "progress_percent": 100,
              "metrics": metrics, "artifacts": {}, "routing": routing,
              "image_dimensions": {"source": {"width": source.shape[1], "height": source.shape[0]},
                                   "reference": {"width": reference.shape[1], "height": reference.shape[0]}},
              "review_reasons": list(filtered.rejection_reasons)}
    if metrics["inlier_ratio"] < 0.70:
        result["review_reasons"].append("Geometric inliers / original feature matches is below 70% after refinement")
    if not filtered.accepted or metrics["inlier_ratio"] < 0.70:
        # Phase 5 explicitly forbids passing rejected correspondences into TPS.
        if filtered.homography is not None and filtered.inlier_count:
            pts = matches.pts_src[filtered.inlier_mask]
            projected = cv2.perspectiveTransform(pts[:, None], filtered.homography)[:, 0]
            metrics["rmse_px"] = float(np.sqrt(np.mean(np.sum((projected - matches.pts_ref[filtered.inlier_mask])**2, axis=1))))
        metrics["rmse_basis"] = "rejected_homography_consensus_diagnostic"
        return result

    stage(75, "tps_alignment")
    warp = solve_tps_warp(filtered.pts_src, filtered.pts_ref)
    validation = filtered.inlier_mask & ~filtered.retained_mask
    validated = int(validation.sum()) >= 8
    if validated:
        rmse = warp.rmse(matches.pts_src[validation], matches.pts_ref[validation])
    else:
        rmse = warp.fitting_rmse_px
        result["review_reasons"].append("Fewer than eight withheld correspondences; independent validation unavailable")
    metrics.update(rmse_px=rmse, fitting_rmse_px=warp.fitting_rmse_px,
                   validation_matches=int(validation.sum()),
                   rmse_basis="withheld_feature_correspondences" if validated else "fitting_only")
    result["status"] = quality_status(rmse, filtered.vsui, validated=validated)
    if rmse > 0.5:
        result["review_reasons"].append(f"RMSE {rmse:.6f} px exceeds 0.50 px")
    aligned = warp.warp(source, reference.shape[:2])
    residuals = np.linalg.norm(warp.forward(filtered.pts_src) - filtered.pts_ref, axis=1)
    ids = [f"LG-{i+1:04d}" for i in range(filtered.retained_count)]
    result["tie_points"] = [
        {"id": identifier, "source_x": float(src[0]), "source_y": float(src[1]),
         "reference_x": float(ref[0]), "reference_y": float(ref[1]),
         "confidence": float(score), "status": "accepted"}
        for identifier, src, ref, score in zip(ids, filtered.pts_src, filtered.pts_ref, filtered.confidences)
    ]
    stage(90, "export")
    output = directory / "artifacts"
    output.mkdir(exist_ok=True)
    if params.reference_grid:
        # OpenCV BGR -> standard TIFF RGB; original band values are preserved.
        raster = aligned
        if aligned.ndim == 3:
            raster = cv2.cvtColor(aligned, cv2.COLOR_BGRA2RGBA if aligned.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        export_registration_bundle(
            raster, output, **params.reference_grid.model_dump(), tie_points=filtered.tie_points,
            residuals_px=residuals, confidences=filtered.confidences, point_ids=ids,
            job_id=job_id, rmse_px=rmse, vsui=filtered.vsui, inlier_ratio=metrics["inlier_ratio"],
            rmse_basis=metrics["rmse_basis"], job_telemetry=params.model_dump(mode="json"),
            pipeline_timeline=[{"stage": a["stage"], "duration_ms": (b["elapsed_seconds"] - a["elapsed_seconds"]) * 1000}
                               for a, b in zip(timeline, timeline[1:])],
        )
    else:
        if not cv2.imwrite(str(output / "registered_output.tif"), aligned):
            raise ValueError("Registered TIFF encoding failed")
        with (output / "tiepoints.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            for identifier, tie, residual, score in zip(ids, filtered.tie_points, residuals, filtered.confidences):
                writer.writerow([identifier, *tie, "", "", residual, score])
        (output / "registration_dossier.json").write_text(json.dumps(
            {"job_id": job_id, "georeferenced": False, "metrics": metrics,
             "job_telemetry": params.model_dump(mode="json"), "pipeline_timeline": timeline}, allow_nan=False), encoding="utf-8")
    result["artifacts"] = {
        "registered_preview_base64": preview_png(aligned),
        "geotiff_download_url": f"/api/v1/artifacts/{job_id}.tif",
        "tie_points_csv_url": f"/api/v1/artifacts/{job_id}_tiepoints.csv",
        "dossier_url": f"/api/v1/artifacts/{job_id}_dossier.json",
        "georeferenced": params.reference_grid is not None,
    }
    result["source_preview"], result["reference_preview"] = preview_png(source), preview_png(reference)
    metrics["runtime_seconds"] = perf_counter() - started
    return result
