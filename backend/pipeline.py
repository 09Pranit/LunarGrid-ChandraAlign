"""Phase 1–7 orchestration, preserving raw pixels and honest quality metrics."""
from __future__ import annotations

import base64
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__:
    from .job_models import ValidatedJobParams
    from .lunar_core.io.ingestion import read_tile, stretch
    from .lunar_core.io.window_exporter import write_bundle, write_dossier
    from .lunar_core.matching import LightGlueMatcher, MatchResult, SIFTMatcher, route_pair
    from .lunar_core.preprocess.wallis import apply_wallis_filter
    from .lunar_core.registration.outlier_rejection import RegistrationRejected, filter_magsac_and_vsui
    from .lunar_core.registration.tps_warper import solve_tps_warp
    from .registration import spatial_coverage
else:
    from job_models import ValidatedJobParams
    from lunar_core.io.ingestion import read_tile, stretch
    from lunar_core.io.window_exporter import write_bundle, write_dossier
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
    params = ValidatedJobParams.model_validate(raw_params)
    job_id = directory.name
    timeline = []

    def stage(percent, name):
        if perf_counter() - started > params.worker_timeout:
            raise ValueError("Worker time quota exceeded")
        progress(percent, name)
        timeline.append({"stage": name, "elapsed_seconds": perf_counter() - started})

    stage(10, "ingestion")
    pixels = sum(t.width * t.height for t in (params.source_tile, params.reference_tile))
    if pixels * 512 + 256 * 1024**2 > params.max_memory_bytes:
        raise ValueError("Worker allocation estimate exceeds configured memory budget")
    src_data = read_tile(directory / "source.img", params.source_record, params.source_tile, max_pixels=params.max_tile_pixels)
    ref_data = read_tile(directory / "reference.img", params.reference_record, params.reference_tile, max_pixels=params.max_tile_pixels)
    source, reference = src_data.raw, ref_data.raw
    src_gray, ref_gray = src_data.display, ref_data.display
    src_mask, ref_mask = src_data.valid, ref_data.valid
    if not src_mask.any() or not ref_mask.any():
        raise ValueError("All-null tile cannot be registered")
    stage(20, "routing")
    metadata = (params.source_record, params.reference_record)
    trustworthy = all(getattr(m, field) is not None and 'unverified' not in m.field_provenance.get(field, '')
                      for m in metadata for field in ('gsd_meters', 'incidence_angle_deg'))
    if trustworthy:
        decision = route_pair(src_gray, ref_gray, *metadata, source_mask=src_mask, reference_mask=ref_mask)
        routing = decision.model_dump(mode="json")
        selected = decision.selected_engine
    else:
        selected = "sift_flann"
        routing = {"selected_engine": selected, "rationales": ["insufficient telemetry", "Bounded SIFT/FLANN image-only fallback; no GSD or incidence invented."]}
    if params.engine != "auto":
        selected = params.engine
        routing["explicit_engine_override"] = selected
    stage(30, "conditioning")
    src_conditioned, ref_conditioned = apply_wallis_filter(src_gray, mask=src_mask), apply_wallis_filter(ref_gray, mask=ref_mask)
    stage(40, "matching")
    with _inference_lock:
        matches = matcher_for(selected).match(src_conditioned, ref_conditioned, mask_src=src_mask, mask_ref=ref_mask)
    raw_count = len(matches)
    matches = refine_matches(matches, src_conditioned, ref_conditioned)
    keep = np.ones(len(matches), dtype=bool)
    for pts, mask in ((matches.pts_src, src_mask), (matches.pts_ref, ref_mask)):
        safe = cv2.erode(mask.astype(np.uint8), np.ones((23, 23), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1)
        xy = np.rint(pts).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        ok = np.zeros(len(pts), dtype=bool)
        ok[inside] = safe[xy[inside, 1], xy[inside, 0]] > 0
        keep &= ok
    matches = MatchResult(matches.pts_src[keep], matches.pts_ref[keep], matches.confidences[keep], matches.execution_time_ms, matches.engine_name)
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
              "schema_version": 2, "metadata": {"source": params.source_record.model_dump(mode="json"), "reference": params.reference_record.model_dump(mode="json")},
              "tiles": {"source": params.source_tile.model_dump(), "reference": params.reference_tile.model_dump()},
              "overlap_status": params.overlap_status,
              "matching_mode": "map_grid_assisted" if params.overlap_status == "verified_grid_overlap" else "image_only",
              "independent_ground_validation": False,
              "image_dimensions": {"source": {"width": source.shape[1], "height": source.shape[0]},
                                   "reference": {"width": reference.shape[1], "height": reference.shape[0]}},
              "review_reasons": list(filtered.rejection_reasons)}
    result["coordinate_transforms"] = {}
    for name, record, tile in (("source", params.source_record, params.source_tile), ("reference", params.reference_record, params.reference_tile)):
        original = record.lineage.get("full_image_origin", [0, 0])
        result["coordinate_transforms"][name] = {"tile_to_full_scene": [[1,0,tile.x+original[0]],[0,1,tile.y+original[1]],[0,0,1]], "convention": "zero-based pixel centers"}
    if params.overlap_status == "unknown":
        result["review_reasons"].append("Overlap unknown: explicit tile image-only matching requires visual review; no lunar ground accuracy established")
    if metrics["inlier_ratio"] < 0.70:
        result["review_reasons"].append("Geometric inliers / original feature matches is below 70% after refinement")
    if not filtered.accepted or metrics["inlier_ratio"] < 0.70:
        # Phase 5 explicitly forbids passing rejected correspondences into TPS.
        if filtered.homography is not None and filtered.inlier_count:
            pts = matches.pts_src[filtered.inlier_mask]
            projected = cv2.perspectiveTransform(pts[:, None], filtered.homography)[:, 0]
            metrics["rmse_px"] = float(np.sqrt(np.mean(np.sum((projected - matches.pts_ref[filtered.inlier_mask])**2, axis=1))))
        metrics["rmse_basis"] = "rejected_homography_consensus_diagnostic"
        result["artifacts"] = {"dossier_url": f"/api/v1/artifacts/{job_id}_dossier.json", "georeferenced": False}
        write_dossier(directory / "artifacts", result)
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
    result["status"] = quality_status(rmse, filtered.vsui, validated=validated and params.overlap_status != "unknown")
    if rmse > 0.5:
        result["review_reasons"].append(f"RMSE {rmse:.6f} px exceeds 0.50 px")
    # Bilinear support must be entirely valid; no interpolation across nulls.
    map_x, map_y = warp.build_remap(reference.shape[:2])
    aligned = cv2.remap(np.where(src_mask, source, 0).astype(np.float64), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.remap(src_mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_mask = (support >= 1 - 1e-6) & ref_mask & np.isfinite(aligned)
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
    write_bundle(output, aligned, aligned_mask, result, params, filtered.tie_points, residuals, filtered.confidences, ids)
    result["artifacts"] = {
        "registered_preview_base64": preview_png(stretch(aligned, aligned_mask)),
        "geotiff_download_url": f"/api/v1/artifacts/{job_id}.tif",
        "tie_points_csv_url": f"/api/v1/artifacts/{job_id}_tiepoints.csv",
        "dossier_url": f"/api/v1/artifacts/{job_id}_dossier.json",
        "georeferenced": params.reference_record.grid is not None,
    }
    result["source_preview"], result["reference_preview"] = preview_png(src_gray), preview_png(ref_gray)
    metrics["runtime_seconds"] = perf_counter() - started
    stage(99, "finalizing")
    # Keep image blobs out of the portable provenance dossier.
    write_dossier(output, {**result, "source_preview": None, "reference_preview": None,
        "artifacts": {**result["artifacts"], "registered_preview_base64": None}, "pipeline_timeline": timeline})
    return result
