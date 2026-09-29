from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

if __package__:
    from .lunar_core.preprocess.wallis import apply_wallis_filter
else:
    from lunar_core.preprocess.wallis import apply_wallis_filter


class RegistrationError(RuntimeError):
    pass


@dataclass
class RegistrationResult:
    aligned: np.ndarray
    conditioned_source: np.ndarray
    matches_preview: np.ndarray
    transform: np.ndarray
    metrics: dict
    tie_points: list[dict]


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise RegistrationError("The uploaded image could not be decoded.")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def wallis_filter(image: np.ndarray, window: int = 41, target_mean: float = 128.0,
                  target_std: float = 52.0, contrast: float = 0.80,
                  brightness: float = 0.20) -> np.ndarray:
    """Compatibility entry point for the shared Phase 2 implementation."""
    return apply_wallis_filter(
        _gray(image), window_size=window, target_mean=target_mean,
        target_std=target_std, contrast=contrast, brightness=brightness,
    )


def spatial_coverage(points: np.ndarray, width: int, height: int, grid: int = 5) -> float:
    if len(points) < 3:
        return 0.0
    occupied = set()
    for x, y in points:
        occupied.add((min(grid - 1, int(x / max(width, 1) * grid)), min(grid - 1, int(y / max(height, 1) * grid))))
    hull = cv2.convexHull(points.astype(np.float32))
    hull_ratio = cv2.contourArea(hull) / max(float(width * height), 1.0)
    grid_ratio = len(occupied) / float(grid * grid)
    return float(np.clip(0.55 * hull_ratio + 0.45 * grid_ratio, 0.0, 1.0))


def register_images(source: np.ndarray, reference: np.ndarray, ratio: float = 0.72,
                    max_features: int = 8000) -> RegistrationResult:
    started = time.perf_counter()
    src_gray = _gray(source)
    ref_gray = _gray(reference)
    src_conditioned = wallis_filter(src_gray)
    ref_conditioned = wallis_filter(ref_gray)

    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.018, edgeThreshold=12)
    key_src, desc_src = sift.detectAndCompute(src_conditioned, None)
    key_ref, desc_ref = sift.detectAndCompute(ref_conditioned, None)
    if desc_src is None or desc_ref is None or len(key_src) < 8 or len(key_ref) < 8:
        raise RegistrationError("Not enough stable features were detected. Try a larger overlap or a less shadowed reference.")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
    pairs = matcher.knnMatch(desc_src, desc_ref, k=2)
    good = [a for a, b in pairs if a.distance < ratio * b.distance]
    if len(good) < 8:
        raise RegistrationError(f"Only {len(good)} candidate correspondences survived the ratio test.")

    src_pts = np.float32([key_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ref_pts = np.float32([key_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    transform, mask = cv2.findHomography(src_pts, ref_pts, method, 2.5, maxIters=10000, confidence=0.999)
    if transform is None or mask is None:
        raise RegistrationError("A stable geometric model could not be estimated.")
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 6:
        raise RegistrationError("Too few geometrically consistent matches remain after robust filtering.")

    projected = cv2.perspectiveTransform(src_pts[inlier_mask], transform)
    residual = projected.reshape(-1, 2) - ref_pts[inlier_mask].reshape(-1, 2)
    distances = np.linalg.norm(residual, axis=1)
    rmse = math.sqrt(float(np.mean(distances * distances)))
    coverage = spatial_coverage(ref_pts[inlier_mask].reshape(-1, 2), ref_gray.shape[1], ref_gray.shape[0])
    aligned = cv2.warpPerspective(source, transform, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_CUBIC)
    preview = cv2.drawMatches(source, key_src, reference, key_ref, good[:240], None, matchesMask=mask.ravel().tolist()[:240], flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    runtime = time.perf_counter() - started
    metrics = {
        "engine": "SIFT + FLANN + USAC_MAGSAC",
        "candidate_matches": len(good),
        "accepted_matches": inliers,
        "rejected_matches": len(good) - inliers,
        "registration_method": "Homography (USAC_MAGSAC)",
        "inlier_ratio": round(inliers / len(good), 4),
        "rmse_px": round(rmse, 4),
        "median_error_px": round(float(np.median(distances)), 4),
        "p95_error_px": round(float(np.percentile(distances, 95)), 4),
        "spatial_coverage": round(coverage, 4),
        "runtime_seconds": round(runtime, 4),
        "validation_note": "Residuals are computed on model inliers. For mission claims, add independent held-out checkpoints.",
    }
    tie_points = [
        {
            "id": f"LG-{i + 1:04d}",
            "source_x": round(float(src_pts[i, 0, 0]), 3),
            "source_y": round(float(src_pts[i, 0, 1]), 3),
            "reference_x": round(float(ref_pts[i, 0, 0]), 3),
            "reference_y": round(float(ref_pts[i, 0, 1]), 3),
            "confidence": None,  # This classical matcher supplies no calibrated probability.
            "status": "accepted" if inlier_mask[i] else "rejected",
        }
        for i in range(len(good))
    ]
    return RegistrationResult(aligned, src_conditioned, preview, transform, metrics, tie_points)
