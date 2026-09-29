"""Phase 5: MAGSAC++ consensus, confidence Grid-NMS, and bounded Voronoi VSUI.

Coordinates are (x, y); image_shape is the reference canvas (height, width).
Spatial statistics and the 8x8 grid use reference coordinates. OpenCV supplies
MAGSAC++'s sigma marginalization internally; findHomography does not expose an
Epanechnikov-kernel option. Its 3.5-pixel threshold still controls the reported
consensus/termination, so it must not be described as entirely threshold-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from numbers import Integral

import cv2
import numpy as np
from scipy.spatial import QhullError, Voronoi

logger = logging.getLogger(__name__)

MIN_INLIER_RATIO = 0.70
MIN_INLIERS = 15
MIN_VSUI = 0.75
REPROJECTION_THRESHOLD = 3.5


@dataclass
class FilteringResult:
    """Filtered pairs in input order, with masks aligned to the original input.

    inlier_mask is geometric consensus before Grid-NMS (unique endpoints only).
    retained_mask is its spatially thinned subset. inlier_ratio always divides
    by the ORIGINAL match count, including duplicates. homography maps source
    to reference and is estimated from all consensus points before thinning.
    On rejection, this object is diagnostic only and is attached to the error.
    """

    homography: np.ndarray | None
    pts_src: np.ndarray
    pts_ref: np.ndarray
    confidences: np.ndarray
    inlier_mask: np.ndarray
    retained_mask: np.ndarray
    vsui_before: float
    vsui: float
    rejection_reasons: tuple[str, ...]

    @property
    def initial_count(self) -> int:
        return len(self.inlier_mask)

    @property
    def inlier_count(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def retained_count(self) -> int:
        return int(self.retained_mask.sum())

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / self.initial_count if self.initial_count else 0.0

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    @property
    def tie_points(self) -> np.ndarray:
        return np.concatenate((self.pts_src, self.pts_ref), axis=1)


class RegistrationRejected(RuntimeError):
    """A quality gate failed; result preserves masks and measured diagnostics."""

    def __init__(self, result: FilteringResult):
        self.result = result
        super().__init__("Registration rejected: " + "; ".join(result.rejection_reasons))


class SpatialGeometryError(RuntimeError):
    """The bounded Voronoi partition could not be computed reliably."""


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _canvas(image_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_shape, (tuple, list, np.ndarray)) or len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    return (_positive_integer(image_shape[0], "height"),
            _positive_integer(image_shape[1], "width"))


def _points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "uif":
        raise ValueError(f"{name} must be a real numeric array of shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return np.array(values, dtype=np.float64, order="C", copy=True)


def _on_canvas(points: np.ndarray, shape: tuple[int, int], name: str) -> None:
    height, width = shape
    if np.any(points < 0) or np.any(points > (width, height)):
        raise ValueError(f"{name} must lie within [0, width] x [0, height]")


def _scores(confidences: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(confidences)
    if values.shape != (count,) or values.dtype.kind not in "uif":
        raise ValueError("confidences must be a real numeric array of shape (N,)")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("confidences must be finite and in [0, 1]")
    return np.array(values, dtype=np.float64, copy=True)


def _clip_axis(polygon: np.ndarray, axis: int, bound: float,
               keep_greater: bool) -> np.ndarray:
    """Sutherland-Hodgman clip against one closed, axis-aligned half-plane."""
    if len(polygon) == 0:
        return polygon
    output = []
    previous = polygon[-1]
    previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
    for current in polygon:
        inside = current[axis] >= bound if keep_greater else current[axis] <= bound
        if inside != previous_inside:
            fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
            crossing = previous + fraction * (current - previous)
            crossing[axis] = bound
            output.append(crossing)
        if inside:
            output.append(current)
        previous, previous_inside = current, inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    # Translate first to avoid cancellation for small cells far from the origin.
    local = polygon - polygon[0]
    x, y = local.T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def bounded_voronoi_cells(points: np.ndarray,
                          image_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Return finite Voronoi polygons clipped to [0,W] x [0,H].

    One polygon is returned per UNIQUE site in lexicographic (x,y) order.
    Duplicate sites do not create fictitious cells. Empty input returns ().
    Four distant guard sites make real regions finite without affecting nearest
    neighbors anywhere on the canvas. Exact clipping, not guard-cell areas,
    supplies the boundary. Isotropic normalization preserves Euclidean geometry
    on rectangular images. No random Qhull jitter or finite-region omission.
    """
    shape = _canvas(image_shape)
    sites = _points(points, "points")
    _on_canvas(sites, shape, "points")
    sites = np.unique(sites, axis=0)
    if not len(sites):
        return ()
    height, width = shape
    scale = float(max(width, height))
    center = np.array([width / 2, height / 2])
    normalized = (sites - center) / scale
    # Canvas is contained in [-0.5,0.5]^2, diameter <= sqrt(2). Every guard
    # is at least 3.5*sqrt(2) from it; no guard can steal a real cell's area.
    guards = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]], dtype=np.float64)
    try:
        diagram = Voronoi(np.vstack((normalized, guards)))
    except QhullError as error:
        raise SpatialGeometryError("Voronoi construction failed") from error
    lower, upper = -center / scale, center / scale
    cells = []
    for region_index in diagram.point_region[:len(sites)]:
        region = diagram.regions[region_index] if region_index >= 0 else []
        if not region or -1 in region:
            raise SpatialGeometryError("Voronoi site has no finite region")
        polygon = diagram.vertices[region]
        for axis in (0, 1):
            polygon = _clip_axis(polygon, axis, lower[axis], True)
            polygon = _clip_axis(polygon, axis, upper[axis], False)
        polygon = np.clip(polygon * scale + center, (0, 0), (width, height))
        if not np.isfinite(polygon).all() or _polygon_area(polygon) <= 0:
            raise SpatialGeometryError("Voronoi cell is empty or non-finite")
        cells.append(polygon)
    total_area = sum(_polygon_area(cell) for cell in cells)
    if not np.isclose(total_area, float(width) * height, rtol=1e-8, atol=1e-8):
        raise SpatialGeometryError("Voronoi cells do not partition the canvas")
    return tuple(cells)


def compute_vsui(points: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Return clipped 1 - population_std(cell_area)/(mean(cell_area)+1e-6).

    Empty input scores zero. Repeated coordinates count as one site. A single
    site's mathematical score is one; minimum-count/geometry gates are separate.
    Negative values of the requested formula are clamped to zero, never rescaled.
    """
    cells = bounded_voronoi_cells(points, image_shape)
    if not cells:
        return 0.0
    areas = np.array([_polygon_area(cell) for cell in cells])
    return float(np.clip(1.0 - areas.std(ddof=0) / (areas.mean() + 1e-6), 0.0, 1.0))


def grid_nms(points: np.ndarray, confidences: np.ndarray,
             image_shape: tuple[int, int], *, top_k: int = 1,
             grid_size: int = 8) -> np.ndarray:
    """Return original indices of the top-K confidence sites in each grid cell.

    Ties prefer the earliest input index; selected indices are in input order.
    Exact duplicates consume only one slot. Closed right/bottom canvas edges
    belong to the last column/row. Grid-NMS cannot populate uncovered cells.
    """
    shape = _canvas(image_shape)
    sites = _points(points, "points")
    _on_canvas(sites, shape, "points")
    scores = _scores(confidences, len(sites))
    top_k = _positive_integer(top_k, "top_k")
    grid_size = _positive_integer(grid_size, "grid_size")
    height, width = shape
    grid = np.minimum((sites / (width, height) * grid_size).astype(np.int64), grid_size - 1)
    occupancy: dict[tuple[int, int], int] = {}
    seen = set()
    kept = []
    for index in np.argsort(-scores, kind="stable"):
        site = tuple(sites[index])
        cell = tuple(grid[index])
        if site not in seen and occupancy.get(cell, 0) < top_k:
            seen.add(site)
            occupancy[cell] = occupancy.get(cell, 0) + 1
            kept.append(index)
    return np.sort(np.asarray(kept, dtype=np.int64))


def _unique_pairs(source: np.ndarray, reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Keep the strongest one-to-one endpoints so repeats cannot inflate consensus."""
    source_seen, reference_seen = set(), set()
    kept = []
    for index in np.argsort(-scores, kind="stable"):
        src, ref = tuple(source[index]), tuple(reference[index])
        if src not in source_seen and ref not in reference_seen:
            kept.append(index)
            source_seen.add(src)
            reference_seen.add(ref)
    return np.sort(np.asarray(kept, dtype=np.int64))


def _spans_plane(points: np.ndarray) -> bool:
    return len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 2


def filter_magsac_and_vsui(pts_src: np.ndarray, pts_ref: np.ndarray,
                          confidences: np.ndarray, image_shape: tuple[int, int], *,
                          top_k: int = 1,
                          source_shape: tuple[int, int] | None = None) -> FilteringResult:
    """Estimate source->reference H, remove outliers, then homogenize inliers.

    Run cv2.findHomography(..., cv2.USAC_MAGSAC, 3.5), with 10,000 iterations
    and 0.999 confidence. Return only when consensus/original >= 0.70, at least
    15 points survive Grid-NMS, both sets span a plane, and final VSUI >= 0.75.
    The 8x8 grid uses reference coordinates, default K=1 (up to 64 tie points).
    No refit changes the MAGSAC++ model after spatial selection.

    Raise RegistrationRejected with .result diagnostics on any quality failure,
    ValueError for malformed inputs, and RuntimeError if MAGSAC++ is unavailable.
    Source pixels need not share the reference extent; source_shape optionally
    validates their canvas. All inputs are preserved. Empty (0,2) arrays yield
    a rejected diagnostic result. No fallback to ordinary RANSAC is permitted.
    """
    shape = _canvas(image_shape)
    source, reference = _points(pts_src, "pts_src"), _points(pts_ref, "pts_ref")
    if source.shape != reference.shape:
        raise ValueError("pts_ref must have the same (N, 2) shape as pts_src")
    _on_canvas(reference, shape, "pts_ref")
    if np.any(source < 0):
        raise ValueError("pts_src must have nonnegative pixel coordinates")
    if source_shape is not None:
        _on_canvas(source, _canvas(source_shape), "pts_src")
    scores = _scores(confidences, len(source))
    top_k = _positive_integer(top_k, "top_k")
    if not hasattr(cv2, "USAC_MAGSAC"):
        raise RuntimeError("OpenCV with USAC_MAGSAC support is required")

    reasons = []
    indices = _unique_pairs(source, reference, scores)
    inlier_mask = np.zeros(len(source), dtype=bool)
    retained_mask = np.zeros(len(source), dtype=bool)
    homography = None
    vsui_before = vsui = 0.0
    if len(indices) < 4 or not all(_spans_plane(p[indices]) for p in (source, reference)):
        reasons.append("at least four distinct, non-collinear correspondences are required")
    else:
        try:
            homography, mask = cv2.findHomography(
                source[indices], reference[indices], cv2.USAC_MAGSAC,
                REPROJECTION_THRESHOLD, maxIters=10000, confidence=0.999,
            )
        except cv2.error as error:
            raise RuntimeError("OpenCV MAGSAC++ estimation failed") from error
        if (homography is None or mask is None or not np.isfinite(homography).all()
                or np.linalg.matrix_rank(homography) < 3):
            homography = None
            reasons.append("MAGSAC++ could not estimate a finite, nonsingular homography")
        else:
            # Also exclude projections at infinity; OpenCV's perspectiveTransform
            # reports zero coordinates for these, which could masquerade as inliers.
            homogeneous = np.column_stack((source[indices], np.ones(len(indices)))) @ homography.T
            denominator = homogeneous[:, 2]
            valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(denominator) > 1e-12)
            projected = np.full((len(indices), 2), np.inf)
            projected[valid] = homogeneous[valid, :2] / denominator[valid, None]
            residuals = np.linalg.norm(projected - reference[indices], axis=1)
            consensus = mask.ravel().astype(bool) & valid & (residuals <= REPROJECTION_THRESHOLD)
            inlier_mask[indices[consensus]] = True

    inlier_indices = np.flatnonzero(inlier_mask)
    selected = grid_nms(reference[inlier_indices], scores[inlier_indices], shape, top_k=top_k)
    retained_mask[inlier_indices[selected]] = True
    try:
        vsui_before = compute_vsui(reference[inlier_mask], shape)
        vsui = compute_vsui(reference[retained_mask], shape)
    except SpatialGeometryError as error:
        reasons.append(str(error))
    ratio = len(inlier_indices) / len(source) if len(source) else 0.0
    if ratio < MIN_INLIER_RATIO:
        reasons.append(f"inlier ratio {ratio:.2%} < {MIN_INLIER_RATIO:.0%}")
    if retained_mask.sum() < MIN_INLIERS:
        reasons.append(f"valid inlier count after Grid-NMS {retained_mask.sum()} < {MIN_INLIERS}")
    elif not all(_spans_plane(p[retained_mask]) for p in (source, reference)):
        reasons.append("retained correspondences do not span a plane")
    if vsui < MIN_VSUI:
        reasons.append(f"VSUI {vsui:.6f} < {MIN_VSUI:.2f}")
    result = FilteringResult(
        homography, source[retained_mask], reference[retained_mask], scores[retained_mask],
        inlier_mask, retained_mask, vsui_before, vsui, tuple(reasons),
    )
    logger.info(
        "Initial points: %d | Inliers retained: %d (MAGSAC++), %d (Grid-NMS) | "
        "Inlier Ratio: %.2f%% (required >=70%%) | VSUI index: %.6f (before: %.6f) | %s",
        result.initial_count, result.inlier_count, result.retained_count,
        100 * result.inlier_ratio, result.vsui, result.vsui_before,
        "ACCEPTED" if result.accepted else "REJECTED: " + "; ".join(reasons),
    )
    if not result.accepted:
        raise RegistrationRejected(result)
    return result
