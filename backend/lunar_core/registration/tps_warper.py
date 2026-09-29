"""Regularized thin-plate splines in pixel coordinates and backward raster warps.

Points are (x, y), canvases are (height, width), and integer coordinates denote
pixel centers. The reverse spline is fitted independently: it approximates the
inverse for smooth, one-to-one deformations, not an arbitrary folded mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from numbers import Integral, Real

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

logger = logging.getLogger(__name__)

DEFAULT_SMOOTHING = 0.05
LARGE_IMAGE_PIXELS = 512 * 512
_KERNEL_BUDGET = 262_144
_REMAP_LIMIT = 32767  # OpenCV requires both source and destination sizes < this.


def _points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "uif":
        raise ValueError(f"{name} must be a real numeric array of shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return np.array(values, dtype=np.float64, order="C", copy=True)


def _canvas(target_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(target_shape, (tuple, list, np.ndarray)) or len(target_shape) != 2:
        raise ValueError("target_shape must be (height, width)")
    for value in target_shape:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("target_shape dimensions must be positive integers")
        if value >= _REMAP_LIMIT:
            raise ValueError("OpenCV remap dimensions must be less than 32767; tile larger rasters")
    return int(target_shape[0]), int(target_shape[1])


def _control_points(points: np.ndarray, name: str) -> None:
    if len(points) < 3:
        raise ValueError("at least three distinct, non-collinear tie-points are required")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name} must contain distinct points (duplicate endpoints)")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) != 2:
        raise ValueError(f"{name} must span a plane (non-collinear points)")


def _rmse(predicted: np.ndarray, expected: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((predicted - expected) ** 2, axis=1))))


@dataclass(frozen=True)
class TPSWarp:
    """Reusable source->reference and reference->source TPS models.

    Construct with solve_tps_warp(). Fitting residuals describe training data;
    use rmse() with independent checkpoints to measure registration accuracy.
    No universal accuracy guarantee or fold prevention is implied by fitting.
    """

    _forward: RBFInterpolator
    _backward: RBFInterpolator
    smoothing: float
    control_point_count: int
    fitting_rmse_px: float
    backward_fitting_rmse_px: float

    def _displacement(self, model: RBFInterpolator, points: np.ndarray) -> np.ndarray:
        # Bound evaluation workspace even for dense coordinate grids. The solve
        # itself is global O(N^2) storage, appropriate for Phase 5's ~64 points.
        batch_size = max(1, min(4096, _KERNEL_BUDGET // self.control_point_count))
        result = np.empty_like(points)
        for start in range(0, len(points), batch_size):
            result[start:start + batch_size] = model(points[start:start + batch_size])
        if not np.isfinite(result).all():
            raise ValueError("TPS evaluation produced non-finite coordinates")
        return result

    def forward(self, pts_src: np.ndarray) -> np.ndarray:
        """Evaluate source->reference coordinates as float64 (N,2)."""
        points = _points(pts_src, "pts_src")
        return points + self._displacement(self._forward, points)

    def backward(self, pts_ref: np.ndarray) -> np.ndarray:
        """Evaluate the independently fitted reference->source spline."""
        points = _points(pts_ref, "pts_ref")
        return points + self._displacement(self._backward, points)

    def rmse(self, pts_src: np.ndarray, pts_ref: np.ndarray) -> float:
        """Radial RMSE in pixels: sqrt(mean(dx**2 + dy**2)), not per-axis RMSE."""
        source, reference = _points(pts_src, "pts_src"), _points(pts_ref, "pts_ref")
        if source.shape != reference.shape or not len(source):
            raise ValueError("RMSE requires nonempty paired arrays of the same (N, 2) shape")
        return _rmse(self.forward(source), reference)

    def build_remap(self, target_shape: tuple[int, int], *,
                    grid_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Build float32 source-coordinate maps for every reference pixel.

        None selects step 8 for canvases >=512*512 pixels, otherwise step 1.
        Explicit 1 gives dense evaluation; 8 uses bilinear cv2.resize on coarse
        displacement fields. Half-pixel-aligned samples and a one-cell halo
        preserve affine maps at boundaries and on non-multiple-of-eight sizes.
        Only maps are resized; raster resampling is always bicubic in warp().
        """
        height, width = _canvas(target_shape)
        if grid_step is None:
            grid_step = 8 if height * width >= LARGE_IMAGE_PIXELS else 1
        if (isinstance(grid_step, bool) or not isinstance(grid_step, Integral)
                or grid_step not in (1, 8)):
            raise ValueError("grid_step must be 1, 8, or None")

        if grid_step == 1:
            # Fill maps in row blocks instead of materializing a full float64
            # H*W*2 mesh and H*W*N radial-basis matrix at once.
            map_x = np.empty((height, width), dtype=np.float32)
            map_y = np.empty_like(map_x)
            rows = max(1, 4096 // width)
            for start in range(0, height, rows):
                stop = min(start + rows, height)
                xx, yy = np.meshgrid(np.arange(width, dtype=np.float64),
                                     np.arange(start, stop, dtype=np.float64))
                coords = np.column_stack((xx.ravel(), yy.ravel()))
                mapped = self.backward(coords).reshape(stop - start, width, 2)
                map_x[start:stop], map_y[start:stop] = mapped[..., 0], mapped[..., 1]
        else:
            coarse_h, coarse_w = (height + 7) // 8 + 2, (width + 7) // 8 + 2
            # resize maps destination j to (j+.5)/8-.5. After cropping the
            # halo, this corresponds exactly to physical coordinates j.
            xx, yy = np.meshgrid((np.arange(coarse_w) - 0.5) * 8 - 0.5,
                                 (np.arange(coarse_h) - 0.5) * 8 - 0.5)
            coords = np.column_stack((xx.ravel(), yy.ravel()))
            delta = self._displacement(self._backward, coords).reshape(coarse_h, coarse_w, 2)
            maps = []
            for axis in (0, 1):
                expanded = cv2.resize(delta[..., axis].astype(np.float32),
                                      (coarse_w * 8, coarse_h * 8),
                                      interpolation=cv2.INTER_LINEAR)
                maps.append(np.ascontiguousarray(expanded[8:8 + height, 8:8 + width]))
            map_x, map_y = maps
            map_x += np.arange(width, dtype=np.float32)[None, :]
            map_y += np.arange(height, dtype=np.float32)[:, None]
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValueError("TPS maps exceed finite float32 pixel coordinates")
        return map_x, map_y

    def warp(self, source: np.ndarray, target_shape: tuple[int, int], *,
             grid_step: int | None = None) -> np.ndarray:
        """Backward sample onto exactly target_shape using cubic interpolation.

        BORDER_CONSTANT fills outside the source with zero. Supports grayscale
        and HWC arrays (1-4 channels), uint8/uint16/int16/float32/float64; dtype
        and channel count are preserved. No intensity normalization is applied.
        Cubic interpolation can overshoot floating-point intensity ranges.
        """
        image = np.asarray(source)
        if (image.ndim not in (2, 3) or not image.size
                or (image.ndim == 3 and not 1 <= image.shape[2] <= 4)):
            raise ValueError("source must be a nonempty grayscale or HWC image with 1-4 channels")
        if image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
            raise ValueError("source dtype must be uint8, uint16, int16, float32, or float64")
        if not image.dtype.isnative or not np.isfinite(image).all():
            raise ValueError("source must contain finite values in native byte order")
        _canvas(image.shape[:2])
        map_x, map_y = self.build_remap(target_shape, grid_step=grid_step)
        warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if image.ndim == 3 and image.shape[2] == 1:
            warped = warped[..., None]  # OpenCV otherwise collapses singleton channels.
        return warped


def solve_tps_warp(pts_src: np.ndarray, pts_ref: np.ndarray, *,
                   smoothing: float = DEFAULT_SMOOTHING) -> TPSWarp:
    """Fit two global regularized TPS models from paired pixel coordinates.

    RBFInterpolator(kernel='thin_plate_spline', degree=1, epsilon=1) solves
    [K + lambda*I, P; P.T, 0] [W; A] = [V; 0], U(r)=r^2*ln(r), U(0)=0.
    lambda defaults to 0.05 in the ORIGINAL pixel-coordinate units. Fitting
    displacement and adding the identity affine term is algebraically equivalent
    to fitting absolute target coordinates, while reducing cancellation.

    Requires >=3 unique, non-collinear endpoints in BOTH directions. Pass Phase
    5's accepted pts_src/pts_ref; this function does not reject match outliers.
    Neither images nor input coordinates are modified. Smoothing is tunable,
    but no automatic normalization silently changes its physical strength.
    """
    source, reference = _points(pts_src, "pts_src"), _points(pts_ref, "pts_ref")
    if source.shape != reference.shape:
        raise ValueError("pts_ref must have the same (N, 2) shape as pts_src")
    _control_points(source, "pts_src")
    _control_points(reference, "pts_ref")
    if (isinstance(smoothing, bool) or not isinstance(smoothing, Real)
            or not np.isfinite(smoothing) or smoothing < 0):
        raise ValueError("smoothing must be a finite nonnegative number")
    options = dict(kernel="thin_plate_spline", degree=1, epsilon=1.0,
                   smoothing=float(smoothing))
    try:
        forward = RBFInterpolator(source, reference - source, **options)
        backward = RBFInterpolator(reference, source - reference, **options)
        forward_rmse = _rmse(source + forward(source), reference)
        backward_rmse = _rmse(reference + backward(reference), source)
    except np.linalg.LinAlgError as error:
        raise ValueError("TPS system is singular; check tie-point geometry") from error
    if not np.isfinite([forward_rmse, backward_rmse]).all():
        raise ValueError("TPS fit produced non-finite coordinates")
    result = TPSWarp(forward, backward, float(smoothing), len(source),
                     forward_rmse, backward_rmse)
    logger.info("TPS fit: %d tie-points | lambda=%.4g | forward training RMSE=%.6f px | "
                "backward training RMSE=%.6f px", len(source), smoothing,
                forward_rmse, backward_rmse)
    return result
