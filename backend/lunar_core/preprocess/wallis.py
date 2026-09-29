"""Vectorized local Wallis conditioning of grayscale images in [0, 255].

The target statistics are local control parameters, not guaranteed global output
statistics. In particular b=0.2 moves a constant area's mean only 20% toward m_d.
No detail can be recovered from an entirely uniform, clipped shadow.
"""

from __future__ import annotations

from numbers import Integral, Real

import cv2
import numpy as np

_MAX_UNTILED_SIDE = 4096
_STD_FLOOR = 1e-6


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate(image: np.ndarray, window_size: int, tile_size: int) -> tuple[int, int]:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2D grayscale NumPy array")
    if image.dtype.kind not in "uif":
        raise ValueError("image must contain real numeric intensities in [0, 255]")
    window_size = _positive_integer(window_size, "window_size")
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    tile_size = _positive_integer(tile_size, "tile_size")
    if tile_size > _MAX_UNTILED_SIDE:
        raise ValueError(f"tile_size must be at most {_MAX_UNTILED_SIDE}")
    return window_size, tile_size


def _tiles(image: np.ndarray, window_size: int, tile_size: int):
    """Yield halo-extended patches, destination slices and patch core slices.

    Only tiles are iterated in Python; all pixel/window arithmetic is vectorized.
    Reflect padding is applied at real image borders, never at internal seams.
    """
    height, width = image.shape
    if max(height, width) <= _MAX_UNTILED_SIDE:
        whole = (slice(None), slice(None))
        yield image, whole, whole
        return
    radius = window_size // 2
    columns = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    for index in range(rows * columns):
        row, column = divmod(index, columns)
        top, left = row * tile_size, column * tile_size
        bottom, right = min(top + tile_size, height), min(left + tile_size, width)
        y0, x0 = max(0, top - radius), max(0, left - radius)
        y1, x1 = min(height, bottom + radius), min(width, right + radius)
        destination = (slice(top, bottom), slice(left, right))
        core = (slice(top - y0, bottom - y0), slice(left - x0, right - x0))
        yield image[y0:y1, x0:x1], destination, core


def _local_stats(patch: np.ndarray, window_size: int):
    # Float64 moments avoid overflow and cancellation in nearly constant regions.
    values = np.asarray(patch, dtype=np.float64, order="C")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
        raise ValueError("image intensities must be finite and in [0, 255]")
    kernel = (window_size, window_size)
    mean = cv2.boxFilter(values, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101)
    variance = cv2.boxFilter(values * values, -1, kernel, normalize=True,
                             borderType=cv2.BORDER_REFLECT_101)
    variance -= mean * mean
    np.maximum(variance, 0, out=variance)
    np.sqrt(variance, out=variance)
    np.maximum(variance, _STD_FLOOR, out=variance)
    return values, mean, variance


def compute_local_stats(
    image: np.ndarray, window_size: int = 41, *, tile_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Return local mean and population std using two normalized boxFilter passes.

    The odd window uses REFLECT_101 borders. Std is floored at 1e-6. Images
    exceeding 4096 on either axis are tiled, with a window-radius halo. Returned
    float64 mean/std maps necessarily occupy 16 bytes per input pixel; temporary
    allocations are limited to a tile plus halo on the tiled path.
    """
    window_size, tile_size = _validate(image, window_size, tile_size)
    if max(image.shape) <= _MAX_UNTILED_SIDE:
        _, mean, std = _local_stats(image, window_size)
        return mean, std
    means = np.empty(image.shape, dtype=np.float64)
    stds = np.empty(image.shape, dtype=np.float64)
    for patch, destination, core in _tiles(image, window_size, tile_size):
        _, mean, std = _local_stats(patch, window_size)
        means[destination], stds[destination] = mean[core], std[core]
    return means, stds


def apply_wallis_filter(
    image: np.ndarray,
    window_size: int = 41,
    *,
    target_mean: float = 128.0,
    target_std: float = 52.0,
    contrast: float = 0.80,
    brightness: float = 0.20,
    tile_size: int = 1024,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the exact local Wallis equation and return clipped uint8 output.

    Inputs are nonempty 2D grayscale arrays already scaled to [0, 255]. Contrast
    is in [0.7, 0.95], brightness in [0.1, 0.3], target_mean in [0, 255], and
    target_std is in (0, 255]. Raw sensor DN values must be scaled by
    the caller. Input is never modified. Large images allocate only the uint8
    output at full resolution; all floating-point work uses haloed tiles.

    At the default contrast, gain is bounded by c/(1-c)=4 even in black areas.
    This limits noise amplification but does not constitute denoising.
    """
    window_size, tile_size = _validate(image, window_size, tile_size)
    parameters = (target_mean, target_std, contrast, brightness)
    if any(isinstance(p, bool) or not isinstance(p, Real) or not np.isfinite(p)
           for p in parameters):
        raise ValueError("Wallis parameters must be finite real numbers")
    if not 0 <= target_mean <= 255 or not 0 < target_std <= 255:
        raise ValueError("target_mean must be in [0, 255]; target_std in (0, 255]")
    if not 0.7 <= contrast <= 0.95:
        raise ValueError("contrast must be in [0.7, 0.95]")
    if not 0.1 <= brightness <= 0.3:
        raise ValueError("brightness must be in [0.1, 0.3]")

    if mask is not None:
        if mask.shape != image.shape or mask.dtype != bool:
            raise ValueError("Wallis mask must be boolean and match image")
        values = image.astype(np.float64)
        weights = mask.astype(np.float64)
        kernel = (window_size, window_size)
        mass = cv2.boxFilter(weights, -1, kernel, normalize=False)
        mass = np.maximum(mass, 1)
        mean = cv2.boxFilter(values * weights, -1, kernel, normalize=False) / mass
        variance = cv2.boxFilter(values * values * weights, -1, kernel, normalize=False) / mass - mean * mean
        std = np.sqrt(np.maximum(variance, _STD_FLOOR**2))
        gain = (contrast * target_std) / (contrast * std + (1 - contrast) * target_std)
        output = np.clip((values - mean) * gain + brightness * target_mean + (1 - brightness) * mean, 0, 255).astype(np.uint8)
        output[~mask] = 0
        return output
    output = np.empty(image.shape, dtype=np.uint8)
    for patch, destination, core in _tiles(image, window_size, tile_size):
        values, mean, std = _local_stats(patch, window_size)
        values, mean, std = values[core], mean[core], std[core]
        gain = (contrast * target_std) / (contrast * std + (1 - contrast) * target_std)
        result = (values - mean) * gain + brightness * target_mean + (1 - brightness) * mean
        np.clip(result, 0, 255, out=result)
        output[destination] = result.astype(np.uint8)
    return output
