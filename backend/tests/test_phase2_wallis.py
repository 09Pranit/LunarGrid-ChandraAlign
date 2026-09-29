"""Phase 2 correctness tests and standalone synthetic-image benchmark.

Run pytest with -s to print histograms and timings, or execute this file directly.
The two original global-statistics gates are explicit expected failures: the
user chose the exact equation over additional global normalization.
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.preprocess import wallis
from lunar_core.preprocess.wallis import apply_wallis_filter, compute_local_stats


def synthetic_lunar_image() -> np.ndarray:
    """Seeded regolith (mean 60, std 5), graded shadow <=10 and bright rim >=240."""
    rng = np.random.default_rng(2026)
    image = np.clip(rng.normal(60, 5, (1024, 1024)), 0, 255).astype(np.float32)
    image[128:384, 128:384] = np.linspace(1, 9, 256, dtype=np.float32)[None, :]
    cv2.circle(image, (720, 512), 140, 245.0, thickness=12)
    return image


def test_local_stats_against_independent_window_reference():
    image = np.random.default_rng(12).uniform(0, 255, (17, 19))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(image, 2, mode="reflect"), (5, 5))
    mean, std = compute_local_stats(image, window_size=5)
    np.testing.assert_allclose(mean, windows.mean(axis=(-2, -1)), atol=1e-10)
    np.testing.assert_allclose(std, windows.std(axis=(-2, -1)), atol=1e-10)


def test_exact_wallis_equation_and_input_preservation():
    image = np.random.default_rng(3).uniform(0, 255, (23, 25))
    original = image.copy()
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(image, 3, mode="reflect"), (7, 7))
    mean = windows.mean(axis=(-2, -1))
    std = np.maximum(windows.std(axis=(-2, -1)), 1e-6)
    expected = (image - mean) * (0.8 * 52 / (0.8 * std + 0.2 * 52)) + 0.2 * 128 + 0.8 * mean
    actual = apply_wallis_filter(image, window_size=7)
    np.testing.assert_array_equal(actual, np.clip(expected, 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(image, original)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("intensity", [0, 10, 60, 128, 255])
def test_constant_image_is_finite_and_obeys_partial_brightness_forcing(intensity):
    image = np.full((48, 53), intensity, dtype=np.uint8)
    mean, std = compute_local_stats(image)
    np.testing.assert_allclose(mean, intensity, atol=1e-12)
    np.testing.assert_allclose(std, 1e-6, atol=1e-12)
    result = apply_wallis_filter(image)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, np.full_like(image, int(0.2 * 128 + 0.8 * intensity)))


def test_nearly_constant_float_region_and_clipping():
    image = np.full((128, 128), 60.125, dtype=np.float64)
    image[64, 64] += 1e-4
    _, std = compute_local_stats(image)
    assert np.isfinite(std).all() and std.max() < 1e-4
    image[64, 64] = 255
    image[32, 32] = 0
    result = apply_wallis_filter(image)
    assert result[64, 64] == 255
    assert result[32, 32] == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 63), (63, 1), (11, 17)])
def test_small_and_noncontiguous_images(shape):
    source = np.arange(shape[0] * shape[1] * 2, dtype=np.float64).reshape(shape[0], shape[1] * 2) % 256
    image = source[:, ::2]
    result = apply_wallis_filter(image)
    assert result.shape == shape and result.dtype == np.uint8


def test_synthetic_regions_brighten_and_retain_shadow_gradients():
    image = synthetic_lunar_image()
    result = apply_wallis_filter(image)
    assert image[128:384, 128:384].max() <= 10
    assert image.max() >= 240
    background = image[800:, :]
    assert abs(background.mean() - 60) < 0.1
    assert abs(background.std() - 5) < 0.1
    shadow = result[150:360, 150:360]
    assert shadow.mean() > image[150:360, 150:360].mean()
    assert np.abs(cv2.Sobel(shadow, cv2.CV_64F, 1, 0)).max() > 0
    assert abs(result.mean() - 128) < abs(image.mean() - 128)
    assert 0 <= result.min() <= result.max() <= 255


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Exact b=0.2 equation does not force global mean to 128; normalization declined")
def test_original_global_mean_acceptance_gate():
    assert abs(apply_wallis_filter(synthetic_lunar_image()).mean() - 128) <= 10


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Local target_std=52 does not guarantee global std=52; normalization declined")
def test_original_global_std_acceptance_gate():
    assert abs(apply_wallis_filter(synthetic_lunar_image()).std() - 52) <= 8


@pytest.mark.parametrize("shape,tile_size", [((129, 151), 32), ((37, 129), 17), ((129, 1), 13)])
def test_tiled_and_untiled_paths_match_including_seams(monkeypatch, shape, tile_size):
    image = np.random.default_rng(42).uniform(0, 255, shape)
    expected_mean, expected_std = compute_local_stats(image)
    expected = apply_wallis_filter(image)
    monkeypatch.setattr(wallis, "_MAX_UNTILED_SIDE", 64)
    mean, std = compute_local_stats(image, tile_size=tile_size)
    actual = apply_wallis_filter(image, tile_size=tile_size)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-9)
    np.testing.assert_allclose(std, expected_std, atol=1e-9)
    np.testing.assert_array_equal(actual, expected)


def test_real_large_image_uses_bounded_tiles(monkeypatch):
    image = np.full((4097, 4097), 60, dtype=np.uint8)
    stats = wallis._local_stats
    patch_shapes = []

    def record_patch(patch, window_size):
        patch_shapes.append(patch.shape)
        return stats(patch, window_size)

    monkeypatch.setattr(wallis, "_local_stats", record_patch)
    result = apply_wallis_filter(image)
    assert len(patch_shapes) == 25
    assert max(max(shape) for shape in patch_shapes) <= 1024 + 40
    assert result.shape == image.shape and np.all(result == 73)


@pytest.mark.parametrize("image", [np.array([]), np.zeros((3, 3, 3)), np.array([[np.nan]]),
                                     np.array([[np.inf]]), np.array([[-1]]), np.array([[256]]),
                                     np.array([[1j]]), np.array([[True]])])
def test_invalid_images_rejected(image):
    with pytest.raises(ValueError):
        apply_wallis_filter(image)
    with pytest.raises(ValueError):
        compute_local_stats(image)


@pytest.mark.parametrize("kwargs", [{"window_size": 0}, {"window_size": 4}, {"window_size": 3.5},
                                      {"tile_size": 0}, {"tile_size": 4097}, {"contrast": 0.69},
                                      {"contrast": 0.96}, {"brightness": 0.09}, {"brightness": 0.31},
                                      {"target_std": 0}, {"target_mean": 256}, {"contrast": np.nan}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        apply_wallis_filter(np.zeros((3, 3)), **kwargs)


def test_registration_entry_point_uses_shared_filter():
    from registration import wallis_filter

    gray = np.random.default_rng(5).integers(0, 256, (53, 59), dtype=np.uint8)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    np.testing.assert_array_equal(wallis_filter(color), apply_wallis_filter(gray))


def run_benchmark() -> float:
    image = synthetic_lunar_image()
    apply_wallis_filter(image)  # Warm OpenCV and allocator; imports excluded.
    timings = []
    for _ in range(7):
        started = perf_counter()
        result = apply_wallis_filter(image)
        timings.append((perf_counter() - started) * 1000)
    edges = np.arange(0, 257, 16)
    print(f"\n1024x1024; window=41; NumPy={np.__version__}; OpenCV={cv2.__version__}")
    print(f"Before: range=[{image.min():.2f}, {image.max():.2f}], mean={image.mean():.3f}, std={image.std():.3f}")
    print(f"After:  range=[{result.min()}, {result.max()}], mean={result.mean():.3f}, std={result.std():.3f}")
    print(f"Histogram edges: {edges.tolist()}")
    print(f"Before counts: {np.histogram(image, bins=edges)[0].tolist()}")
    print(f"After counts:  {np.histogram(result, bins=edges)[0].tolist()}")
    median = float(np.median(timings))
    print(f"Runtime ms: median={median:.2f}, min={min(timings):.2f}, max={max(timings):.2f}; limit=150")
    print("Global mean/std gates: not guaranteed by the exact equation (documented expected failures).")
    return median


def test_1024_benchmark_under_150_ms():
    assert run_benchmark() < 150, "Median of seven warmed runs exceeded 150 ms on this machine"


if __name__ == "__main__":
    assert run_benchmark() < 150
