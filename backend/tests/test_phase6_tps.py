"""Phase 6 acceptance checks using real SciPy TPS and OpenCV resampling.

Run directly: python backend/tests/test_phase6_tps.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import pytest
from scipy.linalg import solve
from scipy.spatial.distance import cdist

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.registration import TPSWarp, solve_tps_warp

logger = logging.getLogger(__name__)
SHAPE = (384, 512)


def grid(shape=SHAPE, count=8):
    height, width = shape
    xx, yy = np.meshgrid(np.linspace(0, width - 1, count),
                         np.linspace(0, height - 1, count))
    return np.column_stack((xx.ravel(), yy.ravel()))


def relief(points):
    x, y = np.asarray(points).T
    return np.column_stack((x + 3.5 * np.sin(y / 50.0),
                            y + 2.0 * np.cos(x / 50.0)))


def radial_rmse(actual, expected):
    return float(np.sqrt(np.mean(np.sum((actual - expected) ** 2, axis=1))))


def texture(points):
    x, y = points.T
    return 120 + 35 * np.sin(x / 8) + 28 * np.cos(y / 11) + 15 * np.sin((x + y) / 19)


@pytest.mark.parametrize("seed", [6, 42, 73])
def test_required_sinusoidal_relief_on_500_unseen_points(seed):
    source = grid()
    reference = relief(source)
    warp = solve_tps_warp(source, reference)
    assert isinstance(warp, TPSWarp)
    assert warp.control_point_count == 64
    assert warp.smoothing == 0.05
    evaluation = np.random.default_rng(seed).uniform((0, 0), (511, 383), (500, 2))
    assert cdist(evaluation, source).min() > 0
    expected = relief(evaluation)
    forward_error = warp.rmse(evaluation, expected)
    reverse_error = radial_rmse(warp.backward(expected), evaluation)
    homography, _ = cv2.findHomography(source, reference, method=0)
    planar = cv2.perspectiveTransform(evaluation[None, ...], homography)[0]
    planar_error = radial_rmse(planar, expected)
    logger.info("VERIFIED seed=%d: 64 ties, 500 unseen points | TPS RMSE=%.6f px "
                "(<0.45 validation, <0.50 acceptance; production target <0.42) | "
                "backward RMSE=%.6f px | homography RMSE=%.6f px",
                seed, forward_error, reverse_error, planar_error)
    assert forward_error < 0.45
    assert reverse_error < 0.45
    assert planar_error > 0.50
    assert forward_error < planar_error / 3


@pytest.mark.parametrize("step", [1, 8])
def test_backward_raster_corrects_relief_and_matches_canvas(step):
    source_points = grid()
    warp = solve_tps_warp(source_points, relief(source_points))
    xx, yy = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    # Analytic continuous reference radiance, sampled at true forward locations.
    # This avoids generating test truth with the spline being tested.
    source = texture(relief(pixels)).reshape(SHAPE).astype(np.float32)
    reference = texture(pixels).reshape(SHAPE).astype(np.float32)
    original = source.copy()
    warped = warp.warp(source, SHAPE, grid_step=step)
    interior = np.s_[8:-8, 8:-8]
    before = float(np.sqrt(np.mean((source[interior] - reference[interior]) ** 2)))
    after = float(np.sqrt(np.mean((warped[interior] - reference[interior]) ** 2)))
    logger.info("Raster step=%d: target=%s, actual=%s | radiance RMSE %.6f -> %.6f",
                step, SHAPE, warped.shape, before, after)
    assert warped.shape == SHAPE
    assert warped.dtype == source.dtype
    assert after < 1.0
    assert after < before / 8
    np.testing.assert_array_equal(source, original)


def kernel(query, controls):
    distances = cdist(query, controls)
    values = np.zeros_like(distances)
    positive = distances > 0
    values[positive] = distances[positive] ** 2 * np.log(distances[positive])
    return values


def test_default_regularization_matches_explicit_block_system():
    source = grid((3, 3), count=4)
    reference = source + np.random.default_rng(1).normal(0, 0.08, source.shape)
    count = len(source)
    polynomial = np.column_stack((np.ones(count), source))
    system = np.block([[kernel(source, source) + 0.05 * np.eye(count), polynomial],
                       [polynomial.T, np.zeros((3, 3))]])
    weights = solve(system, np.vstack((reference, np.zeros((3, 2)))), assume_a="sym")
    unseen = np.random.default_rng(2).uniform(0, 2, (100, 2))
    expected = (kernel(unseen, source) @ weights[:count]
                + np.column_stack((np.ones(len(unseen)), unseen)) @ weights[count:])
    warp = solve_tps_warp(source, reference)
    np.testing.assert_allclose(warp.forward(unseen), expected, atol=1e-12)
    np.testing.assert_allclose(polynomial.T @ weights[:count], 0, atol=1e-12)
    interpolating = solve_tps_warp(source, reference, smoothing=0)
    assert interpolating.fitting_rmse_px < 1e-12
    assert warp.fitting_rmse_px > 1e-4  # lambda is used, not ignored.
    assert warp.rmse(unseen, unseen) < interpolating.rmse(unseen, unseen)


@pytest.mark.parametrize("shape", [(1, 1), (1, 73), (65, 1), (37, 53), (513, 777)])
def test_coarse_affine_maps_have_no_half_pixel_or_edge_bias(shape):
    source = grid((700, 900), count=3)
    affine = np.array([[1.06, 0.09], [-0.04, 0.93]])
    offset = np.array([13.3, -8.6])
    reference = source @ affine.T + offset
    warp = solve_tps_warp(source, reference)
    map_x, map_y = warp.build_remap(shape, grid_step=8)
    xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    expected = ((np.column_stack((xx.ravel(), yy.ravel())) - offset)
                @ np.linalg.inv(affine).T).reshape(*shape, 2)
    np.testing.assert_allclose(map_x, expected[..., 0], atol=1e-4, rtol=0)
    np.testing.assert_allclose(map_y, expected[..., 1], atol=1e-4, rtol=0)
    assert map_x.dtype == map_y.dtype == np.float32
    assert map_x.flags.c_contiguous and map_y.flags.c_contiguous


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_identity_preserves_dtype_channels_and_pixels(dtype, channels):
    shape = (31, 47) if channels is None else (31, 47, channels)
    source = np.random.default_rng(4).integers(1, 255, shape).astype(dtype)
    warp = solve_tps_warp(grid((31, 47), 3), grid((31, 47), 3))
    output = warp.warp(source, shape[:2], grid_step=8)
    assert output.shape == shape
    assert output.dtype == source.dtype
    np.testing.assert_array_equal(output, source)


@pytest.mark.parametrize("step", [1, 8])
def test_translation_uses_backward_sampling_and_zero_border(step):
    points = grid((23, 31), 3)
    warp = solve_tps_warp(points, points + (5, 7))
    source = np.arange(23 * 31, dtype=np.uint16).reshape(23, 31) + 1
    output = warp.warp(source, (37, 53), grid_step=step)
    expected = np.zeros((37, 53), dtype=np.uint16)
    expected[7:30, 5:36] = source
    assert output.shape == (37, 53)
    np.testing.assert_array_equal(output, expected)


def test_fractional_translation_is_bicubic():
    points = grid((32, 48), 3)
    warp = solve_tps_warp(points, points + (0.375, -0.625))
    source = np.random.default_rng(7).uniform(0, 255, (32, 48)).astype(np.float32)
    xx, yy = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(32, dtype=np.float32))
    expected = cv2.remap(source, xx - 0.375, yy + 0.625, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    linear = cv2.remap(source, xx - 0.375, yy + 0.625, cv2.INTER_LINEAR)
    np.testing.assert_allclose(warp.warp(source, source.shape), expected, atol=1e-5)
    assert np.max(np.abs(expected - linear)) > 1


def test_large_canvas_uses_coarse_grid_with_small_interpolation_error(monkeypatch):
    shape = (1025, 1537)
    points = grid(shape)
    warp = solve_tps_warp(points, relief(points))
    evaluation_count = 0
    real_displacement = TPSWarp._displacement

    def counted(self, model, coordinates):
        nonlocal evaluation_count
        evaluation_count += len(coordinates)
        return real_displacement(self, model, coordinates)

    monkeypatch.setattr(TPSWarp, "_displacement", counted)
    started = perf_counter()
    map_x, map_y = warp.build_remap(shape)
    elapsed = perf_counter() - started
    assert evaluation_count == ((shape[0] + 7) // 8 + 2) * ((shape[1] + 7) // 8 + 2)
    assert evaluation_count < np.prod(shape) / 50
    rng = np.random.default_rng(81)
    xy = np.column_stack((rng.integers(0, shape[1], 500), rng.integers(0, shape[0], 500)))
    xy[:4] = [[0, 0], [shape[1] - 1, 0], [0, shape[0] - 1], [shape[1] - 1, shape[0] - 1]]
    actual = np.column_stack((map_x[xy[:, 1], xy[:, 0]], map_y[xy[:, 1], xy[:, 0]]))
    approximation_rmse = radial_rmse(actual, warp.backward(xy))
    logger.info("Large canvas %s: %d TPS samples for %d pixels | map time %.4f s | "
                "coarse-vs-dense RMSE %.6f px", shape, evaluation_count - len(xy),
                np.prod(shape), elapsed, approximation_rmse)
    assert approximation_rmse < 0.03


def test_large_raster_runtime_and_exact_dimensions():
    shape = (1024, 1024)
    points = grid(shape)
    source = np.full(shape, 100, dtype=np.uint16)
    started = perf_counter()
    warp = solve_tps_warp(points, relief(points))
    output = warp.warp(source, shape)
    elapsed = perf_counter() - started
    assert output.shape == shape
    assert np.all(output[10:-10, 10:-10] == 100)  # No unmapped interior holes.
    logger.info("1024x1024 uint16 raster: solve + maps + cubic remap %.4f s | "
                "target=%s actual=%s | sub-second observed=%s",
                elapsed, shape, output.shape, elapsed < 1.0)
    # Timing is reported, not a flaky hardware-dependent CI assertion.


def test_inputs_are_preserved_and_empty_queries_supported():
    points = grid()
    reference = relief(points)
    saved_points, saved_reference = points.copy(), reference.copy()
    warp = solve_tps_warp(points, reference)
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(reference, saved_reference)
    expected = warp.forward(saved_points)
    points[:] = 0
    reference[:] = 0
    np.testing.assert_array_equal(warp.forward(saved_points), expected)
    assert warp.forward(np.empty((0, 2))).shape == (0, 2)
    assert warp.backward(np.empty((0, 2))).shape == (0, 2)


def test_radial_rmse_is_not_per_axis_rmse():
    points = grid((4, 4), 3)
    warp = solve_tps_warp(points, points)
    assert warp.rmse(points, points + (3, 4)) == pytest.approx(5)
    with pytest.raises(ValueError, match="nonempty"):
        warp.rmse(np.empty((0, 2)), np.empty((0, 2)))
    with pytest.raises(ValueError, match="same"):
        warp.rmse(points, points[:-1])


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((4, 2)),
                                  np.array([[0, 0], [1, 1], [2, 2]]),
                                  np.ones((4, 3)), np.full((4, 2), np.nan),
                                  np.full((4, 2), np.inf), np.ones((4, 2), dtype=complex)])
def test_invalid_control_points_rejected_in_both_directions(bad):
    good = grid((3, 3), 2)
    for source, reference in ((bad, good), (good, bad), (bad, bad)):
        with pytest.raises(ValueError):
            solve_tps_warp(source, reference)


@pytest.mark.parametrize("smoothing", [-1, np.nan, np.inf, True, "0.05", [0.05]])
def test_invalid_smoothing_rejected(smoothing):
    points = grid()
    with pytest.raises(ValueError, match="smoothing"):
        solve_tps_warp(points, points, smoothing=smoothing)


@pytest.mark.parametrize("shape", [(0, 2), (2, -1), (2.5, 3), (True, 3),
                                    (2, 3, 4), (32767, 1), "32", None])
def test_invalid_canvases_rejected(shape):
    warp = solve_tps_warp(grid(), grid())
    with pytest.raises(ValueError):
        warp.build_remap(shape)


@pytest.mark.parametrize("step", [0, -1, 2, True, 8.0, "8"])
def test_invalid_grid_step_rejected(step):
    warp = solve_tps_warp(grid(), grid())
    with pytest.raises(ValueError, match="grid_step"):
        warp.build_remap((13, 23), grid_step=step)


@pytest.mark.parametrize("source", [np.zeros((0, 2)), np.zeros((2, 3), dtype=np.int32),
                                     np.zeros((2, 3), dtype=bool), np.zeros((2, 3, 5)),
                                     np.zeros((2, 3, 1, 1)), np.full((2, 3), np.nan),
                                     np.zeros((32767, 1), dtype=np.uint8)])
def test_invalid_rasters_rejected(source):
    warp = solve_tps_warp(grid(), grid())
    with pytest.raises(ValueError):
        warp.warp(source, (13, 23))


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "--log-cli-level=INFO", *sys.argv[1:],
    ]))
