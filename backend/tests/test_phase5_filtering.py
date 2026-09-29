"""Phase 5 standalone acceptance tests; real OpenCV MAGSAC++ and SciPy Voronoi.

    python backend/tests/test_phase5_filtering.py

The 60%-outlier stress set MUST fail the 70% registration gate. Its exception
preserves the actual consensus mask for measuring outlier rejection separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.registration import (
    RegistrationRejected,
    SpatialGeometryError,
    bounded_voronoi_cells,
    compute_vsui,
    filter_magsac_and_vsui,
    grid_nms,
)
from lunar_core.registration import outlier_rejection as filtering

logger = logging.getLogger(__name__)
SHAPE = (800, 800)
KNOWN_H = np.array([[0.97, -0.02, 12], [0.015, 0.98, 8], [1e-5, -2e-5, 1.0]])


def uniform_points(shape=SHAPE, rows=8, columns=8):
    height, width = shape
    xx, yy = np.meshgrid((np.arange(columns) + 0.5) * width / columns,
                         (np.arange(rows) + 0.5) * height / rows)
    return np.column_stack((xx.ravel(), yy.ravel()))


def project(points, homography):
    transformed = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return transformed[:, :2] / transformed[:, 2, None]


def contaminated_matches(outlier_count, seed=73):
    rng = np.random.default_rng(seed)
    reference = uniform_points()
    source = project(reference, np.linalg.inv(KNOWN_H))
    source += rng.normal(0, 0.08, source.shape)
    source = np.vstack((source, rng.uniform(0, 800, (outlier_count, 2))))
    reference = np.vstack((reference, rng.uniform(0, 800, (outlier_count, 2))))
    outliers = np.arange(len(source)) >= 64
    # Outliers receive high scores too: confidence cannot identify ground truth.
    scores = rng.uniform(0.5, 1.0, len(source))
    order = rng.permutation(len(source))
    return source[order], reference[order], scores[order], outliers[order]


@pytest.mark.parametrize("seed", [73, 11, 29])
def test_required_sixty_percent_outliers_rejected(seed):
    source, reference, scores, true_outliers = contaminated_matches(96, seed)
    assert true_outliers.mean() == 0.60
    with pytest.raises(RegistrationRejected, match="inlier ratio") as rejected:
        filter_magsac_and_vsui(source, reference, scores, SHAPE)
    result = rejected.value.result
    removal = np.mean(~result.inlier_mask[true_outliers])
    recovery = np.mean(result.inlier_mask[~true_outliers])
    logger.info("60%% contamination seed=%d: synthetic outliers removed=%.2f%%; "
                "true inliers recovered=%.2f%%", seed, 100 * removal, 100 * recovery)
    assert removal >= 0.95
    assert recovery >= 0.95  # Prevent an all-rejected mask from passing the test.
    assert not result.accepted
    assert result.inlier_ratio < 0.70
    assert result.homography is not None
    error = np.linalg.norm(project(uniform_points(), result.homography)
                           - project(uniform_points(), KNOWN_H), axis=1)
    assert np.median(error) < 0.4
    assert 0 <= result.vsui <= 1


def test_required_accepted_registration_and_determinism():
    source, reference, scores, outliers = contaminated_matches(16)
    originals = [array.copy() for array in (source, reference, scores)]
    result = filter_magsac_and_vsui(source, reference, scores, SHAPE)
    again = filter_magsac_and_vsui(source, reference, scores, SHAPE)
    assert result.accepted and result.initial_count == 80
    assert result.inlier_count == result.retained_count == 64
    assert result.inlier_ratio == 0.8
    assert result.vsui >= 0.75
    assert 0 <= result.vsui <= 1
    assert result.vsui == again.vsui
    np.testing.assert_array_equal(result.inlier_mask, again.inlier_mask)
    np.testing.assert_array_equal(result.retained_mask, again.retained_mask)
    np.testing.assert_array_equal(result.homography, again.homography)
    assert not result.inlier_mask[outliers].any()
    assert np.all(result.retained_mask <= result.inlier_mask)
    np.testing.assert_array_equal(result.pts_src, source[result.retained_mask])
    np.testing.assert_array_equal(result.pts_ref, reference[result.retained_mask])
    np.testing.assert_array_equal(result.confidences, scores[result.retained_mask])
    np.testing.assert_array_equal(result.tie_points, np.column_stack((result.pts_src, result.pts_ref)))
    for original, array in zip(originals, (source, reference, scores)):
        np.testing.assert_array_equal(array, original)


def test_required_grid_nms_cluster_culling_and_uniform_preservation():
    uniform = uniform_points()
    cluster = np.random.default_rng(15).uniform(20, 70, (100, 2))  # 50x50, one cell
    points = np.vstack((uniform, cluster))
    scores = np.r_[np.ones(64), np.linspace(0.1, 0.9, 100)]
    kept = grid_nms(points, scores, SHAPE, top_k=3)
    np.testing.assert_array_equal(kept[:64], np.arange(64))
    # Uniform point consumes one of the cluster cell's three slots.
    np.testing.assert_array_equal(kept[64:], [162, 163])
    result = filter_magsac_and_vsui(points, points.copy(), scores, SHAPE)
    assert result.accepted and result.inlier_count == 164
    assert result.retained_count == 64 and result.inlier_ratio == 1
    np.testing.assert_array_equal(result.pts_ref, uniform)
    assert result.vsui > result.vsui_before
    assert result.vsui == pytest.approx(1.0)
    logger.info("Cluster Grid-NMS: localized cluster 100 -> 2 with K=3; "
                "uniform sites 64 -> 64; pipeline K=1 VSUI %.6f -> %.6f",
                result.vsui_before, result.vsui)


def test_grid_nms_highest_confidence_ties_duplicates_and_closed_edges():
    points = np.array([[10, 10], [20, 20], [30, 30], [800, 800],
                       [799, 799], [0, 800], [800, 0], [20, 20]], dtype=float)
    scores = np.array([0.1, 0.8, 0.8, 0.2, 0.9, 1, 1, 0.95])
    np.testing.assert_array_equal(grid_nms(points, scores, SHAPE), [4, 5, 6, 7])
    np.testing.assert_array_equal(grid_nms(points[:3], scores[:3], SHAPE), [1])
    np.testing.assert_array_equal(grid_nms(points[:3], scores[:3], SHAPE, top_k=2), [1, 2])
    kept = grid_nms(points, scores, SHAPE, top_k=2)
    assert 7 in kept and 1 not in kept  # duplicate cannot consume a second slot


def area(cell):
    # Independent triangle fan measurement (not the implementation's shoelace).
    a, b = cell[1:-1] - cell[0], cell[2:] - cell[0]
    return float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2)


@pytest.mark.parametrize("shape", [(800, 800), (300, 1200), (50, 4000)])
def test_voronoi_uniform_rectangular_canvas_areas(shape):
    points = uniform_points(shape)
    cells = bounded_voronoi_cells(points, shape)
    assert len(cells) == len(points)
    np.testing.assert_allclose([area(cell) for cell in cells], np.prod(shape) / 64, rtol=1e-9)
    assert compute_vsui(points, shape) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("points,expected_areas", [
    ([[0, 0], [0, 100], [100, 0], [100, 100]], [2500] * 4),
    ([[10, 50], [50, 50]], [3000, 7000]),
    ([[12.5, 50], [37.5, 50], [62.5, 50], [87.5, 50]], [2500] * 4),
    ([[0, 0]], [10000]),
    ([[20, 50], [20, 50], [80, 50]], [5000, 5000]),
])
def test_voronoi_analytic_boundary_sparse_collinear_and_duplicate_sites(points, expected_areas):
    cells = bounded_voronoi_cells(np.array(points, dtype=float), (100, 100))
    np.testing.assert_allclose([area(cell) for cell in cells], expected_areas, rtol=1e-10)
    for cell in cells:
        assert np.isfinite(cell).all()
        assert np.all((cell >= 0) & (cell <= 100))


def test_voronoi_unequal_area_formula_is_not_an_occupancy_score():
    points = np.array([[10, 50], [50, 50]], dtype=float)
    expected = 1 - 2000 / (5000 + 1e-6)
    assert compute_vsui(points, (100, 100)) == pytest.approx(expected, abs=1e-12)


def test_voronoi_random_partition_nearest_sites_and_determinism():
    shape = (217, 953)
    points = np.random.default_rng(3).uniform((0, 0), (953, 217), (50, 2))
    points = np.vstack((points, [[0, 0], [953, 0], [0, 217], [953, 217], [953, 80]]))
    ordered_sites = np.unique(points, axis=0)
    cells = bounded_voronoi_cells(points, shape)
    repeated = bounded_voronoi_cells(points[::-1], shape)
    assert sum(map(area, cells)) == pytest.approx(217 * 953, rel=1e-10)
    for site, cell, again in zip(ordered_sites, cells, repeated):
        np.testing.assert_array_equal(cell, again)
        assert np.all((cell >= 0) & (cell <= [953, 217]))
        own_distance = np.sum((cell - site) ** 2, axis=1)
        all_distances = np.sum((cell[:, None, :] - ordered_sites[None, :, :]) ** 2, axis=2)
        # Vertices of a true Voronoi cell are closest to that cell's generator.
        assert np.all(own_distance[:, None] <= all_distances + 1e-7)
    score = compute_vsui(points, shape)
    assert 0 <= score <= 1
    assert score == compute_vsui(points[::-1], shape)


def test_empty_vsui_and_negative_raw_formula_clamp():
    empty = np.empty((0, 2))
    assert bounded_voronoi_cells(empty, SHAPE) == ()
    assert compute_vsui(empty, SHAPE) == 0
    assert grid_nms(empty, np.empty(0), SHAPE).shape == (0,)
    cluster = np.random.default_rng(4).uniform(5, 10, (100, 2))
    areas = np.array([area(cell) for cell in bounded_voronoi_cells(cluster, SHAPE)])
    assert 1 - areas.std() / (areas.mean() + 1e-6) < 0
    assert compute_vsui(cluster, SHAPE) == 0


@pytest.mark.parametrize("count", [0, 1, 3, 14])
def test_too_few_points_reject_even_if_small_set_vsui_is_high(count):
    points = uniform_points()[:count]
    with pytest.raises(RegistrationRejected, match="valid inlier count") as rejected:
        filter_magsac_and_vsui(points, points, np.ones(count), SHAPE)
    result = rejected.value.result
    assert not result.accepted and result.retained_count < 15
    assert result.pts_src.shape == result.pts_ref.shape == (result.retained_count, 2)
    assert 0 <= result.vsui <= 1


def test_fifteen_point_count_boundary_is_accepted():
    points = uniform_points(rows=3, columns=5)
    result = filter_magsac_and_vsui(points, points, np.ones(15), SHAPE)
    assert result.accepted and result.retained_count == 15
    assert result.vsui == pytest.approx(1)


def test_seventy_percent_ratio_boundary_is_accepted():
    rng = np.random.default_rng(101)
    truth = uniform_points(rows=5, columns=7)  # 35 real pairs of 50 -> exactly 70%
    source = np.vstack((truth, rng.uniform(0, 800, (15, 2))))
    reference = np.vstack((truth, rng.uniform(0, 800, (15, 2))))
    result = filter_magsac_and_vsui(source, reference, np.ones(50), SHAPE)
    assert result.inlier_ratio == 0.70 and result.retained_count == 35


def test_count_gate_is_rechecked_after_grid_nms():
    points = np.random.default_rng(91).uniform(10, 60, (100, 2))
    with pytest.raises(RegistrationRejected, match="valid inlier count after Grid-NMS 1") as rejected:
        filter_magsac_and_vsui(points, points, np.ones(100), SHAPE)
    assert rejected.value.result.inlier_count == 100
    assert rejected.value.result.inlier_ratio == 1


def test_low_vsui_rejects_despite_sufficient_inliers():
    # 32 occupied cells in only half of the tile: NMS cannot repair coverage.
    points = uniform_points().reshape(8, 8, 2)[:, :4].reshape(-1, 2)
    with pytest.raises(RegistrationRejected, match="VSUI") as rejected:
        filter_magsac_and_vsui(points, points, np.ones(len(points)), SHAPE)
    assert rejected.value.result.inlier_count == rejected.value.result.retained_count == 32
    assert rejected.value.result.inlier_ratio == 1
    assert rejected.value.result.vsui < 0.75


def test_duplicate_endpoints_do_not_inflate_consensus():
    # Four non-collinear unique pairs could fit a model, but repeating each 25
    # times must not pass either the 15-point or 70% gate.
    points = np.tile(uniform_points(rows=2, columns=2), (25, 1))
    with pytest.raises(RegistrationRejected) as rejected:
        filter_magsac_and_vsui(points, points, np.ones(100), SHAPE, top_k=100)
    assert rejected.value.result.inlier_count <= 4
    assert rejected.value.result.inlier_ratio <= 0.04


def test_collinear_geometry_rejected():
    points = np.column_stack((np.linspace(0, 800, 64), np.linspace(0, 800, 64)))
    with pytest.raises(RegistrationRejected, match="non-collinear"):
        filter_magsac_and_vsui(points, points, np.ones(64), SHAPE)


@pytest.mark.parametrize("bad_shape", [(0, 800), (-1, 800), (True, 800), (1.5, 800), (800,), (800, 800, 3)])
def test_invalid_canvas_shapes(bad_shape):
    with pytest.raises(ValueError):
        compute_vsui(uniform_points(), bad_shape)


@pytest.mark.parametrize("bad_points", [
    [[np.nan, 1]], [[1, np.inf]], [[-1, 0]], [[801, 0]], [1, 2],
    [[1j, 2]], [[True, False]], [["1", "2"]],
])
def test_invalid_spatial_points(bad_points):
    with pytest.raises(ValueError):
        compute_vsui(np.asarray(bad_points), SHAPE)


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5])
def test_invalid_grid_configuration(bad_k):
    with pytest.raises(ValueError):
        grid_nms(uniform_points(), np.ones(64), SHAPE, top_k=bad_k)
    with pytest.raises(ValueError):
        grid_nms(uniform_points(), np.ones(64), SHAPE, grid_size=bad_k)


@pytest.mark.parametrize("bad_scores", [np.ones(63), np.ones((64, 1)), np.full(64, np.nan),
                                      np.full(64, -0.1), np.full(64, 1.1)])
def test_invalid_confidences(bad_scores):
    with pytest.raises(ValueError, match="confidences"):
        filter_magsac_and_vsui(uniform_points(), uniform_points(), bad_scores, SHAPE)


def test_misaligned_coordinates_and_optional_source_canvas():
    points = uniform_points()
    with pytest.raises(ValueError, match="same"):
        filter_magsac_and_vsui(points, points[:-1], np.ones(64), SHAPE)
    with pytest.raises(ValueError, match="pts_src"):
        filter_magsac_and_vsui(points + 800, points, np.ones(64), SHAPE, source_shape=SHAPE)
    result = filter_magsac_and_vsui(points + 800, points, np.ones(64), SHAPE,
                                   source_shape=(1600, 1600))
    assert result.accepted


def test_missing_magsac_fails_without_ransac_fallback(monkeypatch):
    monkeypatch.delattr(cv2, "USAC_MAGSAC")
    with pytest.raises(RuntimeError, match="USAC_MAGSAC"):
        filter_magsac_and_vsui(uniform_points(), uniform_points(), np.ones(64), SHAPE)


@pytest.mark.parametrize("homography", [None, np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_failed_homography_cannot_be_accepted(monkeypatch, homography):
    monkeypatch.setattr(cv2, "findHomography", lambda *args, **kwargs: (homography, np.ones((64, 1))))
    with pytest.raises(RegistrationRejected, match="homography") as rejected:
        filter_magsac_and_vsui(uniform_points(), uniform_points(), np.ones(64), SHAPE)
    assert rejected.value.result.homography is None


def test_voronoi_failure_rejects_registration(monkeypatch):
    def fail(*args, **kwargs):
        raise SpatialGeometryError("Voronoi partition failed")
    monkeypatch.setattr(filtering, "compute_vsui", fail)
    with pytest.raises(RegistrationRejected, match="Voronoi partition"):
        filter_magsac_and_vsui(uniform_points(), uniform_points(), np.ones(64), SHAPE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "--log-cli-level=INFO", *sys.argv[1:],
    ]))
