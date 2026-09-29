"""Phase 3 deterministic tests, also executable with visible CLI metric logs.

    python backend/tests/test_phase3_router.py
    python -m pytest backend/tests/test_phase3_router.py -q --log-cli-level=INFO
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
from pydantic import ValidationError
import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.io.pds4_parser import LunarTelemetryMetadata
from lunar_core.matching import router
from lunar_core.matching.router import (
    RouterConfig,
    RoutingDecision,
    compute_geometric_disparity,
    compute_illumination_entropy,
    load_router_config,
    route_pair,
)


def telemetry(gsd=1.0, incidence=20.0):
    return LunarTelemetryMetadata(
        gsd_meters=gsd,
        incidence_angle_deg=incidence,
        emission_angle_deg=0.0,
        solar_azimuth_deg=0.0,
    )


@pytest.fixture
def image():
    # Sixteen equiprobable, non-null intensities have exactly 4 bits of entropy.
    return np.tile(np.arange(1, 17, dtype=np.uint8), (16, 1))


@pytest.mark.parametrize(
    "gsd_src,gsd_ref,inc_src,inc_ref,expected_g,expected_engine",
    [
        (1.0, 1.0, 20.0, 20.0, 0.0, "sift_flann"),
        (5.0, 0.25, 20.0, 20.0, 9.5, "superpoint_lightglue"),
        (1.0, 1.0, 15.0, 80.0, 32.5, "superpoint_lightglue"),
    ],
    ids=["A_identical_pair", "B_20x_GSD", "C_65_degree_incidence"],
)
def test_required_decision_scenarios(
    image, caplog, gsd_src, gsd_ref, inc_src, inc_ref, expected_g, expected_engine
):
    with caplog.at_level(logging.INFO, logger=router.__name__):
        decision = route_pair(image, image.copy(), telemetry(gsd_src, inc_src), telemetry(gsd_ref, inc_ref))
    assert isinstance(decision, RoutingDecision)
    assert decision.entropy_src == decision.entropy_ref == 4.0
    assert decision.delta_h == 0.0
    assert decision.delta_g == expected_g
    assert decision.selected_engine == expected_engine
    assert decision.gsd_ratio == gsd_src / gsd_ref
    assert decision.incidence_delta_deg == abs(inc_src - inc_ref)
    assert len(decision.rationales) == 3
    assert ("<=" if expected_g <= 3 else ">") in decision.rationales[1]
    assert expected_engine in decision.rationales[2]
    assert f"delta_g={expected_g:.6f}" in caplog.text
    assert f"engine={expected_engine}" in caplog.text
    assert "H_src=4.000000 H_ref=4.000000 delta_h=0.000000" in caplog.text


@pytest.mark.parametrize("incidence", [20.0, 85.0], ids=["entropy_only", "both_metrics"])
def test_illumination_disparity_routes_deep(image, incidence):
    decision = route_pair(image, np.full_like(image, 60), telemetry(incidence=incidence), telemetry())
    assert decision.delta_h == 4.0
    assert decision.delta_g == (0.0 if incidence == 20 else 32.5)
    assert decision.selected_engine == "superpoint_lightglue"
    assert ">" in decision.rationales[0]


@pytest.mark.parametrize("bright_count,expected", [(3, "sift_flann"), (4, "superpoint_lightglue")])
def test_entropy_on_each_side_of_default_point_eight(bright_count, expected):
    source = np.full((1, 16), 10, dtype=np.uint8)
    source[:, :bright_count] = 200
    decision = route_pair(source, np.full_like(source, 10), telemetry(), telemetry())
    # Independent binary entropy: 3/16 -> 0.6962 bits; 4/16 -> 0.8113 bits.
    p = bright_count / 16
    expected_h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    assert decision.delta_h == pytest.approx(expected_h)
    assert decision.selected_engine == expected


@pytest.mark.parametrize("incidence,expected", [
    (6.0 - 1e-9, "sift_flann"), (6.0, "sift_flann"), (6.0 + 1e-9, "superpoint_lightglue"),
])
def test_geometry_threshold_is_inclusive_without_rounding(image, incidence, expected):
    decision = route_pair(image, image, telemetry(incidence=incidence), telemetry(incidence=0.0))
    assert decision.selected_engine == expected


@pytest.mark.parametrize("limit,expected", [
    (np.nextafter(1.0, 0.0), "superpoint_lightglue"),
    (1.0, "sift_flann"),
    (np.nextafter(1.0, np.inf), "sift_flann"),
])
def test_entropy_threshold_is_inclusive_with_geometry_at_limit(limit, expected):
    decision = route_pair(
        np.array([[1, 2]], dtype=np.uint8), np.array([[1]], dtype=np.uint8),
        telemetry(gsd=7.0), telemetry(gsd=1.0),
        config=RouterConfig(entropy_threshold=float(limit)),
    )
    assert decision.delta_h == 1.0
    assert decision.delta_g == 3.0
    assert decision.selected_engine == expected


def test_directional_gsd_formula_is_preserved(image):
    assert compute_geometric_disparity(5.0, 0.25, 20, 20) == 9.5
    assert compute_geometric_disparity(0.25, 5.0, 20, 20) == 0.475
    reverse = route_pair(image, image, telemetry(0.25), telemetry(5.0))
    assert reverse.delta_g == 0.475
    assert reverse.selected_engine == "sift_flann"
    assert compute_geometric_disparity(4.0, 1.0, 20, 30, w1=0.25, w2=2.0) == 20.75


def test_entropy_known_distribution_and_zero_padding():
    # P(10)=1/4, P(20)=3/4; zeros have no contribution or normalization weight.
    original = np.array([[0, 10, 20], [0, 20, 20]], dtype=np.uint8)
    saved = original.copy()
    expected = -0.25 * np.log2(0.25) - 0.75 * np.log2(0.75)
    assert compute_illumination_entropy(original) == pytest.approx(expected)
    assert compute_illumination_entropy(np.pad(original, 100)) == pytest.approx(expected)
    np.testing.assert_array_equal(original, saved)
    assert compute_illumination_entropy(np.array([[0, 255]], dtype=np.uint8)) == 0.0


def test_null_padding_does_not_change_routing(image):
    decision = route_pair(image, np.pad(image, 100), telemetry(), telemetry())
    assert decision.delta_h == decision.delta_g == 0.0
    assert decision.selected_engine == "sift_flann"


def test_float_bins_preserve_positive_subunit_samples_and_strided_images():
    image = np.array([[0.0, 0.25, 0.75, 1.0, 1.75, 255.0, 255.0]])
    assert compute_illumination_entropy(image) == pytest.approx(np.log2(3))
    all_levels = np.arange(1, 256, dtype=np.uint8)[None, :]
    assert compute_illumination_entropy(all_levels) == pytest.approx(np.log2(255))
    assert compute_illumination_entropy(all_levels[:, ::2]) == 7.0


@pytest.mark.parametrize("image", [
    np.array([]), np.zeros((2, 2, 3)), np.zeros((2, 2)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1]]),
    np.array([[256]]), np.array([[True]]), np.array([[1j]]), [[1, 2]],
])
def test_invalid_or_all_null_images_are_rejected(image):
    with pytest.raises(ValueError):
        compute_illumination_entropy(image)


@pytest.mark.parametrize("null_source", [True, False])
def test_all_null_pair_member_cannot_produce_a_decision(image, null_source):
    nulls = np.zeros_like(image)
    with pytest.raises(ValueError, match="no nonzero pixels"):
        route_pair(nulls if null_source else image, image if null_source else nulls, telemetry(), telemetry())


@pytest.mark.parametrize("overrides", [
    {"gsd_src": 0}, {"gsd_ref": 0}, {"gsd_src": -1}, {"gsd_ref": np.inf},
    {"gsd_src": np.nan}, {"gsd_src": True}, {"incidence_src_deg": -1},
    {"incidence_ref_deg": 91}, {"incidence_ref_deg": np.nan},
    {"w1": -1}, {"w2": np.inf}, {"w2": True},
    {"gsd_src": 1e308, "gsd_ref": 1e-308},
    {"gsd_src": 1e-308, "gsd_ref": 1e308},
    {"incidence_src_deg": 90, "w2": 1e308},
])
def test_invalid_geometry_is_rejected(overrides):
    values = dict(gsd_src=1.0, gsd_ref=1.0, incidence_src_deg=20.0, incidence_ref_deg=20.0)
    values.update(overrides)
    with pytest.raises(ValueError):
        compute_geometric_disparity(**values)


def test_router_rejects_nonfinite_phase1_gsd(image):
    # Ingestion now rejects non-finite telemetry before the router.
    with pytest.raises(ValueError, match="finite"):
        route_pair(image, image, telemetry(gsd=np.inf), telemetry())


def test_shipped_configuration_and_cwd_independence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_router_config() == RouterConfig()
    assert load_router_config().entropy_threshold == 0.8
    assert load_router_config().geometry_threshold == 3.0


@pytest.mark.parametrize("field,value,expected", [
    ("geometry_threshold", 1.0, "sift_flann"),
    ("geometry_threshold", 0.999, "superpoint_lightglue"),
    ("entropy_threshold", 0.999, "superpoint_lightglue"),
])
def test_yaml_thresholds_and_weights_control_routing(tmp_path, monkeypatch, field, value, expected):
    settings = dict(entropy_threshold=1.0, geometry_threshold=1.0, w1=0.25, w2=0.125)
    settings[field] = value
    path = tmp_path / "config.yaml"
    path.write_text("routing:\n" + "".join(f"  {key}: {val}\n" for key, val in settings.items()), encoding="utf-8")
    monkeypatch.setattr(router, "DEFAULT_CONFIG_PATH", path)
    decision = route_pair(
        np.array([[1, 2]], dtype=np.uint8), np.array([[1]], dtype=np.uint8),
        telemetry(3.0, 24.0), telemetry(1.0, 20.0),
    )
    assert decision.delta_h == decision.delta_g == 1.0
    assert decision.selected_engine == expected
    assert decision.config == load_router_config(path)
    # File changes are picked up on the next call; omitted fields use defaults.
    path.write_text("routing:\n  entropy_threshold: 2.0\n", encoding="utf-8")
    assert load_router_config().entropy_threshold == 2.0
    assert load_router_config().w1 == load_router_config().w2 == 0.5


@pytest.mark.parametrize("content", [
    "", "[]", "unrelated: {}", "routing: null", "routing: []", "routing: [",
    "routing: {entropy_threshold: -1}", "routing: {geometry_threshold: .nan}",
    "routing: {w1: -0.1}", "routing: {w2: .inf}", "routing: {w1: true}",
    "routing: {entropy_threshold: '0.8'}", "routing: {entropy_treshold: 0.8}",
    "routing: !!python/object:builtins.object {}",
])
def test_invalid_yaml_configuration_is_rejected(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_router_config(path)


def test_missing_configuration_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_router_config(tmp_path / "missing.yaml")


def test_decision_is_deterministic_serializable_and_preserves_inputs(image):
    saved = image.copy()
    source, reference = telemetry(), telemetry()
    first = route_pair(image, image, source, reference)
    second = route_pair(image, image, source, reference)
    assert first == second
    assert RoutingDecision.model_validate_json(first.model_dump_json()) == first
    payload = json.loads(first.model_dump_json())
    assert payload["selected_engine"] == "sift_flann"
    assert payload["config"]["w1"] == 0.5
    assert len(payload["rationales"]) == 3
    with pytest.raises(ValidationError):
        first.selected_engine = "superpoint_lightglue"
    np.testing.assert_array_equal(image, saved)


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "--log-cli-level=INFO",
        "--log-cli-format=%(levelname)s %(message)s", *sys.argv[1:],
    ]))
