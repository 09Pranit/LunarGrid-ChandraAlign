"""Phase 4 acceptance checks using real pretrained networks (no skips/mocks).

    python backend/tests/test_phase4_matching.py

Install the Phase 4 optional dependencies from README.md first. The first neural
construction needs network access to download weights unless TORCH_HOME already
contains them. Ground truth is used only for scoring *all* returned tie points.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_core.matching import BaseMatcher, LightGlueMatcher, MatchResult, SIFTMatcher
from lunar_core.matching.matcher import _select_device
from lunar_core.preprocess.wallis import apply_wallis_filter

logger = logging.getLogger(__name__)


def synthetic_crater_image(size: int = 640, seed: int = 42) -> np.ndarray:
    """Seeded crater field with irregular illuminated rims and multiscale regolith."""
    rng = np.random.default_rng(seed)
    fine = rng.normal(0, 1, (size, size)).astype(np.float32)
    terrain = 110 + 18 * cv2.GaussianBlur(fine, (0, 0), 0.8)
    coarse = rng.normal(0, 1, (40, 40)).astype(np.float32)
    terrain += 12 * cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(220):
        cx, cy = rng.uniform(20, size - 20, 2)
        radius = rng.uniform(5, 29)
        reach = int(np.ceil(1.6 * radius))
        x0, x1 = max(0, int(cx) - reach), min(size, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(size, int(cy) + reach + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = (xx - cx) / radius, (yy - cy) / (radius * rng.uniform(0.8, 1.2))
        angle = np.arctan2(dy, dx)
        distance = np.hypot(dx, dy)
        distance *= 1 + 0.06 * np.sin(rng.integers(3, 8) * angle + rng.uniform(0, 6.28))
        rim = np.exp(-((distance - 1) / rng.uniform(0.08, 0.16)) ** 2)
        bowl = np.exp(-(distance / 0.72) ** 4)
        illumination = np.cos(angle - 0.7)
        amplitude = rng.uniform(32, 65)
        terrain[y0:y1, x0:x1] += amplitude * (rim * (0.65 + illumination) - 0.45 * bowl)
    return np.clip(terrain, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def crater_pair():
    source = synthetic_crater_image()
    height, width = source.shape
    # OpenCV rotation convention, centred on the image; added translation is
    # exactly (+30, -20), with the centre compensation retained in matrix[:, 2].
    affine = cv2.getRotationMatrix2D((width / 2, height / 2), 15.0, 1.2)
    affine[:, 2] += (30.0, -20.0)
    reference = cv2.warpAffine(source, affine, (width, height), flags=cv2.INTER_LINEAR)
    return source, reference, affine


@pytest.fixture(scope="module", params=["sift", "lightglue"])
def engine(request):
    # Required neural test is explicitly CPU, even on a GPU-equipped CI host.
    if request.param == "sift":
        return SIFTMatcher()
    return LightGlueMatcher(device="cpu", max_num_keypoints=2048)


def assert_contract(result, engine_name):
    assert isinstance(result, MatchResult)
    count = len(result)
    assert result.pts_src.shape == result.pts_ref.shape == (count, 2)
    assert result.confidences.shape == (count,)
    assert result.tie_points.shape == (count, 4)
    for values in (result.pts_src, result.pts_ref, result.confidences, result.tie_points):
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
    np.testing.assert_array_equal(result.tie_points[:, :2], result.pts_src)
    np.testing.assert_array_equal(result.tie_points[:, 2:], result.pts_ref)
    assert np.all((result.confidences >= 0) & (result.confidences <= 1))
    assert np.isfinite(result.execution_time_ms) and result.execution_time_ms >= 0
    assert result.engine_name == engine_name


@pytest.mark.parametrize("conditioned", [False, True], ids=["raw", "wallis"])
def test_required_affine_acceptance(engine, crater_pair, conditioned, monkeypatch):
    source, reference, affine = crater_pair
    if conditioned:
        source, reference = apply_wallis_filter(source), apply_wallis_filter(reference)
    saved_source, saved_reference = source.copy(), reference.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Acceptance must run before any RANSAC/geometric filtering")

    for name in ("findHomography", "estimateAffine2D", "estimateAffinePartial2D", "findFundamentalMat"):
        monkeypatch.setattr(cv2, name, forbidden)
    if isinstance(engine, LightGlueMatcher):
        monkeypatch.setattr(engine._torch.cuda, "synchronize", forbidden)
        monkeypatch.setattr(engine._torch.mps, "synchronize", forbidden)
        assert engine.device.type == "cpu"
        assert not engine.extractor.training and not engine.matcher.training
    cv2.setRNGSeed(42)
    result = engine.match(source, reference)
    assert_contract(result, engine.engine_name)
    assert len(result) >= 50
    projected = result.pts_src.astype(np.float64) @ affine[:, :2].T + affine[:, 2]
    errors = np.linalg.norm(projected - result.pts_ref, axis=1)
    mean_error = float(errors.mean())
    logger.info("%s %s: matches=%d mean_transfer_error=%.4f px time=%.2f ms",
                engine.engine_name, "wallis" if conditioned else "raw", len(result),
                mean_error, result.execution_time_ms)
    assert mean_error < 2.0, f"All-match mean transfer error: {mean_error:.4f} px"
    np.testing.assert_array_equal(source, saved_source)
    np.testing.assert_array_equal(reference, saved_reference)


@pytest.mark.parametrize("shape", [(64, 64), (7, 5), (64, 64, 3)])
def test_textureless_images_return_correct_empty_shapes(engine, shape):
    result = engine(np.zeros(shape, dtype=np.uint8), np.full(shape, 120, dtype=np.uint8))
    assert_contract(result, engine.engine_name)
    assert len(result) == 0


@pytest.mark.parametrize("representation", ["unit_float", "byte_float", "bgr", "bgra", "single_channel", "strided"])
def test_input_representations_and_original_coordinates(engine, representation):
    image = synthetic_crater_image(size=240)
    if representation == "unit_float":
        image = image.astype(np.float32) / 255
    elif representation == "byte_float":
        image = image.astype(np.float64)
    elif representation == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif representation == "bgra":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif representation == "single_channel":
        image = image[..., None]
    elif representation == "strided":
        image = image[:, ::-1]
    original = image.copy()
    result = engine(image, image.copy())
    assert_contract(result, engine.engine_name)
    assert len(result) >= 20
    np.testing.assert_allclose(result.pts_src, result.pts_ref, atol=0.01)
    for points in (result.pts_src, result.pts_ref):
        assert np.all((points[:, 0] >= 0) & (points[:, 0] < image.shape[1]))
        assert np.all((points[:, 1] >= 0) & (points[:, 1] < image.shape[0]))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("bad", [
    None, [[1, 2]], np.empty((0, 2)), np.zeros(4), np.zeros((2, 2, 2)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1]]),
    np.array([[256]]), np.array([[True]]), np.array([[1j]]),
])
def test_invalid_images_fail_clearly(engine, bad):
    with pytest.raises(ValueError):
        engine(bad, np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError):
        engine(np.zeros((64, 64), dtype=np.uint8), bad)


def test_base_class_is_abstract_and_result_rejects_misaligned_arrays():
    with pytest.raises(TypeError):
        BaseMatcher()
    with pytest.raises(ValueError, match="pts_ref"):
        MatchResult(np.zeros((2, 2)), np.zeros((1, 2)), np.ones(2), 0, "test")
    with pytest.raises(ValueError, match="confidences"):
        MatchResult(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 1)), 0, "test")


def test_sparse_flann_neighbors_and_equal_descriptor_distances():
    matcher = SIFTMatcher()
    # FLANN can return fewer than k neighbors; identical second-best descriptors
    # must be rejected as ambiguous without dividing by a zero distance.
    matcher.matcher = SimpleNamespace(knnMatch=lambda *args, **kwargs: [
        (), (cv2.DMatch(0, 0, 0.0),),
        (cv2.DMatch(0, 0, 0.0), cv2.DMatch(0, 1, 0.0)),
    ])
    assert matcher._ratio_matches(np.zeros((2, 128), np.float32),
                                  np.zeros((2, 128), np.float32)) == {}


def test_learned_no_accepted_matches_preserves_empty_shapes(engine, monkeypatch):
    if not isinstance(engine, LightGlueMatcher):
        # Exercise the real classical detector on texture versus a blank image.
        result = engine(synthetic_crater_image(240), np.zeros((240, 240), np.uint8))
    else:
        # Still executes the real extractor and attention network; this score
        # threshold deliberately rejects all matches at the final network filter.
        monkeypatch.setattr(engine.matcher.conf, "filter_threshold", 1.0)
        image = synthetic_crater_image(240)
        result = engine(image, image.copy())
    assert_contract(result, engine.engine_name)
    assert len(result) == 0


@pytest.mark.parametrize("options", [
    {"nfeatures": 0}, {"nfeatures": True}, {"ratio_threshold": 0},
    {"ratio_threshold": 1}, {"ratio_threshold": np.nan}, {"mutual_check": "yes"},
])
def test_sift_invalid_configuration(options):
    with pytest.raises(ValueError):
        SIFTMatcher(**options)


@pytest.mark.parametrize("options", [
    {"max_num_keypoints": 0}, {"max_num_keypoints": True},
    {"filter_threshold": -0.1}, {"filter_threshold": np.nan},
])
def test_neural_invalid_configuration(options):
    with pytest.raises(ValueError):
        LightGlueMatcher(**options)


@pytest.mark.parametrize("cuda,mps,requested,expected,fallback", [
    (True, True, "auto", "cuda", False), (False, True, "auto", "mps", False),
    (False, False, "auto", "cpu", False), (True, True, "cpu", "cpu", False),
    (False, False, "cuda", "cpu", True), (False, False, "mps", "cpu", True),
    (True, False, "cuda:1", "cuda:1", False), (True, False, "cuda:9", "cpu", True),
])
def test_device_selection_without_requiring_accelerator(cuda, mps, requested, expected, fallback):
    import torch

    # Only availability probes are simulated; neural acceptance always uses the
    # real pretrained models. This cannot claim actual CUDA/MPS execution.
    capabilities = SimpleNamespace(
        device=torch.device,
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 2),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    if fallback:
        with pytest.warns(RuntimeWarning, match="using CPU"):
            selected = _select_device(capabilities, requested)
    else:
        selected = _select_device(capabilities, requested)
    assert str(selected) == expected


def test_classical_import_and_execution_without_neural_dependencies():
    backend = str(Path(__file__).resolve().parents[1])
    script = """
import importlib.abc
import sys
class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'lightglue', 'kornia'}:
            raise ImportError('neural dependency deliberately unavailable')
sys.meta_path.insert(0, BlockNeural())
import numpy as np
from lunar_core.matching import SIFTMatcher, LightGlueMatcher
result = SIFTMatcher().match(np.zeros((32,32), dtype=np.uint8), np.zeros((32,32), dtype=np.uint8))
assert result.tie_points.shape == (0,4)
try:
    LightGlueMatcher()
except ImportError as error:
    assert 'requires PyTorch' in str(error)
else:
    raise AssertionError('Missing neural dependencies must fail explicitly')
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=backend,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        str(Path(__file__).resolve()), "-q", "--log-cli-level=INFO", *sys.argv[1:],
    ]))
