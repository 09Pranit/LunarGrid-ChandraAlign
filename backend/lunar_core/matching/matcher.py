"""Classical and learned feature matching in original image pixel coordinates.

Neural dependencies are imported only when constructing LightGlueMatcher. No
engine estimates a transform, runs RANSAC, or uses ground-truth geometry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import ClassVar
import warnings

import cv2
import numpy as np


@dataclass
class MatchResult:
    """Correspondences in (x, y) order; confidences are engine-specific scores.

    SIFT scores are 1 - descriptor distance ratio, not probabilities. LightGlue
    scores are the network's match scores. They are not calibrated across engines.
    Timing covers image preparation and inference, excluding model construction.
    """

    pts_src: np.ndarray
    pts_ref: np.ndarray
    confidences: np.ndarray
    execution_time_ms: float
    engine_name: str

    def __post_init__(self) -> None:
        self.pts_src = np.array(self.pts_src, dtype=np.float32, copy=True)
        self.pts_ref = np.array(self.pts_ref, dtype=np.float32, copy=True)
        self.confidences = np.array(self.confidences, dtype=np.float32, copy=True)
        if self.pts_src.ndim != 2 or self.pts_src.shape[1] != 2:
            raise ValueError("pts_src must have shape (N, 2)")
        if self.pts_ref.shape != self.pts_src.shape:
            raise ValueError("pts_ref must have the same (N, 2) shape as pts_src")
        if self.confidences.shape != (len(self.pts_src),):
            raise ValueError("confidences must have shape (N,)")
        if not all(np.isfinite(a).all() for a in (self.pts_src, self.pts_ref, self.confidences)):
            raise ValueError("coordinates and confidences must be finite")
        if np.any((self.confidences < 0) | (self.confidences > 1)):
            raise ValueError("confidences must be in [0, 1]")
        self.execution_time_ms = float(self.execution_time_ms)
        if not np.isfinite(self.execution_time_ms) or self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must be finite and nonnegative")
        if not isinstance(self.engine_name, str) or not self.engine_name:
            raise ValueError("engine_name must be a nonempty string")

    @property
    def tie_points(self) -> np.ndarray:
        """Return (N, 4) columns [x_src, y_src, x_ref, y_ref]."""
        return np.concatenate((self.pts_src, self.pts_ref), axis=1)

    def __len__(self) -> int:
        return len(self.pts_src)


def _gray_float(image: np.ndarray) -> np.ndarray:
    """Validate a raw/Wallis array and return contiguous grayscale float32 [0, 1].

    Accept (H,W), (H,W,1), BGR and BGRA arrays (OpenCV channel order). Intensities
    must be [0,255], or [0,1] for normalized floating-point input. Sensor DN values
    beyond 255 must first be scaled by the caller, as in the Wallis/router APIs.
    No per-image contrast stretching or input mutation is performed.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a nonempty NumPy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("image must have shape (H,W), (H,W,1), (H,W,3), or (H,W,4)")
    if image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("image must contain finite real numeric intensities")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("image intensities must be in [0,255] or floating-point [0,1]")
    unit_range = image.dtype.kind == "f" and image.max() <= 1
    values = np.array(image, dtype=np.float32, order="C", copy=True)
    if not unit_range:
        values /= 255.0
    if values.ndim == 3:
        if values.shape[2] == 1:
            values = values[..., 0]
        else:
            conversion = cv2.COLOR_BGR2GRAY if values.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            values = cv2.cvtColor(values, conversion)
    return np.ascontiguousarray(values)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _threshold(value: float, name: str, *, inclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if not (0 <= value <= 1 if inclusive else 0 < value < 1):
        raise ValueError(f"{name} must be in {'[0,1]' if inclusive else '(0,1)'}")
    return float(value)


class BaseMatcher(ABC):
    """Match source pixels to reference pixels without geometric verification."""

    engine_name: ClassVar[str]

    @abstractmethod
    def match(self, image_src: np.ndarray, image_ref: np.ndarray) -> MatchResult:
        """Return correspondences; insufficient texture yields correctly shaped empties."""

    def __call__(self, image_src: np.ndarray, image_ref: np.ndarray) -> MatchResult:
        return self.match(image_src, image_ref)

    def _empty(self, started: float) -> MatchResult:
        return MatchResult(
            np.empty((0, 2)), np.empty((0, 2)), np.empty(0),
            (perf_counter() - started) * 1000, self.engine_name,
        )


class SIFTMatcher(BaseMatcher):
    """SIFT + KD-tree FLANN, Lowe ratio filtering and optional reciprocal checks.

    Reciprocal checks (enabled by default) require both directional ratio tests
    to agree. They reduce ambiguous repeated-crater matches using descriptors
    alone. Set mutual_check=False for the one-way Lowe pipeline.
    """

    engine_name = "sift_flann"

    def __init__(self, *, nfeatures: int = 10000, ratio_threshold: float = 0.75,
                 mutual_check: bool = True) -> None:
        self.nfeatures = _positive_integer(nfeatures, "nfeatures")
        self.ratio_threshold = _threshold(ratio_threshold, "ratio_threshold")
        if not isinstance(mutual_check, bool):
            raise ValueError("mutual_check must be a bool")
        self.mutual_check = mutual_check
        self.detector = cv2.SIFT_create(nfeatures=self.nfeatures, contrastThreshold=0.03)
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def _ratio_matches(self, source: np.ndarray, reference: np.ndarray) -> dict:
        accepted = {}
        for neighbors in self.matcher.knnMatch(source, reference, k=2):
            if len(neighbors) != 2:
                continue
            best, second = neighbors
            if best.distance < self.ratio_threshold * second.distance:
                accepted[best.queryIdx] = (best.trainIdx, 1.0 - best.distance / second.distance)
        return accepted

    def match(self, image_src: np.ndarray, image_ref: np.ndarray) -> MatchResult:
        started = perf_counter()
        src = np.rint(_gray_float(image_src) * 255).astype(np.uint8)
        ref = np.rint(_gray_float(image_ref) * 255).astype(np.uint8)
        key_src, desc_src = self.detector.detectAndCompute(src, None)
        key_ref, desc_ref = self.detector.detectAndCompute(ref, None)
        if desc_src is None or desc_ref is None or len(desc_ref) < 2:
            return self._empty(started)
        if self.mutual_check and len(desc_src) < 2:
            return self._empty(started)
        forward = self._ratio_matches(desc_src, desc_ref)
        reverse = self._ratio_matches(desc_ref, desc_src) if self.mutual_check else {}
        rows = []
        for source, (reference, confidence) in forward.items():
            if self.mutual_check:
                backward = reverse.get(reference)
                if backward is None or backward[0] != source:
                    continue
                confidence = min(confidence, backward[1])
            rows.append((*key_src[source].pt, *key_ref[reference].pt, confidence))
        if not rows:
            return self._empty(started)
        points = np.asarray(rows, dtype=np.float32)
        return MatchResult(points[:, :2], points[:, 2:4], points[:, 4],
                           (perf_counter() - started) * 1000, self.engine_name)


def _select_device(torch, requested):
    """Prefer CUDA, then MPS, then CPU; warn and use CPU for unavailable hardware."""
    mps = getattr(torch.backends, "mps", None)
    if requested is None or str(requested) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(requested)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("device must be auto, cpu, cuda[:index], or mps") from exc
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda[:index], or mps")
    available = True
    if device.type == "cuda":
        available = torch.cuda.is_available()
        if available and device.index is not None:
            available = device.index < torch.cuda.device_count()
    elif device.type == "mps":
        available = mps is not None and mps.is_available() and device.index in (None, 0)
    if not available:
        warnings.warn(f"Requested device {device} is unavailable; using CPU.", RuntimeWarning,
                      stacklevel=3)
        return torch.device("cpu")
    return device


class LightGlueMatcher(BaseMatcher):
    """Pretrained SuperPoint + LightGlue with CUDA/MPS/CPU device selection.

    Install the optional dependencies following README.md. First construction
    downloads upstream weights into the Torch cache (configure TORCH_HOME for an
    offline/prepopulated cache). Missing dependencies or weights fail explicitly;
    the learned engine never silently substitutes SIFT or random model weights.
    """

    engine_name = "superpoint_lightglue"

    def __init__(self, *, device: str = "auto", max_num_keypoints: int = 2048,
                 filter_threshold: float = 0.1) -> None:
        self.max_num_keypoints = _positive_integer(max_num_keypoints, "max_num_keypoints")
        self.filter_threshold = _threshold(filter_threshold, "filter_threshold", inclusive=True)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("LightGlueMatcher requires PyTorch; see the Phase 4 README setup.") from exc
        self._torch = torch
        self.device = _select_device(torch, device)
        try:
            # Kornia 0.8.3 imports legacy TorchScript helpers even though this
            # pipeline uses eager inference. Torch 2.13 deprecates that decorator.
            # Limit compatibility handling to this exact upstream import warning;
            # all inference/device and unrelated warnings remain visible.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning,
                    message=r"`torch\.jit\.script` is deprecated\..*",
                    module=r"torch\.jit\._script",
                )
                from lightglue import LightGlue, SuperPoint
        except ImportError as exc:
            raise ImportError(
                "LightGlueMatcher requires LightGlue and its optional dependencies; "
                "see the Phase 4 README setup. SIFTMatcher is available independently."
            ) from exc
        self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(
            features="superpoint", filter_threshold=self.filter_threshold,
            flash=self.device.type == "cuda", mp=False,
        ).eval().to(self.device)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self._torch.mps.synchronize()

    def match(self, image_src: np.ndarray, image_ref: np.ndarray) -> MatchResult:
        self._synchronize()
        started = perf_counter()
        src, ref = _gray_float(image_src), _gray_float(image_ref)
        # SuperPoint downsamples three times and cannot process an axis below 8.
        # Constant arrays contain no usable texture, regardless of learned bias.
        if any(min(a.shape) < 8 or np.ptp(a) == 0 for a in (src, ref)):
            return self._empty(started)
        torch = self._torch
        with torch.inference_mode():
            features = [self.extractor.extract(
                torch.from_numpy(a)[None].to(self.device), resize=None,
            ) for a in (src, ref)]
            if any(f["keypoints"].shape[1] == 0 for f in features):
                self._synchronize()
                return self._empty(started)
            prediction = self.matcher({"image0": features[0], "image1": features[1]})
            indices = prediction["matches"][0]
            pts_src = features[0]["keypoints"][0][indices[:, 0]].cpu().numpy()
            pts_ref = features[1]["keypoints"][0][indices[:, 1]].cpu().numpy()
            confidences = prediction["scores"][0].cpu().numpy()
        self._synchronize()
        return MatchResult(pts_src, pts_ref, confidences,
                           (perf_counter() - started) * 1000, self.engine_name)
