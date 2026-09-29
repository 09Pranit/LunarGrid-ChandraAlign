"""Deterministic matcher selection from illumination and orbital geometry.

This module selects an engine; feature extraction and matcher execution belong
to their respective engines. GSD disparity uses the specified directional
source/reference ratio, and incidence angles remain in degrees.
"""

from __future__ import annotations

import logging
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import entropy
import yaml

from ..io.pds4_parser import LunarTelemetryMetadata

logger = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
EngineName = Literal["sift_flann", "superpoint_lightglue"]


class RouterConfig(BaseModel):
    """Inclusive classical-routing limits and nonnegative geometry weights."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)

    entropy_threshold: float = Field(0.8, ge=0.0, description="Maximum delta H in bits")
    geometry_threshold: float = Field(3.0, ge=0.0, description="Maximum weighted delta G")
    w1: float = Field(0.5, ge=0.0, description="Weight of abs(GSD_src/GSD_ref - 1)")
    w2: float = Field(0.5, ge=0.0, description="Weight of incidence difference in degrees")


class RoutingDecision(BaseModel):
    """JSON-serializable metrics, configuration snapshot and routing reasons."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    entropy_src: float = Field(..., ge=0.0, le=8.0, description="Source entropy in bits")
    entropy_ref: float = Field(..., ge=0.0, le=8.0, description="Reference entropy in bits")
    delta_h: float = Field(..., ge=0.0, le=8.0)
    gsd_ratio: float = Field(..., gt=0.0, description="GSD_src / GSD_ref")
    incidence_delta_deg: float = Field(..., ge=0.0, le=90.0)
    delta_g: float = Field(..., ge=0.0)
    selected_engine: EngineName
    rationales: tuple[str, ...] = Field(..., min_length=2)
    config: RouterConfig


def load_router_config(path: str | Path | None = None) -> RouterConfig:
    """Read the ``routing`` section of backend/config.yaml or an explicit file.

    Omitted settings use RouterConfig defaults. Missing files, a missing routing
    mapping, unknown routing keys, and invalid values fail explicitly. Other
    top-level sections are left available for the rest of the backend.
    """
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid router YAML in {config_path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("routing"), dict):
        raise ValueError(f"{config_path} must contain a 'routing' mapping")
    return RouterConfig.model_validate(document["routing"])


def compute_illumination_entropy(image: np.ndarray) -> float:
    """Return Shannon entropy in bits, excluding exactly-zero null pixels.

    Accept a nonempty finite 2D grayscale array already scaled to [0, 255].
    Fractional intensities use unit-width bins [i, i+1) for i=0..255; positive
    values below 1 remain valid samples. No per-image intensity rescaling is
    performed. Empty/all-null images have no probability distribution and raise
    ValueError. Neither the input nor its zero padding is modified.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2D grayscale NumPy array")
    if image.dtype.kind not in "uif":
        raise ValueError("image must contain real numeric intensities in [0, 255]")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("image intensities must be finite and in [0, 255]")
    valid = image[image != 0]
    if valid.size == 0:
        raise ValueError("image contains no nonzero pixels for illumination entropy")
    counts, _ = np.histogram(valid, bins=256, range=(0, 256))
    probabilities = counts.astype(np.float64) / valid.size
    return float(entropy(probabilities, base=2))


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def compute_geometric_disparity(
    gsd_src: float,
    gsd_ref: float,
    incidence_src_deg: float,
    incidence_ref_deg: float,
    *,
    w1: float = 0.5,
    w2: float = 0.5,
) -> float:
    """Compute w1*abs(GSD_src/GSD_ref - 1) + w2*abs(inc_src - inc_ref).

    GSDs must be finite and positive in the same units (normally metres).
    Incidence angles follow Phase 1's [0, 90] degree contract. Weights must be
    finite and nonnegative; they are used as supplied without normalization.
    Source/reference reversal can change the result; this is not a symmetric
    max(GSD)/min(GSD) scale-gap metric.
    """
    gsd_src = _finite_real(gsd_src, "gsd_src")
    gsd_ref = _finite_real(gsd_ref, "gsd_ref")
    incidence_src_deg = _finite_real(incidence_src_deg, "incidence_src_deg")
    incidence_ref_deg = _finite_real(incidence_ref_deg, "incidence_ref_deg")
    w1, w2 = _finite_real(w1, "w1"), _finite_real(w2, "w2")
    if gsd_src <= 0 or gsd_ref <= 0:
        raise ValueError("GSD values must be positive")
    if not 0 <= incidence_src_deg <= 90 or not 0 <= incidence_ref_deg <= 90:
        raise ValueError("incidence angles must be in [0, 90] degrees")
    if w1 < 0 or w2 < 0:
        raise ValueError("geometry weights must be nonnegative")
    ratio = gsd_src / gsd_ref
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("GSD ratio is outside the finite positive float range")
    delta_g = w1 * abs(ratio - 1.0) + w2 * abs(incidence_src_deg - incidence_ref_deg)
    if not np.isfinite(delta_g):
        raise ValueError("geometric disparity exceeds the finite float range")
    return delta_g


def route_pair(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_metadata: LunarTelemetryMetadata,
    reference_metadata: LunarTelemetryMetadata,
    *,
    config: RouterConfig | None = None,
) -> RoutingDecision:
    """Select SIFT+FLANN only when BOTH inclusive disparity limits are met.

    Inputs may have different shapes/resolutions. Pass the Phase 1 telemetry
    models and grayscale images with null pixels still represented by zero
    (before Wallis conditioning, which may brighten padding). Configuration is
    loaded from backend/config.yaml on each call unless explicitly supplied;
    applications may load once and reuse the immutable RouterConfig. Emits one
    INFO record with metrics, limits, weights, selection and reasons.
    """
    settings = load_router_config() if config is None else config
    delta_g = compute_geometric_disparity(
        source_metadata.gsd_meters,
        reference_metadata.gsd_meters,
        source_metadata.incidence_angle_deg,
        reference_metadata.incidence_angle_deg,
        w1=settings.w1,
        w2=settings.w2,
    )
    entropy_src = compute_illumination_entropy(source_image)
    entropy_ref = compute_illumination_entropy(reference_image)
    delta_h = abs(entropy_src - entropy_ref)
    illumination_ok = delta_h <= settings.entropy_threshold
    geometry_ok = delta_g <= settings.geometry_threshold
    engine: EngineName = (
        "sift_flann" if illumination_ok and geometry_ok else "superpoint_lightglue"
    )
    rationales = (
        f"Illumination: delta_h={delta_h!r} bits "
        f"{'<=' if illumination_ok else '>'} threshold={settings.entropy_threshold!r} bits.",
        f"Geometry: delta_g={delta_g!r} "
        f"{'<=' if geometry_ok else '>'} threshold={settings.geometry_threshold!r}.",
        "Both limits are satisfied; selected sift_flann." if engine == "sift_flann"
        else "At least one limit is exceeded; selected superpoint_lightglue.",
    )
    decision = RoutingDecision(
        entropy_src=entropy_src,
        entropy_ref=entropy_ref,
        delta_h=delta_h,
        gsd_ratio=source_metadata.gsd_meters / reference_metadata.gsd_meters,
        incidence_delta_deg=abs(
            source_metadata.incidence_angle_deg - reference_metadata.incidence_angle_deg
        ),
        delta_g=delta_g,
        selected_engine=engine,
        rationales=rationales,
        config=settings,
    )
    logger.info(
        "Routing metrics: H_src=%.6f H_ref=%.6f delta_h=%.6f "
        "gsd_ratio=%.6f incidence_delta_deg=%.6f delta_g=%.6f "
        "entropy_threshold=%.6f geometry_threshold=%.6f w1=%.6f w2=%.6f engine=%s | %s",
        decision.entropy_src, decision.entropy_ref, decision.delta_h,
        decision.gsd_ratio, decision.incidence_delta_deg, decision.delta_g,
        settings.entropy_threshold, settings.geometry_threshold, settings.w1, settings.w2,
        decision.selected_engine, " ".join(decision.rationales),
    )
    return decision
