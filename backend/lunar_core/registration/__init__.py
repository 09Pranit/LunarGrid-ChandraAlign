"""Geometric filtering, spatial uniformity, and non-rigid TPS registration."""

from .outlier_rejection import (
    FilteringResult,
    RegistrationRejected,
    SpatialGeometryError,
    bounded_voronoi_cells,
    compute_vsui,
    filter_magsac_and_vsui,
    grid_nms,
)
from .tps_warper import TPSWarp, solve_tps_warp

__all__ = [
    "FilteringResult",
    "RegistrationRejected",
    "SpatialGeometryError",
    "bounded_voronoi_cells",
    "compute_vsui",
    "filter_magsac_and_vsui",
    "grid_nms",
    "TPSWarp",
    "solve_tps_warp",
]
