"""Adaptive routing and dual-engine lunar image matching."""

from .matcher import BaseMatcher, LightGlueMatcher, MatchResult, SIFTMatcher

from .router import (
    RouterConfig,
    RoutingDecision,
    compute_geometric_disparity,
    compute_illumination_entropy,
    load_router_config,
    route_pair,
)

__all__ = [
    "BaseMatcher",
    "MatchResult",
    "SIFTMatcher",
    "LightGlueMatcher",
    "RouterConfig",
    "RoutingDecision",
    "compute_geometric_disparity",
    "compute_illumination_entropy",
    "load_router_config",
    "route_pair",
]
