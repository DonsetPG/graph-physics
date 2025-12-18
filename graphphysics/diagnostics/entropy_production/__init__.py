"""Entropy production + oversmoothing diagnostics for graphphysics."""

from .graph_ep import EPEstimationConfig, estimate_graph_ep_all_orders

__all__ = [
    "EPEstimationConfig",
    "estimate_graph_ep_all_orders",
]
