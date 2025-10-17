"""Bezier specific utilities."""

from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import Data

from named_features import NamedData, make_x_layout
from graphphysics.utils.nodetype import NodeType


def _pick_feature(graph: Data, candidates: Sequence[str], index: int) -> torch.Tensor:
    if hasattr(graph, "x_sel"):
        for name in candidates:
            try:
                return graph.x_sel(name).squeeze(-1)
            except KeyError:
                continue
    return graph.x[:, index]


def _ensure_unique_name(existing: Sequence[str], base: str) -> str:
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def add_bezier_node_type(graph: Data) -> Data:
    """Append a node type feature derived from Bezier boundary annotations."""

    bn = _pick_feature(graph, ["bn", "boundary"], 3)
    a1 = _pick_feature(graph, ["a1"], 4)
    a2 = _pick_feature(graph, ["a2"], 5)
    a3 = _pick_feature(graph, ["a3"], 6)
    a4 = _pick_feature(graph, ["a4"], 7)

    node_type = torch.zeros_like(bn)

    wall_mask = torch.logical_and(bn == 1.0, a1 == 0.0)
    wall_mask = torch.logical_and(wall_mask, a2 == 0.0)
    wall_mask = torch.logical_and(wall_mask, a3 == 0.0)
    wall_mask = torch.logical_and(wall_mask, a4 == 0.0)

    inflow_mask = a1 == 1.0
    outflow_mask = a3 == 1.0

    node_type[wall_mask] = NodeType.WALL_BOUNDARY
    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    graph.x = torch.cat((graph.x, node_type.unsqueeze(1)), dim=1)

    if isinstance(graph, NamedData):
        sizes = dict(graph.x_layout.sizes())
        names = list(graph.x_names())
        feature_name = _ensure_unique_name(names, "node_type")
        names.append(feature_name)
        sizes[feature_name] = 1
        graph.x_layout = make_x_layout(names, sizes)
        graph.validate_x()

    return graph
