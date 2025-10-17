"""Panel specific feature builders."""

from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import Data

from named_features import NamedData, make_x_layout

device = "cuda" if torch.cuda.is_available() else "cpu"


def _select_feature(graph: Data, name: str, fallback_slice: slice) -> torch.Tensor:
    """Return a feature either by semantic name or by numeric slice."""

    if hasattr(graph, "x_sel"):
        try:
            return graph.x_sel(name)
        except KeyError:
            pass
    return graph.x[:, fallback_slice]


def _single_feature(
    graph: Data, name: str, fallback_index: int, squeeze: bool = True
) -> torch.Tensor:
    """Return a single-column feature as a ``(N, 1)`` tensor."""

    if hasattr(graph, "x_sel"):
        try:
            tensor = graph.x_sel(name)
            if tensor.ndim == graph.x.ndim:
                tensor = tensor.unsqueeze(-1) if tensor.shape[-1] == 0 else tensor
            if tensor.shape[-1] == 1:
                return tensor
        except KeyError:
            pass
    column = graph.x[:, fallback_index]
    return column.unsqueeze(-1) if squeeze else column


def _update_layout(graph: NamedData, names: Sequence[str], sizes: Sequence[int]) -> None:
    """Update ``graph.x_layout`` to match the provided ``names`` and ``sizes``."""

    layout = make_x_layout(list(names), {name: size for name, size in zip(names, sizes)})
    graph.x_layout = layout
    graph.validate_x()


def build_features(graph: Data) -> Data:
    """Construct panel features using semantic selections when available."""

    current_velocity = _select_feature(graph, "Vitesse", slice(0, 2))
    if current_velocity.shape[-1] > 2:
        current_velocity = current_velocity[..., :2]

    pressure = _single_feature(graph, "Pression", 3)
    levelset = _single_feature(graph, "LevelSetObject", 4)
    nodetype = _single_feature(graph, "NodeType", 5)

    new_features = torch.cat(
        (
            current_velocity,
            pressure,
            levelset,
            graph.pos,
            nodetype,
        ),
        dim=-1,
    )

    graph.x = new_features

    if isinstance(graph, NamedData):
        feature_names = [
            "Vitesse",
            "Pression",
            "LevelSetObject",
            "mesh_pos",
            "NodeType",
        ]
        feature_sizes = [
            current_velocity.shape[-1],
            pressure.shape[-1],
            levelset.shape[-1],
            graph.pos.shape[-1],
            nodetype.shape[-1],
        ]
        _update_layout(graph, feature_names, feature_sizes)

    # hide Vz in target
    if hasattr(graph, "y") and graph.y is not None:
        graph.y = graph.y[:, : current_velocity.shape[-1]]

    return graph
