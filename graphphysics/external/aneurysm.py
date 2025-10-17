"""Aneurysm specific feature helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from named_features import NamedData, make_x_layout

device = "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_named_feature(
    graph: Data, candidates: Sequence[str], index: int
) -> torch.Tensor:
    """Return the first matching named feature or fall back to column ``index``."""

    if hasattr(graph, "x_sel"):
        for name in candidates:
            try:
                return graph.x_sel(name)
            except KeyError:
                continue
    return graph.x[:, index : index + 1]


def aneurysm_node_type(graph: Data) -> torch.Tensor:
    """Infer node types for aneurysm meshes using named selections when possible."""

    velocity = _maybe_named_feature(graph, ["Vitesse", "velocity"], 0)
    wall_inputs = _maybe_named_feature(graph, ["wall_mask", "wall"], 3)

    node_type = torch.zeros(
        velocity.shape[0], device=velocity.device, dtype=velocity.dtype
    )

    wall_mask = wall_inputs.squeeze(-1) == 1.0

    inflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] <= 0)
    outflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] >= 0)

    node_type[wall_mask] = NodeType.WALL_BOUNDARY
    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    return node_type.to(device)


def build_features(graph: Data) -> Data:
    """Augment aneurysm graphs with acceleration and statistics."""

    node_type = aneurysm_node_type(graph)

    if hasattr(graph, "x_sel"):
        current_velocity = graph.x_sel("Vitesse")
    else:
        current_velocity = graph.x[:, 0:3]

    target_velocity = graph.y[:, : current_velocity.shape[-1]]
    previous_velocity = torch.as_tensor(
        graph.previous_data["Vitesse"], device=device, dtype=current_velocity.dtype
    )

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    not_inflow_mask = node_type != NodeType.INFLOW
    next_acceleration[not_inflow_mask] = 0
    next_acceleration_unique = next_acceleration.unique()

    mean_next_accel = torch.full_like(node_type, torch.mean(next_acceleration_unique))
    min_next_accel = torch.full_like(node_type, torch.min(next_acceleration_unique))
    max_next_accel = torch.full_like(node_type, torch.max(next_acceleration_unique))

    extras = [
        ("acceleration", acceleration),
        ("mesh_pos", graph.pos),
        ("mean_next_accel", mean_next_accel.unsqueeze(-1)),
        ("min_next_accel", min_next_accel.unsqueeze(-1)),
        ("max_next_accel", max_next_accel.unsqueeze(-1)),
        ("node_type", node_type.unsqueeze(-1)),
    ]

    graph.x = torch.cat([graph.x] + [value for _, value in extras], dim=-1)

    if isinstance(graph, NamedData):
        sizes = dict(graph.x_layout.sizes())
        names = list(graph.x_names())
        for base_name, value in extras:
            candidate = base_name
            suffix = 1
            while candidate in sizes:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            names.append(candidate)
            sizes[candidate] = value.shape[-1]
        graph.x_layout = make_x_layout(names, sizes)
        graph.validate_x()

    return graph
