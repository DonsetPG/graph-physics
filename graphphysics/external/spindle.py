import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType


def _upper_die_height_gap(mesh_pos: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
    upper_pos = mesh_pos[node_type == NodeType.OBSTACLE]
    closest = torch.cdist(mesh_pos[:, :2], upper_pos[:, :2]).argmin(dim=1)
    gap = upper_pos[closest, 2:3] - mesh_pos[:, 2:3]
    gap[node_type != NodeType.NORMAL] = 0
    return gap


def build_mesh_pos_features(
    graph: Data,
    mesh_pos_index_start: int = 0,
    mesh_pos_index_end: int = 3,
    node_type_index: int | None = 3,
) -> Data:
    """Use mesh position and node type."""
    mesh_pos = graph.x[:, mesh_pos_index_start:mesh_pos_index_end]
    features = [mesh_pos]

    node_type = graph.x[:, node_type_index].reshape(-1, 1)

    features.append(node_type)

    graph.x = torch.cat(features, dim=1)
    return graph


def build_features(
    graph: Data,
    mesh_pos_index_start: int = 0,
    mesh_pos_index_end: int = 3,
    node_type_index: int = 3,
    pressure_index_start: int = 7,
    pressure_index_end: int = 8,
    velocity_index_start: int = 9,
    velocity_index_end: int = 12,
    surface_normal_index_start: int = 14,
    surface_normal_index_end: int = 17,
) -> Data:
    """Build spindle inputs from the raw XDMF node-feature layout.

    The world-position preprocessing runs after this function and inserts the
    prescribed die displacement after ``mesh_pos``. The resulting layout is:
    ``mesh_pos, die_displacement, velocity, pressure, surface_normal, node_type``.
    """
    mesh_pos = graph.x[:, mesh_pos_index_start:mesh_pos_index_end]
    pressure = graph.x[:, pressure_index_start:pressure_index_end]
    velocity = graph.x[:, velocity_index_start:velocity_index_end]
    surface_normal = graph.x[
        :, surface_normal_index_start:surface_normal_index_end
    ]
    node_type = graph.x[:, node_type_index].reshape(-1, 1)

    graph.x = torch.cat(
        [mesh_pos, velocity, pressure, surface_normal, node_type],
        dim=1,
    )
    return graph
