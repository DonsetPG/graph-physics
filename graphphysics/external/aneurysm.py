import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType

device = "cuda" if torch.cuda.is_available() else "cpu"


def aneurysm_node_type(graph: Data) -> torch.Tensor:
    v_x = graph.x[:, 0]
    wall_inputs = graph.x[:, 3]
    node_type = torch.zeros(v_x.shape)

    wall_mask = wall_inputs == 1.0

    inflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] <= 0)

    outflow_mask = torch.logical_and(graph.pos[:, 1] == 0.0, graph.pos[:, 0] >= 0)

    node_type[wall_mask] = NodeType.WALL_BOUNDARY
    node_type[inflow_mask] = NodeType.INFLOW
    node_type[outflow_mask] = NodeType.OUTFLOW

    return node_type.to(device)


def build_features(graph: Data) -> Data:
    node_type = graph.x[:, 3]

    velocity_4DFlow = graph.x[:, 0:3]

    graph.x = torch.cat(
        (
            velocity_4DFlow,
            graph.pos,
            node_type.to(device).unsqueeze(1),
        ),
        dim=1,
    )

    return graph
