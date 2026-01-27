import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType


def add_bezier_node_type(graph: Data) -> Data:
    bn = graph.x[:, 3]
    a1 = graph.x[:, 4]
    a2 = graph.x[:, 5]
    a3 = graph.x[:, 6]
    a4 = graph.x[:, 7]

    node_type = torch.zeros(bn.shape)

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

    return graph
