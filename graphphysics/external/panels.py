import torch
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_features(graph: Data) -> Data:
    # construct features
    current_velocity = graph.x[:, 0:2]
    pressure = graph.x[:, 3].unsqueeze(1)
    levelset = graph.x[:, 4].unsqueeze(1)
    nodetype = graph.x[:, 5].unsqueeze(1)

    graph.x = torch.cat(
        (
            current_velocity,
            pressure,
            levelset,
            graph.pos,
            nodetype,
        ),
        dim=1,
    )

    # hide Vz in target
    graph.y = graph.y[:, 0:2]

    return graph
