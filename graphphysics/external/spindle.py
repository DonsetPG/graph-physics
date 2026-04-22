import torch
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_features(graph: Data) -> Data:
    # construct features
    current_displacement = graph.x[:, 0:3]
    position = graph.pos
    graph.pos = current_displacement + position
    nodetype = graph.x[:, -2].unsqueeze(1)

    mask = graph.x[:, -2] == 0

    if mask.any():
        diameter = graph.pos[mask, 1].max().item() * 2

    diameter = torch.tensor(diameter, device=graph.x.device)

    diameter_feature = diameter.repeat(graph.x.size(0), 1)

    graph.x = torch.cat(
        (
            current_displacement,
            graph.pos,
            diameter_feature,
            nodetype,
        ),
        dim=1,
    )

    return graph
