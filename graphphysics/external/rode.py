import torch
from torch_geometric.data import Data

def rode(graph: Data) -> Data:
    device = graph.x.device
    world_pos = graph.x[:, 0:3]
    # prev_pos  = torch.as_tensor(graph.previous_data["world_pos"], device=device)
    # last_disp = world_pos - prev_pos
    rest_pos  = graph.pos
    node_type = graph.x[:, 6]

    next_pos = graph.y[:, 0:3].to(device=device, dtype=world_pos.dtype)

    mask = (node_type == 3)
    prescribed_disp = torch.zeros_like(world_pos)
    prescribed_disp[mask] = next_pos[mask] - world_pos[mask]


    parts = [world_pos, rest_pos, prescribed_disp]
    node_type_col = node_type.unsqueeze(1)
    if node_type_col is not None:
        parts.append(node_type_col)
    graph.x = torch.cat(parts, dim=1)
    return graph
