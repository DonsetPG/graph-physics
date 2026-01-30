import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType

device = "cuda" if torch.cuda.is_available() else "cpu"

STENTS = {
    "S1": {"wire_number": 36, "wire_radius": 0.015, "wire_distance": 0.358},
    "S2": {"wire_number": 24, "wire_radius": 0.015, "wire_distance": 0.540},
    "S3": {"wire_number": 16, "wire_radius": 0.015, "wire_distance": 0.805},
}


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


def build_stent_features(graph: Data, node_type: torch.Tensor) -> Data:

    stent_id = graph.id.split('_')[-1]
    stent_params = STENTS[stent_id]
    stent = torch.Tensor([stent_params.values()], device=device)
    stent_features = torch.zeros((graph.x.shape[0], 3), device=device)

    stent_mask = node_type == NodeType.STENT
    stent_features[stent_mask] = stent

    return stent_features


def build_features(graph: Data) -> Data:
    node_type = graph.x[:, 3]
    lvlset_inlet = graph.x[:, 4]

    current_velocity = graph.x[:, 0:3]
    target_velocity = graph.y[:, 0:3]
    previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    not_inflow_mask = node_type != NodeType.INFLOW
    next_acceleration[not_inflow_mask] = 0
    next_acceleration_unique = next_acceleration.unique()

    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        next_acceleration_unique
    )
    min_next_accel = torch.ones(node_type.shape, device=device) * torch.min(
        next_acceleration_unique
    )
    max_next_accel = torch.ones(node_type.shape, device=device) * torch.max(
        next_acceleration_unique
    )

    if NodeType.STENT in torch.unique(node_type):
        timestep = graph.x[:, 6]
        lvlset_stent = graph.x[:, 5]
        stent_features = build_stent_features(graph)
    else:
        timestep = graph.x[:, 5]
        lvlset_stent = torch.zeros_like(lvlset_inlet)
        stent_features = torch.zeros((graph.x.shape[0], 3), device=device)

    graph.x = torch.cat(
        (
            current_velocity,
            timestep.to(device).unsqueeze(1),
            acceleration,
            graph.pos,
            mean_next_accel.unsqueeze(1),
            min_next_accel.unsqueeze(1),
            max_next_accel.unsqueeze(1),
            lvlset_inlet.to(device).unsqueeze(1),
            lvlset_stent.to(device).unsqueeze(1),
            stent_features,
            node_type.to(device).unsqueeze(1),
        ),
        dim=1,
    )
    # print(graph.x[1000])

    return graph
