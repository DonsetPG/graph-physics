import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.vectorial_operators import compute_gradient


def rode_node_type(graph: Data) -> torch.Tensor:
    appartient1    = graph.x[:, 4]
    appartientACT  = graph.x[:, 5]

    node_type = torch.zeros_like(appartient1, device=graph.x.device)

    node_type[(appartient1 == 0.0) & (appartientACT == 0.0)] = NodeType.NORMAL   # Normal
    node_type[(appartient1 == 0.0) & (appartientACT == 1.0)] = NodeType.HANDLE   # Force
    node_type[(appartient1 == 1.0) & (appartientACT == 0.0)] = NodeType.OBSTACLE # Support

    return node_type

def compute_strain_from_displacement(
    graph: Data,
    method: str = "least_squares",
    device: str = "cpu",
    target: bool = False,
) -> torch.Tensor:
    u = graph.x[:, 0:3].to(device)  if not target else graph.y[:, 0:3].to(device)  # [N, 3]
    grad_u = compute_gradient(graph, u, method=method, device=device)  # [N,3,D]

    strain = 0.5 * (grad_u + grad_u.transpose(1, 2))  # [N,3,3]
    return strain

def compute_sigma_from_strain(
    strain: torch.Tensor,
    pressure: torch.Tensor,
    lame_lambda: torch.Tensor,
    mu: torch.Tensor | None = None,
    nu: float = 0.499,
) -> torch.Tensor:
    pressure = pressure.view(-1)
    lame_lambda = lame_lambda.view(-1)

    if mu is None:
        mu = lame_lambda * (1.0 - 2.0 * nu) / (2.0 * nu)
    else:
        mu = mu.view(-1)  # [N]

    tr_eps = strain.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)  # [N,1]
    I = torch.eye(3, device=strain.device, dtype=strain.dtype).unsqueeze(0)  # [1,3,3]
    dev_eps = strain - (tr_eps / 3.0).unsqueeze(-1) * I  # [N,3,3]

    sigma = (
        pressure.unsqueeze(-1).unsqueeze(-1) * I
        - 2.0 * mu.unsqueeze(-1).unsqueeze(-1) * dev_eps
    )  # [N,3,3]

    return sigma

def rode(graph: Data) -> Data:
    device = graph.x.device
    node_type = rode_node_type(graph)
    target_dt = graph.target_dt

    current_displacement = graph.x[:, 0:3]   # u^n

    previous_displacement = torch.tensor(
        graph.previous_data["Displacement"],
        device=device,
        dtype=current_displacement.dtype,
    )  # u^{n-1}

    previous_previous_displacement = torch.tensor(
        graph.previous_previous_data["Displacement"],
        device=device,
        dtype=current_displacement.dtype,
    )  # u^{n-2}

    velocity = (current_displacement - previous_displacement) / target_dt
    graph.velocity = velocity

    acceleration = (
        current_displacement
        - 2.0 * previous_displacement
        + previous_previous_displacement
    ) / (target_dt ** 2)

    graph.acceleration = acceleration
    graph.target_acceleration = acceleration.clone()

    rest_pos = graph.pos

    pressure    = graph.x[:, 3]
    lame_lambda = graph.x[:, 7]

    poisson_ratio = 0.499
    mu = lame_lambda * (1.0 - 2.0 * poisson_ratio) / (2.0 * poisson_ratio)
    graph.pressure = pressure
    graph.lame_lambda = lame_lambda
    graph.mu = mu

    # Prescribed displacement on HANDLE nodes to simulate applying force like in DeformingPlate
    handle_mask = (node_type == NodeType.HANDLE)
    prescribed_disp = torch.zeros_like(current_displacement)

    target_displacement = graph.y[:, 0:3]
    prescribed_disp[handle_mask] = (
        target_displacement[handle_mask] - current_displacement[handle_mask]
    )

    strain = compute_strain_from_displacement(graph, target=False, method="least_squares", device=device)

    sigma = compute_sigma_from_strain(
        strain=strain,
        pressure=pressure,
        lame_lambda=lame_lambda,
        mu=mu,
        nu=poisson_ratio,
    )

    target_strain = compute_strain_from_displacement(
        graph,
        target=True,
        method="least_squares",
        device=device,
    )
    target_sigma = compute_sigma_from_strain(
        strain=target_strain,
        pressure=pressure,
        lame_lambda=lame_lambda,
        mu=mu,
        nu=poisson_ratio,
    )

    graph.target_strain = target_strain  # [N,3,3]
    graph.target_sigma  = target_sigma   # [N,3,3]

    # Storing for loss computation
    graph.strain = strain  # [N,3,3]
    graph.sigma  = sigma   # [N,3,3]
    graph.rho = graph.x[:, 6]  # Density

    # New node feat
    parts = [current_displacement, velocity, rest_pos, prescribed_disp]
    node_type_col = node_type.unsqueeze(1).float()
    parts.append(node_type_col)

    graph.x = torch.cat(parts, dim=1)
    graph.y = graph.y[:, 0:3]
    graph.node_type = node_type
    return graph
