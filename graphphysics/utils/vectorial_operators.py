import torch
from torch_geometric.data import Data


def compute_gradient_weighted_least_squares(
    graph: Data, field: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute gradient using weighted least squares (similar to VTK approach).
    More accurate than simple finite differences for irregular meshes.

    Args:
        graph: Graph data with pos and edge_index
        field: Vector field (N, F)
        k_neighbors: Number of neighbors to use for each node
        device: Computation device

    Returns:
        gradients: Tensor of shape (N, F, D)
    """
    device = torch.device(device)

    # Move inputs to device
    points = graph.pos.to(device)  # (N, 3)
    field = field.to(device)  # (N,), (N,2), or (N,3)

    # Ensure field is at least 2D: (N, dim_u)
    if field.ndim == 1:
        field = field.unsqueeze(1)

    dim_x = points.shape[1]
    dim_u = field.shape[1]  # field dimension

    # Get element connectivity
    elements = graph.face.T.to(device)  # (M, D+1)

    D = elements.shape[1] - 1  # 2 for triangle, 3 for tetrahedron
    N = points.shape[0]

    # Coordinates of element nodes (M, D+1, 3)
    elem_points = points[elements]
    # Field values at element nodes (M, D+1, dim_u)
    elem_field = field[elements]

    # Build difference matrices relative to first vertex
    A = elem_points[:, 1:, :] - elem_points[:, 0:1, :]  # (M, D, 3)
    B = elem_field[:, 1:, :] - elem_field[:, 0:1, :]  # (M, D, dim_u)

    # Solve A @ grad^T ≈ B  => grad ≈ B^T @ A⁺
    # grad_elems: (M, dim_u, 3)
    grad_elems = torch.linalg.lstsq(A, B).solution.transpose(1, 2)

    # --- Element measure (area or volume) ---
    if D == 2:  # triangle area
        v1 = A[:, 0, :]  # (M, 3)
        v2 = A[:, 1, :]  # (M, 3)
        # TODO: 2nd option works both ways no?
        if v1.shape[1] == 3:
            cross = torch.cross(v1, v2, dim=1)  # (M, 3)
            volume = 0.5 * torch.norm(cross, dim=1)  # (M,)
        if v1.shape[1] == 2:
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            volume = 0.5 * torch.abs(cross)
    elif D == 3:  # tetrahedron volume
        volume = torch.abs(torch.linalg.det(A)) / 6.0  # (M,)
    else:
        raise ValueError(f"Unsupported element dimension D={D}")

    # Accumulate contributions to nodes
    gradients = torch.zeros((N, dim_u, dim_x), device=device)
    weights = torch.zeros((N, 1), device=device)

    for i in range(D + 1):
        idx = elements[:, i]
        gradients.index_add_(0, idx, grad_elems * volume[:, None, None])
        weights.index_add_(0, idx, volume[:, None])

    gradients /= weights.clamp(min=1e-12).view(-1, 1, 1)

    return gradients


def compute_gradient_green_gauss(
    graph: Data,
    field: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Green-Gauss gradient computation (cell-based, similar to what is used in CIMLIB).
    Requires face information or cell connectivity.

    Args:
        graph: Graph data with pos and edge_index
        field: Vector field (N, F)
        device: Computation device

    Returns:
        gradients: Tensor of shape (N, F, D)
    """
    pos = graph.pos
    edge_index = graph.edge_index
    N, D = pos.shape
    _, F = field.shape

    # Remove duplicate edges
    edges = torch.unique(torch.sort(edge_index.T, dim=1)[0], dim=0).T
    i, j = edges[0], edges[1]

    # Create cell info: face fields and face vectors
    face_fields = 0.5 * (field[i] + field[j])  # (E, F)
    edge_vectors = pos[j] - pos[i]  # (E, D)
    # TODO: compute for real face_areas
    if D == 2:
        # For 2D: face normal is perpendicular to edge (ensure orientation is outward from i to j)
        face_normals = torch.stack([edge_vectors[:, 1], -edge_vectors[:, 0]], dim=1)
        face_areas_computed = torch.norm(edge_vectors, dim=1)
    else:
        # For 3D: would need actual face normals from mesh topology
        face_normals = edge_vectors / (
            torch.norm(edge_vectors, dim=1, keepdim=True) + 1e-8
        )
        face_areas_computed = torch.ones(edges.shape[1], device=device)

    # Compute cell volumes (approximate using Voronoi cells)
    cell_volumes = torch.zeros(N, device=device)
    for node in range(N):
        connected_edges = (i == node) | (j == node)
        if connected_edges.any():
            # Approximate cell volume as sum of connected edge contributions
            cell_volumes[node] = face_areas_computed[connected_edges].sum() / 2.0

    cell_volumes = torch.clamp(cell_volumes, min=1e-8)

    # Green-Gauss: grad(phi) = (1/V) sum( phi_face * n_face * A_face)
    gradients = torch.zeros((N, F, D), device=device)

    for e in range(edges.shape[1]):
        node_i, node_j = i[e], j[e]
        face_area = face_areas_computed[e]
        face_normal = face_normals[e]
        face_field = face_fields[e]  # (F,)

        # Contribution to each adjacent cell
        contrib = torch.outer(face_field, face_normal) * face_area  # (F, D)

        gradients[node_i] += contrib / cell_volumes[node_i]
        gradients[node_j] += (
            contrib / cell_volumes[node_j]
        )  # Same direction for undirected edges

    return gradients


def compute_gradient_finite_differences(
    graph: Data, field: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Finite difference gradient computation (original one we have in graphphysics).
    Args:
        graph: Graph data with pos and edge_index
        field: Vector field (N, F)
        device: Computation device

    Returns:
        gradients: Tensor of shape (N, F, D)
    """
    pos = graph.pos
    edges = graph.edge_index
    edges = torch.unique(torch.sort(edges.T, dim=1)[0], dim=0).T

    N, D = pos.shape
    _, F = field.shape
    i, j = edges[0], edges[1]

    # Coordinate and field differences
    dx = pos[j] - pos[i]
    du = field[j] - field[i]
    distances = torch.norm(dx, dim=1)

    # VTK-style weighting: inverse distance squared
    weights = 1.0 / (distances**2 + 1e-8)

    # Weighted gradient computation
    gradient_edges = torch.matmul(du.unsqueeze(2), dx.unsqueeze(1)) * weights.view(
        -1, 1, 1
    )
    gradient_edges = gradient_edges / (distances.view(-1, 1, 1) ** 2 + 1e-8)

    # Accumulate and normalize
    gradient = torch.zeros((N, F, D), device=device)
    weight_sums = torch.zeros((N, F, D), device=device)

    gradient.index_add_(0, i, gradient_edges * weights.view(-1, 1, 1))
    gradient.index_add_(0, j, gradient_edges * weights.view(-1, 1, 1))
    weight_sums.index_add_(0, i, weights.view(-1, 1, 1).expand(-1, F, D))
    weight_sums.index_add_(0, j, weights.view(-1, 1, 1).expand(-1, F, D))

    gradient = gradient / (weight_sums + 1e-8)
    return gradient


def compute_gradient(
    graph: Data,
    field: torch.Tensor,
    method: str = "least_squares",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Different gradient computation methods.

    Args:
        graph: Graph data with pos and edge_index
        field: Vector field (N, F)
        method: "least_squares", "green_gauss", or "finite_diff"
        device: Computation device

    Returns:
        gradients: Tensor of shape (N, F, D)
    """
    if method == "least_squares":
        return compute_gradient_weighted_least_squares(graph, field, device=device)
    elif method == "green_gauss":
        return compute_gradient_green_gauss(graph, field, device=device)
    elif method == "finite_diff":
        return compute_gradient_finite_differences(graph, field, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_vector_gradient_product(
    graph: Data,
    field: torch.Tensor,
    gradient: torch.Tensor = None,
    method: str = "finite_diff",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the product of a vector field with its gradient (e.g., u * grad(u)).

    Args:
        graph (Data): Data object, should have 'pos' and 'edge_index' attributes.
        field (torch.Tensor): Vector field (N, F).
        gradient (torch.Tensor, optional): Gradient of field (N, F, D).
        method (str): Method to compute the gradient.
        device (str): Device to perform the computation on.

    Returns:
        product (torch.Tensor): Tensor of shape (N, F) representing the product u * grad(u).
    """
    if gradient is None:
        gradient = compute_gradient(
            graph, field, method=method, device=device
        )  # Shape: (N, F, D)
    product = torch.einsum(
        "nf,nfd->nf", field, gradient
    )  # Element-wise product and sum over D
    return product


def compute_divergence(
    graph: Data,
    field: torch.Tensor,
    gradient: torch.Tensor = None,
    method: str = "finite_diff",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the divergence of a vector field on an unstructured graph.

    Args:
        graph (Data): Data object, should have 'pos' and 'edge_index' attributes.
        field (torch.Tensor): Vector field (N, F).
        gradient (torch.Tensor, optional): Gradient of field (N, F, D).
        method (str): Method to compute the gradient.
        device (str): Device to perform the computation on.

    Returns:
        divergence (torch.Tensor): Tensor of shape (N,) representing the divergence of the vector field.
    """
    if gradient is None:
        gradient = compute_gradient(
            graph, field, method=method, device=device
        )  # (N, F, D)

    divergence = gradient.diagonal(dim1=1, dim2=2).sum(dim=-1)
    return divergence
