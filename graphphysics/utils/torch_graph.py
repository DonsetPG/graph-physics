from typing import Dict, List, Optional, Union

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from meshio import Mesh
from torch_geometric.data import Data


device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_k_hop_edge_index(
    edge_index: torch.Tensor,
    num_hops: int,
    num_nodes: int,
) -> torch.Tensor:
    """Computes the k-hop edge index for a given edge index tensor.

    Parameters:
        edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges].
        num_hops (int): The number of hops.
        num_nodes (int): The number of nodes.

    Returns:
        torch.Tensor: The edge index tensor representing the k-hop edges.
    """
    # Build the sparse adjacency matrix
    adj = torch.sparse_coo_tensor(
        edge_index,
        values=torch.ones(edge_index.size(1), dtype=torch.float32, device=device),
        size=(num_nodes, num_nodes),
    ).coalesce()

    adj_k = adj.clone()
    for _ in range(num_hops - 1):
        adj_k = adj_k + torch.sparse.mm(adj_k, adj)
        adj_k = adj_k.coalesce()

        # Remove self-loops
        indices = adj_k.indices()
        mask = indices[0] != indices[1]
        adj_k = torch.sparse_coo_tensor(
            indices=indices[:, mask],
            values=adj_k.values()[mask],
            size=adj_k.size(),
        ).coalesce()

    khop_edge_index = adj_k.indices()
    return khop_edge_index


def compute_k_hop_graph(
    graph: Data,
    num_hops: int,
    add_edge_features_to_khop: bool = False,
    device: str = "cpu",
    world_pos_index_start: int = 0,
    world_pos_index_end: int = 3,
) -> Data:
    """Builds a k-hop mesh graph.

    This implementation constructs the sparse adjacency matrix associated with the mesh graph
    and computes its powers in a sparse manner.

    Parameters:
        graph (Data): The input graph data.
        num_hops (int): The number of hops.
        add_edge_features_to_khop (bool): Whether to compute edge features for the k-hop graph.
        device (str): The device to move tensors to.

    Returns:
        Data: The k-hop graph data.
    """
    if num_hops == 1:
        return graph

    edge_index = graph.edge_index
    num_nodes = graph.num_nodes

    khop_edge_index = compute_k_hop_edge_index(
        edge_index=edge_index,
        num_hops=num_hops,
        num_nodes=num_nodes,
    ).to(device)

    # Build k-hop graph
    khop_mesh_graph = Data(
        x=graph.x, edge_index=khop_edge_index, pos=graph.pos, y=graph.y, face=graph.face
    )

    # Optionally compute edge features
    if add_edge_features_to_khop:
        transforms = [
            T.Cartesian(norm=False),
            T.Distance(norm=False),
        ]
        edge_feature_computer = T.Compose(transforms)
        khop_mesh_graph = edge_feature_computer(khop_mesh_graph).to(device)

    return khop_mesh_graph


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
    pos = graph.pos
    N, D = pos.shape
    _, F = field.shape

    edge_index = graph.edge_index

    gradients = torch.zeros((N, F, D), device=device)

    for node in range(N):
        # Find neighbors
        neighbor_mask = (edge_index[0] == node) | (edge_index[1] == node)
        if not neighbor_mask.any():
            continue

        # Get neighbor indices
        edges_local = edge_index[:, neighbor_mask]
        neighbors = torch.cat([edges_local[0], edges_local[1]])
        neighbors = neighbors[neighbors != node].unique()

        if len(neighbors) < D:  # Need at least D neighbors
            continue

        # Coordinate differences
        delta_pos = pos[neighbors] - pos[node]  # (n_neighbors, D)
        delta_field = field[neighbors] - field[node]  # (n_neighbors, F)

        # Weights based on inverse distance
        distances = torch.norm(delta_pos, dim=1)
        weights = 1.0 / (distances + 1e-8)
        W = torch.diag(weights)

        # Solve weighted least squares: W * A * grad = W * b
        A = delta_pos  # (n_neighbors, D)

        try:
            # Normal equation: (A^T W A) grad = A^T W b
            AtWA = A.t() @ W @ A  # (D, D)
            AtWA_inv = torch.inverse(AtWA + 1e-6 * torch.eye(D, device=device))

            for f in range(F):
                b = delta_field[:, f]  # (n_neighbors,)
                grad_f = AtWA_inv @ A.t() @ W @ b  # (D,)
                gradients[node, f, :] = grad_f

        except Exception:
            # Fallback to simple averaging if matrix is singular
            if len(neighbors) > 0:
                avg_grad = torch.mean(delta_field / (delta_pos + 1e-8), dim=0)
                gradients[node, :, :] = avg_grad.unsqueeze(1).repeat(1, D)

    return gradients


def compute_gradient_green_gauss(
    graph: Data,
    field: torch.Tensor,
    face_areas: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Green-Gauss gradient computation (cell-based, similar to what is used in CIMLIB).
    Requires face information or cell connectivity.

    Args:
        graph: Graph data with pos and edge_index
        field: Vector field (N, F)
        face_areas: Face areas if available (for 3D meshes)
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
    if D == 2:
        # For 2D: face normal is perpendicular to edge
        face_normals = torch.stack([-edge_vectors[:, 1], edge_vectors[:, 0]], dim=1)
        face_areas_computed = torch.norm(edge_vectors, dim=1)
    else:
        # For 3D: would need actual face normals from mesh topology
        face_normals = edge_vectors / (
            torch.norm(edge_vectors, dim=1, keepdim=True) + 1e-8
        )
        face_areas_computed = torch.ones(edges.shape[1], device=device)
    if face_areas is not None:
        face_areas_computed = face_areas

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
        gradients[node_j] -= contrib / cell_volumes[node_j]  # Opposite direction

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
    graph: Data, field: torch.Tensor, method: str = "finite_diff", device: str = "cpu"
) -> torch.Tensor:
    """
    Compute the product of a vector field with its gradient (e.g., u * grad(u)).

    Args:
        graph (Data): Data object, should have 'pos' and 'edge_index' attributes.
        field (torch.Tensor): Vector field (N, F).
        method (str): Method to compute the gradient.
        device (str): Device to perform the computation on.

    Returns:
        product (torch.Tensor): Tensor of shape (N, F) representing the product u * grad(u).
    """
    gradient = compute_gradient(
        graph, field, method=method, device=device
    )  # Shape: (N, F, D)
    product = torch.einsum(
        "nf,nfd->nf", field, gradient
    )  # Element-wise product and sum over D
    return product


def compute_divergence(
    graph: Data, field: torch.Tensor, method: str = "finite_diff", device: str = "cpu"
) -> torch.Tensor:
    """
    Compute the divergence of a vector field on an unstructured graph.

    Args:
        graph (Data): Data object, should have 'pos' and 'edge_index' attributes.
        field (torch.Tensor): Vector field (N, F).
        method (str): Method to compute the gradient.
        device (str): Device to perform the computation on.

    Returns:
        divergence (torch.Tensor): Tensor of shape (N,) representing the divergence of the vector field.
    """
    gradient = compute_gradient(graph, field, method=method, device=device)  # (N, F, D)
    # divergence = torch.einsum("nii->n", gradient)
    divergence = gradient[:, 0, 0] + gradient[:, 1, 1]  # Assuming 2D field
    return divergence


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[np.ndarray] = None,
    return_only_node_features: bool = False,
    id: Optional[str] = None,
) -> Data:
    """Converts mesh data into a PyTorch Geometric Data object.

    Parameters:
        points (np.ndarray): The coordinates of the mesh points.
        cells (np.ndarray): The connectivity of the mesh (how points form cells); either triangles or tetrahedras.
        point_data (Dict[str, np.ndarray]): A dictionary of point-associated data.
        time (int or float): A scalar value representing the time step.
        target (np.ndarray, optional): An optional target tensor.
        return_only_node_features (bool): Whether to return only node features.
        id (str, optional): An optional mesh id to link graph to original dataset mesh.

    Returns:
        Data: A PyTorch Geometric Data object representing the mesh.
    """
    # Combine all point data into a single array
    if point_data is not None:
        if any(data.ndim > 1 for data in point_data.values()):
            # if any(data.shape[1] > 1 for data in point_data.values()):
            node_features = np.hstack(
                [data for data in point_data.values()]
                + [np.full((len(points),), time).reshape((-1, 1))]
            )
            node_features = torch.tensor(node_features, dtype=torch.float32)
        else:
            node_features = np.vstack(
                [data for data in point_data.values()] + [np.full((len(points),), time)]
            ).T
            node_features = torch.tensor(node_features, dtype=torch.float32)
    else:
        node_features = torch.zeros((len(points), 1), dtype=torch.float32)

    if return_only_node_features:
        return node_features

    # Convert target to tensor if provided
    if target is not None:
        if any(data.ndim > 1 for data in target.values()):
            # if any(data.shape[1] > 1 for data in target.values()):
            target_features = np.hstack([data for data in target.values()])
            target_features = torch.tensor(target_features, dtype=torch.float32)
        else:
            target_features = np.vstack([data for data in target.values()]).T
            target_features = torch.tensor(target_features, dtype=torch.float32)
    else:
        target_features = None

    # Get tetrahedras and triangles from cells
    tetra = None
    cells = cells.T
    cells = torch.tensor(cells)
    if cells.shape[0] == 4:
        tetra = cells
        face = torch.cat(
            [
                cells[0:3],
                cells[1:4],
                torch.stack([cells[2], cells[3], cells[0]], dim=0),
                torch.stack([cells[3], cells[0], cells[1]], dim=0),
            ],
            dim=1,
        )
    if cells.shape[0] == 3:
        face = torch.tensor(cells)

    return Data(
        x=node_features,
        face=face,
        tetra=tetra,
        y=target_features,
        pos=torch.tensor(points, dtype=torch.float32),
        id=id,
    )


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
) -> Data:
    """Converts mesh and optional target mesh data into a PyTorch Geometric Data object.

    Parameters:
        mesh (Mesh): A Mesh object containing the mesh data.
        time (int or float): A scalar value representing the time step.
        target_mesh (Mesh, optional): An optional Mesh object containing target data.
        target_fields (List[str], optional): Fields from the target_mesh to be used as the target data.

    Returns:
        Data: A PyTorch Geometric Data object representing the mesh with optional target data.
    """
    # Prepare target data if a target mesh is provided
    target = None
    if target_mesh is not None and target_fields:
        target_data = [target_mesh.point_data[field] for field in target_fields]
        target = np.hstack(target_data)

    # Extract cells of type 'triangle' and 'quad'
    cells = np.vstack(
        [v for k, v in mesh.cells_dict.items() if k in ["triangle", "quad"]]
    )

    return meshdata_to_graph(
        points=mesh.points,
        cells=cells,
        point_data=mesh.point_data,
        time=time,
        target=target,
    )


def torch_graph_to_mesh(graph: Data, node_features_mapping: dict[str, int]) -> Mesh:
    """Converts a PyTorch Geometric graph to a meshio Mesh object.

    This function takes a graph represented in PyTorch Geometric's `Data` format and
    converts it into a meshio Mesh object. It extracts the positions, faces, and specified
    node features from the graph and constructs a Mesh object.

    Parameters:
        - graph (Data): The graph to convert, represented as a PyTorch Geometric `Data` object.
                      It should contain node positions in `graph.pos` and connectivity
                      (faces) in `graph.face`.
        - node_features_mapping (dict[str, int]): A dictionary mapping feature names to their
                                                corresponding column indices in `graph.x`.
                                                This allows selective inclusion of node features
                                                in the resulting Mesh object's point data.

    Returns:
        - Mesh: A meshio Mesh object containing the graph's geometric and feature data.

    Note:
    The function detaches tensors and moves them to CPU before converting to NumPy arrays,
    ensuring compatibility with meshio and avoiding GPU memory issues.
    """
    point_data = {
        f: graph.x[:, indx].detach().cpu().numpy()
        for f, indx in node_features_mapping.items()
    }

    cells = graph.face.detach().cpu().numpy()
    if graph.pos.shape[1] == 2:
        extra_shape = 3
        cells_type = "triangle"
    elif graph.pos.shape[1] == 3:
        extra_shape = 4
        cells_type = "tetra"
    else:
        raise ValueError(
            f"Graph Pos does not have the right shape. Expected shape[1] to be 2 or 3. Got {graph.pos.shape[1]}"
        )

    if cells.shape[-1] != extra_shape:
        cells = cells.T

    return meshio.Mesh(
        graph.pos.detach().cpu().numpy(),
        [(cells_type, cells)],
        point_data=point_data,
    )


def get_masked_indexes(graph: Data, masking_ratio: float = 0.15) -> torch.Tensor:
    """Generate masked indices for the input graph based on the masking ratio.

    Args:
        graph (Data): The input graph data.
        masking_ratio (float): The ratio of nodes to mask.

    Returns:
        selected_indices (Tensor): The indices of nodes to keep after masking.
    """
    n, _ = graph.x.shape
    nodes_to_keep = 1 - masking_ratio
    num_rows_to_sample = int(nodes_to_keep * n)
    # Generate random indices
    random_indices = torch.randperm(n)
    selected_indices = random_indices[:num_rows_to_sample]

    return selected_indices
