from functools import partial
from typing import Dict, List, Optional, Union

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from meshio import Mesh
from torch_geometric.data import Data

from graphphysics.dataset.preprocessing import add_world_pos_features

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
