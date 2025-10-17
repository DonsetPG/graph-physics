from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import meshio
import numpy as np
import torch
import torch_geometric.transforms as T
from meshio import Mesh
from torch_geometric.data import Data

from named_features import NamedData, XFeatureLayout

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


def _stack_point_data(
    point_data: Mapping[str, np.ndarray],
    *,
    order: Optional[Sequence[str]] = None,
    num_nodes: int,
) -> np.ndarray:
    if not point_data:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    if order is None:
        items = list(point_data.items())
    else:
        items = [(name, point_data[name]) for name in order if name in point_data]

    arrays = []
    detected_nodes = None
    for name, array in items:
        arr = np.asarray(array)
        if arr.ndim == 1:
            arr = arr[:, None]
        if detected_nodes is None:
            detected_nodes = arr.shape[0]
        elif arr.shape[0] != detected_nodes:
            raise ValueError(
                f"Feature '{name}' has inconsistent node count: expected {detected_nodes}, got {arr.shape[0]}."
            )
        arrays.append(arr.astype(np.float32))

    if not arrays:
        return np.zeros((num_nodes, 0), dtype=np.float32)

    return np.concatenate(arrays, axis=1)


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Mapping[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[Mapping[str, np.ndarray]] = None,
    return_only_node_features: bool = False,
    id: Optional[str] = None,
    next_data: Optional[Mapping[str, np.ndarray]] = None,
    *,
    x_layout: Optional[XFeatureLayout] = None,
    x_coords: Optional[Mapping[str, object]] = None,
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
    feature_order: Optional[Sequence[str]] = None
    if x_layout is not None:
        feature_order = x_layout.names()

    if point_data is not None:
        stacked = _stack_point_data(
            point_data, order=feature_order, num_nodes=len(points)
        )
        node_features = torch.tensor(stacked, dtype=torch.float32)
    else:
        node_features = torch.zeros((len(points), 0), dtype=torch.float32)

    if x_layout is None:
        time_column = torch.full((len(points), 1), float(time), dtype=torch.float32)
        if node_features.numel() == 0:
            node_features = time_column
        else:
            node_features = torch.cat([node_features, time_column], dim=1)

    if return_only_node_features:
        return node_features

    # Convert target to tensor if provided
    if target is not None and len(target) > 0:
        target_features = torch.tensor(
            _stack_point_data(target, order=feature_order, num_nodes=len(points)),
            dtype=torch.float32,
        )
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
        face = cells

    data_kwargs = dict(
        face=face,
        tetra=tetra,
        y=target_features,
        pos=torch.tensor(points, dtype=torch.float32),
        id=id,
        next_data=next_data,
        time=time,
    )

    if x_layout is not None:
        return NamedData(x=node_features, x_layout=x_layout, x_coords=x_coords, **data_kwargs)

    return Data(x=node_features, **data_kwargs)


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
    *,
    x_layout: Optional[XFeatureLayout] = None,
    x_coords: Optional[Mapping[str, object]] = None,
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
        x_layout=x_layout,
        x_coords=x_coords,
    )


def _named_point_data(graph: NamedData, names: Iterable[str]) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for name in names:
        tensor = graph.x_sel(name)
        array = tensor.detach().cpu().numpy()
        if array.ndim == 2 and array.shape[1] == 1:
            data[name] = array[:, 0]
        else:
            data[name] = array
    return data


def torch_graph_to_mesh(
    graph: Data, node_features_mapping: Optional[Union[Mapping[str, int], Sequence[str]]] = None
) -> Mesh:
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
    point_data: Dict[str, np.ndarray] = {}

    if node_features_mapping is None:
        if isinstance(graph, NamedData):
            point_data = _named_point_data(graph, graph.x_names())
        elif getattr(graph, "x", None) is not None:
            for i in range(graph.x.shape[1]):
                point_data[f"x{i}"] = graph.x[:, i].detach().cpu().numpy()
    elif isinstance(node_features_mapping, Mapping):
        for feature, index in node_features_mapping.items():
            point_data[feature] = graph.x[:, index].detach().cpu().numpy()
    else:
        if not hasattr(graph, "x_sel"):
            raise ValueError(
                "Name-based export requires a graph with named features (NamedData)."
            )
        point_data = _named_point_data(graph, node_features_mapping)

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
