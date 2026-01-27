from typing import Dict, List, Optional, Union

import meshio
import numpy as np
import jax
import jax.numpy as jnp
import jraph
from meshio import Mesh
import jax.numpy as jnp
import numpy as np
import jraph
from typing import Optional, Dict, Union, Tuple, Any, List


def compute_k_hop_edge_index(
    edge_index: jnp.ndarray,
    num_hops: int,
    num_nodes: int,
) -> jnp.ndarray:
    """Computes the k-hop edge index for a given edge index array.

    Parameters:
        edge_index (jnp.ndarray): The edge index array of shape [2, num_edges].
        num_hops (int): The number of hops.
        num_nodes (int): The number of nodes.

    Returns:
        jnp.ndarray: The edge index array representing the k-hop edges.
    """
    # Build the adjacency matrix as a boolean matrix
    adj = jnp.zeros((num_nodes, num_nodes), dtype=bool)
    adj = adj.at[edge_index[0], edge_index[1]].set(True)

    adj_k = adj.copy()

    for _ in range(num_hops - 1):
        adj_k = adj_k | (adj_k @ adj)
        # Remove self-loops
        adj_k = adj_k.at[jnp.diag_indices(num_nodes)].set(False)

    # Get the indices where adj_k is True
    khop_edge_index = jnp.array(jnp.nonzero(adj_k))
    return khop_edge_index


def compute_k_hop_graph(
    graph: jraph.GraphsTuple,
    num_hops: int,
) -> jraph.GraphsTuple:
    """Builds a k-hop mesh graph.

    Parameters:
        graph (jraph.GraphsTuple): The input graph data.
        num_hops (int): The number of hops.
        add_edge_features_to_khop (bool): Whether to compute edge features for the k-hop graph.

    Returns:
        jraph.GraphsTuple: The k-hop graph data.
    """
    if num_hops == 1:
        return graph

    edge_index = jnp.vstack([graph.senders, graph.receivers])
    num_nodes = graph.nodes["features"].shape[0]

    khop_edge_index = compute_k_hop_edge_index(
        edge_index=edge_index,
        num_hops=num_hops,
        num_nodes=num_nodes,
    )

    # Build k-hop graph
    khop_graph = jraph.GraphsTuple(
        nodes=graph.nodes,
        edges=None,
        senders=khop_edge_index[0],
        receivers=khop_edge_index[1],
        n_node=graph.n_node,
        n_edge=jnp.array([khop_edge_index.shape[1]]),
        globals=graph.globals,
    )

    return khop_graph


def face_to_edge(faces: jnp.ndarray):
    """
    Converts mesh faces to edge indices.

    Args:
        faces (jnp.ndarray): An array of shape [num_faces, nodes_per_face], e.g., [num_faces, 3] for triangles.

    Returns:
        senders (jnp.ndarray): Sender node indices of shape [num_edges].
        receivers (jnp.ndarray): Receiver node indices of shape [num_edges].
    """
    # Transpose faces to match the shape [nodes_per_face, num_faces]
    face = faces.T  # Shape: [nodes_per_face, num_faces]

    # Extract edges from faces
    edge_index = jnp.concatenate(
        [
            face[:2],  # Edges between nodes 0 and 1
            face[1:],  # Edges between nodes 1 and 2
            face[::2],  # Edges between nodes 0 and 2
        ],
        axis=1,
    )  # Shape: [2, num_edges]

    senders = edge_index[0]
    receivers = edge_index[1]

    # Make edges undirected by adding reversed edges
    undirected_senders = jnp.concatenate([senders, receivers])
    undirected_receivers = jnp.concatenate([receivers, senders])

    # Remove duplicate edges
    edges = jnp.stack([undirected_senders, undirected_receivers], axis=1)
    # Sort each edge to ensure (min, max) ordering for undirected edges
    edges = jnp.sort(edges, axis=1)
    # Get unique edges
    unique_edges = jnp.unique(edges, axis=0)

    # Final senders and receivers
    senders = unique_edges[:, 0]
    receivers = unique_edges[:, 1]

    return senders, receivers


def meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]],
    time: Union[int, float] = 1,
    target: Optional[Dict[str, np.ndarray]] = None,
    return_only_node_features: bool = False,
) -> Union[jraph.GraphsTuple, jnp.ndarray]:
    """Converts mesh data into a Jraph GraphsTuple object.

    Parameters:
        points (np.ndarray): The coordinates of the mesh points.
        cells (np.ndarray): The connectivity of the mesh (how points form cells).
        point_data (Dict[str, np.ndarray]): A dictionary of point-associated data.
        time (int or float): A scalar value representing the time step.
        target (Dict[str, np.ndarray], optional): An optional dictionary of target data.
        return_only_node_features (bool): Whether to return only node features.

    Returns:
        Union[jraph.GraphsTuple, jnp.ndarray]: A Jraph GraphsTuple object representing the mesh, or node features.
    """
    if point_data is not None:
        if any(data.ndim > 1 for data in point_data.values()):
            # if any(data.shape[1] > 1 for data in point_data.values()):
            node_features = np.hstack(
                [data for data in point_data.values()]
                + [np.full((len(points),), time).reshape((-1, 1))]
            )
            node_features = jnp.array(node_features, dtype=jnp.float32)
        else:
            node_features = np.vstack(
                [data for data in point_data.values()] + [np.full((len(points),), time)]
            ).T
            node_features = jnp.array(node_features, dtype=jnp.float32)
    else:
        node_features = jnp.zeros((len(points), 1), dtype=jnp.float32)

    if return_only_node_features:
        return node_features

    # Convert target to tensor if provided
    if target is not None:
        if any(data.ndim > 1 for data in target.values()):
            # if any(data.shape[1] > 1 for data in target.values()):
            target_features = np.hstack([data for data in target.values()])
            target_features = jnp.array(target_features, dtype=jnp.float32)
            globals_dict = {"target_features": target_features}
        else:
            target_features = np.vstack([data for data in target.values()]).T
            target_features = jnp.array(target_features, dtype=jnp.float32)
            globals_dict = {"target_features": target_features}
    else:
        globals_dict = {}
    # Generate edges from cells
    senders, receivers = face_to_edge(cells)

    # Create the graph
    nodes = {"features": node_features, "pos": jnp.array(points, dtype=jnp.float32)}
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([len(points)]),
        n_edge=jnp.array([len(senders)]),
        globals=globals_dict,
    )

    return graph


def batched_meshdata_to_graph(
    points: np.ndarray,
    cells: np.ndarray,
    point_data: Optional[Dict[str, np.ndarray]],
    time: Union[int, float, np.ndarray] = 1,
    target: Optional[Dict[str, np.ndarray]] = None,
    return_only_node_features: bool = False,
) -> jraph.GraphsTuple:
    """
    Converts batched mesh data into a batched Jraph GraphsTuple object.

    Parameters:
        points (np.ndarray): The coordinates of the mesh points, shape [batch_size, num_points, num_dims].
        cells (np.ndarray): The connectivity of the mesh, shape [batch_size, num_cells, num_points_per_cell].
        point_data (Dict[str, np.ndarray]): A dictionary of point-associated data, each of shape [batch_size, num_points, feature_dim].
        time (int or float or np.ndarray): A scalar value or array representing the time step, shape [batch_size].
        target (Dict[str, np.ndarray], optional): An optional dictionary of target data, each of shape [batch_size, ...].
        return_only_node_features (bool): Whether to return only node features.

    Returns:
        Union[jraph.GraphsTuple, jnp.ndarray]: A batched Jraph GraphsTuple object representing the meshes, or batched node features.
    """
    batch_size = points.shape[0]
    all_graphs = []
    all_node_features = []

    for i in range(batch_size):
        # Get data for the i-th graph
        graph_points = points[i]  # shape [num_points, num_dims]
        graph_cells = cells[i]  # shape [num_cells, num_points_per_cell]
        graph_point_data = (
            {k: v[i] for k, v in point_data.items()} if point_data is not None else None
        )
        graph_time = time if isinstance(time, (int, float)) else time[i]
        graph_target = (
            {k: v[i] for k, v in target.items()} if target is not None else None
        )

        # Compute node features
        if graph_point_data is not None:
            # Concatenate point data features and time
            features_list = list(graph_point_data.values())
            time_feature = np.full((len(graph_points), 1), graph_time)
            features_list.append(time_feature)
            node_features = np.hstack(features_list)
            node_features = jnp.array(node_features, dtype=jnp.float32)
        else:
            node_features = jnp.full(
                (len(graph_points), 1), graph_time, dtype=jnp.float32
            )

        if return_only_node_features:
            # Collect node features
            all_node_features.append(node_features)
            continue  # Proceed to the next graph

        # Process target features
        if graph_target is not None:
            target_features = np.hstack(list(graph_target.values()))
            target_features = jnp.array(target_features, dtype=jnp.float32)
            globals_dict = {"target_features": target_features}
        else:
            globals_dict = {}

        # Generate edges from cells
        senders, receivers = face_to_edge(graph_cells)

        # Nodes
        nodes = {
            "features": node_features,
            "pos": jnp.array(graph_points, dtype=jnp.float32),
        }

        # Create the graph
        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([len(graph_points)]),
            n_edge=jnp.array([len(senders)]),
            globals=globals_dict,
        )

        # Append the graph to the list
        all_graphs.append(graph)

    if return_only_node_features:
        # Concatenate node features along the batch dimension
        batched_node_features = jnp.stack(all_node_features, axis=0)
        return batched_node_features

    # Batch the graphs
    batched_graph = jraph.batch(all_graphs)

    return batched_graph


def mesh_to_graph(
    mesh: Mesh,
    time: Union[int, float] = 1,
    target_mesh: Optional[Mesh] = None,
    target_fields: Optional[List[str]] = None,
) -> jraph.GraphsTuple:
    """Converts mesh and optional target mesh data into a Jraph GraphsTuple object.

    Parameters:
        mesh (Mesh): A Mesh object containing the mesh data.
        time (int or float): A scalar value representing the time step.
        target_mesh (Mesh, optional): An optional Mesh object containing target data.
        target_fields (List[str], optional): Fields from the target_mesh to be used as the target data.

    Returns:
        jraph.GraphsTuple: A Jraph GraphsTuple object representing the mesh with optional target data.
    """
    # Prepare target data if a target mesh is provided
    target = None
    if target_mesh is not None and target_fields:
        target_data = {field: target_mesh.point_data[field] for field in target_fields}
        target = target_data

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


def graph_to_mesh(
    graph: jraph.GraphsTuple, node_features_mapping: Dict[str, int]
) -> Mesh:
    """Converts a Jraph GraphsTuple graph to a meshio Mesh object.

    Parameters:
        graph (jraph.GraphsTuple): The graph to convert.
        node_features_mapping (Dict[str, int]): Mapping from feature names to indices in the node features.

    Returns:
        Mesh: A meshio Mesh object containing the graph's geometric and feature data.
    """
    # Extract point data
    node_features = graph.nodes["features"]
    point_data = {
        key: node_features[:, idx] for key, idx in node_features_mapping.items()
    }

    points = np.array(graph.nodes["pos"])

    # Reconstruct cells (this is non-trivial without original cell information)
    # Here, we assume that senders and receivers can reconstruct the faces
    # For simplicity, we'll leave cells empty or you can customize as needed
    cells = []

    return meshio.Mesh(
        points,
        cells,
        point_data=point_data,
    )


def get_masked_indexes(
    graph: jraph.GraphsTuple, masking_ratio: float = 0.15
) -> jnp.ndarray:
    """Generate masked indices for the input graph based on the masking ratio.

    Args:
        graph (jraph.GraphsTuple): The input graph data.
        masking_ratio (float): The ratio of nodes to mask.

    Returns:
        selected_indices (jnp.ndarray): The indices of nodes to keep after masking.
    """
    n = graph.nodes["features"].shape[0]
    nodes_to_keep = 1 - masking_ratio
    num_rows_to_sample = int(nodes_to_keep * n)
    # Generate random indices
    random_indices = jax.random.permutation(jax.random.PRNGKey(0), n)
    selected_indices = random_indices[:num_rows_to_sample]

    return selected_indices
