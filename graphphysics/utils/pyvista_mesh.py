import numpy as np
import pyvista as pv
from torch_geometric.data import Data


def convert_to_pyvista_mesh(graph: Data, add_all_data: bool = False) -> pv.PolyData:
    """
    Converts a PyTorch Geometric graph to a PyVista PolyData mesh.

    Args:
        graph (Data): The graph data to convert.
        add_all_data (bool, optional): If True, adds all node features from graph.x to the mesh point data.

    Returns:
        pv.PolyData: The converted PyVista mesh.
    """
    # Ensure 'pos' attribute exists
    if not hasattr(graph, "pos") or graph.pos is None:
        raise ValueError("Graph must have 'pos' attribute with node positions.")

    # Extract node positions and ensure they have three coordinates
    vertices = graph.pos.cpu().numpy()
    num_coords = vertices.shape[1]
    if num_coords < 3:
        # Pad with zeros to make it 3D
        padding = np.zeros((vertices.shape[0], 3 - num_coords), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding])
    elif num_coords > 3:
        raise ValueError(f"Unsupported vertex dimension: {num_coords}")

    # Ensure 'edge_index' attribute exists
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("Graph must have 'edge_index' attribute with edge indices.")

    # Extract edges and create lines for PyVista
    edges = graph.edge_index.t().cpu().numpy()
    num_edges = edges.shape[0]
    lines = np.hstack([np.full((num_edges, 1), 2, dtype=np.int64), edges]).flatten()

    # Create PyVista mesh
    mesh = pv.PolyData(vertices, lines=lines)

    # Optionally add all node features as point data
    if add_all_data and hasattr(graph, "x") and graph.x is not None:
        x_data = graph.x.cpu().numpy()
        for i in range(x_data.shape[1]):
            mesh.point_data[f"x{i}"] = x_data[:, i]

    return mesh
