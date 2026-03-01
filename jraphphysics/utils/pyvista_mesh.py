from __future__ import annotations

import numpy as np
import pyvista as pv
import jraph


def convert_to_pyvista_mesh(
    graph: jraph.GraphsTuple,
    add_all_data: bool = False,
) -> pv.PolyData:
    if "pos" not in graph.nodes:
        raise ValueError("Graph must have 'pos' in nodes.")

    vertices = np.asarray(graph.nodes["pos"])
    if vertices.shape[1] < 3:
        padding = np.zeros((vertices.shape[0], 3 - vertices.shape[1]), dtype=vertices.dtype)
        vertices = np.hstack([vertices, padding])
    elif vertices.shape[1] > 3:
        raise ValueError(f"Unsupported vertex dimension: {vertices.shape[1]}")

    senders = np.asarray(graph.senders)
    receivers = np.asarray(graph.receivers)
    edges = np.stack([senders, receivers], axis=1)
    num_edges = edges.shape[0]
    lines = np.hstack([np.full((num_edges, 1), 2, dtype=np.int64), edges]).reshape(-1)

    mesh = pv.PolyData(vertices, lines=lines)
    if add_all_data and "features" in graph.nodes:
        x_data = np.asarray(graph.nodes["features"])
        for idx in range(x_data.shape[1]):
            mesh.point_data[f"x{idx}"] = x_data[:, idx]
    return mesh
