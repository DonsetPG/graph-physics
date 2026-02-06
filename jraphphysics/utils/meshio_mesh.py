from __future__ import annotations

from typing import Dict, List

import meshio
import numpy as np
import jraph


def _infer_faces_from_edges(senders: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    # Conservative fallback: encode edges as degenerate triangles for export.
    return np.stack([senders, receivers, receivers], axis=1)


def convert_to_meshio_vtu(
    graph: jraph.GraphsTuple,
    add_all_data: bool = False,
) -> meshio.Mesh:
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
    faces = _infer_faces_from_edges(senders, receivers)
    cells = [("triangle", faces)]
    mesh = meshio.Mesh(vertices, cells)

    if add_all_data and "features" in graph.nodes:
        x_data = np.asarray(graph.nodes["features"])
        for idx in range(x_data.shape[1]):
            mesh.point_data[f"x{idx}"] = x_data[:, idx]

    if add_all_data and graph.globals and "target_features" in graph.globals:
        y_data = np.asarray(graph.globals["target_features"])
        for idx in range(y_data.shape[1]):
            mesh.point_data[f"y{idx}"] = y_data[:, idx]

    return mesh


def vtu_to_xdmf(
    filename: str,
    files_list: List[str],
    timestep: float = 1.0,
    remove_vtus: bool = True,
) -> None:
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    init_vtu = meshio.read(files_list[0])
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        writer.write_points_cells(init_vtu.points, init_vtu.cells)
        t = 0.0
        for file in files_list:
            mesh = meshio.read(file)
            writer.write_data(t, point_data=mesh.point_data, cell_data=mesh.cell_data)
            t += timestep

    # Move generated h5 archive next to xdmf file if required.
    generated_h5 = xdmf_filename.replace(".xdmf", ".h5")
    if generated_h5 != h5_filename:
        import shutil

        shutil.move(generated_h5, h5_filename)

    if remove_vtus:
        import os

        for file in files_list:
            os.remove(file)
