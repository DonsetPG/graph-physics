import os
import shutil
from typing import List

import meshio
import numpy as np
from torch_geometric.data import Data


def convert_to_meshio_vtu(graph: Data, add_all_data: bool = False) -> meshio.Mesh:
    """
    Converts a PyTorch Geometric graph to a Meshio mesh.

    Args:
        graph (Data): The graph data to convert.
        add_all_data (bool, optional): If True, adds all node features from graph.x to the mesh point data.

    Returns:
        meshio.Mesh: The converted Meshio mesh.
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

    # Ensure 'faces' attribute exists
    if not hasattr(graph, "face") or graph.face is None:
        raise ValueError("Graph must have 'face' attribute with face indices.")

    # Extract faces
    faces = (graph.tetra if getattr(graph, "tetra", None) is not None else graph.face).cpu().numpy().T
    cells = [("tetra" if getattr(graph, "tetra", None) is not None else "triangle", faces)]

    # Create Meshio mesh
    mesh = meshio.Mesh(vertices, cells)

    # Optionally add all node features as point data
    if add_all_data and hasattr(graph, "x") and graph.x is not None:
        x_data = graph.x.cpu().numpy()
        for i in range(x_data.shape[1]):
            mesh.point_data[f"x{i}"] = x_data[:, i]

    return mesh


def vtu_to_xdmf(
    filename: str, files_list: List[str], timestep=1, remove_vtus: bool = True
) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format.

    Args:
        filename (str): Name for the XDMF/HDF5 file without the extension.
        files_list (List[str]): List of the files' paths to compress.
        timestep (float, optional): Timestep between to consecutive timeframes.
        remove_vtus (bool, optional): If True, remove the original vtu files.

    Returns:
        None: XDMF/HDF5 file is saved to the path filename.
    """
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    init_vtu = meshio.read(files_list[0])
    points = init_vtu.points
    cells = init_vtu.cells

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        t = 0
        for file in files_list:
            mesh = meshio.read(file)
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # The H5 archive is systematically created in cwd, we just need to move it
    shutil.move(
        src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename
    )

    # Remove the original vtu files
    if remove_vtus:
        for file in files_list:
            os.remove(file)
