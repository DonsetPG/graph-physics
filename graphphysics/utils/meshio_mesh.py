import copy
import os
import shutil
from typing import List

import h5py
import meshio
import numpy as np
from lxml import etree
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
    faces = (
        (graph.tetra if getattr(graph, "tetra", None) is not None else graph.face)
        .cpu()
        .numpy()
        .T
    )
    cells = [
        ("tetra" if getattr(graph, "tetra", None) is not None else "triangle", faces)
    ]

    # Create Meshio mesh
    mesh = meshio.Mesh(vertices, cells)

    # Optionally add all node features as point data
    if add_all_data and hasattr(graph, "x") and graph.x is not None:
        x_data = graph.x.cpu().numpy()
        for i in range(x_data.shape[1]):
            mesh.point_data[f"x{i}"] = x_data[:, i]

    # Optionally add node targets as point data
    if add_all_data and hasattr(graph, "y") and graph.y is not None:
        y_data = graph.y.cpu().numpy()
        for i in range(y_data.shape[1]):
            mesh.point_data[f"y{i}"] = y_data[:, i]

    return mesh


def vtu_to_xdmf(
    filename: str, files_list: List[str], timestep=1, remove_vtus: bool = True
) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format from VTU files.

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


def meshes_to_xdmf(
    filename: str,
    meshes: List[meshio.Mesh],
    timestep=1,
) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format from meshio.Mesh objects.

    Args:
        filename (str): Name for the XDMF/HDF5 file without the extension.
        meshes (List[meshio.Mesh]): List of the meshes to compress.
        timestep (float, optional): Timestep between to consecutive timeframes.

    Returns:
        None: XDMF/HDF5 file is saved to the path filename.
    """

    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    points = meshes[0].points
    cells = meshes[0].cells

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        t = 0
        for mesh in meshes:
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # The H5 archive is systematically created in cwd, we just need to move it
    shutil.move(
        src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename
    )


def append_mesh_to_xdmf(
    filename: str, mesh: meshio.Mesh, timestep: float = 1.0, compress=False
) -> None:
    """
    Appends a single timeframe to an existing XDMF/HDF5 time series archive,
    without loading the existing data into RAM.

    Args:
        filename (str): Path to the existing archive (without extension).
        mesh (meshio.Mesh): Mesh object for the new timeframe.
                            Must share the same points and cells as the existing archive.
        timestep (float): Time increment added to the last recorded timestep.

    Returns:
        None: XDMF/HDF5 file is saved to the path filename.
    """

    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    if not os.path.exists(h5_filename) or not os.path.exists(xdmf_filename):
        raise FileNotFoundError(f"XDMF/HDF5 file not found: {filename}")

    # Get temporal grid to get the last timestep grid
    tree = etree.parse(xdmf_filename)
    root = tree.getroot()
    temporal_grid = root.find(
        ".//{*}Grid[@GridType='Collection'][@CollectionType='Temporal']"
    )
    if temporal_grid is None:
        raise ValueError(
            "Could not find the temporal grid collection in the XDMF file."
        )

    time_grids = temporal_grid.findall("Grid")

    last_grid = time_grids[-1]
    last_time = float(last_grid.find("Time").get("Value"))
    new_time = last_time + timestep

    # Add data to the H5 file.
    with h5py.File(h5_filename, "a") as h5:
        existing_h5_keys = h5.keys()
        last_h5_key_idx = max(int(k.replace("data", "")) for k in existing_h5_keys)
        h5_new_keys_mapping = {}
        for i, (field_name, field_values) in enumerate(mesh.point_data.items()):
            new_key = f"data{last_h5_key_idx + (i+1)}"
            if compress:
                h5.create_dataset(
                    new_key,
                    data=np.asarray(field_values),
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,  # Default value for meshio, ranges from 0 to 9
                )
            else:
                h5.create_dataset(new_key, data=np.asarray(field_values))
            h5_new_keys_mapping[field_name] = new_key

    # Add a new grid to the XDMF file.
    new_grid = copy.deepcopy(last_grid)
    new_grid.find("Time").set("Value", str(new_time))

    for data_item in new_grid.iter():
        if data_item.tag == "Attribute":
            next_h5_key = h5_new_keys_mapping[data_item.attrib["Name"]]
        if data_item.tag == "DataItem":
            data_item.text = f"{os.path.basename(h5_filename)}:/{next_h5_key}"

    temporal_grid.append(new_grid)
    tree.write(
        xdmf_filename, pretty_print=False, xml_declaration=False, encoding="UTF-8"
    )
