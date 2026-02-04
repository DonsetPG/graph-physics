import os
import shutil
import meshio
import pathlib
import re

MOCK_TFR_PATH = "tests/mock_tfr"
MOCK_TR_SPLIT = "train"
MOCK_H5_SAVE_PATH = "tests/mock_h5/write_mock.h5"
MOCK_H5_TEST_PATH = "tests/mock_h5/write_mock_test.h5"
MOCK_H5_META_SAVE_PATH = "tests/mock_h5/meta.json"
MOCK_H5_TARGETS = ["velocity"]
MOCK_H5_META10_SAVE_PATH = "tests/mock_h5/meta10.json"
MOCK_VTU_FOLDER_PATH = "tests/mock_vtu"
MOCK_VTU_PATH = os.path.join(MOCK_VTU_FOLDER_PATH, "cylinder_0.vtu")
MOCK_VTU_ANEURYSM_FOLDER_PATH = "tests/mock_vtu_aneurysm"
MOCK_XDMF_FOLDER = "tests/mock_xdmf"
MOCK_XDMF_FOLDER_TEST = "tests/mock_xdmf_test"
os.makedirs(MOCK_XDMF_FOLDER, exist_ok=True)
os.makedirs(MOCK_XDMF_FOLDER_TEST, exist_ok=True)
MOCK_XDMF_PATH = os.path.join(MOCK_XDMF_FOLDER, "mock.xdmf")
MOCK_XDMF_H5_PATH = os.path.join(MOCK_XDMF_FOLDER, "mock.h5")
MOCK_XDMF_TARGETS = ["velocity_x", "velocity_y"]


def get_meshs_from_vtu():
    file_list = [
        os.path.join(MOCK_VTU_FOLDER_PATH, f)
        for f in os.listdir(MOCK_VTU_FOLDER_PATH)
        if os.path.isfile(os.path.join(MOCK_VTU_FOLDER_PATH, f))
    ]
    return [meshio.read(file) for file in file_list]


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)


def regroup_h5(xdmf_file_path: str):
    filename = pathlib.Path(xdmf_file_path)
    if str(filename.parent) != ".":
        shutil.move(f"{filename.stem}.h5", str(filename.parent))


def files_to_xdmf(folder_name: str, path: str, time_step=1):
    """
    Converts a series of mesh files in a folder to a single XDMF file.

    This function reads mesh files from a specified folder, assuming they represent a
    time series or sequence of meshes, and writes them to an XDMF file.

    Parameters:
        - folder_name: The name of the folder containing the mesh files to be converted.
        - path: The file path where the XDMF file will be saved.
        - time_step: The time between each step.

    Note:
    The function assumes that all Mesh objects in the list have the same points and cells
    structure, using the first Mesh object in the list as a reference for writing the shared
    points and cells data.
    """
    file_list = [
        os.path.join(folder_name, f)
        for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f))
    ]

    if not file_list:
        raise FileNotFoundError("No files found in the specified folder.")

    sort_nicely(file_list)

    init_mesh = meshio.read(file_list[0])
    points, cells = init_mesh.points, init_mesh.cells

    with meshio.xdmf.TimeSeriesWriter(path) as writer:
        writer.write_points_cells(points, cells)
        for t, mesh_file in enumerate(file_list):
            mesh = meshio.read(mesh_file)
            writer.write_data(
                t * time_step, point_data=mesh.point_data, cell_data=mesh.cell_data
            )
    regroup_h5(path)
