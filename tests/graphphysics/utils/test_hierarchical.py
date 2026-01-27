from graphphysics.utils.hierarchical import (
    get_frame_as_graph,
    get_frame_as_mesh,
    get_h5_dataset,
    get_traj_as_meshes,
)
from tests.mock import MOCK_H5_META_SAVE_PATH, MOCK_H5_SAVE_PATH, MOCK_H5_TARGETS


def test_get_h5_dataset():
    file_handle, datasets_index, size_dataset, meta = get_h5_dataset(
        MOCK_H5_SAVE_PATH, MOCK_H5_META_SAVE_PATH
    )
    assert size_dataset == 1


def test_get_traj_as_meshs():
    file_handle, datasets_index, size_dataset, meta = get_h5_dataset(
        MOCK_H5_SAVE_PATH, MOCK_H5_META_SAVE_PATH
    )
    traj = get_traj_as_meshes(file_handle, "0", meta)
    assert traj["mesh_pos"].shape == (1, 1876, 2)
    assert traj["cells"].shape == (1, 3518, 3)
    assert traj["velocity"].shape == (600, 1876, 2)
    assert traj["pressure"].shape == (600, 1876, 1)


def test_get_frame_as_mesh():
    file_handle, datasets_index, size_dataset, meta = get_h5_dataset(
        MOCK_H5_SAVE_PATH, MOCK_H5_META_SAVE_PATH
    )
    traj = get_traj_as_meshes(file_handle, "0", meta)
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj=traj, frame=0, targets=MOCK_H5_TARGETS
    )
    assert points.shape[0] == 1876
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj=traj, frame=0, targets=["velocity"], frame_target=1
    )
    assert target["velocity"].shape[1] == 2
    assert "pressure" in next_data
    assert next_data["pressure"].shape[1] == 1
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj=traj, frame=0, targets=["pressure"], frame_target=1
    )
    assert target["pressure"].shape[1] == 1
    assert "velocity" in next_data
    assert next_data["velocity"].shape[1] == 2
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj=traj, frame=0, targets=["velocity", "pressure"], frame_target=1
    )
    assert target["velocity"].shape[1] == 2
    assert target["pressure"].shape[1] == 1
    assert next_data == {}


def test_get_frame_as_graph():
    file_handle, datasets_index, size_dataset, meta = get_h5_dataset(
        MOCK_H5_SAVE_PATH, MOCK_H5_META_SAVE_PATH
    )
    traj = get_traj_as_meshes(file_handle, "0", meta)
    graph = get_frame_as_graph(traj=traj, frame=0, meta=meta, targets=MOCK_H5_TARGETS)

    assert graph.x.shape[0] == 1876
