import json
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import jraph

from jraphphysics.utils.jax_graph import meshdata_to_graph


def read_h5_metadata(
    dataset_path: str, meta_path: str
) -> Tuple[List[str], int, Dict[str, Any]]:
    with h5py.File(dataset_path, "r") as file_handle:
        datasets_index = list(file_handle.keys())
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return datasets_index, len(datasets_index), meta


def get_h5_dataset(
    dataset_path: str, meta_path: str
) -> Tuple[h5py.File, List[str], int, Dict[str, Any]]:
    file_handle = h5py.File(dataset_path, "r")
    datasets_index = list(file_handle.keys())
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return file_handle, datasets_index, len(datasets_index), meta


def get_traj_as_meshes(
    file_handle: h5py.File, traj_number: str, meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    features = file_handle[traj_number]
    meshes = {}
    for key, field in meta["features"].items():
        data = features[key][()].astype(field["dtype"])
        data = data.reshape(field["shape"])
        meshes[key] = data
    return meshes


def get_frame_as_mesh(
    traj: Dict[str, np.ndarray],
    frame: int,
    targets: list[str] = None,
    frame_target: Optional[int] = None,
):
    target_point_data = None
    next_data = None

    if frame_target is not None and targets is not None:
        target_point_data = {key: traj[key][frame_target] for key in targets}
        next_data = {
            key: traj[key][frame_target]
            for key in traj.keys()
            if key not in ["mesh_pos", "cells", "node_type"] and key not in targets
        }

    point_data = {
        key: traj[key][frame]
        for key in traj.keys()
        if key not in ["mesh_pos", "cells", "node_type"]
    }
    point_data["node_type"] = traj["node_type"][0]

    mesh_pos = (
        traj["mesh_pos"][frame] if traj["mesh_pos"].ndim > 1 else traj["mesh_pos"]
    )
    cells = traj["cells"][frame] if traj["cells"].ndim > 1 else traj["cells"]
    return mesh_pos, cells, point_data, target_point_data, next_data


def get_frame_as_graph(
    traj: Dict[str, np.ndarray],
    frame: int,
    meta: Dict[str, Any],
    targets: list[str] = None,
    frame_target: Optional[int] = None,
) -> jraph.GraphsTuple:
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj, frame, targets, frame_target
    )
    del next_data
    time = frame * meta.get("dt", 1)
    return meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        time=time,
        target=target,
    )
