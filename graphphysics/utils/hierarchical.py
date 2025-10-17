from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in lightweight environments
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
from torch_geometric.data import Data

from graphphysics.utils.torch_graph import meshdata_to_graph

try:  # Optional import to avoid circular dependency during typing
    from named_features import XFeatureLayout
except ImportError:  # pragma: no cover - fallback when named_features unavailable
    XFeatureLayout = None  # type: ignore

try:  # pragma: no cover - used in IO heavy paths
    import h5py
except ImportError:  # pragma: no cover - allow graceful skipping when unavailable
    h5py = None  # type: ignore


def _require_h5py() -> None:
    if h5py is None:  # pragma: no cover - runtime guard
        raise RuntimeError("h5py is required for hierarchical dataset utilities.")


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - runtime guard
        raise RuntimeError("NumPy is required for hierarchical dataset utilities.")


def read_h5_metadata(
    dataset_path: str, meta_path: str
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Reads trajectory indices and metadata without keeping the file open."""

    _require_h5py()
    with h5py.File(dataset_path, "r") as file_handle:
        datasets_index = list(file_handle.keys())

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    return datasets_index, len(datasets_index), meta


def get_h5_dataset(
    dataset_path: str, meta_path: str
) -> Tuple[h5py.File, List[str], int, Dict[str, Any]]:
    """Opens an H5 file and retrieves its dataset indices.

    This function opens an H5 file for reading, collects the keys of all datasets
    contained within, and returns the file handle, a list of these dataset keys,
    and the total number of datasets.

    Parameters:
        dataset_path (str): The file path of the H5 file to be opened.
        meta_path (str): The file path to the JSON file with info about the dataset.

    Returns:
        tuple: A tuple containing the following four elements:
            - h5py.File: The file handle for the opened H5 file.
            - List[str]: A list of keys representing datasets within the H5 file.
            - int: The total number of datasets within the H5 file.
            - Dict[str, Any]: The metadata dictionary loaded from the JSON file.
    """
    _require_h5py()
    file_handle = h5py.File(dataset_path, "r")
    datasets_index = list(file_handle.keys())
    with open(meta_path, "r") as fp:
        meta = json.load(fp)
    return file_handle, datasets_index, len(datasets_index), meta


def get_traj_as_meshes(
    file_handle: h5py.File, traj_number: str, meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Retrieves mesh data for an entire trajectory from an H5 file.

    This function iterates over the specified trajectory in the H5 file, converting
    each feature into its appropriate data type and shape as defined in the metadata,
    and collects them into a dictionary.

    Parameters:
        file_handle (h5py.File): An open H5 file handle.
        traj_number (str): The key of the trajectory to retrieve.
        meta (Dict[str, Any]): A dictionary containing metadata about the dataset.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are feature names and values are
        NumPy arrays containing the data for each feature across the entire trajectory.
    """
    _require_numpy()
    features = file_handle[traj_number]
    meshes = {}

    for key, field in meta["features"].items():
        data = features[key][()].astype(field["dtype"])
        data = data.reshape(field["shape"])
        meshes[key] = data

    return meshes


def _normalise_feature_array(array: np.ndarray) -> np.ndarray:
    """Ensure feature arrays are 2-D and float32."""

    if array.ndim == 1:
        array = array[:, None]
    return array.astype(np.float32, copy=False)


def _extract_frame(traj_value: np.ndarray, frame: int) -> np.ndarray:
    """Return the frame slice for the provided trajectory array."""

    if traj_value.ndim == 1:
        return traj_value
    return traj_value[frame]


def _ordered_dynamic_keys(
    keys: Iterable[str],
    *,
    feature_order: Optional[Sequence[str]] = None,
) -> List[str]:
    if feature_order:
        seen = set()
        ordered = []
        for name in feature_order:
            if name in keys and name not in seen:
                ordered.append(name)
                seen.add(name)
        remaining = sorted(name for name in keys if name not in seen)
        return ordered + remaining
    return sorted(keys)


def get_frame_as_mesh(
    traj: Dict[str, np.ndarray],
    frame: int,
    targets: Optional[Sequence[str]] = None,
    frame_target: Optional[int] = None,
    *,
    feature_layout: Optional["XFeatureLayout"] = None,
    time_value: Optional[float] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    "OrderedDict[str, np.ndarray]",
    Optional["OrderedDict[str, np.ndarray]"],
    Optional["OrderedDict[str, np.ndarray]"],
]:
    """Retrieves mesh data for a given frame from an H5 file.

    This function extracts mesh position, cell data, and additional point data
    (e.g., node type, velocity, pressure) for a specified frame. If a target frame is
    provided, it also retrieves the target frame's data.

    Parameters:
        traj (Dict[str, np.ndarray]): A dictionary where keys are feature names and values
            are NumPy arrays containing the data for each feature across the entire trajectory.
        frame (int): The index of the frame to retrieve data for.
        targets (list[str]): A list of target names to retrieve.
        frame_target (int, optional): The index of the target frame to retrieve data for.

    Returns:
        Tuple: A tuple containing the following elements:
            - np.ndarray: The positions of the mesh points.
            - np.ndarray: The indices of points forming each cell.
            - Dict[str, np.ndarray]: A dictionary containing point data.
            - Optional[Dict[str, np.ndarray]]: A dictionary containing the target frame's point data,
              similar to point_data.
    """
    _require_numpy()
    targets = list(targets or [])
    mesh_pos_source = traj["mesh_pos"]
    mesh_pos = (
        mesh_pos_source[frame] if mesh_pos_source.ndim > 1 else mesh_pos_source
    )
    num_nodes = mesh_pos.shape[0]
    cells_source = traj["cells"]
    cells = cells_source[frame] if cells_source.ndim > 1 else cells_source

    dynamic_keys = [
        key
        for key in traj.keys()
        if key not in {"mesh_pos", "cells", "node_type"}
    ]

    layout_names: Optional[List[str]] = None
    layout_sizes: Mapping[str, int] = {}
    if feature_layout is not None:
        layout_names = list(feature_layout.names())
        layout_sizes = feature_layout.sizes()

    ordered_point_data: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _zeros_for(name: str) -> np.ndarray:
        size = layout_sizes.get(name, 1)
        return np.zeros((num_nodes, size), dtype=np.float32)

    def _time_feature(size: int) -> np.ndarray:
        value = 0.0 if time_value is None else float(time_value)
        return np.full((num_nodes, size), value, dtype=np.float32)

    processed_dynamic = {key: _normalise_feature_array(_extract_frame(traj[key], frame)) for key in dynamic_keys}

    if layout_names:
        for name in layout_names:
            if name == "node_type":
                node_type_source = traj["node_type"][0]
                ordered_point_data[name] = _normalise_feature_array(node_type_source)
            elif name == "time":
                ordered_point_data[name] = _time_feature(layout_sizes.get(name, 1))
            elif name in processed_dynamic:
                ordered_point_data[name] = processed_dynamic.pop(name)
            else:
                ordered_point_data[name] = _zeros_for(name)
        for leftover in _ordered_dynamic_keys(processed_dynamic.keys()):
            ordered_point_data[leftover] = processed_dynamic[leftover]
    else:
        ordered_dynamic_keys = _ordered_dynamic_keys(processed_dynamic.keys())
        for name in ordered_dynamic_keys:
            ordered_point_data[name] = processed_dynamic[name]
        ordered_point_data["node_type"] = _normalise_feature_array(traj["node_type"][0])

    target_point_data: Optional["OrderedDict[str, np.ndarray]"] = None
    next_data: Optional["OrderedDict[str, np.ndarray]"] = None

    if frame_target is not None:
        target_point_data = OrderedDict()
        next_candidates: Dict[str, np.ndarray] = {}
        for key in targets:
            target_point_data[key] = _normalise_feature_array(
                _extract_frame(traj[key], frame_target)
            )

        for key in dynamic_keys:
            if key in targets:
                continue
            next_candidates[key] = _normalise_feature_array(
                _extract_frame(traj[key], frame_target)
            )

        if layout_names:
            ordered = [name for name in layout_names if name in next_candidates]
            next_data = OrderedDict(
                (name, next_candidates.pop(name)) for name in ordered
            )
        else:
            next_data = OrderedDict()

        if next_candidates:
            for name in _ordered_dynamic_keys(next_candidates.keys()):
                next_data[name] = next_candidates[name]

    return mesh_pos, cells, ordered_point_data, target_point_data, next_data


def get_frame_as_graph(
    traj: Dict[str, np.ndarray],
    frame: int,
    meta: Dict[str, Any],
    targets: Optional[Sequence[str]] = None,
    frame_target: Optional[int] = None,
    *,
    feature_layout: Optional["XFeatureLayout"] = None,
    x_coords: Optional[Mapping[str, Any]] = None,
) -> Data:
    """Converts mesh data for a given frame into a graph representation.

    This function first retrieves mesh data using `get_frame_as_mesh` and then
    converts this data into a graph representation using the `meshdata_to_graph`
    function from the `torch_graph` module.

    Parameters:
        traj (Dict[str, np.ndarray]): A dictionary where keys are feature names and values
            are NumPy arrays containing the data for each feature across the entire trajectory.
        frame (int): The index of the frame to retrieve and convert.
        meta (Dict[str, Any]): A dictionary containing metadata about the dataset.
        targets (list[str]): A list of target names to retrieve.
        frame_target (int, optional): The index of the target frame to retrieve and convert.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object representing the graph.
    """
    _require_numpy()
    time = frame * meta.get("dt", 1)
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj,
        frame,
        targets,
        frame_target,
        feature_layout=feature_layout,
        time_value=time,
    )
    return meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        time=time,
        target=target,
        next_data=next_data,
        x_layout=feature_layout,
        x_coords=x_coords,
    )
