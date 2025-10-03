from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import get_worker_info
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.hierarchical import (
    get_frame_as_graph,
    get_traj_as_meshes,
    read_h5_metadata,
)

try:
    import h5py
except ImportError as exc:
    raise RuntimeError("h5py is required to use H5Dataset") from exc


class H5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        world_pos_parameters: Optional[dict] = None,
        cache_size: int = 8,
    ):
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            world_pos_parameters=world_pos_parameters,
        )

        self.type = "h5"

        if switch_to_val:
            h5_path = h5_path.replace("train", "test")

        self.h5_path = h5_path
        self.meta_path = meta_path
        self.cache_size = cache_size

        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "The dataset has a timestep set to 0. Fallback to dt=1 to ensure xdmf can be saved."
            )

        (
            self.datasets_index,
            self._size_dataset,
            self.meta,
        ) = read_h5_metadata(dataset_path=h5_path, meta_path=meta_path)

        self._file_handles: dict[str, h5py.File] = {}
        self._trajectory_cache: OrderedDict[str, dict] = OrderedDict()
        self._frame_cache: OrderedDict[
            tuple[str, int], Tuple[Data, Optional[torch.Tensor]]
        ] = OrderedDict()

    @property
    def size_dataset(self) -> int:
        """Returns the number of trajectories in the dataset."""
        return self._size_dataset

    def _close_file_handles(self):
        for handle in self._file_handles.values():
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        self._file_handles.clear()

    def _get_file_handle(self) -> h5py.File:
        worker_info = get_worker_info()
        worker_id = str(worker_info.id) if worker_info is not None else "main"
        handle = self._file_handles.get(worker_id)
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._file_handles[worker_id] = handle
        return handle

    def _get_trajectory(self, traj_number: str) -> dict:
        cached = self._trajectory_cache.get(traj_number)
        if cached is not None:
            self._trajectory_cache.move_to_end(traj_number)
            return cached

        file_handle = self._get_file_handle()
        trajectory = get_traj_as_meshes(
            file_handle=file_handle, traj_number=traj_number, meta=self.meta
        )
        self._trajectory_cache[traj_number] = trajectory
        if len(self._trajectory_cache) > self.cache_size:
            self._trajectory_cache.popitem(last=False)
        return trajectory

    def _cache_graph(
        self,
        key: tuple[str, int],
        graph: Data,
        selected_indices: Optional[torch.Tensor],
    ) -> None:
        self._frame_cache[key] = (graph, selected_indices)
        self._frame_cache.move_to_end(key)
        if len(self._frame_cache) > self.cache_size * 2:
            self._frame_cache.popitem(last=False)
        return None

    def _build_node_features(self, traj: dict, frame: int) -> torch.Tensor:
        time = frame * self.meta.get("dt", 1)

        point_data = {
            key: (traj[key][frame] if traj[key].ndim > 1 else traj[key])
            for key in traj.keys()
            if key not in ["mesh_pos", "cells", "node_type"]
        }
        point_data["node_type"] = traj["node_type"][0]

        arrays = []
        for data in point_data.values():
            arr = data
            if arr.ndim == 1:
                arr = arr[:, None]
            arrays.append(arr.astype(np.float32))

        if arrays:
            node_features = np.concatenate(arrays, axis=1)
        else:
            node_features = np.zeros((traj["mesh_pos"].shape[-2], 0), dtype=np.float32)

        time_column = np.full((node_features.shape[0], 1), time, dtype=np.float32)
        node_features = np.concatenate([node_features, time_column], axis=1)

        return torch.from_numpy(node_features)

    def _get_processed_graph(
        self,
        traj_index: int,
        frame: int,
        traj: Optional[dict] = None,
    ) -> Tuple[Data, Optional[torch.Tensor]]:
        traj_number = self.datasets_index[traj_index]
        cache_key = (traj_number, frame)

        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            graph, selected_indices = cached
        else:
            if traj is None:
                traj = self._get_trajectory(traj_number)

            graph = get_frame_as_graph(
                traj=traj,
                frame=frame,
                meta=self.meta,
                targets=self.targets,
                frame_target=frame + 1,
            )

            graph = self._apply_preprocessing(graph)
            graph = self._apply_k_hop(graph, traj_index)
            graph = self._may_remove_edges_attr(graph)
            graph = self._add_random_edges(graph)
            selected_indices = self._get_masked_indexes(graph)

            graph.edge_index = (
                graph.edge_index.long() if graph.edge_index is not None else None
            )

            self._cache_graph(cache_key, graph, selected_indices)

        graph_out = graph.clone()
        selected_out = (
            selected_indices.clone() if selected_indices is not None else None
        )
        return graph_out, selected_out

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """Retrieve a graph representation of a frame from a trajectory.

        This method extracts a single frame from a trajectory based on the index provided.
        It first determines the trajectory and frame number using `get_traj_frame` method.
        Then, it retrieves the trajectory data as meshes and converts the specified frame
        into a graph representation.

        Parameters:
            index (int): The index of the item in the dataset.

        Returns:
            Union[Data, Tuple[Data, torch.Tensor]]: A graph representation of the specified frame in the trajectory,
            optionally along with selected indices if masking is applied.
        """
        traj_index, frame = self.get_traj_frame(index=index)
        traj_number = self.datasets_index[traj_index]
        traj = self._get_trajectory(traj_number)

        graph, selected_indices = self._get_processed_graph(
            traj_index=traj_index, frame=frame, traj=traj
        )

        next_features = self._build_node_features(traj, frame + 1)
        graph.next_data = next_features

        if self.use_previous_data:
            previous_features = self._build_node_features(traj, frame - 1)
            graph.previous_data = previous_features

        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph

    def __del__(self):
        """Ensure that the H5 file is properly closed."""
        self._close_file_handles()
