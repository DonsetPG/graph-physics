import os
import random
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.torch_graph import meshdata_to_graph


class XDMFDataset(BaseDataset):
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        random_prev: int = 1,  # If we use previous data, we will fetch one previous frame between [-1, -random_prev]
        random_next: int = 1,  # The target will be the frame : t + [1, random_next]
        cache_size: int = 8,
    ):
        super().__init__(
            meta_path=meta_path,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
        )

        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "The dataset has a timestep set to 0. Fallback to dt=1 to ensure xdmf can be saved."
            )
        self.random_next = random_next
        self.random_prev = random_prev

        if switch_to_val:
            xdmf_folder = xdmf_folder.replace("train", "test")
            self.random_next = 1
            self.random_prev = 1

        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path
        self.cache_size = cache_size

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._size_dataset: int = len(self.file_paths)
        self._trajectory_cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
        self._frame_cache: OrderedDict[
            tuple[str, int, int, int], Tuple[Data, Optional[torch.Tensor]]
        ] = OrderedDict()

    @property
    def size_dataset(self) -> int:
        """Returns the number of trajectories in the dataset."""
        return self._size_dataset

    def _load_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        path = self.file_paths[traj_index]
        cached = self._trajectory_cache.get(path)
        if cached is not None:
            self._trajectory_cache.move_to_end(path)
            return cached

        with meshio.xdmf.TimeSeriesReader(path) as reader:
            points, cells = reader.read_points_cells()
            point_data = []
            times = []
            for step in range(reader.num_steps):
                time, frame_point_data, _ = reader.read_data(step)
                times.append(time)
                frame_dict = {
                    k: np.array(frame_point_data[k]) for k in frame_point_data.keys()
                }
                point_data.append(frame_dict)

        trajectory = {
            "points": points.astype(np.float32),
            "cells": cells,
            "point_data": point_data,
            "num_steps": len(point_data),
            "times": times,
        }

        self._trajectory_cache[path] = trajectory
        if len(self._trajectory_cache) > self.cache_size:
            self._trajectory_cache.popitem(last=False)

        return trajectory

    def _cache_graph(
        self,
        cache_key: tuple[str, int, int, int],
        graph: Data,
        selected_indices: Optional[torch.Tensor],
    ) -> None:
        self._frame_cache[cache_key] = (graph, selected_indices)
        self._frame_cache.move_to_end(cache_key)
        if len(self._frame_cache) > self.cache_size * 2:
            self._frame_cache.popitem(last=False)
        return None

    def _get_processed_graph(
        self,
        traj_index: int,
        frame: int,
        target_delta: int,
        previous_delta: int,
        trajectory: Dict[str, np.ndarray],
        mesh_id: str,
    ) -> Tuple[Data, Optional[torch.Tensor]]:
        xdmf_file = self.file_paths[traj_index]
        cache_key = (xdmf_file, frame, target_delta, previous_delta)

        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            graph, selected_indices = cached
        else:
            mesh = meshio.Mesh(
                trajectory["points"],
                trajectory["cells"],
                point_data=trajectory["point_data"][frame],
            )

            if "triangle" in mesh.cells_dict:
                cells = mesh.cells_dict["triangle"]
            elif "tetra" in mesh.cells_dict:
                cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
            else:
                raise ValueError(
                    "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
                )

            point_data = {
                k: np.array(mesh.point_data[k]).astype(
                    self.meta["features"][k]["dtype"]
                )
                for k in self.meta["features"]
                if k in mesh.point_data.keys()
            }

            target_frame = trajectory["point_data"][frame + target_delta]
            target_data = {
                k: np.array(target_frame[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in target_frame.keys()
                and self.meta["features"][k]["type"] == "dynamic"
            }

            def _reshape_array(a: dict):
                for k, v in a.items():
                    if v.ndim == 1:
                        a[k] = v.reshape(-1, 1)

            _reshape_array(point_data)
            _reshape_array(target_data)

            points = trajectory["points"]
            graph = meshdata_to_graph(
                points=points.astype(np.float32),
                cells=cells,
                point_data=point_data,
                time=trajectory["times"][frame],
                target=target_data,
                id=mesh_id,
            )

            graph.target_dt = target_delta * self.dt

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
        xdmf_file = self.file_paths[traj_index]
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]

        # Fetch index for previous_data and target
        _target_data_index = random.randint(1, self.random_next)
        _previous_data_index = random.randint(1, self.random_prev)

        trajectory = self._load_trajectory(traj_index)
        num_steps = trajectory["num_steps"]

        if frame - _previous_data_index < 0:
            _previous_data_index = 1
        if frame + _target_data_index > num_steps - 1:
            _target_data_index = 1

        if frame >= num_steps - 1:
            raise IndexError(
                f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
            )

        graph, selected_indices = self._get_processed_graph(
            traj_index=traj_index,
            frame=frame,
            target_delta=_target_data_index,
            previous_delta=_previous_data_index,
            trajectory=trajectory,
            mesh_id=mesh_id,
        )

        if self.use_previous_data:
            prev_frame = trajectory["point_data"][frame - _previous_data_index]
            previous = {
                k: np.array(prev_frame[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in prev_frame.keys()
                and self.meta["features"][k]["type"] == "dynamic"
            }

            for k, v in previous.items():
                if v.ndim == 1:
                    previous[k] = v.reshape(-1, 1)

            graph.previous_data = previous
            graph.previous_dt = -_previous_data_index * self.dt

        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
