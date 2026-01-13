import math
import os
import random
from bisect import bisect_right
from typing import Callable, List, Optional, Tuple, Union, Dict

import meshio
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.torch_graph import meshdata_to_graph, create_subgraphs


class XDMFDataset(BaseDataset):
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
        random_prev: int = 1,  # If we use previous data, we will fetch one previous frame between [-1, -random_prev]
        random_next: int = 1,  # The target will be the frame : t + [1, random_next]
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
            use_partitioning=use_partitioning,
        )
        if use_partitioning:
            if num_partitions is not None and max_nodes_per_partition is not None:
                raise ValueError(
                    "Please specify either 'num_partitions' or 'max_nodes_per_partition', not both."
                )
            if num_partitions is None and max_nodes_per_partition is None:
                raise ValueError(
                    "If 'use_partitioning' is True, please specify either 'num_partitions' or 'max_nodes_per_partition'."
                )

        self.num_partitions = num_partitions
        self.max_nodes_per_partition = max_nodes_per_partition
        self.partitions_edge_index_cache: Dict[int, List[torch.Tensor]] = (
            {}
        )  # Cache for partitioned edge indices per trajectory

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = "xdmf"

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

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._build_index_map()

    def __len__(self) -> int:
        return self._size_dataset

    def _build_index_map(self):
        """
        Builds a map from a flat index to trajectory, frame, and partition indices.
        This is necessary because trajectories can have different numbers of nodes,
        leading to a variable number of partitions per trajectory.
        """
        self.partitions_per_trajectory: Dict[int, int] = {}
        self.frames_per_trajectory: Dict[int, int] = {}
        self.cumulative_samples: List[int] = [0]
        self._size_dataset = 0

        for traj_index, file_path in enumerate(self.file_paths):
            with meshio.xdmf.TimeSeriesReader(file_path) as reader:
                points, _ = reader.read_points_cells()
                num_nodes = len(points)
                num_steps = reader.num_steps

            if self.use_partitioning:
                if self.num_partitions is not None:
                    num_partitions = self.num_partitions
                else:
                    num_partitions = math.ceil(num_nodes / self.max_nodes_per_partition)
            else:
                num_partitions = 1

            self.partitions_per_trajectory[traj_index] = num_partitions
            self.frames_per_trajectory[traj_index] = num_steps

            num_valid_frames = num_steps - self.random_next - self.random_prev
            if num_valid_frames < 0:
                logger.warning(
                    f"Trajectory {traj_index} has too few frames ({num_steps}) to be used. Skipping."
                )
                num_valid_frames = 0

            total_samples_in_traj = num_valid_frames * num_partitions
            self._size_dataset += total_samples_in_traj
            self.cumulative_samples.append(self._size_dataset)

    def _get_indices(self, index: int) -> Tuple[int, int, int]:
        """
        From a global sample index, find the corresponding trajectory, frame, and subgraph indices.
        """
        traj_index = bisect_right(self.cumulative_samples, index) - 1
        local_index = index - self.cumulative_samples[traj_index]
        num_partitions = self.partitions_per_trajectory[traj_index]
        frame_in_traj = local_index // num_partitions
        subgraph_idx = local_index % num_partitions
        frame = frame_in_traj + int(self.use_previous_data)
        return traj_index, frame, subgraph_idx

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
        traj_index, frame, subgraph_idx = self._get_indices(index)
        xdmf_file = self.file_paths[traj_index]
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]

        _target_data_index = random.randint(1, self.random_next)
        _previous_data_index = random.randint(1, self.random_prev)

        with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
            num_steps = reader.num_steps

            if frame - _previous_data_index < 0:
                _previous_data_index = 1
            if frame + _target_data_index > num_steps - 1:
                _target_data_index = 1

            if frame >= num_steps - 1:
                raise IndexError(
                    f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
                )

            points, cells = reader.read_points_cells()
            time, point_data, _ = reader.read_data(frame)
            _, target_point_data, _ = reader.read_data(frame + _target_data_index)

            if self.use_previous_data:
                _, previous_data, _ = reader.read_data(frame - _previous_data_index)

        # Prepare the mesh data
        mesh = meshio.Mesh(points, cells, point_data=point_data)

        # Get faces or cells
        if "triangle" in mesh.cells_dict:
            cells = mesh.cells_dict["triangle"]
        elif "tetra" in mesh.cells_dict:
            cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
        else:
            raise ValueError(
                "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
            )

        # Process point data and target data
        point_data = {
            k: np.array(mesh.point_data[k]).astype(self.meta["features"][k]["dtype"])
            for k in self.meta["features"]
            if k in mesh.point_data.keys()
        }

        target_data = {}
        next_data = {}
        for k in self.meta["features"]:
            if k in self.targets:
                target_data[k] = np.array(target_point_data[k]).astype(
                    self.meta["features"][k]["dtype"]
                )
            else:
                if (
                    k in target_point_data.keys()
                    and self.meta["features"][k]["type"] == "dynamic"
                ):
                    next_data[k] = np.array(target_point_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )

        def _reshape_array(a: dict):
            for k, v in a.items():
                if v.ndim == 1:
                    a[k] = v.reshape(-1, 1)

        _reshape_array(point_data)
        _reshape_array(target_data)

        # Create graph from mesh data
        graph = meshdata_to_graph(
            points=points.astype(np.float32),
            cells=cells,
            point_data=point_data,
            time=time,
            target=target_data,
            id=mesh_id,
            next_data=next_data,
        )
        # TODO: add target_dt and previous_dt as features per node.
        graph.target_dt = _target_data_index * self.dt

        if self.use_previous_data:
            previous = {
                k: np.array(previous_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in previous_data.keys()
                and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_array(previous)
            graph.previous_data = previous
            graph.previous_dt = -_previous_data_index * self.dt

        graph = graph.to(self.device)

        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph)
        selected_indices = self._get_masked_indexes(graph)

        graph.edge_index = (
            graph.edge_index.long() if graph.edge_index is not None else None
        )

        del graph.next_data
        del graph.previous_data
        graph.traj_index = traj_index

        # TODO: not working with masking and selected_indices yet
        if self.use_partitioning:
            if traj_index not in self.partitions_nodes_cache:
                num_partitions = self.partitions_per_trajectory[traj_index]
                loader, node_ids = create_subgraphs(graph, num_partitions)
                self.partitions_nodes_cache[traj_index] = node_ids

            partitioned_node_ids = self.partitions_nodes_cache[traj_index][
                subgraph_idx
            ].to(self.device)
            graph = self._apply_partition(graph, partitioned_node_ids)

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
