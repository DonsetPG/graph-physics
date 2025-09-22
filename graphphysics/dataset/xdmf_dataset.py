import os
import random
from typing import Callable, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.torch_graph import meshdata_to_graph
from graphphysics.utils.bsms_wrappers import calculate_multi_mesh


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
        model_type: str = None,
        message_passing_num: int = 0,
        switch_to_val: bool = False,
        random_prev: int = 1,  # If we use previous data, we will fetch one previous frame between [-1, -random_prev]
        random_next: int = 1,  # The target will be the frame : t + [1, random_next]
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

        self.model_type = model_type
        self.message_passing_num = message_passing_num

        self.data_dir = xdmf_folder.replace("train", "")
        if switch_to_val:
            xdmf_folder = xdmf_folder.replace("train", "test")
            self.random_next = 1
            self.random_prev = 1

        self.mm_layer_dir = os.path.join(
            self.data_dir, f"mm_layers_{self.message_passing_num}"
        )
        os.makedirs(self.mm_layer_dir, exist_ok=True)

        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._size_dataset: int = len(self.file_paths)

    @property
    def size_dataset(self) -> int:
        """Returns the number of trajectories in the dataset."""
        return self._size_dataset

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

        # Read XDMF file
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
            if self.model_type == "bsms" and self.message_passing_num > 0:
                calculate_multi_mesh(
                    mesh_pos=points,
                    cells=cells,
                    mesh_type="triangle",
                    message_passing_num=self.message_passing_num,
                    mm_layer_dir=self.mm_layer_dir,
                    mesh_id=mesh_id,
                )
        elif "tetra" in mesh.cells_dict:
            cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
            if self.model_type == "bsms" and self.message_passing_num > 0:
                calculate_multi_mesh(
                    mesh_pos=points,
                    cells=cells,
                    mesh_type="tetra",
                    message_passing_num=self.message_passing_num,
                    mm_layer_dir=self.mm_layer_dir,
                    mesh_id=mesh_id,
                )
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

        target_data = {
            k: np.array(target_point_data[k]).astype(self.meta["features"][k]["dtype"])
            for k in self.meta["features"]
            if k in target_point_data.keys()
            and self.meta["features"][k]["type"] == "dynamic"
        }

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

        del graph.previous_data
        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
