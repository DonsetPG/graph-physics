import os
from typing import Callable, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
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
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
    ):
        super().__init__(
            meta_path=meta_path,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
        )
        if switch_to_val:
            xdmf_folder = xdmf_folder.replace("train", "test")
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
        traj_id = xdmf_file.split("/")[-1].split(".")[0].split("_")[-1]
        reader = meshio.xdmf.TimeSeriesReader(xdmf_file)

        num_steps = reader.num_steps
        if frame >= num_steps - 1:
            raise IndexError(
                f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
            )

        points, cells = reader.read_points_cells()
        time, point_data, _ = reader.read_data(frame)
        _, target_point_data, _ = reader.read_data(frame + 1)

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
            k: np.array(v).astype(self.meta["features"][k]["dtype"])
            for k, v in point_data.items()
            if k in self.meta["features"]
        }

        target_data = {
            k: np.array(v).astype(self.meta["features"][k]["dtype"])
            for k, v in target_point_data.items()
            if k in self.meta["features"]
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
            id=traj_id,
        )

        if self.use_previous_data:
            _, previous_data, _ = reader.read_data(frame - 1)
            previous = {
                k: np.array(v).astype(self.meta["features"][k]["dtype"])
                for k, v in previous_data.items()
                if k in self.meta["features"]
                and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_array(previous)
            graph.previous_data = previous

        graph = graph.to(self.device)

        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        selected_indices = self._get_masked_indexes(graph)

        del graph.previous_data
        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
