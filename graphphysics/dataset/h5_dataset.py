from typing import Callable, Optional, Tuple, Union

import torch
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.hierarchical import (
    get_frame_as_graph,
    get_h5_dataset,
    get_traj_as_meshes,
)


class H5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path: str,
        meta_path: str,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        #idem que dataset.py
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
            h5_path = h5_path.replace("train", "test")

        self.h5_path = h5_path
        self.meta_path = meta_path

        # Open the H5 file and load metadata
        (
            self.file_handle,
            self.datasets_index,
            self._size_dataset,
            self.meta,
        ) = get_h5_dataset(dataset_path=h5_path, meta_path=meta_path)

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
        # Modifier ça pour avoir uniquement le graphe de base 
        traj_index, frame = self.get_traj_frame(index=index)
        traj_number = self.datasets_index[traj_index]

        # Retrieve the trajectory data
        traj = get_traj_as_meshes(
            file_handle=self.file_handle, traj_number=traj_number, meta=self.meta
        )

        # Get the graph for the specified frame : est ce que c'est ça ce qu'on veut ?
        graph = get_frame_as_graph(
            traj=traj, frame=frame, meta=self.meta, frame_target=frame + 1
        )

        # A virer ???
        if self.use_previous_data:
            previous_graph = get_frame_as_graph(
                traj=traj, frame=frame - 1, meta=self.meta, frame_target=None
            )
            graph.previous_data = previous_graph.x

        graph = graph.to(self.device)

        # Je pense qu'il faut garder ça pour avoir le graphe de base
        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        selected_indices = self._get_masked_indexes(graph)

        # A virer ???
        del graph.previous_data
        graph.traj_index = traj_index

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph

    def __del__(self):
        """Ensure that the H5 file is properly closed."""
        if hasattr(self, "file_handle") and self.file_handle is not None:
            self.file_handle.close()
