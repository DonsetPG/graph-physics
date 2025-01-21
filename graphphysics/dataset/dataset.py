import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset

from graphphysics.utils.torch_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    get_masked_indexes,
)


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        meta_path: str,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
    ):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.meta: Dict[str, Any] = meta

        self.trajectory_length: int = self.meta["trajectory_length"]
        self.num_trajectories: Optional[int] = None
        self.khop_edge_index_cache: Dict[int, torch.Tensor] = (
            {}
        )  # Cache for k-hop edge indices per trajectory
        self.khop_edge_attr_cache: Dict[int, torch.Tensor] = (
            {}
        )  # Cache for edge attributes if possible

        self.preprocessing = preprocessing
        self.masking_ratio = masking_ratio
        self.khop = khop
        self.add_edge_features = add_edge_features

        self.use_previous_data = use_previous_data

    @property
    @abstractmethod
    def size_dataset(self) -> int:
        """Should return the number of trajectories in the dataset."""

    def get_traj_frame(self, index: int) -> Tuple[int, int]:
        """Calculate the trajectory and frame number based on the given index.

        This method divides the dataset into trajectories and frames. It calculates
        which trajectory and frame the given index corresponds to, considering the
        length of each trajectory.

        Parameters:
            index (int): The index of the item in the dataset.

        Returns:
            Tuple[int, int]: A tuple containing the trajectory number and the frame number within that trajectory.
        """
        traj = index // (self.trajectory_length - 1)
        frame = index % (self.trajectory_length - 1) + int(self.use_previous_data)
        return traj, frame

    def __len__(self) -> int:
        return self.size_dataset * (self.trajectory_length - 1)

    @abstractmethod
    def __getitem__(self, index: int) -> Data:
        """Abstract method to retrieve a data sample."""
        raise NotImplementedError

    def _apply_preprocessing(self, graph: Data) -> Data:
        """Applies preprocessing transforms to the graph if provided.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Data: The preprocessed graph data.
        """
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)
        return graph

    def _apply_k_hop(self, graph: Data, traj_index: int) -> Data:
        """Applies k-hop expansion to the graph and caches the result.

        Parameters:
            graph (Data): The input graph data.
            traj_index (int): The index of the trajectory.

        Returns:
            Data: The graph with k-hop edges.
        """
        if self.khop > 1:
            if traj_index in self.khop_edge_index_cache:
                khop_edge_index = self.khop_edge_index_cache[traj_index]
                graph.edge_index = khop_edge_index.to(graph.edge_index.device)
                if self.add_edge_features:
                    khop_edge_attr = self.khop_edge_attr_cache[traj_index]
                    graph.edge_attr = khop_edge_attr.to(graph.edge_attr.device)
            else:
                # Compute k-hop edge indices and cache them
                if self.add_edge_features:
                    graph = compute_k_hop_graph(
                        graph,
                        num_hops=self.khop,
                        add_edge_features_to_khop=True,
                        device=self.device,
                    )
                    self.khop_edge_index_cache[traj_index] = graph.edge_index.cpu()
                    self.khop_edge_attr_cache[traj_index] = graph.edge_attr.cpu()
                else:
                    khop_edge_index = compute_k_hop_edge_index(
                        graph.edge_index, self.khop, graph.num_nodes
                    )
                    self.khop_edge_index_cache[traj_index] = khop_edge_index.cpu()
                    graph.edge_index = khop_edge_index.to(graph.edge_index.device)
        return graph

    def _may_remove_edges_attr(self, graph: Data) -> Data:
        """Removes edge attributes if they are not needed.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Data: The graph with edge attributes removed if not needed.
        """
        if not self.add_edge_features:
            graph.edge_attr = None
        return graph

    def _get_masked_indexes(self, graph: Data) -> Optional[torch.Tensor]:
        """Gets masked indices based on the masking ratio.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Optional[torch.Tensor]: The selected indices or None if masking is not applied.
        """
        if self.masking_ratio is not None:
            selected_indices = get_masked_indexes(graph, self.masking_ratio)
            return selected_indices
        else:
            return None
