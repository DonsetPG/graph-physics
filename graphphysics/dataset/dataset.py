import json
from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch_geometric.transforms as T
from loguru import logger
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_random_edge, subgraph

from graphphysics.utils.torch_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    get_masked_indexes,
    create_subgraphs,
)


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        meta_path: str,
        targets: list[str],
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        world_pos_parameters: Optional[dict] = None,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
    ):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        # Dataset preprocessing stays on CPU to let DataLoader handle device transfers lazily.
        self.device = torch.device("cpu")

        self.meta: Dict[str, Any] = meta

        # Check targets are properly defined
        if targets is None or len(targets) == 0:
            raise ValueError("At least one target must be specified.")
        for target in targets:
            if target not in self.meta["features"]:
                raise ValueError(f"Target {target} not found in available fields.")
            if self.meta["features"][target]["type"] != "dynamic":
                raise ValueError(f"Target {target} is not a dynamic field.")
        self.targets = targets

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
        self.new_edges_ratio = new_edges_ratio
        self.add_edge_features = add_edge_features
        self.use_previous_data = use_previous_data
        self.use_partitioning = use_partitioning

        if use_partitioning:
            if num_partitions is not None and max_nodes_per_partition is not None:
                raise ValueError(
                    "Specify either 'num_partitions' or 'max_nodes_per_partition', not both."
                )
            if num_partitions is None and max_nodes_per_partition is None:
                raise ValueError(
                    "If 'use_partitioning' is True, specify either 'num_partitions' or 'max_nodes_per_partition'."
                )

        self.num_partitions = num_partitions
        self.max_nodes_per_partition = max_nodes_per_partition
        self.partitions_node_ids_cache: Dict[int, List[torch.Tensor]] = (
            {}
        )  # Cache for partitioned edge indices per trajectory
        self.partitions_per_trajectory: Dict[int, int] = {}
        self.cumulative_samples: List[int] = [0]
        self._size_dataset = 0
        self._len_dataset = 0

        self.world_pos_index_start = None
        self.world_pos_index_end = None
        if world_pos_parameters is not None:
            self.world_pos_index_start = world_pos_parameters.get(
                "world_pos_index_start"
            )
            self.world_pos_index_end = world_pos_parameters.get("world_pos_index_end")

    @property
    def size_dataset(self) -> int:
        """Should return the number of trajectories in the dataset."""
        return self._size_dataset

    @abstractmethod
    def _build_index_map(self):
        """Abstract method to build the index map for the dataset."""
        raise NotImplementedError

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

    def __len__(self) -> int:
        return self._len_dataset

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

    def _add_random_edges(self, graph: Data) -> Data:
        """Add p random edges to the adjacency matrix to simulate random jumps.

        Parameters:
            graph (Data): The input graph data.

        Returns:
            Data: The graph with added random edges.
        """
        new_edges_ratio = self.new_edges_ratio
        if new_edges_ratio <= 0.0 or new_edges_ratio > 1.0:
            return graph
        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            logger.warning(
                "You are trying to add random edges but your graph doesn't have any."
            )
            return graph

        new_edge_index, _ = add_random_edge(
            edge_index, p=new_edges_ratio, force_undirected=True
        )
        graph.edge_index = new_edge_index
        if self.add_edge_features:
            graph.edge_attr = None
            edge_feature_computer = T.Compose(
                [
                    T.Cartesian(norm=False),
                    T.Distance(norm=False),
                ]
            )
            graph = edge_feature_computer(graph).to(self.device)

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
                        world_pos_index_start=self.world_pos_index_start,
                        world_pos_index_end=self.world_pos_index_end,
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

    def _apply_partition(self, graph: Data, node_ids: torch.Tensor):
        """
        Applies partitioning to the graph based on the provided node IDs.
        Parameters:
            graph (Data): The input graph data.
            node_ids (torch.Tensor): The node IDs to include in the subgraph.
        Returns:
            Data: The partitioned subgraph.
        """
        node_ids, _ = node_ids.sort()
        x = graph.x[node_ids]
        y = graph.y[node_ids]
        pos = graph.pos[node_ids]

        edge_index, edge_attr = subgraph(
            subset=node_ids,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            relabel_nodes=True,
            num_nodes=graph.num_nodes,
        )

        # face = None
        # if hasattr(graph, "face") and graph.face is not None:
        #     # graph.face: [k, num_faces]
        #     faces = graph.face

        #     # Mask faces where *all* vertices are in node_ids
        #     node_mask = torch.isin(faces, node_ids)
        #     face_mask = node_mask.all(dim=0)

        #     faces = faces[:, face_mask]

        #     # Relabel node indices
        #     # Build mapping old_index -> new_index
        #     new_index = torch.full(
        #         (graph.num_nodes,),
        #         -1,
        #         dtype=torch.long,
        #         device=node_ids.device,
        #     )
        #     new_index[node_ids] = torch.arange(node_ids.size(0), device=node_ids.device)

        #     face = new_index[faces]

        sub_graph = Data(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # face=face,
            num_nodes=x.size(0),
            traj_index=graph.traj_index,
        )
        return sub_graph

    def _get_partition(self, graph, traj_index, subgraph_idx):
        """
        Gets the partitioned subgraph for a specific trajectory and subgraph index.

        Parameters:
            graph (Data): The input graph data.
            traj_index (int): The index of the trajectory.
            subgraph_idx (int): The index of the subgraph.

        Returns:
            Data: The partitioned subgraph.
        """
        if traj_index not in self.partitions_node_ids_cache:
            num_partitions = self.partitions_per_trajectory[traj_index]
            loader, node_ids = create_subgraphs(graph, num_partitions)
            self.partitions_node_ids_cache[traj_index] = node_ids

        partitioned_node_ids = self.partitions_node_ids_cache[traj_index][
            subgraph_idx
        ].to(self.device)
        graph = self._apply_partition(graph, partitioned_node_ids)

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
