import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch_geometric.transforms as T
from loguru import logger
from pathlib import Path
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_random_edge

from graphphysics.models.utils_csr import build_fixed_fanout_k8, edge_index_to_csr
from graphphysics.utils.torch_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    get_masked_indexes,
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

        self.idx_cache_dir = Path(meta_path).resolve().parent / "idx_k8_cache"
        self.idx_cache_enabled = (
            self.idx_cache_dir is not None and self.new_edges_ratio <= 0.0
        )

        self.world_pos_index_start = None
        self.world_pos_index_end = None
        if world_pos_parameters is not None:
            self.world_pos_index_start = world_pos_parameters.get(
                "world_pos_index_start"
            )
            self.world_pos_index_end = world_pos_parameters.get("world_pos_index_end")

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

    def _attach_idx_k8(
        self, graph: Data, cache_key: Optional[str] = None
    ) -> Data:
        """Attach or load idx_k8/CSR tensors on the graph."""
        if getattr(graph, "edge_index", None) is None:
            return graph
        if getattr(graph, "idx_k8", None) is not None:
            return graph

        cache_path: Optional[Path] = None
        if (
            self.idx_cache_enabled
            and cache_key
            and self.idx_cache_dir is not None
        ):
            cache_path = self.idx_cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu")
                graph.idx_k8 = payload["idx_k8"]
                graph.rowptr = payload.get("rowptr")
                graph.col = payload.get("col")
                return graph

        rowptr, col = edge_index_to_csr(graph.edge_index, graph.num_nodes)
        idx_k8 = build_fixed_fanout_k8(
            rowptr=rowptr,
            col=col,
            include_self=True,
            sort_neighbors=True,
        )
        graph.rowptr = rowptr
        graph.col = col
        graph.idx_k8 = idx_k8

        if cache_path is not None:
            payload = {
                "idx_k8": idx_k8.cpu(),
                "rowptr": rowptr.cpu(),
                "col": col.cpu(),
            }
            torch.save(payload, cache_path)

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
