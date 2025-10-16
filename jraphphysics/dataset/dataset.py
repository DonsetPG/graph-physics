import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch.utils import data
import jraph
import jax.numpy as jnp

from jraphphysics.utils.jax_graph import (
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    get_masked_indexes,
)


class BaseDataset(data.Dataset):
    def __init__(
        self,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[
            Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
        ] = None,
        khop: int = 1,
        use_previous_data: bool = False,
    ):
        with open(meta_path, "r") as fp:
            meta = json.load(fp)

        self.meta: Dict[str, Any] = meta

        # Check targets are properly defined
        if targets is not None:
            for target in targets:
                if target not in self.meta["features"]:
                    raise ValueError(f"Target {target} not found in available fields.")
                if self.meta["features"][target]["type"] != "dynamic":
                    raise ValueError(f"Target {target} is not a dynamic field.")
            self.targets = targets
        else:
            raise ValueError("Please provide a list of target properties to predict.")

        self.trajectory_length: int = self.meta["trajectory_length"]
        self.num_trajectories: Optional[int] = None
        self.khop_edge_index_cache: Dict[int, Tuple[jnp.ndarray, jnp.ndarray]] = {}
        self.khop_edge_attr_cache: Dict[int, jnp.ndarray] = {}

        self.preprocessing = preprocessing
        self.khop = khop
        self.use_previous_data = use_previous_data

    @property
    @abstractmethod
    def size_dataset(self) -> int:
        """Should return the number of trajectories in the dataset."""
        pass

    def get_traj_frame(self, index: int) -> Tuple[int, int]:
        traj = index // (self.trajectory_length - 1)
        frame = index % (self.trajectory_length - 1) + int(self.use_previous_data)
        return traj, frame

    def __len__(self) -> int:
        return self.size_dataset * (self.trajectory_length - 1)

    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[jraph.GraphsTuple, Tuple[jraph.GraphsTuple, jnp.ndarray]]:
        """Abstract method to retrieve a data sample."""
        raise NotImplementedError

    def _apply_preprocessing(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if self.preprocessing is not None:
            graph = self.preprocessing(graph)
        return graph

    def _apply_k_hop(
        self, graph: jraph.GraphsTuple, traj_index: int
    ) -> jraph.GraphsTuple:
        if self.khop > 1:
            if traj_index in self.khop_edge_index_cache:
                senders, receivers = self.khop_edge_index_cache[traj_index]
                graph = graph._replace(senders=senders, receivers=receivers)
                if self.add_edge_features:
                    edges = self.khop_edge_attr_cache[traj_index]
                    graph = graph._replace(edges=edges)
            else:
                if self.add_edge_features:
                    graph = compute_k_hop_graph(
                        graph,
                        num_hops=self.khop,
                        add_edge_features_to_khop=True,
                    )
                    self.khop_edge_index_cache[traj_index] = (
                        graph.senders,
                        graph.receivers,
                    )
                    self.khop_edge_attr_cache[traj_index] = graph.edges
                else:
                    khop_edge_index = compute_k_hop_edge_index(
                        jnp.vstack([graph.senders, graph.receivers]),
                        self.khop,
                        graph.nodes["features"].shape[0],
                    )
                    senders, receivers = khop_edge_index
                    self.khop_edge_index_cache[traj_index] = (senders, receivers)
                    graph = graph._replace(senders=senders, receivers=receivers)
        return graph

    def _may_remove_edges_attr(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        if not self.add_edge_features:
            graph = graph._replace(edges=None)
        return graph

    def _get_masked_indexes(self, graph: jraph.GraphsTuple) -> Optional[jnp.ndarray]:
        if self.masking_ratio is not None:
            selected_indices = get_masked_indexes(graph, self.masking_ratio)
            return selected_indices
        else:
            return None
