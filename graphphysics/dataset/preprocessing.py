from __future__ import annotations

import math
import random
from functools import partial
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch_geometric.transforms as T
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from graphphysics.utils.nodetype import NodeType

try:  # Optional import during typing in lightweight environments
    from named_features import XFeatureLayout
except ImportError:  # pragma: no cover - fallback when named features are unavailable
    XFeatureLayout = None  # type: ignore


def _get_layout(graph: Data) -> "XFeatureLayout":
    layout = getattr(graph, "x_layout", None)
    if layout is None:
        raise ValueError(
            "This transformation requires graph.x_layout to resolve named features."
        )
    return layout


def _select_feature(graph: Data, name: str) -> torch.Tensor:
    if hasattr(graph, "x_sel"):
        return graph.x_sel(name)
    layout = _get_layout(graph)
    return graph.x[:, layout.slc(name)]


def _assign_feature(graph: Data, name: str, value: torch.Tensor) -> Data:
    if hasattr(graph, "x_assign"):
        return graph.x_assign({name: value}, inplace=True)
    layout = _get_layout(graph)
    graph.x[:, layout.slc(name)] = value
    return graph


def _feature_slice_from_indices(
    layout: Optional["XFeatureLayout"], start: int, end: int
) -> Optional[str]:
    if layout is None:
        return None
    for block in layout.blocks:
        if block.start == start and block.end == end:
            return block.name
    return None


def _feature_name_from_index(layout: Optional["XFeatureLayout"], index: int) -> Optional[str]:
    if layout is None:
        return None
    for block in layout.blocks:
        if block.start <= index < block.end and block.size == 1:
            return block.name
    return None


def _ensure_sequence(value: Union[Sequence[int], int]) -> List[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(v) for v in value]
    return [int(value)]


def _ensure_float_sequence(
    value: Union[Sequence[float], float], expected: int
) -> List[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(v) for v in value]
        if len(values) != expected:
            raise ValueError(
                "noise_scale must have the same length as the number of target features."
            )
        return values
    return [float(value)] * expected


def _translate_world_params(
    params: Optional[Mapping[str, Any]], layout: Optional["XFeatureLayout"]
) -> Optional[dict]:
    if params is None:
        return None

    result = dict(params)
    if layout is None:
        return result

    world_name = result.get("world_pos_feature")
    if world_name and (
        result.get("world_pos_index_start") is None
        or result.get("world_pos_index_end") is None
    ):
        slc = layout.slc(world_name)
        result.setdefault("world_pos_index_start", slc.start)
        result.setdefault("world_pos_index_end", slc.stop)
    elif (
        "world_pos_feature" not in result
        and result.get("world_pos_index_start") is not None
        and result.get("world_pos_index_end") is not None
    ):
        name = _feature_slice_from_indices(
            layout,
            int(result["world_pos_index_start"]),
            int(result["world_pos_index_end"]),
        )
        if name:
            result["world_pos_feature"] = name

    node_type_name = result.get("node_type_feature")
    if node_type_name and result.get("node_type_index") is None:
        slc = layout.slc(node_type_name)
        if slc.stop - slc.start != 1:
            raise ValueError(
                "node_type_feature must reference a single channel in the layout."
            )
        result.setdefault("node_type_index", slc.start)
    elif (
        "node_type_feature" not in result and result.get("node_type_index") is not None
    ):
        name = _feature_name_from_index(layout, int(result["node_type_index"]))
        if name:
            result["node_type_feature"] = name

    displacement = result.get("displacement_feature")
    if displacement and (
        result.get("displacement_index_start") is None
        or result.get("displacement_index_end") is None
    ):
        slc = layout.slc(displacement)
        result.setdefault("displacement_index_start", slc.start)
        result.setdefault("displacement_index_end", slc.stop)

    return result


def _translate_noise_params(
    params: Optional[Mapping[str, Any]], layout: Optional["XFeatureLayout"]
) -> Optional[dict]:
    if params is None:
        return None

    result = dict(params)
    if layout is None:
        return result

    features = result.get("noise_features")
    if features is not None and (
        result.get("noise_index_start") is None or result.get("noise_index_end") is None
    ):
        if isinstance(features, (str, bytes)):
            feature_names = [features]
        elif isinstance(features, Iterable):
            feature_names = [str(name) for name in features]
        else:  # pragma: no cover - guard for unexpected input
            raise TypeError("noise_features must be a string or sequence of strings.")
        starts: List[int] = []
        ends: List[int] = []
        for name in feature_names:
            slc = layout.slc(name)
            starts.append(slc.start)
            ends.append(slc.stop)
        result.setdefault(
            "noise_index_start", starts if len(starts) > 1 else starts[0]
        )
        result.setdefault("noise_index_end", ends if len(ends) > 1 else ends[0])
    elif (
        "noise_features" not in result
        and result.get("noise_index_start") is not None
        and result.get("noise_index_end") is not None
    ):
        starts = _ensure_sequence(result["noise_index_start"])
        ends = _ensure_sequence(result["noise_index_end"])
        if len(starts) == len(ends):
            names: List[str] = []
            for start, end in zip(starts, ends):
                name = _feature_slice_from_indices(layout, start, end)
                if name is None:
                    names = []
                    break
                names.append(name)
            if names:
                result["noise_features"] = names if len(names) > 1 else names[0]

    node_type_name = result.get("node_type_feature")
    if node_type_name and result.get("node_type_index") is None:
        slc = layout.slc(node_type_name)
        if slc.stop - slc.start != 1:
            raise ValueError(
                "node_type_feature must reference a single channel in the layout."
            )
        result.setdefault("node_type_index", slc.start)
    elif (
        "node_type_feature" not in result and result.get("node_type_index") is not None
    ):
        name = _feature_name_from_index(layout, int(result["node_type_index"]))
        if name:
            result["node_type_feature"] = name

    return result


def add_edge_features() -> List[Callable[[Data], Data]]:
    """
    Returns a list of PyTorch Geometric transforms to add edge features to a graph.

    Returns:
        List[Callable[[Data], Data]]: List of transforms to add edge features.
    """
    return [T.Cartesian(norm=False), T.Distance(norm=False)]


def _3d_face_to_edge(graph: Data) -> Data:
    """
    Converts 3D quadrilateral faces to triangular faces.

    Parameters:
        graph (Data): The input graph data.

    Returns:
        Data: The graph with updated faces.
    """
    face = graph.face
    graph.face = torch.cat(
        [
            face[0:3],
            face[1:4],
            torch.stack([face[2], face[3], face[0]], dim=0),
            torch.stack([face[3], face[0], face[1]], dim=0),
        ],
        dim=1,
    )
    return graph


def add_obstacles_next_pos(
    graph: Data,
    *,
    world_pos_feature: Optional[str] = None,
    target_world_pos_feature: Optional[str] = None,
    displacement_feature: Optional[str] = None,
    node_type_feature: Optional[str] = None,
    world_pos_index_start: Optional[int] = None,
    world_pos_index_end: Optional[int] = None,
    node_type_index: Optional[int] = None,
) -> Data:
    """Adds obstacle displacement information to the node features.

    Parameters
    ----------
    graph:
        Input graph.
    world_pos_feature:
        Name of the feature in ``graph.x`` representing the current world position.
    target_world_pos_feature:
        Name of the feature representing the target/next world position. Defaults to
        ``world_pos_feature`` when omitted.
    displacement_feature:
        Feature name that should receive the computed displacement. Required when
        ``world_pos_feature`` is provided.
    node_type_feature:
        Feature name representing node types. Defaults to ``"node_type"`` when
        available.
    world_pos_index_start/world_pos_index_end/node_type_index:
        Legacy numeric configuration kept for backwards compatibility when named
        features are unavailable.
    """

    use_named = world_pos_feature is not None or node_type_feature is not None

    if use_named:
        if world_pos_feature is None:
            raise ValueError(
                "Named usage of add_obstacles_next_pos requires 'world_pos_feature'."
            )
        if displacement_feature is None:
            raise ValueError(
                "Named usage of add_obstacles_next_pos requires 'displacement_feature'."
            )

        layout = _get_layout(graph)
        target_name = target_world_pos_feature or world_pos_feature
        world_pos = _select_feature(graph, world_pos_feature)
        try:
            target_world_pos = graph.y[:, layout.slc(target_name)]
        except Exception as exc:  # pragma: no cover - guard for mismatched targets
            raise ValueError(
                f"Unable to index target feature '{target_name}' using the x-layout slices."
            ) from exc

        displacement = target_world_pos - world_pos
        node_type_name = node_type_feature or "node_type"
        node_type_values = _select_feature(graph, node_type_name).squeeze(-1)

        obstacle_mask = node_type_values == NodeType.OBSTACLE
        if obstacle_mask.any():
            mean_obstacle = displacement[obstacle_mask].mean(dim=0)
        else:  # pragma: no cover - rare case when dataset has no obstacles
            mean_obstacle = torch.zeros_like(displacement[0])
        displacement[~obstacle_mask] = mean_obstacle

        _assign_feature(graph, displacement_feature, displacement)
        return graph

    if world_pos_index_start is None or world_pos_index_end is None:
        raise ValueError(
            "Legacy usage of add_obstacles_next_pos requires 'world_pos_index_start' and 'world_pos_index_end'."
        )
    if node_type_index is None:
        raise ValueError(
            "Legacy usage of add_obstacles_next_pos requires 'node_type_index'."
        )

    world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    other_features = graph.x[:, world_pos_index_end:]
    target_world_pos = graph.y[:, world_pos_index_start:world_pos_index_end]
    obstacle_displacement = target_world_pos - world_pos
    node_type = graph.x[:, node_type_index - 3]

    only_obstacle_displacement = obstacle_displacement[node_type == NodeType.OBSTACLE]
    mean_obstacle_displacement = torch.mean(only_obstacle_displacement, dim=0)
    obstacle_displacement[node_type != NodeType.OBSTACLE] = mean_obstacle_displacement

    graph.x = torch.cat([world_pos, obstacle_displacement, other_features], dim=1)
    return graph


def add_world_edges(
    graph: Data,
    world_pos_index_start: Optional[int] = None,
    world_pos_index_end: Optional[int] = None,
    node_type_index: Optional[int] = None,
    *,
    radius: float = 0.03,
    world_pos_feature: Optional[str] = None,
    node_type_feature: Optional[str] = None,
) -> Data:
    """
    Adds world edges to the graph based on proximity in world position.

    Parameters:
        graph (Data): The input graph data.
        world_pos_index_start (int): The starting index of world position in node features.
        world_pos_index_end (int): The ending index of world position in node features.
        node_type_index (int): The index of the node type feature.
        radius (float): The radius within which to connect nodes.

    Returns:
        Data: The graph with added world edges.
    """

    # Extract world positions
    def _close_pairs_ckdtree(X, max_d):
        tree = cKDTree(X.cpu().numpy())
        pairs = tree.query_pairs(max_d, output_type="ndarray")
        return torch.Tensor(pairs.T).long()

    use_named = world_pos_feature is not None or node_type_feature is not None

    if use_named:
        if world_pos_feature is None:
            raise ValueError(
                "Named usage of add_world_edges requires 'world_pos_feature'."
            )
        world_pos = _select_feature(graph, world_pos_feature)
        node_type_name = node_type_feature or "node_type"
        node_types = _select_feature(graph, node_type_name).squeeze(-1)
    else:
        if world_pos_index_start is None or world_pos_index_end is None:
            raise ValueError(
                "Legacy usage of add_world_edges requires 'world_pos_index_start' and 'world_pos_index_end'."
            )
        if node_type_index is None:
            raise ValueError(
                "Legacy usage of add_world_edges requires 'node_type_index'."
            )
        world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
        node_types = graph.x[:, node_type_index]

    added_edges = _close_pairs_ckdtree(world_pos, radius).to(graph.x.device)

    type = node_types

    m1 = torch.gather(type, -1, added_edges[0]) == NodeType.OBSTACLE
    m2 = torch.gather(type, -1, added_edges[1]) == NodeType.NORMAL
    mask1 = torch.logical_and(m1, m2)

    m1 = torch.gather(type, -1, added_edges[0]) == NodeType.NORMAL
    m2 = torch.gather(type, -1, added_edges[1]) == NodeType.OBSTACLE
    mask2 = torch.logical_and(m1, m2)

    mask = torch.logical_or(mask1, mask2)

    added_edges = added_edges[:, mask]

    edge_index = torch.cat([added_edges, graph.edge_index], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

    graph.edge_index = edge_index
    return graph


def add_world_pos_features(
    graph: Data,
    world_pos_index_start: Optional[int] = None,
    world_pos_index_end: Optional[int] = None,
    *,
    world_pos_feature: Optional[str] = None,
) -> Data:
    """
    Adds world position features to the graph's edge attributes.

    Parameters:
        graph (Data): The input graph data.
        world_pos_index_start (int): The starting index of world position in node features.
        world_pos_index_end (int): The ending index of world position in node features.

    Returns:
        Data: The graph with updated edge attributes.
    """
    if world_pos_feature is not None:
        world_pos = _select_feature(graph, world_pos_feature)
    else:
        if world_pos_index_start is None or world_pos_index_end is None:
            raise ValueError(
                "Legacy usage of add_world_pos_features requires 'world_pos_index_start' and 'world_pos_index_end'."
            )
        world_pos = graph.x[:, world_pos_index_start:world_pos_index_end]
    senders, receivers = graph.edge_index

    relative_world_pos = world_pos[senders] - world_pos[receivers]
    relative_world_pos_norm = torch.norm(relative_world_pos, p=2, dim=-1, keepdim=True)

    graph.edge_attr = torch.cat(
        [
            graph.edge_attr,
            relative_world_pos.type_as(graph.edge_attr),
            relative_world_pos_norm.type_as(graph.edge_attr),
        ],
        dim=-1,
    )

    return graph


def add_noise(
    graph: Data,
    noise_index_start: Union[int, List[int], None] = None,
    noise_index_end: Union[int, List[int], None] = None,
    noise_scale: Union[float, List[float]] = 0.0,
    node_type_index: Optional[int] = None,
    *,
    noise_features: Optional[Union[str, Sequence[str]]] = None,
    node_type_feature: Optional[str] = None,
    t: Optional[float] = None,
) -> Data:
    """
    Adds Gaussian noise to the specified features of the graph's nodes.

    Parameters:
        graph (Data): The graph to modify.
        noise_index_start (Union[int, List[int]]): The starting index or indices for noise addition.
        noise_index_end (Union[int, List[int]]): The ending index or indices for noise addition.
        noise_scale (Union[float, List[float]]): The standard deviation(s) of the Gaussian noise.
        node_type_index (int): The index of the node type feature.
        t (float): If defined, we add a curicullum of noise instead of a fixed one. We follow the fol-
            lowing formula: noise(t) = 10*std*(1+cos(t*pi))

    Returns:
        Data: The modified graph with noise added to node features.
    """
    use_named = noise_features is not None or node_type_feature is not None

    if use_named:
        if noise_features is None:
            raise ValueError(
                "Named usage of add_noise requires 'noise_features'."
            )

        if isinstance(noise_features, (str, bytes)):
            target_features = [noise_features]
        elif isinstance(noise_features, Iterable):
            target_features = [str(name) for name in noise_features]
        else:  # pragma: no cover - guard for unexpected input
            raise TypeError("noise_features must be a string or sequence of strings.")

        scales = _ensure_float_sequence(noise_scale, len(target_features))
        node_type_name = node_type_feature or "node_type"
        node_type_tensor = _select_feature(graph, node_type_name).squeeze(-1)
        mask = node_type_tensor != NodeType.NORMAL

        for feature_name, scale in zip(target_features, scales):
            feature_tensor = _select_feature(graph, feature_name)
            scale_value = (
                10 * scale * (1 + math.cos(t * math.pi)) if t is not None else scale
            )
            noise = torch.randn_like(feature_tensor) * scale_value
            noise[mask] = 0
            updated = feature_tensor + noise
            _assign_feature(graph, feature_name, updated)

        return graph

    if noise_index_start is None or noise_index_end is None:
        raise ValueError(
            "Legacy usage of add_noise requires 'noise_index_start' and 'noise_index_end'."
        )
    if node_type_index is None:
        raise ValueError(
            "Legacy usage of add_noise requires 'node_type_index'."
        )

    noise_index_start = _ensure_sequence(noise_index_start)
    noise_index_end = _ensure_sequence(noise_index_end)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError(
            "noise_index_start and noise_index_end must have the same length."
        )

    noise_scale_values = _ensure_float_sequence(noise_scale, len(noise_index_start))

    node_type = graph.x[:, node_type_index]

    mask = node_type != NodeType.NORMAL

    for start, end, scale in zip(
        noise_index_start, noise_index_end, noise_scale_values
    ):
        feature = graph.x[:, start:end]
        scale_ = 10 * scale * (1 + math.cos(t * math.pi)) if t is not None else scale
        noise = torch.randn_like(feature) * scale_
        noise[mask] = 0
        graph.x[:, start:end] = feature + noise

    return graph


def compute_min_distance_to_type(
    graph: Data, target_type: NodeType, node_types: torch.Tensor
):
    """
    Computes the minimum distance from each node to any node of the specified type.

    Parameters:
        graph (Data): The graph to modify.
        target_type (NodeType): Nodes to compare to.
        node_types (torch.Tensor): The node type features

    Returns:
        torch.Tensor: Tensor of shape [num_nodes] containing minimum distances
    """
    # Get masks for target type nodes
    type_a_mask = node_types == target_type

    # Get positions
    pos = graph.pos  # [num_nodes, 3]

    # Expand dimensions for broadcasting
    # [num_nodes, 1, 3] and [1, num_type_a_nodes, 3]
    pos_expanded = pos.unsqueeze(1)
    pos_type_a = pos[type_a_mask].unsqueeze(0)

    # Compute pairwise distances
    # Using broadcasting to compute differences
    # Result shape: [num_nodes, num_type_a_nodes]
    distances = torch.sqrt(torch.sum((pos_expanded - pos_type_a) ** 2, dim=-1))

    # Get minimum distance for each node
    min_distances = torch.min(distances, dim=1)[0]

    return min_distances


class Random3DRotate(BaseTransform):
    """
    Applies random 3D rotation to node positions and specified feature sets.

    Args:
        feature_indices (List[Tuple[int, int]]): List of (start_idx, end_idx) tuples
            indicating which features in graph.x should be rotated as 3D coordinates.
            Each tuple specifies a range of 3 consecutive features representing x,y,z coordinates.
    """

    def __init__(self, feature_indices: List[Tuple[int, int]] = None) -> None:
        self.feature_indices = feature_indices or []
        # Validate that each range spans 3 features (x,y,z coordinates)
        for start_idx, end_idx in self.feature_indices:
            assert end_idx - start_idx == 3, (
                f"Each feature range must span exactly 3 features for xyz coordinates. "
                f"Got range {start_idx}-{end_idx}"
            )

    def _get_random_angles(self):
        """Generate random rotation angles in degrees and convert to radians."""
        angles = [random.uniform(-180, 180) for _ in range(3)]
        return [math.radians(angle) for angle in angles]

    def _build_rotation_matrix(self, alpha, beta, gamma):
        """Build the complete 3D rotation matrix using the given angles.

        Args:
            alpha (float): Rotation angle around z-axis (yaw) in radians
            beta (float): Rotation angle around y-axis (pitch) in radians
            gamma (float): Rotation angle around x-axis (roll) in radians

        Returns:
            torch.Tensor: 3x3 rotation matrix
        """
        # Compute trigonometric functions for all angles
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        cos_b, sin_b = math.cos(beta), math.sin(beta)
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)

        # Build the complete rotation matrix according to the formula
        matrix = [
            [
                cos_a * cos_b,
                cos_a * sin_b * sin_g + sin_a * cos_g,
                -cos_a * sin_b * cos_g + sin_a * sin_g,
            ],
            [
                -sin_a * cos_b,
                -sin_a * sin_b * sin_g + cos_a * cos_g,
                sin_a * sin_b * cos_g + cos_a * sin_g,
            ],
            [sin_b, -cos_b * sin_g, cos_b * cos_g],
        ]

        return torch.tensor(matrix)

    def _rotate_features(self, x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Rotate specified feature sets using the rotation matrix."""
        for start_idx, end_idx in self.feature_indices:
            feat = x[:, start_idx:end_idx]
            # Apply rotation
            rotated_feat = feat @ matrix.to(feat.device, feat.dtype)
            # Update the features
            x[:, start_idx:end_idx] = rotated_feat
        return x

    def forward(self, data: Union[Data, Batch]) -> Union[Data, Batch]:
        # Generate random angles and build rotation matrix
        alpha, beta, gamma = self._get_random_angles()
        rotation_matrix = self._build_rotation_matrix(alpha, beta, gamma)

        # First rotate the node positions if they exist
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
            assert pos.size(-1) == 3, "Node positions must be 3-dimensional"
            data.pos = pos @ rotation_matrix.to(pos.device, pos.dtype)

        # Then rotate the specified feature sets if they exist
        if hasattr(data, "x") and data.x is not None and self.feature_indices:
            data.x = self._rotate_features(data.x, rotation_matrix)

        if hasattr(data, "y") and data.x is not None:
            target = data.y[:, 0:3]
            # Apply rotation
            rotated_target = target @ rotation_matrix.to(target.device, target.dtype)
            # Update the target
            data.y = rotated_target

        return data


def build_preprocessing(
    noise_parameters: Optional[dict] = None,
    world_pos_parameters: Optional[dict] = None,
    add_edges_features: bool = True,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    *,
    x_layout: Optional["XFeatureLayout"] = None,
) -> T.Compose:
    """
    Builds a preprocessing transform pipeline for the graph data.

    Parameters:
        noise_parameters (dict, optional): Parameters for adding noise.
        world_pos_parameters (dict, optional): Parameters for adding world position features.
        add_edges_features (bool): Whether to add edge features.
        extra_node_features (Callable or List[Callable], optional): Extra node feature functions to apply first.
        extra_edge_features (Callable or List[Callable], optional): Extra edge feature functions to apply last.
        x_layout (XFeatureLayout, optional): Feature layout used to translate legacy indices into names.

    Returns:
        T.Compose: A composition of graph transformations.
    """
    preprocessing: List[Callable[[Data], Data]] = []

    # Add extra node features functions at the beginning
    if extra_node_features is not None:
        if not isinstance(extra_node_features, list):
            extra_node_features = [extra_node_features]
        preprocessing.extend(extra_node_features)

    if world_pos_parameters is not None:
        world_pos_parameters = _translate_world_params(world_pos_parameters, x_layout)
        if not world_pos_parameters.pop("use", True):
            world_pos_parameters = None

    if world_pos_parameters is not None:
        if "world_pos_feature" in world_pos_parameters and "displacement_feature" not in world_pos_parameters:
            raise ValueError(
                "World position preprocessing requires 'displacement_feature' when using named features."
            )
        preprocessing.extend(
            [
                partial(
                    add_obstacles_next_pos,
                    **(
                        {
                            "world_pos_feature": world_pos_parameters["world_pos_feature"],
                            "target_world_pos_feature": world_pos_parameters.get(
                                "target_feature"
                            ),
                            "displacement_feature": world_pos_parameters[
                                "displacement_feature"
                            ],
                            "node_type_feature": world_pos_parameters.get(
                                "node_type_feature"
                            ),
                        }
                        if "world_pos_feature" in world_pos_parameters
                        else {
                            "world_pos_index_start": world_pos_parameters[
                                "world_pos_index_start"
                            ],
                            "world_pos_index_end": world_pos_parameters[
                                "world_pos_index_end"
                            ],
                            "node_type_index": world_pos_parameters["node_type_index"],
                        }
                    ),
                ),
                T.FaceToEdge(remove_faces=False),
                partial(
                    add_world_edges,
                    **(
                        {
                            "world_pos_feature": world_pos_parameters["world_pos_feature"],
                            "node_type_feature": world_pos_parameters.get(
                                "node_type_feature"
                            ),
                        }
                        if "world_pos_feature" in world_pos_parameters
                        else {
                            "world_pos_index_start": world_pos_parameters[
                                "world_pos_index_start"
                            ],
                            "world_pos_index_end": world_pos_parameters[
                                "world_pos_index_end"
                            ],
                            "node_type_index": world_pos_parameters["node_type_index"],
                        }
                    ),
                    radius=world_pos_parameters.get("radius", 0.03),
                ),
            ]
        )
        preprocessing.extend(add_edge_features())
    else:
        preprocessing.append(T.FaceToEdge(remove_faces=False))
        if add_edges_features:
            preprocessing.extend(add_edge_features())

    if noise_parameters is not None:
        noise_parameters = _translate_noise_params(noise_parameters, x_layout)
        if "noise_features" in noise_parameters and not noise_parameters["noise_features"]:
            raise ValueError("Named noise configuration requires non-empty 'noise_features'.")
        add_noise_transform = partial(
            add_noise,
            **(
                {
                    "noise_features": noise_parameters["noise_features"],
                    "noise_scale": noise_parameters.get("noise_scale", 0.0),
                    "node_type_feature": noise_parameters.get("node_type_feature"),
                }
                if "noise_features" in noise_parameters
                else {
                    "noise_index_start": noise_parameters["noise_index_start"],
                    "noise_index_end": noise_parameters["noise_index_end"],
                    "noise_scale": noise_parameters["noise_scale"],
                    "node_type_index": noise_parameters["node_type_index"],
                }
            ),
        )
        # Insert after the first transform
        preprocessing.insert(1, add_noise_transform)

    # Append extra edge features functions at the end
    if extra_edge_features is not None:
        if not isinstance(extra_edge_features, list):
            extra_edge_features = [extra_edge_features]
        preprocessing.extend(extra_edge_features)

    return T.Compose(preprocessing)
