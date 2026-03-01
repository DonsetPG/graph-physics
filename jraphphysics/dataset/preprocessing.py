from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jraph

from graphphysics.utils.nodetype import NodeType


def add_edge_features(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    pos = graph.nodes["pos"]
    delta = pos[graph.senders] - pos[graph.receivers]
    dist = jnp.linalg.norm(delta, axis=1, keepdims=True)
    return graph._replace(edges=jnp.concatenate([delta, dist], axis=1))


def add_world_pos_features(
    graph: jraph.GraphsTuple,
    world_pos_index_start: int,
    world_pos_index_end: int,
) -> jraph.GraphsTuple:
    world_pos = graph.nodes["features"][:, world_pos_index_start:world_pos_index_end]
    relative = world_pos[graph.senders] - world_pos[graph.receivers]
    relative_norm = jnp.linalg.norm(relative, axis=1, keepdims=True)
    edge_attr = graph.edges
    if edge_attr is None:
        edge_attr = jnp.zeros((graph.senders.shape[0], 0), dtype=relative.dtype)
    edge_attr = jnp.concatenate([edge_attr, relative, relative_norm], axis=1)
    return graph._replace(edges=edge_attr)


def add_noise(
    graph: jraph.GraphsTuple,
    noise_index_start: Union[int, Tuple[int, ...]],
    noise_index_end: Union[int, Tuple[int, ...]],
    noise_scale: Union[float, Tuple[float, ...]],
    node_type_index: int,
    key: jax.Array,
    node_type_normal_value: int = int(NodeType.NORMAL),
) -> Tuple[jraph.GraphsTuple, jax.Array]:
    if isinstance(noise_index_start, int):
        noise_index_start = (noise_index_start,)
    if isinstance(noise_index_end, int):
        noise_index_end = (noise_index_end,)

    if isinstance(noise_scale, float):
        noise_scale = (noise_scale,) * len(noise_index_start)

    if len(noise_index_start) != len(noise_index_end):
        raise ValueError(
            "noise_index_start and noise_index_end must have the same length."
        )
    if len(noise_scale) != len(noise_index_start):
        raise ValueError(
            "noise_scale must have the same length as noise_index_start and noise_index_end."
        )

    features = graph.nodes["features"]
    node_type = features[:, node_type_index]
    mask = (node_type == node_type_normal_value).astype(features.dtype)[:, None]

    new_features = features
    for start, end, scale in zip(noise_index_start, noise_index_end, noise_scale):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=(features.shape[0], end - start))
        noise = noise * scale * mask
        new_features = new_features.at[:, start:end].add(noise)

    new_nodes = dict(graph.nodes)
    new_nodes["features"] = new_features
    return graph._replace(nodes=new_nodes), key


def build_preprocessing(
    noise_parameters: Optional[dict] = None,
    world_pos_parameters: Optional[dict] = None,
    add_edges_features: bool = True,
    extra_node_features: Optional[Union[Callable, Iterable[Callable]]] = None,
    extra_edge_features: Optional[Union[Callable, Iterable[Callable]]] = None,
):
    node_feature_fns = []
    edge_feature_fns = []

    if extra_node_features is not None:
        if isinstance(extra_node_features, (list, tuple)):
            node_feature_fns.extend(extra_node_features)
        else:
            node_feature_fns.append(extra_node_features)

    if add_edges_features:
        edge_feature_fns.append(add_edge_features)

    if world_pos_parameters is not None:
        edge_feature_fns.append(
            lambda graph: add_world_pos_features(
                graph=graph,
                world_pos_index_start=world_pos_parameters["world_pos_index_start"],
                world_pos_index_end=world_pos_parameters["world_pos_index_end"],
            )
        )

    if extra_edge_features is not None:
        if isinstance(extra_edge_features, (list, tuple)):
            edge_feature_fns.extend(extra_edge_features)
        else:
            edge_feature_fns.append(extra_edge_features)

    def _preprocess(
        graph: jraph.GraphsTuple,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        g = graph
        if noise_parameters is not None:
            g, key = add_noise(
                graph=g,
                noise_index_start=noise_parameters["noise_index_start"],
                noise_index_end=noise_parameters["noise_index_end"],
                noise_scale=noise_parameters["noise_scale"],
                node_type_index=noise_parameters["node_type_index"],
                key=key,
            )

        for fn in node_feature_fns:
            g = fn(g)
        for fn in edge_feature_fns:
            g = fn(g)

        return g, key

    return _preprocess
