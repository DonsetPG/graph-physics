from __future__ import annotations

import jax.numpy as jnp
import jraph


def filter_edges(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    node_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
):
    num_nodes = int(jnp.maximum(jnp.max(senders), jnp.max(receivers))) + 1
    mask = -jnp.ones((num_nodes,), dtype=jnp.int32)
    mask = mask.at[node_index].set(jnp.arange(node_index.shape[0], dtype=jnp.int32))

    s = mask[senders]
    r = mask[receivers]
    keep = jnp.logical_and(s >= 0, r >= 0)
    filtered_s = s[keep]
    filtered_r = r[keep]
    filtered_attr = edge_attr[keep] if edge_attr is not None else None
    return filtered_s, filtered_r, filtered_attr, keep


def build_masked_graph(
    graph: jraph.GraphsTuple,
    selected_indexes: jnp.ndarray,
):
    edge_attr = graph.edges
    s, r, edge_attr, edge_mask = filter_edges(
        graph.senders, graph.receivers, selected_indexes, edge_attr
    )
    nodes = dict(graph.nodes)
    nodes["features"] = nodes["features"][selected_indexes]
    if "pos" in nodes:
        nodes["pos"] = nodes["pos"][selected_indexes]
    masked = graph._replace(
        nodes=nodes,
        senders=s,
        receivers=r,
        edges=edge_attr,
        n_node=jnp.array([selected_indexes.shape[0]], dtype=jnp.int32),
        n_edge=jnp.array([s.shape[0]], dtype=jnp.int32),
    )
    return masked, edge_mask


def reconstruct_graph(
    graph: jraph.GraphsTuple,
    latent_masked_graph: jraph.GraphsTuple,
    selected_indexes: jnp.ndarray,
    node_mask_token: jnp.ndarray,
    edges_mask: jnp.ndarray,
    edge_encoder=None,
    edge_mask_token: jnp.ndarray | None = None,
) -> jraph.GraphsTuple:
    features = graph.nodes["features"]
    n, f = features.shape
    latent_features = jnp.broadcast_to(node_mask_token.reshape(1, -1), (n, f))
    latent_features = latent_features.at[selected_indexes].set(
        latent_masked_graph.nodes["features"]
    )
    nodes = dict(graph.nodes)
    nodes["features"] = latent_features

    edges = graph.edges
    if edges is not None and edge_encoder is not None and edge_mask_token is not None:
        latent_edges = edge_encoder(edges)
        latent_edges = latent_edges + edge_mask_token.reshape(1, -1)
        latent_edges = latent_edges.at[edges_mask].set(latent_masked_graph.edges)
    else:
        latent_edges = edges

    return graph._replace(nodes=nodes, edges=latent_edges)
