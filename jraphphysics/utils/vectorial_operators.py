from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp
import numpy as np
import jraph


def _undirected_unique_edges(senders: jnp.ndarray, receivers: jnp.ndarray) -> jnp.ndarray:
    edges = jnp.stack([senders, receivers], axis=1)
    edges = jnp.sort(edges, axis=1)
    return jnp.unique(edges, axis=0)


def _neighbors_from_edges(num_nodes: int, edges: np.ndarray) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for i, j in edges:
        i_int = int(i)
        j_int = int(j)
        neighbors[i_int].append(j_int)
        neighbors[j_int].append(i_int)
    return neighbors


def compute_gradient_weighted_least_squares(
    graph: jraph.GraphsTuple,
    field: jnp.ndarray,
) -> jnp.ndarray:
    pos = np.asarray(graph.nodes["pos"])
    field_np = np.asarray(field)
    if field_np.ndim == 1:
        field_np = field_np[:, None]

    num_nodes = pos.shape[0]
    dim_x = pos.shape[1]
    dim_u = field_np.shape[1]

    edges = np.asarray(_undirected_unique_edges(graph.senders, graph.receivers))
    neighbors = _neighbors_from_edges(num_nodes, edges)

    gradients = np.zeros((num_nodes, dim_u, dim_x), dtype=np.float32)
    for node in range(num_nodes):
        neigh = neighbors[node]
        if len(neigh) < dim_x:
            continue
        a = pos[neigh] - pos[node]  # (k, D)
        b = field_np[neigh] - field_np[node]  # (k, F)
        try:
            solution, *_ = np.linalg.lstsq(a, b, rcond=None)  # (D, F)
            gradients[node] = solution.T
        except np.linalg.LinAlgError:
            continue

    return jnp.asarray(gradients)


def compute_gradient_finite_differences(
    graph: jraph.GraphsTuple,
    field: jnp.ndarray,
) -> jnp.ndarray:
    pos = graph.nodes["pos"]
    if field.ndim == 1:
        field = field[:, None]

    edges = _undirected_unique_edges(graph.senders, graph.receivers)
    i, j = edges[:, 0], edges[:, 1]

    dx = pos[j] - pos[i]  # (E, D)
    du = field[j] - field[i]  # (E, F)
    distances = jnp.linalg.norm(dx, axis=1)
    eps = 1e-8

    gradient_edges = jnp.einsum("ef,ed->efd", du, dx) / (
        distances[:, None, None] ** 2 + eps
    )
    weight_edges = 1.0 / (distances**2 + eps)

    num_nodes = pos.shape[0]
    num_features = field.shape[1]
    dim = pos.shape[1]

    weighted_edges = gradient_edges * weight_edges[:, None, None]

    gradient = jnp.zeros((num_nodes, num_features, dim), dtype=field.dtype)
    gradient = gradient.at[i].add(weighted_edges)
    gradient = gradient.at[j].add(weighted_edges)

    weight_sums = jnp.zeros((num_nodes, num_features, dim), dtype=field.dtype)
    expanded_w = jnp.broadcast_to(weight_edges[:, None, None], (i.shape[0], num_features, dim))
    weight_sums = weight_sums.at[i].add(expanded_w)
    weight_sums = weight_sums.at[j].add(expanded_w)

    return gradient / (weight_sums + eps)


def compute_gradient(
    graph: jraph.GraphsTuple,
    field: jnp.ndarray,
    method: str = "least_squares",
) -> jnp.ndarray:
    if method == "least_squares":
        return compute_gradient_weighted_least_squares(graph, field)
    if method == "finite_diff":
        return compute_gradient_finite_differences(graph, field)
    raise ValueError(f"Unknown method: {method}")


def compute_vector_gradient_product(
    graph: jraph.GraphsTuple,
    field: jnp.ndarray,
    gradient: jnp.ndarray | None = None,
    method: str = "finite_diff",
) -> jnp.ndarray:
    if gradient is None:
        gradient = compute_gradient(graph, field, method=method)
    return jnp.einsum("nf,nfd->nf", field, gradient)


def compute_divergence(
    graph: jraph.GraphsTuple,
    field: jnp.ndarray,
    gradient: jnp.ndarray | None = None,
    method: str = "finite_diff",
) -> jnp.ndarray:
    if gradient is None:
        gradient = compute_gradient(graph, field, method=method)
    diag_dim = min(gradient.shape[1], gradient.shape[2])
    idx = jnp.arange(diag_dim)
    return gradient[:, idx, idx].sum(axis=-1)
