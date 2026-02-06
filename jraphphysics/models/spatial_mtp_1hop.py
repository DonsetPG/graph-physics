from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from flax import nnx


class SpatialMTP1Hop(nnx.Module):
    """
    JAX-compatible 1-hop Spatial MTP approximation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 1,
        assume_undirected: bool = True,
        max_neighbors: Optional[int] = None,
    ):
        del d_model, num_heads, num_layers
        self.assume_undirected = assume_undirected
        self.max_neighbors = max_neighbors

    def _make_edges(self, edge_index: jnp.ndarray) -> jnp.ndarray:
        if self.assume_undirected:
            return edge_index.astype(jnp.int32)
        rev = jnp.stack([edge_index[1], edge_index[0]], axis=0)
        return jnp.concatenate([edge_index, rev], axis=1).astype(jnp.int32)

    def forward(
        self,
        H: jnp.ndarray,
        edge_index: jnp.ndarray,
        centers: jnp.ndarray,
        out_head: nnx.Module,
        target: jnp.ndarray,
        reduction: str = "mean_per_center",
        row_ptr: jnp.ndarray | None = None,
        dst_sorted: jnp.ndarray | None = None,
        H_neigh: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        del row_ptr, dst_sorted
        centers = centers.astype(jnp.int32)
        if centers.shape[0] == 0:
            zero = jnp.asarray(0.0, dtype=H.dtype)
            return zero, {"sp_mtp/centers": zero, "sp_mtp/pairs": zero}

        edges = self._make_edges(edge_index)
        src = edges[0]
        dst = edges[1]

        owners = []
        neighbors = []
        for idx, c in enumerate(centers.tolist()):
            neigh = dst[src == c]
            if self.max_neighbors is not None:
                neigh = neigh[: self.max_neighbors]
            if neigh.shape[0] == 0:
                continue
            owners.append(jnp.full((neigh.shape[0],), idx, dtype=jnp.int32))
            neighbors.append(neigh)

        if len(neighbors) == 0:
            zero = jnp.asarray(0.0, dtype=H.dtype)
            return zero, {
                "sp_mtp/centers": jnp.asarray(float(centers.shape[0]), dtype=H.dtype),
                "sp_mtp/pairs": zero,
            }

        owners = jnp.concatenate(owners, axis=0)
        targets = jnp.concatenate(neighbors, axis=0).astype(jnp.int32)
        z_frontier = (H_neigh if H_neigh is not None else H)[targets]

        y_hat = out_head(z_frontier)
        y_true = target[targets]
        err = jnp.mean((y_hat - y_true) ** 2, axis=-1)

        if reduction == "mean":
            aux_loss = jnp.mean(err)
        else:
            num_centers = centers.shape[0]
            loss_sum = jnp.zeros((num_centers,), dtype=err.dtype).at[owners].add(err)
            cnt = jnp.zeros((num_centers,), dtype=err.dtype).at[owners].add(1.0)
            aux_loss = jnp.mean(loss_sum / jnp.maximum(cnt, 1.0))

        stats = {
            "sp_mtp/centers": jnp.asarray(float(centers.shape[0]), dtype=err.dtype),
            "sp_mtp/pairs": jnp.asarray(float(targets.shape[0]), dtype=err.dtype),
            "sp_mtp/mean_pair_loss": jnp.mean(err),
            "sp_mtp/max_deg": jnp.asarray(
                float(max(int((src == c).sum()) for c in centers.tolist())),
                dtype=err.dtype,
            ),
        }
        return aux_loss, stats

    __call__ = forward
