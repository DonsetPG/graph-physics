from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from flax import nnx


@dataclass
class PooledBatch:
    x: jnp.ndarray
    pos: jnp.ndarray
    batch: jnp.ndarray


class UpSampler(nnx.Module):
    def __init__(self, d_in: int, d_out: int, k: int = 6, *, rngs: nnx.Rngs):
        self.k = k
        self.lin = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(
        self,
        x_coarse: jnp.ndarray,
        pos_coarse: jnp.ndarray,
        pos_fine: jnp.ndarray,
        batch_coarse: Optional[jnp.ndarray] = None,
        batch_fine: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del batch_coarse, batch_fine
        distances = jnp.linalg.norm(
            pos_fine[:, None, :] - pos_coarse[None, :, :],
            axis=-1,
        )
        knn_idx = jnp.argsort(distances, axis=1)[:, : self.k]
        gathered = x_coarse[knn_idx]
        interp = jnp.mean(gathered, axis=1)
        return self.lin(interp)


class DownSampler(nnx.Module):
    def __init__(self, d_in: int, d_out: int, ratio: float = 0.25, *, rngs: nnx.Rngs):
        self.ratio = ratio
        self.lin = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        batch: jnp.ndarray,
        attn: Optional[jnp.ndarray] = None,
    ) -> PooledBatch:
        scores = jnp.linalg.norm(attn if attn is not None else x, axis=1)
        num_nodes = x.shape[0]
        keep = max(1, int(self.ratio * num_nodes))
        perm = jnp.argsort(scores)[-keep:]

        x_c = self.lin(x[perm])
        pos_c = pos[perm]
        batch_c = batch[perm]
        return PooledBatch(x=x_c, pos=pos_c, batch=batch_c)
