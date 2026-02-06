from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from jax.experimental import sparse as jsparse

from jraphphysics.models.layers import TemporalAttention, Transformer, build_mlp


def gumbel_softmax(logits: jnp.ndarray, tau: float = 1.0, hard: bool = False) -> jnp.ndarray:
    del hard
    eps = 1e-8
    u = jnp.clip(nnx.sigmoid(logits), eps, 1.0 - eps)
    gumbel_noise = -jnp.log(-jnp.log(u))
    y = (logits + gumbel_noise) / tau
    return nnx.softmax(y, axis=-1)


class TransolverPlusBlock(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        last_layer: bool = False,
        out_dim: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.last_layer = last_layer
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attn = Transformer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            rngs=rngs,
        )
        self.mlp = build_mlp(
            in_size=hidden_dim,
            hidden_size=hidden_dim,
            out_size=hidden_dim,
            nb_of_layers=2,
            layer_norm=False,
            act="gelu",
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(hidden_dim, out_dim, rngs=rngs) if last_layer else None

    def __call__(self, x: jnp.ndarray, adj: jsparse.BCOO) -> jnp.ndarray:
        x = x + self.attn(self.norm1(x), adj)
        x = x + self.mlp(self.norm2(x))
        if self.out_proj is not None:
            return self.out_proj(x)
        return x


class Model(nnx.Module):
    def __init__(
        self,
        space_dim: int = 0,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 1,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        del space_dim, dropout, act, mlp_ratio, slice_num, ref, unified_pos
        del use_rope_embeddings, use_gated_attention, rope_pos_dimension, rope_base
        self.use_temporal_block = use_temporal_block
        self.preprocess = build_mlp(
            in_size=fun_dim,
            hidden_size=2 * n_hidden,
            out_size=n_hidden,
            nb_of_layers=2,
            layer_norm=False,
            act="gelu",
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [
                TransolverPlusBlock(
                    hidden_dim=n_hidden,
                    num_heads=n_head,
                    last_layer=(not use_temporal_block) and (idx == n_layers - 1),
                    out_dim=out_dim,
                    rngs=rngs,
                )
                for idx in range(n_layers)
            ]
        )
        self.output_proj = nnx.Linear(n_hidden, out_dim, rngs=rngs) if use_temporal_block else None
        self.temporal_block = (
            TemporalAttention(hidden_size=n_hidden, num_heads=n_head, rngs=rngs)
            if use_temporal_block
            else None
        )

    def __call__(
        self,
        x: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        pos: jnp.ndarray | None = None,
        condition: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del pos, condition
        n = x.shape[0]
        adj = jsparse.BCOO(
            (
                jnp.ones_like(senders, dtype=jnp.float32),
                jnp.stack([senders, receivers], axis=-1),
            ),
            shape=(n, n),
        )
        h = self.preprocess(x)
        prev_h = h
        for block in self.blocks:
            prev_h = h
            h = block(h, adj)
        if self.temporal_block is not None and self.output_proj is not None:
            h = self.temporal_block(prev_h, h, adj)
            h = self.output_proj(h)
        return h
