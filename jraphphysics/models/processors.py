import jax
import jax.numpy as jnp
import jraph
from flax import nnx
from jax.experimental import sparse as jsparse

from jraphphysics.models.layers import (
    GraphNetBlock,
    TemporalAttention,
    Transformer,
    build_mlp,
)
from jraphphysics.models.transolver import Model as TransolverModel


class EncodeProcessDecode(nnx.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        use_gated_mlp: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.use_temporal_block = use_temporal_block
        self.edge_input_size = edge_input_size

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
                rngs=rngs,
            )
            self.edges_encoder = build_mlp(
                in_size=max(edge_input_size, 1),
                hidden_size=hidden_size,
                out_size=hidden_size,
                rngs=rngs,
            )
            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
                rngs=rngs,
            )

        self.processor_list = nnx.List(
            [
            GraphNetBlock(
                hidden_size=hidden_size,
                use_gated_mlp=use_gated_mlp,
                use_rope=use_rope_embeddings,
                rope_axes=rope_pos_dimension,
                rope_base=rope_base,
                use_gate=use_gated_attention,
                rngs=rngs,
            )
            for _ in range(message_passing_num)
            ]
        )
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size, rngs=rngs)
            if use_temporal_block
            else None
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        if self.only_processor:
            x = graph.nodes["features"]
            edge_attr = graph.edges
            if edge_attr is None:
                edge_attr = jnp.zeros(
                    (graph.senders.shape[0], self.hidden_size),
                    dtype=x.dtype,
                )
        else:
            x = self.nodes_encoder(graph.nodes["features"])
            if graph.edges is None:
                edge_inputs = jnp.zeros(
                    (graph.senders.shape[0], max(self.edge_input_size, 1)),
                    dtype=x.dtype,
                )
            else:
                edge_inputs = graph.edges
            edge_attr = self.edges_encoder(edge_inputs)

        prev_x = x
        last_x = x
        pos = graph.nodes.get("pos") if isinstance(graph.nodes, dict) else None
        phi = graph.nodes.get("phi") if isinstance(graph.nodes, dict) else None

        for block in self.processor_list:
            prev_x = x
            x, edge_attr = block(
                x=x,
                senders=graph.senders,
                receivers=graph.receivers,
                edge_attr=edge_attr,
                pos=pos,
                phi=phi,
            )
            last_x = x

        if self.temporal_block is not None:
            x = self.temporal_block(prev_x, last_x)

        if self.only_processor:
            return x
        return self.decode_module(x)


class EncodeTransformDecode(nnx.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        only_processor: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        del use_rope_embeddings, use_gated_attention, rope_pos_dimension, rope_base

        self.only_processor = only_processor
        self.use_temporal_block = use_temporal_block

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
                rngs=rngs,
            )
            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
                rngs=rngs,
            )

        self.processor_list = nnx.List(
            [
            Transformer(
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                rngs=rngs,
            )
            for _ in range(message_passing_num)
            ]
        )
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size, num_heads=num_heads, rngs=rngs)
            if use_temporal_block
            else None
        )

    def _build_adjacency_matrix(self, graph: jraph.GraphsTuple) -> jsparse.BCOO:
        num_nodes = int(graph.n_node.sum())
        indices = jnp.stack([graph.senders, graph.receivers], axis=-1)
        return jsparse.BCOO(
            (jnp.ones_like(graph.senders, dtype=jnp.float32), indices),
            shape=(num_nodes, num_nodes),
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        if self.only_processor:
            x = graph.nodes["features"]
        else:
            x = self.nodes_encoder(graph.nodes["features"])

        adj = self._build_adjacency_matrix(graph)
        prev_x = x
        last_x = x

        for block in self.processor_list:
            prev_x = x
            x = block(x, adj)
            last_x = x

        if self.temporal_block is not None:
            x = self.temporal_block(prev_x, last_x, adj=adj)

        if self.only_processor:
            return x
        return self.decode_module(x)


class TransolverProcessor(nnx.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
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
        del dropout, mlp_ratio, slice_num, ref, unified_pos, use_gated_attention
        del rope_pos_dimension, rope_base
        self.use_rope_embeddings = use_rope_embeddings
        self.model = TransolverModel(
            n_layers=message_passing_num,
            n_hidden=hidden_size,
            n_head=num_heads,
            fun_dim=node_input_size,
            out_dim=output_size,
            use_rope_embeddings=use_rope_embeddings,
            use_temporal_block=use_temporal_block,
            rngs=rngs,
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        pos = graph.nodes.get("pos") if isinstance(graph.nodes, dict) else None
        if self.use_rope_embeddings and pos is None:
            raise ValueError(
                "use_rope_embeddings=True requires 'pos' attribute in the input graph."
            )
        return self.model(
            x=graph.nodes["features"],
            senders=graph.senders,
            receivers=graph.receivers,
            pos=pos,
            condition=None,
        )
