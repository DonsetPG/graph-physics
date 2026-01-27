import jax.numpy as jnp
from flax import nnx
from jax.experimental import sparse as jsparse
import jraph
import jax
from jraphphysics.models.layers import FeedForward, Transformer


class EncodeTransformDecode(nnx.Module):

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):

        self.nodes_encoder = FeedForward(
            features=node_input_size,
            hidden_dim=hidden_size,
            rngs=rngs,
        )
        self.decode_module = FeedForward(
            features=hidden_size,
            hidden_dim=output_size,
            rngs=rngs,
        )

        self.processor_list = [
            Transformer(
                input_dim=hidden_size,
                output_dim=hidden_size,
                num_heads=num_heads,
                rngs=rngs,
            )
            for _ in range(message_passing_num)
        ]

    def _build_adjacency_matrix(self, graph: jraph.GraphsTuple) -> jsparse.BCOO:
        num_nodes = graph.n_node.sum()

        senders = graph.senders
        receivers = graph.receivers
        indices = jnp.stack([senders, receivers], axis=-1)

        return jsparse.BCOO(
            (
                jnp.ones_like(senders, dtype=jnp.float32),  # data
                indices,  # indices
            ),
            shape=(num_nodes, num_nodes),
        )

    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        x = graph.nodes["features"]
        x = self.nodes_encoder(x)

        adj = self._build_adjacency_matrix(graph)

        for block in self.processor_list:
            x = block(x, adj)

        x_decoded = self.decode_module(x)
        return x_decoded
