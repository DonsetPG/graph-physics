import jax
import jax.numpy as jnp
import numpy as np
import jraph
from flax import nnx
import pytest
from jax.experimental import sparse as jsparse
from jraphphysics.models.processors import (
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)
from jraphphysics.models.layers import Transformer


class TestEncodeTransformDecode:
    @pytest.fixture
    def sample_graph(self):
        # Create a sample graph for testing
        n_nodes = 5
        node_features = 10

        nodes = {"features": jnp.ones((n_nodes, node_features))}
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])

        graph = jraph.GraphsTuple(
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([len(senders)]),
            nodes=nodes,
            edges=None,
            senders=senders,
            receivers=receivers,
            globals=None,
        )
        return graph

    @pytest.mark.parametrize(
        "message_passing_num,node_input_size,output_size,hidden_size,num_heads",
        [
            (2, 10, 5, 64, 4),
            (3, 15, 8, 128, 2),
        ],
    )
    def test_encode_transform_decode_output(
        self,
        sample_graph,
        message_passing_num,
        node_input_size,
        output_size,
        hidden_size,
        num_heads,
    ):
        # Ensure the graph matches the specified input size
        sample_graph.nodes["features"] = jnp.ones(
            (sample_graph.n_node[0], node_input_size)
        )

        # Create the model
        model = EncodeTransformDecode(
            message_passing_num=message_passing_num,
            node_input_size=node_input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rngs=nnx.Rngs(params=0, dropout=0),
        )

        # Run the model
        outputs = model(sample_graph)

        # Check output shape
        assert outputs.shape == (sample_graph.n_node[0], output_size)

    def test_adjacency_matrix_creation(self, sample_graph):
        model = EncodeTransformDecode(
            message_passing_num=2,
            node_input_size=10,
            output_size=5,
            hidden_size=64,
            num_heads=4,
            rngs=nnx.Rngs(params=0, dropout=0),
        )

        # Create adjacency matrix
        adj_matrix = model._build_adjacency_matrix(sample_graph)

        # Check matrix properties
        assert isinstance(adj_matrix, jsparse.BCOO)
        assert adj_matrix.shape == (sample_graph.n_node[0], sample_graph.n_node[0])

        # Verify correct indices are present
        assert jnp.all(adj_matrix.indices[:, 0] == sample_graph.senders)
        assert jnp.all(adj_matrix.indices[:, 1] == sample_graph.receivers)

    def test_processor_list_creation(self):
        message_passing_num = 3

        model = EncodeTransformDecode(
            message_passing_num=message_passing_num,
            node_input_size=10,
            output_size=5,
            hidden_size=64,
            num_heads=4,
            rngs=nnx.Rngs(params=0, dropout=0),
        )

        # Check processor list creation
        assert len(model.processor_list) == message_passing_num

        # Verify each processor is a Transformer
        for processor in model.processor_list:
            assert isinstance(processor, Transformer)


class TestEncodeProcessDecode:
    @pytest.fixture
    def sample_graph(self):
        n_nodes = 6
        n_edges = 8
        nodes = {
            "features": jnp.ones((n_nodes, 10)),
            "pos": jnp.ones((n_nodes, 3)),
        }
        edges = jnp.ones((n_edges, 4))
        senders = jnp.array([0, 1, 2, 3, 4, 5, 1, 2], dtype=jnp.int32)
        receivers = jnp.array([1, 2, 3, 4, 5, 0, 3, 4], dtype=jnp.int32)
        return jraph.GraphsTuple(
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([n_edges]),
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            globals=None,
        )

    def test_encode_process_decode_output(self, sample_graph):
        model = EncodeProcessDecode(
            message_passing_num=2,
            node_input_size=10,
            edge_input_size=4,
            output_size=5,
            hidden_size=32,
            rngs=nnx.Rngs(params=0, dropout=0),
        )
        outputs = model(sample_graph)
        assert outputs.shape == (sample_graph.n_node[0], 5)

    def test_encode_process_decode_only_processor(self, sample_graph):
        hidden_size = 32
        sample_graph = sample_graph._replace(
            nodes={
                "features": jnp.ones((sample_graph.n_node[0], hidden_size)),
                "pos": sample_graph.nodes["pos"],
            },
            edges=jnp.ones((sample_graph.n_edge[0], hidden_size)),
        )
        model = EncodeProcessDecode(
            message_passing_num=2,
            node_input_size=10,
            edge_input_size=4,
            output_size=5,
            hidden_size=hidden_size,
            only_processor=True,
            rngs=nnx.Rngs(params=0, dropout=0),
        )
        outputs = model(sample_graph)
        assert outputs.shape == (sample_graph.n_node[0], hidden_size)

    def test_encode_process_decode_rope_requires_pos(self, sample_graph):
        model = EncodeProcessDecode(
            message_passing_num=1,
            node_input_size=10,
            edge_input_size=4,
            output_size=2,
            hidden_size=32,
            use_rope_embeddings=True,
            rngs=nnx.Rngs(params=0, dropout=0),
        )
        bad_graph = sample_graph._replace(nodes={"features": sample_graph.nodes["features"]})
        with pytest.raises(ValueError):
            model(bad_graph)


class TestTransolverProcessor:
    @pytest.fixture
    def sample_graph(self):
        n_nodes = 5
        nodes = {
            "features": jnp.ones((n_nodes, 8)),
            "pos": jnp.ones((n_nodes, 3)),
        }
        senders = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        receivers = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        return jraph.GraphsTuple(
            n_node=jnp.array([n_nodes]),
            n_edge=jnp.array([4]),
            nodes=nodes,
            edges=None,
            senders=senders,
            receivers=receivers,
            globals=None,
        )

    def test_transolver_output(self, sample_graph):
        model = TransolverProcessor(
            message_passing_num=2,
            node_input_size=8,
            output_size=3,
            hidden_size=16,
            num_heads=4,
            rngs=nnx.Rngs(params=0, dropout=0),
        )
        outputs = model(sample_graph)
        assert outputs.shape == (sample_graph.n_node[0], 3)

    def test_transolver_rope_requires_pos(self, sample_graph):
        model = TransolverProcessor(
            message_passing_num=2,
            node_input_size=8,
            output_size=3,
            hidden_size=16,
            num_heads=4,
            use_rope_embeddings=True,
            rngs=nnx.Rngs(params=0, dropout=0),
        )
        bad_graph = sample_graph._replace(nodes={"features": sample_graph.nodes["features"]})
        with pytest.raises(ValueError):
            model(bad_graph)
