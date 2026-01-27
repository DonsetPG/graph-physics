import unittest
import jax
import jax.numpy as jnp
from flax import nnx
import jraph
import numpy as np
from jax import random
from typing import Optional

from jraphphysics.models.layers import Normalizer
from jraphphysics.models.processors import EncodeTransformDecode
from graphphysics.utils.nodetype import NodeType
from jraphphysics.models.simulator import Simulator
from jaxtyping import ArrayLike


class DummyModel(nnx.Module):
    def __call__(self, graph):
        # For testing, simply return the node features summed along axis=-1
        features = graph.nodes["features"]
        return features[:, 0:2]


class TestSimulator(unittest.TestCase):
    def setUp(self):
        # Create a sample graph
        num_nodes = 5
        num_node_features = 4

        # Node features: [feature0, feature1, feature2, node_type]
        node_features = jnp.array(
            [
                [1.0, 2.0, 3.0, 0],
                [4.0, 5.0, 6.0, 1],
                [7.0, 8.0, 9.0, 0],
                [10.0, 11.0, 12.0, 1],
                [13.0, 14.0, 15.0, 0],
            ],
            dtype=jnp.float32,
        )

        # Node positions
        node_positions = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        )

        nodes = {
            "features": node_features,
            "pos": node_positions,
        }

        # Edges (for testing with adjacency)
        senders = jnp.array([0, 1, 2], dtype=jnp.int32)
        receivers = jnp.array([1, 2, 3], dtype=jnp.int32)

        # Globals (including target features)
        globals = {
            "target_features": jnp.array(
                [
                    [0.1, 0.2],
                    [0.2, 0.3],
                    [0.3, 0.4],
                    [0.4, 0.5],
                    [0.5, 0.6],
                ],
                dtype=jnp.float32,
            )
        }

        self.graph_with_edges = jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([len(senders)]),
            globals=globals,
        )

        # Simulator parameters
        self.node_input_size = node_features.shape[1] + 2 + 8
        self.edge_input_size = 0
        self.output_size = 2  # Assuming target features have size 2
        self.feature_index_start = 0
        self.feature_index_end = 3  # Exclude node_type
        self.output_index_start = 0
        self.output_index_end = 2
        self.node_type_index = 3  # Node type is at index 3

        # Create a simple model
        rng = random.PRNGKey(0)
        self.model = DummyModel()

        # Instantiate the Simulator
        self.simulator = Simulator(
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            feature_index_start=self.feature_index_start,
            feature_index_end=self.feature_index_end,
            output_index_start=self.output_index_start,
            output_index_end=self.output_index_end,
            node_type_index=self.node_type_index,
            model=self.model,
            rngs=nnx.Rngs({"params": rng}),
        )

    def test_simulator_with_edges_training(self):
        network_output, target_delta_normalized, outputs = self.simulator(
            self.graph_with_edges, is_training=True
        )

        # Check shapes
        self.assertEqual(network_output.shape, (5, 2))
        self.assertEqual(target_delta_normalized.shape, (5, 2))
        self.assertIsNone(outputs)

    def test_simulator_with_edges_inference(self):
        network_output, target_delta_normalized, outputs = self.simulator(
            self.graph_with_edges, is_training=False
        )

        # Check shapes
        self.assertEqual(network_output.shape, (5, 2))
        self.assertEqual(target_delta_normalized.shape, (5, 2))
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.shape, (5, 2))

    def test_one_hot_node_type(self):
        one_hot = self.simulator._get_one_hot_type(self.graph_with_edges)
        self.assertEqual(one_hot.shape, (5, NodeType.SIZE))

    def test_normalizers(self):
        # Test that normalizers are properly initialized
        self.assertIsInstance(self.simulator._node_normalizer, Normalizer)
        self.assertIsInstance(self.simulator._output_normalizer, Normalizer)


if __name__ == "__main__":
    unittest.main()
