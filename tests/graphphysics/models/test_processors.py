import unittest
import torch
from torch_geometric.data import Data
from graphphysics.models.processors import (
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)


class TestEncodeProcessDecode(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.num_edges = 10
        self.node_input_size = 8
        self.edge_input_size = 4
        self.output_size = 3
        self.hidden_size = 16
        self.message_passing_num = 3

        x = torch.randn(self.num_nodes, self.node_input_size)
        edge_attr = torch.randn(self.num_edges, self.edge_input_size)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def test_encode_process_decode_forward(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        # x_decoded should have shape [num_nodes, output_size]
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_only_processor(self):
        self.graph.x = torch.randn(self.num_nodes, self.hidden_size)
        self.graph.edge_attr = torch.randn(self.num_edges, self.hidden_size)
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,  # Not used when only_processor=True
            edge_input_size=self.edge_input_size,  # Not used when only_processor=True
            output_size=self.output_size,  # Not used when only_processor=True
            hidden_size=self.hidden_size,
            only_processor=True,
        )
        x_updated = model(self.graph)
        # x_updated should have shape [num_nodes, hidden_size]
        self.assertEqual(x_updated.shape, (self.num_nodes, self.hidden_size))

    def test_gradients(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        loss = x_decoded.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)

    def test_multiple_message_passing_steps(self):
        model = EncodeProcessDecode(
            message_passing_num=5,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_temporal_block(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            use_temporal_block=True,
        )
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_rope_requires_pos(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            use_rope_embeddings=True,
        )
        with self.assertRaises(ValueError):
            model(self.graph)

    def test_rope_forward(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            use_rope_embeddings=True,
        )
        self.graph.pos = torch.randn(self.num_nodes, 3)
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_gate_forward(self):
        model = EncodeProcessDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            use_gated_attention=True,
        )
        self.graph.phi = torch.randn(self.num_nodes)
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))


class TestEncodeTransformDecode(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.node_input_size = 8
        self.output_size = 3
        self.hidden_size = 16
        self.num_heads = 4
        self.message_passing_num = 3

        x = torch.randn(self.num_nodes, self.node_input_size)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 2))
        self.graph = Data(x=x, edge_index=edge_index)

    def test_encode_transform_decode_forward(self):
        model = EncodeTransformDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        # x_decoded should have shape [num_nodes, output_size]
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_only_processor(self):
        self.graph.x = torch.randn(self.num_nodes, self.hidden_size)
        model = EncodeTransformDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,  # Not used when only_processor=True
            output_size=self.output_size,  # Not used when only_processor=True
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            only_processor=True,
        )
        x_updated = model(self.graph)
        # x_updated should have shape [num_nodes, hidden_size]
        self.assertEqual(x_updated.shape, (self.num_nodes, self.hidden_size))

    def test_gradients(self):
        model = EncodeTransformDecode(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        loss = x_decoded.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)

    def test_multiple_message_passing_steps(self):
        model = EncodeTransformDecode(
            message_passing_num=5,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            only_processor=False,
        )
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))


class TestTransolverProcessor(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.node_input_size = 8
        self.output_size = 3
        self.hidden_size = 16
        self.num_heads = 4
        self.message_passing_num = 3

        x = torch.randn(self.num_nodes, self.node_input_size)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 2))
        self.graph = Data(x=x, edge_index=edge_index)

    def test_transolver_forward(self):
        model = TransolverProcessor(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        x_decoded = model(self.graph)
        # x_decoded should have shape [num_nodes, output_size]
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_gradients(self):
        model = TransolverProcessor(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        x_decoded = model(self.graph)
        loss = x_decoded.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)

    def test_multiple_message_passing_steps(self):
        model = TransolverProcessor(
            message_passing_num=5,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
        )
        x_decoded = model(self.graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_transolver_with_temporal_block(self):
        graph = Data(
            x=self.graph.x.clone(),
            edge_index=self.graph.edge_index.clone(),
            pos=torch.randn(self.num_nodes, 3),
        )
        model = TransolverProcessor(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            use_temporal_block=True,
        )
        x_decoded = model(graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))

    def test_transolver_rope_requires_pos(self):
        model = TransolverProcessor(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            use_rope_embeddings=True,
        )
        with self.assertRaises(ValueError):
            model(self.graph)

    def test_transolver_rope_with_pos(self):
        graph = Data(
            x=self.graph.x.clone(),
            edge_index=self.graph.edge_index.clone(),
            pos=torch.randn(self.num_nodes, 3),
        )
        model = TransolverProcessor(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            use_rope_embeddings=True,
        )
        x_decoded = model(graph)
        self.assertEqual(x_decoded.shape, (self.num_nodes, self.output_size))


if __name__ == "__main__":
    unittest.main()
