import unittest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from graphphysics.models.simulator import Simulator
from graphphysics.utils.nodetype import NodeType


class MockModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        batch_size = input.x.size(0)
        return torch.randn(batch_size, self.output_size)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.node_input_size = 5
        self.edge_input_size = 4
        self.output_size = 2
        self.feature_index_start = 0
        self.feature_index_end = 5
        self.output_index_start = 0
        self.output_index_end = 2
        self.node_type_index = 5
        self.device = torch.device("cpu")

        # Mock model
        self.mock_model = MockModel(output_size=self.output_size)

        self.simulator = Simulator(
            node_input_size=self.node_input_size + NodeType.SIZE,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            feature_index_start=self.feature_index_start,
            feature_index_end=self.feature_index_end,
            output_index_start=self.output_index_start,
            output_index_end=self.output_index_end,
            node_type_index=self.node_type_index,
            model=self.mock_model,
            model_type="mock",
            device=self.device,
            model_dir="checkpoint/simulator.pth",
        )

        # Create sample input data
        num_nodes = 10
        num_edges = 15
        x = torch.randn(num_nodes, self.node_input_size + 1)
        x[:, 5] = torch.abs(x[:, 5])
        y = torch.randn(num_nodes, self.output_size)
        edge_attr = torch.randn(num_edges, self.edge_input_size)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        pos = torch.randn(num_nodes, 3)

        self.data = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index, pos=pos)

    def test_forward_training(self):
        self.simulator.train()
        output, target_delta_normalized, outputs = self.simulator(self.data)
        # Check that outputs is None during training
        self.assertIsNone(outputs)
        # Check shapes
        self.assertEqual(output.shape, (10, self.output_size))
        self.assertEqual(target_delta_normalized.shape, (10, self.output_size))

    def test_forward_evaluation(self):
        self.simulator.eval()
        output, target_delta_normalized, outputs = self.simulator(self.data)
        # Check that outputs is not None during evaluation
        self.assertIsNotNone(outputs)
        # Check shapes
        self.assertEqual(output.shape, (10, self.output_size))
        self.assertEqual(target_delta_normalized.shape, (10, self.output_size))
        self.assertEqual(outputs.shape, (10, self.output_size))

    def test_get_pre_target(self):
        pre_target = self.simulator._get_pre_target(self.data)
        self.assertEqual(
            pre_target.shape, (10, self.output_index_end - self.output_index_start)
        )

    def test_get_target_normalized(self):
        target_normalized = self.simulator._get_target_normalized(self.data)
        self.assertEqual(target_normalized.shape, (10, self.output_size))

    def test_get_one_hot_type(self):
        one_hot_type = self.simulator._get_one_hot_type(self.data)
        self.assertEqual(one_hot_type.shape, (10, NodeType.SIZE))

    def test_build_node_features(self):
        one_hot_type = self.simulator._get_one_hot_type(self.data)
        node_features = self.simulator._build_node_features(self.data, one_hot_type)
        expected_size = (
            self.feature_index_end - self.feature_index_start
        ) + NodeType.SIZE
        self.assertEqual(node_features.shape, (10, expected_size))

    def test_build_input_graph(self):
        graph, target_delta_normalized = self.simulator._build_input_graph(
            self.data, is_training=True
        )
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape[0], 10)
        self.assertEqual(target_delta_normalized.shape, (10, self.output_size))

    def test_build_outputs(self):
        network_output = torch.randn(10, self.output_size)
        outputs = self.simulator.build_outputs(self.data, network_output)
        self.assertEqual(outputs.shape, (10, self.output_size))


if __name__ == "__main__":
    unittest.main()
