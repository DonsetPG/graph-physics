import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch_geometric.data import Data
from graphphysics.models.classification_simulator import ClassificationSimulator
from graphphysics.utils.nodetype import NodeType


class MockModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        batch_size = input.x.size(0)
        return torch.randn(batch_size, self.output_size)


class TestClassificationSimulator(unittest.TestCase):
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

        self.simulator = ClassificationSimulator(
            node_input_size=self.node_input_size + NodeType.SIZE,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            feature_index_start=self.feature_index_start,
            feature_index_end=self.feature_index_end,
            output_index_start=self.output_index_start,
            output_index_end=self.output_index_end,
            node_type_index=self.node_type_index,
            batch_size=1,
            model=self.mock_model,
            device=self.device,
            model_dir="checkpoint/classification_simulator.pth",
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

    def test_forward(self):
        self.simulator.train()
        output = self.simulator(self.data)
        # Check shapes
        self.assertEqual(output.shape, (10, self.output_size))

    def test_build_input_graph(self):
        graph = self.simulator._build_input_graph(self.data, is_training=True)
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
