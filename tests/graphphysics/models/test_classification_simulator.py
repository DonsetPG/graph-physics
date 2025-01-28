import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch_geometric.data import Data
from graphphysics.models.classification_simulator import ClassificationSimulator
from graphphysics.models.classification_model import (
    PointNetClassifier,
    ClassificationPointNetP2,
    ClassificationPointTransformer,
)


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
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            feature_index_start=self.feature_index_start,
            feature_index_end=self.feature_index_end,
            output_index_start=self.output_index_start,
            output_index_end=self.output_index_end,
            node_type_index=self.node_type_index,
            model=self.mock_model,
            model_type="epd",
            device=self.device,
            model_dir="checkpoint/classification_simulator.pth",
        )

        # Create sample input data
        num_nodes = 10
        num_edges = 15
        x = torch.randn(num_nodes, self.node_input_size)
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


class TestPointNetClassifierSimulator(unittest.TestCase):
    def setUp(self):
        self.node_input_size = 3
        self.output_size = 2
        self.device = torch.device("cpu")

        self.model = PointNetClassifier(
            node_input_size=self.node_input_size,
            hidden_size=64,
            output_size=self.output_size,
        )

        self.simulator = ClassificationSimulator(
            node_input_size=self.node_input_size,
            edge_input_size=0,
            output_size=self.output_size,
            feature_index_start=0,
            feature_index_end=self.node_input_size,
            output_index_start=0,
            output_index_end=self.output_size,
            node_type_index=0,
            model=self.model,
            model_type="pointnet",
            device=self.device,
        )

        num_nodes = 10
        x = torch.randn(num_nodes, self.node_input_size)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 15))
        self.data = Data(x=x, pos=pos, edge_index=edge_index)

    def test_forward(self):
        self.simulator.train()
        output = self.simulator(self.data)
        self.assertEqual(output.shape, (1, self.output_size))


class TestClassificationPointNetP2Simulator(unittest.TestCase):
    def setUp(self):
        self.node_input_size = 3
        self.output_size = 2
        self.device = torch.device("cpu")

        self.model = ClassificationPointNetP2(
            node_input_size=self.node_input_size,
            output_size=self.output_size,
        )

        self.simulator = ClassificationSimulator(
            node_input_size=self.node_input_size,
            edge_input_size=0,
            output_size=self.output_size,
            feature_index_start=0,
            feature_index_end=self.node_input_size,
            output_index_start=0,
            output_index_end=self.output_size,
            node_type_index=0,
            model=self.model,
            model_type="pointnetp2",
            device=self.device,
        )

        num_nodes = 10
        x = torch.randn(num_nodes, self.node_input_size)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 15))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        self.data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)

    def test_forward(self):
        self.simulator.train()
        output = self.simulator(self.data)
        self.assertEqual(output.shape, (1, self.output_size))


class TestClassificationPointTransformerSimulator(unittest.TestCase):
    def setUp(self):
        self.node_input_size = 3
        self.output_size = 2
        self.device = torch.device("cpu")

        self.model = ClassificationPointTransformer(
            in_channels=self.node_input_size,
            dim_model=[32, 64, 128],
            out_channels=self.output_size,
        )

        self.simulator = ClassificationSimulator(
            node_input_size=self.node_input_size,
            edge_input_size=0,
            output_size=self.output_size,
            feature_index_start=0,
            feature_index_end=self.node_input_size,
            output_index_start=0,
            output_index_end=self.output_size,
            node_type_index=0,
            model=self.model,
            model_type="pointtransformer",
            device=self.device,
        )

        num_nodes = 10
        x = torch.randn(num_nodes, self.node_input_size)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 15))
        self.data = Data(x=x, pos=pos, edge_index=edge_index)

    def test_forward(self):
        self.simulator.train()
        output = self.simulator(self.data)
        self.assertEqual(output.shape, (1, self.output_size))


if __name__ == "__main__":
    unittest.main()
