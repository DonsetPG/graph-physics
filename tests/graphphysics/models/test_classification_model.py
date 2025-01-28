import unittest
import torch
from torch_geometric.data import Data, Batch
from graphphysics.models.classification_model import (
    ClassificationModel,
    Decoder_1,
    PointNetClassifier,
    ClassificationPointNetP2,
    ClassificationPointTransformer,
)


class TestClassificationModel(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.num_edges = 10
        self.node_input_size = 3
        self.edge_input_size = 1
        self.output_size = 1
        self.hidden_size = 16
        self.message_passing_num = 3

        x = torch.randn(self.num_nodes, self.node_input_size)
        edge_attr = torch.randn(self.num_edges, self.edge_input_size)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        pos = torch.randn(self.num_nodes, 3)
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        self.graph = Batch(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, pos=pos
        )

    def test_classification_model_forward(self):
        model = ClassificationModel(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )
        output = model(self.graph)
        # output should have dim  0 (scalar)
        self.assertEqual(output.shape, (1, 1))

    def test_gradients(self):
        model = ClassificationModel(
            message_passing_num=self.message_passing_num,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )
        output = model(self.graph)
        loss = output.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)

    def test_multiple_message_passing_steps(self):
        model = ClassificationModel(
            message_passing_num=5,
            node_input_size=self.node_input_size,
            edge_input_size=self.edge_input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
        )
        output = model(self.graph)
        self.assertEqual(output.shape, (1, 1))


class TestDecoder1(unittest.TestCase):
    def setUp(self):
        self.decoder = Decoder_1(in_size=128, hidden_size=128)
        self.x = torch.randn(100, 128)  # 100 nodes with 128 features each
        self.batch = torch.zeros(
            100, dtype=torch.long
        )  # All nodes belong to the same graph

    def test_forward(self):
        output = self.decoder(self.x, self.batch)
        self.assertEqual(output.shape, (1, 1))


class TestPointNetClassifier(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.node_input_size = 3
        self.hidden_size = 64
        self.output_size = 2

        x = torch.randn(self.num_nodes, self.node_input_size)
        pos = torch.randn(self.num_nodes, 3)
        edge_index = torch.randint(0, self.num_nodes, (2, 10))
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        self.graph = Batch(x=x, pos=pos, edge_index=edge_index, batch=batch)

    def test_forward(self):
        model = PointNetClassifier(
            node_input_size=self.node_input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )
        output = model(self.graph)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_gradients(self):
        model = PointNetClassifier(
            node_input_size=self.node_input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )
        output = model(self.graph)
        loss = output.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)


class TestClassificationPointNetP2(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_nodes = 5
        self.node_input_size = 3
        self.output_size = 2

        x = torch.randn(self.batch_size * self.num_nodes, self.node_input_size)
        pos = torch.randn(self.batch_size * self.num_nodes, 3)
        batch = torch.cat(
            [
                torch.full((self.num_nodes,), i, dtype=torch.long)
                for i in range(self.batch_size)
            ]
        )
        self.graph = Batch(x=x, pos=pos, batch=batch)

    def test_forward(self):
        model = ClassificationPointNetP2(
            node_input_size=self.node_input_size,
            output_size=self.output_size,
        )
        output = model(self.graph)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    def test_gradients(self):
        model = ClassificationPointNetP2(
            node_input_size=self.node_input_size,
            output_size=self.output_size,
        )
        output = model(self.graph)
        loss = output.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)


class TestClassificationPointTransformer(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.node_input_size = 3
        self.output_size = 2
        self.dim_model = [32, 64, 128]

        x = torch.randn(self.num_nodes, self.node_input_size)
        pos = torch.randn(self.num_nodes, 3)
        batch = torch.zeros(self.num_nodes, dtype=torch.long)
        self.graph = Batch(x=x, pos=pos, batch=batch)

    def test_forward(self):
        model = ClassificationPointTransformer(
            in_channels=self.node_input_size,
            dim_model=self.dim_model,
            out_channels=self.output_size,
        )
        output = model(self.graph)
        self.assertEqual(output.shape, (1, self.output_size))

    def test_gradients(self):
        model = ClassificationPointTransformer(
            in_channels=self.node_input_size,
            dim_model=self.dim_model,
            out_channels=self.output_size,
        )
        output = model(self.graph)
        loss = output.sum()
        loss.backward()
        # Check that gradients are computed
        params = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(params) > 0)


if __name__ == "__main__":
    unittest.main()
