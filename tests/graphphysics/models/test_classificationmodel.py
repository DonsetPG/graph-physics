import unittest
import torch
from torch_geometric.data import Data
from graphphysics.models.classification_model import ClassificationModel, decoder_1, decoder_2

class TestClassificationModel(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.num_edges = 10
        self.node_input_size = 8
        self.edge_input_size = 1
        self.output_size = 1
        self.hidden_size = 16
        self.message_passing_num = 3

        x = torch.randn(self.num_nodes, self.node_input_size)
        edge_attr = torch.randn(self.num_edges, self.edge_input_size)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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
        self.assertEqual(output.dim(), 0)

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
        self.assertEqual(output.dim(), 0)

class TestDecoder1(unittest.TestCase):
    def setUp(self):
        self.decoder = decoder_1(in_size=128, hidden_size=128)
        self.x = torch.randn(100, 128)  # 100 nodes with 128 features each

    def test_forward(self):
        output = self.decoder(self.x)
        self.assertEqual(output.shape, (1,))  

class TestDecoder2(unittest.TestCase):
    def setUp(self):
        self.decoder = decoder_2(in_size=100, hidden_size=128)
        self.x = torch.randn(100, 128)  # 100 nodes with 128 features each

    def test_forward(self):
        output = self.decoder(self.x)
        self.assertEqual(output.shape, (1,)) 


if __name__ == "__main__":
    unittest.main()
