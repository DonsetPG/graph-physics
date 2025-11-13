import math
import unittest

import torch
from torch_geometric.data import Data

from graphphysics.models.hierarchical_pooling import DownSampler, UpSampler


class TestHierarchicalPoolingPrimitives(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 32
        self.node_input_size = 8
        self.hidden_size = 16
        self.edge_size = 4
        self.ratio = 0.5
        self.k = 4

        x = torch.randn(self.num_nodes, self.node_input_size)
        pos = torch.randn(self.num_nodes, 3)
        num_edges = self.num_nodes * 2
        edge_index = torch.randint(0, self.num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_size)

        self.graph = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        self.graph.batch = torch.zeros(self.num_nodes, dtype=torch.long)

    def test_downsampler_reduces_node_count(self):
        downsampler = DownSampler(
            d_in=self.node_input_size,
            d_out=self.hidden_size,
            edge_dim=self.edge_size,
            ratio=self.ratio,
            k=self.k,
        )
        coarse_graph = downsampler(self.graph)

        expected_nodes = math.ceil(self.num_nodes * self.ratio)
        self.assertEqual(coarse_graph.x.shape[0], expected_nodes)
        self.assertEqual(coarse_graph.x.shape[1], self.hidden_size)
        self.assertEqual(coarse_graph.edge_attr.shape[1], self.edge_size)

    def test_upsampler_restores_fine_resolution(self):
        downsampler = DownSampler(
            d_in=self.node_input_size,
            d_out=self.hidden_size,
            edge_dim=self.edge_size,
            ratio=self.ratio,
            k=self.k,
        )
        coarse_graph = downsampler(self.graph)

        upsampler = UpSampler(
            d_in=self.hidden_size,
            d_out=self.hidden_size,
            k=self.k,
        )
        upsampled = upsampler(
            x_coarse=coarse_graph.x,
            pos_coarse=coarse_graph.pos,
            pos_fine=self.graph.pos,
            batch_coarse=coarse_graph.batch,
            batch_fine=self.graph.batch,
        )

        self.assertEqual(upsampled.shape, (self.num_nodes, self.hidden_size))


if __name__ == "__main__":
    unittest.main()
