import unittest
import torch
from torch import nn
from torch_geometric.data import Data
from graphphysics.utils.meshmask import (
    filter_edges,
    build_masked_graph,
    reconstruct_graph,
)


class TestGraphFunctions(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.edge_index1 = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
        self.x1 = torch.randn(5, 16)
        self.pos1 = torch.randn(5, 3)
        self.graph1 = Data(x=self.x1, edge_index=self.edge_index1, pos=self.pos1).to(
            self.device
        )

    def test_filter_edges_basic(self):
        """Test filtering edges with a subset of nodes."""
        node_index = torch.tensor([0, 1, 2], dtype=torch.long).to(
            self.device
        )  # Keep nodes 0, 1, 2
        expected_edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ).to(self.device)
        # Edges expected: 0-1, 1-0, 1-2, 2-1 (remapped)

        filtered_ei, mask = filter_edges(self.graph1.edge_index, node_index)

        self.assertTrue(torch.equal(filtered_ei, expected_edge_index))
        self.assertEqual(filtered_ei.device.type, self.device.type)
        self.assertEqual(mask.sum(), 4)  # 4 edges should be kept
        self.assertEqual(
            mask.shape[0], self.graph1.edge_index.shape[1]
        )  # Mask size = num original edges
        original_edges_kept = self.graph1.edge_index[:, mask]
        self.assertTrue(
            (original_edges_kept < 3).all()
        )  # All nodes in kept edges should be < 3

    def test_filter_edges_no_common_edges(self):
        """Test filtering when selected nodes have no edges between them."""
        node_index = torch.tensor([0, 3], dtype=torch.long).to(
            self.device
        )  # Keep nodes 0, 3
        expected_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)

        filtered_ei, mask = filter_edges(self.graph1.edge_index, node_index)

        self.assertTrue(torch.equal(filtered_ei, expected_edge_index))
        self.assertEqual(filtered_ei.device.type, self.device.type)
        self.assertEqual(mask.sum(), 0)

    def test_filter_edges_empty_selection(self):
        """Test filtering with empty node_index."""
        node_index = torch.tensor([], dtype=torch.long).to(self.device)
        expected_edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)

        filtered_ei, mask = filter_edges(self.graph1.edge_index, node_index)

        self.assertTrue(torch.equal(filtered_ei, expected_edge_index))
        self.assertEqual(filtered_ei.device.type, self.device.type)
        self.assertEqual(mask.sum(), 0)

    def test_build_masked_graph_basic(self):
        """Test building a masked graph with features and positions."""
        selected_indexes = torch.tensor([0, 1, 2], dtype=torch.long).to(self.device)
        graph_to_mask = self.graph1.clone()

        masked_graph = build_masked_graph(graph_to_mask, selected_indexes)

        expected_edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ).to(self.device)
        expected_x = self.graph1.x[selected_indexes]
        expected_pos = self.graph1.pos[selected_indexes]

        self.assertEqual(masked_graph.num_nodes, 3)
        torch.testing.assert_close(masked_graph.x, expected_x)
        torch.testing.assert_close(masked_graph.pos, expected_pos)
        self.assertTrue(torch.equal(masked_graph.edge_index, expected_edge_index))
        self.assertEqual(masked_graph.x.device.type, self.device.type)
        self.assertEqual(masked_graph.pos.device.type, self.device.type)
        self.assertEqual(masked_graph.edge_index.device.type, self.device.type)

    def test_build_masked_graph_select_all(self):
        """Test building a masked graph selecting all nodes (should be identity)."""
        selected_indexes = torch.arange(self.graph1.num_nodes, dtype=torch.long).to(
            self.device
        )
        graph_to_mask = self.graph1.clone()

        masked_graph = build_masked_graph(graph_to_mask, selected_indexes)

        expected_edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        ).to(self.device)

        self.assertEqual(masked_graph.num_nodes, self.graph1.num_nodes)
        torch.testing.assert_close(masked_graph.x, self.graph1.x)
        torch.testing.assert_close(masked_graph.pos, self.graph1.pos)
        self.assertTrue(torch.equal(masked_graph.edge_index, expected_edge_index))

    def test_reconstruct_graph_basic(self):
        """Test reconstructing a graph from its masked latent version."""
        original_graph = self.graph1.clone()
        selected_indexes = torch.tensor([0, 2, 4], dtype=torch.long).to(self.device)
        embedding_dim = original_graph.x.shape[1]

        # Simulate a latent masked graph (e.g., output of an encoder)
        # Has features only for selected nodes (3 nodes), potentially modified
        latent_x = torch.randn(3, embedding_dim).to(self.device)
        latent_masked_graph = Data(x=latent_x).to(
            self.device
        )  # Only need x for this function's perspective

        # Mask token
        mask_token = nn.Parameter(torch.zeros(embedding_dim, device=self.device))
        mask_token.data += 0.5  # Give it a non-zero value for easier testing

        reconstructed_graph = reconstruct_graph(
            original_graph, latent_masked_graph, selected_indexes, mask_token
        )

        self.assertEqual(reconstructed_graph.num_nodes, original_graph.num_nodes)
        self.assertEqual(reconstructed_graph.x.shape, original_graph.x.shape)
        self.assertEqual(reconstructed_graph.x.device.type, self.device.type)

        # Check selected nodes have features from latent graph
        torch.testing.assert_close(reconstructed_graph.x[selected_indexes], latent_x)

        # Check unselected nodes have the mask token
        unselected_mask = torch.ones(
            original_graph.num_nodes, dtype=torch.bool, device=self.device
        )
        unselected_mask[selected_indexes] = False
        expected_mask_features = mask_token.data.unsqueeze(0).expand(
            unselected_mask.sum(), -1
        )
        torch.testing.assert_close(
            reconstructed_graph.x[unselected_mask], expected_mask_features
        )

        # Check other attributes are preserved
        self.assertTrue(
            torch.equal(reconstructed_graph.edge_index, original_graph.edge_index)
        )
        torch.testing.assert_close(reconstructed_graph.pos, original_graph.pos)
        self.assertEqual(reconstructed_graph.edge_index.device.type, self.device.type)
        self.assertEqual(reconstructed_graph.pos.device.type, self.device.type)

    def test_reconstruct_graph_select_all(self):
        """Test reconstructing when all nodes were selected."""
        original_graph = self.graph1.clone()
        selected_indexes = torch.arange(original_graph.num_nodes, dtype=torch.long).to(
            self.device
        )
        embedding_dim = original_graph.x.shape[1]

        # Latent graph has features for all nodes
        latent_x = torch.randn(original_graph.num_nodes, embedding_dim).to(self.device)
        latent_masked_graph = Data(x=latent_x).to(self.device)

        mask_token = nn.Parameter(torch.zeros(embedding_dim, device=self.device))

        reconstructed_graph = reconstruct_graph(
            original_graph, latent_masked_graph, selected_indexes, mask_token
        )

        self.assertEqual(reconstructed_graph.num_nodes, original_graph.num_nodes)
        # All features should come from the latent graph
        torch.testing.assert_close(reconstructed_graph.x, latent_x)
        # Check other attributes
        self.assertTrue(
            torch.equal(reconstructed_graph.edge_index, original_graph.edge_index)
        )
        torch.testing.assert_close(reconstructed_graph.pos, original_graph.pos)

    def test_reconstruct_graph_select_none(self):
        """Test reconstructing when no nodes were selected."""
        original_graph = self.graph1.clone()
        selected_indexes = torch.tensor([], dtype=torch.long).to(self.device)
        embedding_dim = original_graph.x.shape[1]

        # Latent graph has features for 0 nodes
        latent_x = torch.empty((0, embedding_dim), device=self.device)
        latent_masked_graph = Data(x=latent_x).to(self.device)

        mask_token = nn.Parameter(
            torch.ones(embedding_dim, device=self.device) * -1.0
        )  # Use a specific value

        reconstructed_graph = reconstruct_graph(
            original_graph, latent_masked_graph, selected_indexes, mask_token
        )

        self.assertEqual(reconstructed_graph.num_nodes, original_graph.num_nodes)
        # All features should be the mask token
        expected_mask_features = mask_token.data.unsqueeze(0).expand(
            original_graph.num_nodes, -1
        )
        torch.testing.assert_close(reconstructed_graph.x, expected_mask_features)
        # Check other attributes
        self.assertTrue(
            torch.equal(reconstructed_graph.edge_index, original_graph.edge_index)
        )
        torch.testing.assert_close(reconstructed_graph.pos, original_graph.pos)


if __name__ == "__main__":
    unittest.main()
