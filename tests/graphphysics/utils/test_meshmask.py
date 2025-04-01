import unittest
import torch
from torch import nn
from torch_geometric.data import Data
from graphphysics.utils.meshmask import (
    filter_edges,
    build_masked_graph,
    reconstruct_graph,
)


class TestGraphFunctionsUpdated(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_feature_dim = 16
        self.edge_feature_dim = 8

        # Sample Graph 1 (5 nodes, 6 edges with attributes)
        self.edge_index1 = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
        )
        self.x1 = torch.randn(5, self.node_feature_dim)
        self.pos1 = torch.randn(5, 3)
        self.edge_attr1 = torch.randn(6, self.edge_feature_dim)  # One attr per edge
        self.graph1 = Data(
            x=self.x1,
            edge_index=self.edge_index1,
            pos=self.pos1,
            edge_attr=self.edge_attr1,
        ).to(self.device)

        # Sample Graph 2 (4 nodes, 2 edges, no edge_attr, no pos)
        self.edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.x2 = torch.randn(4, self.node_feature_dim)
        self.graph2 = Data(x=self.x2, edge_index=self.edge_index2).to(self.device)

        # Reconstruct components
        self.node_mask_token = nn.Parameter(
            torch.randn(1, self.node_feature_dim) * 0.1
        ).to(
            self.device
        )  # Single row, expand later
        self.edge_mask_token = nn.Parameter(
            torch.randn(1, self.edge_feature_dim) * 0.2
        ).to(
            self.device
        )  # Single row, expand later
        # Simple linear encoder for testing edge reconstruction
        self.edge_encoder = nn.Linear(self.edge_feature_dim, self.edge_feature_dim).to(
            self.device
        )
        # Identity encoder for simpler testing
        self.identity_encoder = nn.Identity().to(self.device)

    # --- Tests for filter_edges ---

    def test_filter_edges_with_attr(self):
        """Test filtering edges with attributes."""
        node_index = torch.tensor([0, 1, 2], dtype=torch.long).to(
            self.device
        )  # Keep nodes 0, 1, 2
        expected_ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(
            self.device
        )
        expected_mask = torch.tensor(
            [True, True, True, True, False, False], dtype=torch.bool
        ).to(self.device)
        expected_ea = self.graph1.edge_attr[expected_mask]

        filtered_ei, filtered_ea, mask = filter_edges(
            self.graph1.edge_index, node_index, self.graph1.edge_attr
        )

        self.assertTrue(torch.equal(filtered_ei, expected_ei))
        self.assertEqual(filtered_ei.device.type, self.device.type)
        torch.testing.assert_close(filtered_ea, expected_ea)
        self.assertEqual(filtered_ea.device.type, self.device.type)
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(mask.device.type, self.device.type)

    def test_filter_edges_without_attr(self):
        """Test filtering edges when edge_attr is None."""
        node_index = torch.tensor([0, 1], dtype=torch.long).to(
            self.device
        )  # Keep nodes 0, 1
        expected_ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
        expected_mask = torch.tensor([True, True], dtype=torch.bool).to(self.device)

        # Use graph2 which has no edge_attr
        filtered_ei, filtered_ea, mask = filter_edges(
            self.graph2.edge_index, node_index, None
        )  # Explicitly pass None

        self.assertTrue(torch.equal(filtered_ei, expected_ei))
        self.assertIsNone(filtered_ea)  # Expect None for attributes
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_filter_edges_no_common_edges_with_attr(self):
        """Test filtering with attributes when selected nodes have no common edges."""
        node_index = torch.tensor([0, 3], dtype=torch.long).to(
            self.device
        )  # Keep nodes 0, 3
        expected_ei = torch.empty((2, 0), dtype=torch.long).to(self.device)
        expected_mask = torch.tensor(
            [False, False, False, False, False, False], dtype=torch.bool
        ).to(self.device)
        expected_ea = torch.empty((0, self.edge_feature_dim)).to(
            self.device
        )  # Expect empty tensor

        filtered_ei, filtered_ea, mask = filter_edges(
            self.graph1.edge_index, node_index, self.graph1.edge_attr
        )

        self.assertTrue(torch.equal(filtered_ei, expected_ei))
        torch.testing.assert_close(filtered_ea, expected_ea)
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(filtered_ea.shape[0], 0)  # Check shape explicitly

    def test_build_masked_graph_with_attr(self):
        """Test building a masked graph with edge attributes."""
        selected_indexes = torch.tensor([0, 1, 2], dtype=torch.long).to(self.device)
        graph_to_mask = self.graph1.clone()

        masked_graph, edges_mask = build_masked_graph(graph_to_mask, selected_indexes)

        expected_ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).to(
            self.device
        )
        expected_mask = torch.tensor(
            [True, True, True, True, False, False], dtype=torch.bool
        ).to(self.device)
        expected_x = self.graph1.x[selected_indexes]
        expected_pos = self.graph1.pos[selected_indexes]
        expected_ea = self.graph1.edge_attr[expected_mask]

        self.assertEqual(masked_graph.num_nodes, 3)
        torch.testing.assert_close(masked_graph.x, expected_x)
        torch.testing.assert_close(masked_graph.pos, expected_pos)
        self.assertTrue(torch.equal(masked_graph.edge_index, expected_ei))
        torch.testing.assert_close(masked_graph.edge_attr, expected_ea)
        self.assertTrue(torch.equal(edges_mask, expected_mask))

        # Check device consistency
        self.assertEqual(masked_graph.x.device.type, self.device.type)
        self.assertEqual(masked_graph.pos.device.type, self.device.type)
        self.assertEqual(masked_graph.edge_index.device.type, self.device.type)
        self.assertEqual(masked_graph.edge_attr.device.type, self.device.type)
        self.assertEqual(edges_mask.device.type, self.device.type)

    def test_build_masked_graph_without_attr(self):
        """Test building masked graph when original has no edge attributes."""
        selected_indexes = torch.tensor([0, 1], dtype=torch.long).to(self.device)
        graph_to_mask = self.graph2.clone()  # graph2 has no edge_attr or pos
        masked_graph, edges_mask = build_masked_graph(graph_to_mask, selected_indexes)
        expected_ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
        expected_mask = torch.tensor([True, True], dtype=torch.bool).to(self.device)
        expected_x = self.graph2.x[selected_indexes]

        self.assertEqual(masked_graph.num_nodes, 2)
        torch.testing.assert_close(masked_graph.x, expected_x)
        self.assertTrue(torch.equal(masked_graph.edge_index, expected_ei))
        self.assertFalse(getattr(masked_graph, "pos", None) is not None)
        self.assertFalse(
            getattr(masked_graph, "edge_attr", None) is not None
        )  # Should not have edge_attr
        self.assertTrue(torch.equal(edges_mask, expected_mask))

    def test_reconstruct_graph_with_attr(self):
        """Test reconstructing a graph including edge attributes."""
        original_graph = self.graph1.clone()
        selected_nodes_idx = torch.tensor([0, 2, 4], dtype=torch.long).to(self.device)
        num_selected_nodes = len(selected_nodes_idx)

        # 1. Simulate building the masked graph to get the correct edges_mask and latent structure
        # We only keep edges between selected nodes {0, 2, 4}. In graph1, there are NO edges between just these nodes.
        # Let's change selection to {0, 1, 2} for a more interesting edge case.
        selected_nodes_idx = torch.tensor([0, 1, 2], dtype=torch.long).to(self.device)
        num_selected_nodes = len(selected_nodes_idx)
        # Edges kept: (0,1), (1,0), (1,2), (2,1) -> Indices 0, 1, 2, 3 in original edge list
        expected_edges_mask = torch.tensor(
            [True, True, True, True, False, False], dtype=torch.bool
        ).to(self.device)
        num_kept_edges = expected_edges_mask.sum()

        # 2. Create a *simulated* latent_masked_graph (output of some GNN)
        # It should have node features for selected nodes and edge features for kept edges.
        latent_node_x = torch.randn(num_selected_nodes, self.node_feature_dim).to(
            self.device
        )
        latent_edge_attr = torch.randn(num_kept_edges, self.edge_feature_dim).to(
            self.device
        )  # Use encoded dimension
        # We don't strictly need the correct edge_index/pos in latent_masked_graph for reconstruct_graph, only x and edge_attr
        latent_masked_graph = Data(x=latent_node_x, edge_attr=latent_edge_attr).to(
            self.device
        )

        # 3. Reconstruct
        # Use identity encoder for simplicity here to match dimensions easily
        reconstructed_graph = reconstruct_graph(
            original_graph,
            latent_masked_graph,
            selected_nodes_idx,
            self.node_mask_token,
            expected_edges_mask,  # Pass the correct mask
            edge_encoder=self.identity_encoder,  # Use identity
            edge_mask_token=self.edge_mask_token,
        )

        # --- Assertions ---
        # Node features
        self.assertEqual(reconstructed_graph.num_nodes, original_graph.num_nodes)
        self.assertEqual(reconstructed_graph.x.shape, original_graph.x.shape)
        torch.testing.assert_close(
            reconstructed_graph.x[selected_nodes_idx], latent_node_x
        )
        unselected_nodes_mask = torch.ones(
            original_graph.num_nodes, dtype=torch.bool, device=self.device
        )
        unselected_nodes_mask[selected_nodes_idx] = False
        expected_node_mask_features = self.node_mask_token.expand(
            unselected_nodes_mask.sum(), -1
        )
        torch.testing.assert_close(
            reconstructed_graph.x[unselected_nodes_mask], expected_node_mask_features
        )

        # Edge features
        self.assertTrue(hasattr(reconstructed_graph, "edge_attr"))
        self.assertEqual(
            reconstructed_graph.edge_attr.shape, original_graph.edge_attr.shape
        )

        # - Check unmasked edges (where expected_edges_mask is True)
        torch.testing.assert_close(
            reconstructed_graph.edge_attr[expected_edges_mask], latent_edge_attr
        )

        # - Check masked edges (where expected_edges_mask is False)
        masked_original_attrs = original_graph.edge_attr[~expected_edges_mask]
        # Apply encoder (identity in this case) + mask token
        expected_masked_edge_attrs = self.identity_encoder(
            masked_original_attrs
        ) + self.edge_mask_token.expand(masked_original_attrs.shape[0], -1)
        torch.testing.assert_close(
            reconstructed_graph.edge_attr[~expected_edges_mask],
            expected_masked_edge_attrs,
        )

        # Other attributes (should be same as original)
        self.assertTrue(
            torch.equal(reconstructed_graph.edge_index, original_graph.edge_index)
        )
        torch.testing.assert_close(reconstructed_graph.pos, original_graph.pos)

    def test_reconstruct_graph_without_attr(self):
        """Test reconstructing when the original graph had no edge attributes."""
        original_graph = self.graph2.clone()  # graph2 has no edge_attr
        selected_nodes_idx = torch.tensor([0], dtype=torch.long).to(
            self.device
        )  # Select just one node
        num_selected_nodes = len(selected_nodes_idx)

        # Edges mask: Graph2 edges are (0,1), (1,0). Selecting node 0 keeps no edges.
        expected_edges_mask = torch.tensor([False, False], dtype=torch.bool).to(
            self.device
        )
        num_kept_edges = 0

        latent_node_x = torch.randn(num_selected_nodes, self.node_feature_dim).to(
            self.device
        )
        # No latent edge attributes needed as original had none
        latent_masked_graph = Data(x=latent_node_x).to(self.device)

        reconstructed_graph = reconstruct_graph(
            original_graph,
            latent_masked_graph,
            selected_nodes_idx,
            self.node_mask_token,
            expected_edges_mask,  # Pass the mask
            edge_encoder=None,  # No encoder needed
            edge_mask_token=None,  # No token needed
        )

        # Assertions
        self.assertEqual(reconstructed_graph.num_nodes, original_graph.num_nodes)
        torch.testing.assert_close(
            reconstructed_graph.x[selected_nodes_idx], latent_node_x
        )
        unselected_nodes_mask = torch.ones(
            original_graph.num_nodes, dtype=torch.bool, device=self.device
        )
        unselected_nodes_mask[selected_nodes_idx] = False
        expected_node_mask_features = self.node_mask_token.expand(
            unselected_nodes_mask.sum(), -1
        )
        torch.testing.assert_close(
            reconstructed_graph.x[unselected_nodes_mask], expected_node_mask_features
        )

        # Should NOT have edge attributes
        self.assertFalse(getattr(reconstructed_graph, "edge_attr", None) is not None)

        # Other attributes
        self.assertTrue(
            torch.equal(reconstructed_graph.edge_index, original_graph.edge_index)
        )
        self.assertFalse(
            getattr(reconstructed_graph, "pos", None) is not None
        )  # Graph 2 had no pos


if __name__ == "__main__":
    unittest.main()
