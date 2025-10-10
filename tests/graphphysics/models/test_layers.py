import unittest
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected

from graphphysics.models.layers import (
    RMSNorm,
    build_mlp,
    GatedMLP,
    build_gated_mlp,
    Normalizer,
    scaled_query_key_softmax,
    scaled_dot_product_attention,
    Attention,
    Transformer,
    GraphNetBlock,
    set_use_silu_activation,
)

try:
    import dgl.sparse as dglsp

    HAS_DGL_SPARSE = True
except ImportError:
    HAS_DGL_SPARSE = False
    dglsp = None


class TestTransformerComponents(unittest.TestCase):
    def test_rmsnorm(self):
        d = 10
        x = torch.randn(5, d)
        rms_norm = RMSNorm(d)
        output = rms_norm(x)
        self.assertEqual(output.shape, x.shape)

    def test_build_mlp(self):
        set_use_silu_activation(False)
        in_size = 10
        hidden_size = 20
        out_size = 5
        nb_of_layers = 4
        mlp = build_mlp(in_size, hidden_size, out_size, nb_of_layers)
        x = torch.randn(3, in_size)
        output = mlp(x)
        self.assertEqual(output.shape, (3, out_size))
        self.assertTrue(
            any(isinstance(layer, nn.ReLU) for layer in mlp),
            "MLP should include ReLU activations when SiLU is disabled.",
        )
        set_use_silu_activation(True)
        mlp_silu = build_mlp(in_size, hidden_size, out_size, nb_of_layers)
        self.assertTrue(
            any(isinstance(layer, nn.SiLU) for layer in mlp_silu),
            "MLP should include SiLU activations when SiLU is enabled.",
        )
        set_use_silu_activation(False)

    def test_gated_mlp(self):
        set_use_silu_activation(False)
        in_size = 10
        hidden_size = 20
        expansion_factor = 2
        gated_mlp = GatedMLP(in_size, hidden_size, expansion_factor)
        x = torch.randn(3, in_size)
        output = gated_mlp(x)
        self.assertEqual(output.shape, (3, expansion_factor * hidden_size))
        self.assertIsInstance(
            gated_mlp.activation,
            nn.GELU,
            "GatedMLP should use GELU when SiLU disabled.",
        )
        set_use_silu_activation(True)
        gated_mlp_silu = GatedMLP(in_size, hidden_size, expansion_factor)
        self.assertIsInstance(
            gated_mlp_silu.activation,
            nn.SiLU,
            "GatedMLP should use SiLU when SiLU enabled.",
        )
        set_use_silu_activation(False)

    def test_build_gated_mlp(self):
        in_size = 10
        hidden_size = 20
        out_size = 5
        gated_mlp = build_gated_mlp(in_size, hidden_size, out_size)
        x = torch.randn(3, in_size)
        output = gated_mlp(x)
        self.assertEqual(output.shape, (3, out_size))

    def test_normalizer(self):
        size = 5
        normalizer = Normalizer(size, device="cpu")
        x = torch.randn(10, size)
        normalized_x = normalizer(x)
        self.assertEqual(normalized_x.shape, x.shape)

        reconstructed_x = normalizer.inverse(normalized_x)
        self.assertTrue(torch.allclose(x, reconstructed_x, atol=1e-6))

    def test_scaled_query_key_softmax(self):
        q = torch.randn(5, 10)
        k = torch.randn(5, 10)
        if HAS_DGL_SPARSE:
            adj = dglsp.from_coo(
                torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), shape=(5, 5)
            )
            attn = scaled_query_key_softmax(q, k, adj)
        else:
            attn = scaled_query_key_softmax(q, k, None)
        self.assertEqual(attn.shape[0], q.shape[0])

    def test_scaled_dot_product_attention(self):
        q = torch.randn(5, 10)
        k = torch.randn(5, 10)
        v = torch.randn(5, 15)
        if HAS_DGL_SPARSE:
            adj = dglsp.from_coo(
                torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), shape=(5, 5)
            )
            y = scaled_dot_product_attention(q, k, v, adj)
        else:
            y = scaled_dot_product_attention(q, k, v)
        self.assertEqual(y.shape[0], q.shape[0])
        self.assertEqual(y.shape[1], v.shape[1])

    def test_attention(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        attention = Attention(input_dim, output_dim, num_heads)
        x = torch.randn(5, input_dim)
        if HAS_DGL_SPARSE:
            adj = dglsp.from_coo(
                torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), shape=(5, 5)
            )
            output = attention(x, adj)
        else:
            output = attention(x, None)
        self.assertEqual(output.shape, (5, output_dim))

    def test_attention_with_rope(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        attention = Attention(
            input_dim,
            output_dim,
            num_heads,
            use_rope_embeddings=True,
        )
        x = torch.randn(5, input_dim)
        pos = torch.randn(5, 3)
        output = attention(x, None, pos=pos)
        self.assertEqual(output.shape, (5, output_dim))
        attention_no_pos = Attention(
            input_dim,
            output_dim,
            num_heads,
            use_rope_embeddings=True,
        )
        with self.assertRaises(ValueError):
            attention_no_pos(x, None)

    def test_attention_with_gate(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        attention = Attention(
            input_dim,
            output_dim,
            num_heads,
            use_gated_attention=True,
        )
        x = torch.randn(5, input_dim)
        output = attention(x, None)
        self.assertEqual(output.shape, (5, output_dim))
        self.assertIsNotNone(attention.gate_proj)

    def test_transformer(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        transformer = Transformer(input_dim, output_dim, num_heads)
        x = torch.randn(5, input_dim)
        if HAS_DGL_SPARSE:
            adj = dglsp.from_coo(
                torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), shape=(5, 5)
            )
            output = transformer(x, adj)
        else:
            output = transformer(x, None)
        self.assertEqual(output.shape, (5, output_dim))

    def test_transformer_with_attention_output(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        transformer = Transformer(input_dim, output_dim, num_heads)
        x = torch.randn(5, input_dim)
        if HAS_DGL_SPARSE:
            adj = dglsp.from_coo(
                torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]), shape=(5, 5)
            )
            output, attn = transformer(x, adj, return_attention=True)
        else:
            output, attn = transformer(x, None, return_attention=True)
        self.assertEqual(output.shape, (5, output_dim))
        self.assertIsNotNone(attn)

    def test_transformer_with_rope(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        transformer = Transformer(
            input_dim,
            output_dim,
            num_heads,
            use_rope_embeddings=True,
        )
        x = torch.randn(5, input_dim)
        pos = torch.randn(5, 3)
        output = transformer(x, None, pos=pos)
        self.assertEqual(output.shape, (5, output_dim))
        transformer_no_pos = Transformer(
            input_dim,
            output_dim,
            num_heads,
            use_rope_embeddings=True,
        )
        with self.assertRaises(ValueError):
            transformer_no_pos(x, None)

    def test_transformer_with_gate(self):
        input_dim = 16
        output_dim = 16
        num_heads = 4
        transformer = Transformer(
            input_dim,
            output_dim,
            num_heads,
            use_gated_attention=True,
        )
        x = torch.randn(5, input_dim)
        output = transformer(x, None)
        self.assertEqual(output.shape, (5, output_dim))


class TestGraphNetBlock(unittest.TestCase):
    def setUp(self):
        # Create a simple undirected graph with 4 nodes and 4 edges
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_index = to_undirected(edge_index)

        num_nodes = 4
        hidden_size = 16

        x = torch.randn(num_nodes, hidden_size)

        num_edges = edge_index.size(1)
        edge_attr = torch.randn(num_edges, hidden_size)

        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.hidden_size = hidden_size

    def test_graphnetblock_forward(self):
        block = GraphNetBlock(hidden_size=self.hidden_size)

        x_updated, edge_attr_updated = block(self.x, self.edge_index, self.edge_attr)

        self.assertEqual(x_updated.shape, self.x.shape)
        self.assertEqual(edge_attr_updated.shape, self.edge_attr.shape)

    def test_graphnetblock_gradients(self):
        block = GraphNetBlock(hidden_size=self.hidden_size)
        x = self.x.clone().requires_grad_(True)
        edge_attr = self.edge_attr.clone().requires_grad_(True)

        x_updated, edge_attr_updated = block(x, self.edge_index, edge_attr)

        # Compute a dummy loss
        loss = x_updated.sum() + edge_attr_updated.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(edge_attr.grad)

    def test_graphnetblock_multiple_steps(self):
        block = GraphNetBlock(hidden_size=self.hidden_size)
        x = self.x.clone()
        edge_attr = self.edge_attr.clone()

        # Run multiple steps
        for _ in range(3):
            x, edge_attr = block(x, self.edge_index, edge_attr)

        # Check the shapes
        self.assertEqual(x.shape, self.x.shape)
        self.assertEqual(edge_attr.shape, self.edge_attr.shape)

    def test_graphnetblock_with_layer_norm(self):
        block = GraphNetBlock(hidden_size=self.hidden_size, layer_norm=True)
        x_updated, edge_attr_updated = block(self.x, self.edge_index, self.edge_attr)
        # Check that outputs are computed
        self.assertEqual(x_updated.shape, self.x.shape)
        self.assertEqual(edge_attr_updated.shape, self.edge_attr.shape)


if __name__ == "__main__":
    unittest.main()
