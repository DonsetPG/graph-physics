import unittest
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected

from graphphysics.models.layers import (
    AdaptiveAdjacencyPolicy,
    RMSNorm,
    LoopIndexEmbedding,
    NodeACTHalting,
    NodeMoEFFN,
    RecurrentGraphCore,
    StableInjection,
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

    def test_stable_injection_bounds(self):
        layer = StableInjection(dim=8)
        coeff = layer.get_A()
        self.assertTrue(torch.all(coeff > 0))
        self.assertTrue(torch.all(coeff < 1))

        h = torch.randn(4, 8)
        e = torch.randn(4, 8)
        update = torch.randn(4, 8)
        output = layer(h, e, update)
        self.assertEqual(output.shape, h.shape)

    def test_loop_index_embedding_broadcast(self):
        layer = LoopIndexEmbedding(hidden_size=8, loop_dim=4)
        x = torch.zeros(3, 8)
        y = layer(x, loop_index=2)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y[0] - x[0], y[1] - x[1]))

    def test_node_moe_ffn_shape_and_stats(self):
        moe = NodeMoEFFN(dim=8, n_experts=4, top_k=2, n_shared=1)
        x = torch.randn(6, 8)
        y = moe(x)
        self.assertEqual(y.shape, x.shape)
        self.assertIn("moe/router_entropy", moe.last_stats)
        self.assertIn("moe/expert_0_utilization", moe.last_stats)

    def test_node_act_halting_freezes_halted_nodes(self):
        halting = NodeACTHalting(dim=8, threshold=0.99)
        halting.halt_head.weight.data.zero_()
        halting.halt_head.bias.data.fill_(20.0)

        state = halting.initial_state(
            num_nodes=4,
            hidden_size=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        candidate = torch.randn(4, 8)
        previous = torch.zeros_like(candidate)
        active = torch.ones(4, dtype=torch.bool)

        hidden, state = halting.step(candidate, previous, state, active, loop_index=0)
        self.assertTrue(state["halted"].all())

        next_candidate = torch.randn(4, 8)
        hidden_next, _ = halting.step(
            next_candidate,
            hidden,
            state,
            active,
            loop_index=1,
        )
        self.assertTrue(torch.allclose(hidden, hidden_next))

        final_output, metrics = halting.finalize(
            current_hidden=hidden_next,
            state=state,
            final_depths=torch.ones(4),
        )
        self.assertEqual(final_output.shape, hidden.shape)
        self.assertIn("act/mean_exit_depth", metrics)

    def test_adaptive_adjacency_target_active_mask(self):
        policy = AdaptiveAdjacencyPolicy(hidden_size=8, mode="target_active", pos_dim=0)
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
        hidden = torch.randn(3, 8)
        active_targets = torch.tensor([True, False, True])

        filtered_edge_index, stats = policy(
            base_edge_index=edge_index,
            hidden=hidden,
            active_targets=active_targets,
        )
        self.assertTrue(torch.all(active_targets[filtered_edge_index[1]]))
        self.assertIn("adj/retained_edge_ratio", stats)

    def test_recurrent_graph_core_reports_metrics(self):
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_ffn_metrics = {}

            def forward(self, x, structure, pos=None):
                return x + 0.1

        core = RecurrentGraphCore(
            recurrent_blocks=nn.ModuleList([ResidualBlock()]),
            hidden_size=8,
            max_loops=2,
            eval_loops=2,
            loop_embedding=LoopIndexEmbedding(hidden_size=8, loop_dim=4),
            stable_injection=StableInjection(dim=8),
            halting=None,
            adjacency_policy=None,
            loop_sampling="off",
            mu_rec=2,
            mu_bwd=1,
        )
        encoded = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        output, metrics = core(encoded, edge_index)
        self.assertEqual(output.shape, encoded.shape)
        self.assertIn("loop/mean_sampled_loops", metrics)
        self.assertIn("loop/spectral_radius_max", metrics)


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

        pos = torch.randn(num_nodes, 3)
        phi = torch.randn(num_nodes)

        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.hidden_size = hidden_size
        self.pos = pos
        self.phi = phi

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

    def test_graphnetblock_with_gated_mlp(self):
        block = GraphNetBlock(hidden_size=self.hidden_size, use_gated_mlp=True)
        x = self.x.clone().requires_grad_(True)
        edge_attr = self.edge_attr.clone().requires_grad_(True)

        x_updated, edge_attr_updated = block(x, self.edge_index, edge_attr)

        self.assertEqual(x_updated.shape, x.shape)
        self.assertEqual(edge_attr_updated.shape, edge_attr.shape)

        loss = x_updated.sum() + edge_attr_updated.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(edge_attr.grad)

    def test_graphnetblock_rope_requires_pos(self):
        block = GraphNetBlock(
            hidden_size=self.hidden_size, use_rope=True, rope_axes=3, rope_base=1000.0
        )
        with self.assertRaises(ValueError):
            block(self.x, self.edge_index, self.edge_attr)

    def test_graphnetblock_with_rope_and_gate(self):
        block = GraphNetBlock(
            hidden_size=self.hidden_size,
            use_rope=True,
            rope_axes=3,
            use_gate=True,
        )
        x = self.x.clone().requires_grad_(True)
        edge_attr = self.edge_attr.clone().requires_grad_(True)
        phi = self.phi.clone().requires_grad_(False)

        x_updated, edge_attr_updated = block(
            x,
            self.edge_index,
            edge_attr,
            pos=self.pos,
            phi=phi,
        )

        self.assertEqual(x_updated.shape, x.shape)
        self.assertEqual(edge_attr_updated.shape, edge_attr.shape)

        loss = x_updated.sum() + edge_attr_updated.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(edge_attr.grad)


if __name__ == "__main__":
    unittest.main()
