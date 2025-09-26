import unittest
import torch
from graphphysics.models.transolver import (
    gumbel_softmax,
    Physics_Attention_1D_Eidetic,
    Model,
)


class TestTransolverComponents(unittest.TestCase):
    def test_gumbel_softmax(self):
        logits = torch.randn(3, 5)
        result = gumbel_softmax(logits, tau=1, hard=False)
        self.assertEqual(result.shape, (3, 5))
        self.assertTrue(torch.allclose(result.sum(dim=-1), torch.ones(3)))

    def test_physics_attention_1d_eidetic(self):
        attention_layer = Physics_Attention_1D_Eidetic(
            dim=64, heads=8, dim_head=8, dropout=0.0, slice_num=64
        )
        x = torch.randn(1, 10, 64)
        output = attention_layer(x)
        self.assertEqual(output.shape, (1, 10, 64))

    def test_transolver(self):
        node_input_size = 5
        node_output_size = 3
        transolver_model = Model(
            space_dim=0,
            n_layers=3,
            n_hidden=16,
            fun_dim=node_input_size,
            out_dim=node_output_size,
        )
        x = torch.randn(1, 10, node_input_size)
        pos = None
        condition = None
        output = transolver_model(x, pos, condition)
        self.assertEqual(output.shape, (1, 10, node_output_size))
