import unittest

import torch
from torch_geometric.data import Data

from graphphysics.utils.vectorial_operators import (
    compute_divergence,
    compute_gradient,
    compute_vector_gradient_product,
)


class TestComputeAllGradients(unittest.TestCase):

    def test_compute_gradients_2d(self):
        # 4 nodes in a square, for simplicity
        #   (0,1) --- (1,1)
        #     |    /    |
        #   (0,0) --- (1,0)
        pos = torch.tensor(
            [
                [0.0, 0.0],  # node 0
                [1.0, 0.0],  # node 1
                [0.0, 1.0],  # node 2
                [1.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )

        # Directed edges of a square, one outgoing edge per node
        edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 3, 0, 2, 3]], dtype=torch.long)
        face = torch.tensor([[0, 0], [1, 2], [3, 3]])
        graph = Data(pos=pos, edge_index=edge_index, face=face)
        field = torch.randn((4, 2))

        grad_methods = {
            "finite_differences": lambda: compute_gradient(
                graph, field, method="finite_diff"
            ),
            "weighted_least_squares": lambda: compute_gradient(
                graph, field, method="least_squares"
            ),
            "green_gauss": lambda: compute_gradient(graph, field, method="green_gauss"),
        }

        for name, method in grad_methods.items():
            gradients = method()
            self.assertEqual(gradients.shape, (4, 2, 2))

    def test_compute_gradients_false_2d(self):
        # 4 nodes in a square, for simplicity
        #   (0,1,0) --- (1,1,0)
        #     |     /      |
        #   (0,0,0) --- (1,0,0)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # node 0
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [1.0, 1.0, 0.0],  # node 3
            ],
            dtype=torch.float,
        )

        # Directed edges of a square, one outgoing edge per node
        edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 3, 0, 2, 3]], dtype=torch.long)
        face = torch.tensor([[0, 0], [1, 2], [3, 3]])
        graph = Data(pos=pos, edge_index=edge_index, face=face)
        field = torch.randn((4, 2))

        grad_methods = {
            "finite_differences": lambda: compute_gradient(
                graph, field, method="finite_diff"
            ),
            "weighted_least_squares": lambda: compute_gradient(
                graph, field, method="least_squares"
            ),
            "green_gauss": lambda: compute_gradient(graph, field, method="green_gauss"),
        }

        for name, method in grad_methods.items():
            gradients = method()
            self.assertEqual(gradients.shape, (4, 2, 3))

    def test_compute_gradients_3d(self):
        # 4 nodes building a tetrahedron
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # node 0
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [0.0, 0.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )

        # Directed edges of a square, one outgoing edge per node
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 3, 3], [1, 2, 0, 0, 1, 2]], dtype=torch.long
        )
        face = torch.tensor([[0], [1], [2], [3]])
        graph = Data(pos=pos, edge_index=edge_index, face=face)

        field = torch.randn((4, 3))

        grad_methods = {
            "finite_differences": lambda: compute_gradient(
                graph, field, method="finite_diff"
            ),
            "weighted_least_squares": lambda: compute_gradient(
                graph, field, method="least_squares"
            ),
            "green_gauss": lambda: compute_gradient(graph, field, method="green_gauss"),
        }

        for name, method in grad_methods.items():
            gradients = method()
            self.assertEqual(gradients.shape, (4, 3, 3))


class TestVectorialOperators(unittest.TestCase):
    def setUp(self):

        # Mock 2D graph
        pos_2d = torch.tensor(
            [
                [0.0, 0.0],  # node 0
                [1.0, 0.0],  # node 1
                [0.0, 1.0],  # node 2
                [1.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )
        edge_index_2d = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]], dtype=torch.long)

        self.graph_2d = Data(pos=pos_2d, edge_index=edge_index_2d)
        self.field_2d = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, -2.0],
                [2.0, 3.0],
                [1.0, -4.0],
            ],
            dtype=torch.float,
        )
        self.gradient_2d = torch.randn((4, 2, 2))

        # Mock 3D graph
        pos_3d = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # node 0
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [1.0, 1.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )
        edge_index_3d = torch.tensor(
            [[0, 1, 2, 3, 3, 3], [1, 2, 0, 0, 1, 2]], dtype=torch.long
        )
        face_3d = torch.tensor([[0], [1], [2], [3]])

        self.graph_3d = Data(pos=pos_3d, edge_index=edge_index_3d, face=face_3d)
        self.field_3d = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, -2.0, 1.0],
                [2.0, 3.0, 1.0],
                [1.0, -4.0, 1.0],
            ],
            dtype=torch.float,
        )

    def test_compute_divergence_2d(self):
        divergence = compute_divergence(graph=self.graph_2d, field=self.field_2d)
        self.assertIsNotNone(divergence)
        self.assertEqual(divergence.dim(), 1)
        self.assertEqual(divergence.shape[0], self.field_2d.shape[0])

    def test_compute_divergence_3d(self):
        divergence = compute_divergence(graph=self.graph_3d, field=self.field_3d)
        self.assertIsNotNone(divergence)
        self.assertEqual(divergence.dim(), 1)
        self.assertEqual(divergence.shape[0], self.field_3d.shape[0])

    def test_compute_divergence_w_gradient(self):
        divergence = compute_divergence(
            graph=self.graph_2d, field=self.field_2d, gradient=self.gradient_2d
        )
        self.assertIsNotNone(divergence)
        self.assertEqual(divergence.dim(), 1)
        self.assertEqual(divergence.shape[0], self.field_2d.shape[0])

    def test_compute_vector_gradient_product_2d(self):
        product = compute_vector_gradient_product(
            graph=self.graph_2d, field=self.field_2d
        )
        self.assertIsNotNone(product)
        self.assertEqual(product.shape, self.field_2d.shape)

    def test_compute_vector_gradient_product_3d(self):
        product = compute_vector_gradient_product(
            graph=self.graph_3d, field=self.field_3d
        )
        self.assertIsNotNone(product)
        self.assertEqual(product.shape, self.field_3d.shape)

    def test_compute_vector_gradient_product_w_gradient(self):
        product = compute_vector_gradient_product(
            graph=self.graph_2d, field=self.field_2d, gradient=self.gradient_2d
        )
        self.assertIsNotNone(product)
        self.assertEqual(product.shape, self.field_2d.shape)
