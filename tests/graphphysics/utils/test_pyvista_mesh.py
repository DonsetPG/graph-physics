import unittest
import numpy as np
import torch
from torch_geometric.data import Data

from graphphysics.utils.pyvista_mesh import convert_to_pyvista_mesh


class TestConvertToPyVistaMesh(unittest.TestCase):
    def test_2d_positions(self):
        """Test conversion with 2D positions."""
        pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh points have shape (N, 3)
        self.assertEqual(mesh.points.shape, (2, 3))
        # Check that third coordinate is zero
        np.testing.assert_array_equal(mesh.points[:, 2], np.zeros(2))

        # Check that the number of lines matches the number of edges
        self.assertEqual(mesh.n_lines, edge_index.size(1))

    def test_3d_positions(self):
        """Test conversion with 3D positions."""
        pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh points match the input positions
        np.testing.assert_array_equal(mesh.points, pos.numpy())

    def test_1d_positions(self):
        """Test conversion with 1D positions."""
        pos = torch.tensor([[1.0], [2.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh points have shape (N, 3)
        self.assertEqual(mesh.points.shape, (2, 3))
        # Check that second and third coordinates are zero
        np.testing.assert_array_equal(mesh.points[:, 1:], np.zeros((2, 2)))

    def test_add_all_data(self):
        """Test adding all node features to mesh point data when x is 2D."""
        pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.tensor([[10, 20], [30, 40]])
        graph = Data(pos=pos, edge_index=edge_index, x=x)
        mesh = convert_to_pyvista_mesh(graph, add_all_data=True)

        # Check that point data has the correct keys
        self.assertIn("x0", mesh.point_data)
        self.assertIn("x1", mesh.point_data)
        # Check that data matches the graph's x
        np.testing.assert_array_equal(mesh.point_data["x0"], x[:, 0].numpy())
        np.testing.assert_array_equal(mesh.point_data["x1"], x[:, 1].numpy())

    def test_missing_pos(self):
        """Test error handling when graph.pos is missing."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(edge_index=edge_index)
        with self.assertRaises(ValueError) as context:
            convert_to_pyvista_mesh(graph)
        self.assertIn(
            "Graph must have 'pos' attribute with node positions.",
            str(context.exception),
        )

    def test_missing_edge_index(self):
        """Test error handling when graph.edge_index is missing."""
        pos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        graph = Data(pos=pos)
        with self.assertRaises(ValueError) as context:
            convert_to_pyvista_mesh(graph)
        self.assertIn(
            "Graph must have 'edge_index' attribute with edge indices.",
            str(context.exception),
        )

    def test_unsupported_vertex_dimension(self):
        """Test error handling when pos has more than 3 dimensions."""
        pos = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        with self.assertRaises(ValueError) as context:
            convert_to_pyvista_mesh(graph)
        self.assertIn("Unsupported vertex dimension: 4", str(context.exception))

    def test_empty_graph(self):
        """Test conversion with an empty graph."""
        pos = torch.empty((0, 2))
        edge_index = torch.empty((2, 0), dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh has no points or lines
        self.assertEqual(mesh.n_points, 0)
        self.assertEqual(mesh.n_lines, 0)

    def test_non_standard_dtype(self):
        """Test conversion with non-standard data types."""
        pos = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh points are floats
        self.assertEqual(mesh.points.dtype, np.float32)

    def test_large_graph(self):
        """Test conversion with a large graph."""
        num_nodes = 1000
        pos = torch.randn((num_nodes, 2))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
        graph = Data(pos=pos, edge_index=edge_index)
        mesh = convert_to_pyvista_mesh(graph)

        # Check that mesh points have the correct shape
        self.assertEqual(mesh.points.shape, (num_nodes, 3))
        self.assertEqual(mesh.n_points, num_nodes)
        self.assertEqual(mesh.n_lines, edge_index.size(1))


if __name__ == "__main__":
    unittest.main()
