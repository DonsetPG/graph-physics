import unittest
import meshio
import numpy as np
import os
import shutil
import torch
from torch_geometric.data import Data

from graphphysics.utils.meshio_mesh import convert_to_meshio_vtu, vtu_to_xdmf
from tests.mock import MOCK_VTU_FOLDER_PATH, MOCK_VTU_ANEURYSM_FOLDER_PATH


class TestConvertToMeshioVtu(unittest.TestCase):
    def setUp(self):
        self.pos_2d = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.face_2d = torch.tensor([[0], [1], [2]])
        self.pos_3d = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.face_3d = torch.tensor([[0, 0, 0, 0], [1, 1, 2, 2], [2, 3, 3, 3]])
        self.tetra = torch.tensor([[0], [1], [2], [3]])

    def test_2d_graph(self):
        """Test conversion for a 2D graph."""
        graph = Data(pos=self.pos_2d, face=self.face_2d)
        mesh = convert_to_meshio_vtu(graph)

        # Check that mesh points have shape (N, 3)
        self.assertEqual(mesh.points.shape, (3, 3))
        # Check that third coordinate is zero
        np.testing.assert_array_equal(mesh.points[:, 2], np.zeros(3))
        # Check that the number of faces matches the number of cells
        self.assertEqual(mesh.cells_dict["triangle"].shape[0], self.face_2d.size(1))

    def test_3d_graph(self):
        """Test conversion for a 3D graph."""
        graph = Data(pos=self.pos_3d, face=self.face_3d, tetra=self.tetra)
        mesh = convert_to_meshio_vtu(graph)

        # Check that mesh points have shape (N, 3)
        self.assertEqual(mesh.points.shape, (4, 3))

        # Check that the number of tetrahedras matches the number of cells
        self.assertEqual(mesh.cells_dict["tetra"].shape[0], self.face_2d.size(1))

    def test_add_all_data(self):
        """Test adding all node features to mesh point data when x is 2D."""
        x = torch.tensor([[10, 20], [30, 40], [50, 60]])
        graph = Data(pos=self.pos_2d, face=self.face_2d, x=x)
        mesh = convert_to_meshio_vtu(graph, add_all_data=True)

        # Check that point data has the correct keys
        self.assertIn("x0", mesh.point_data)
        self.assertIn("x1", mesh.point_data)
        # Check that data matches the graph's x
        np.testing.assert_array_equal(mesh.point_data["x0"], x[:, 0].numpy())
        np.testing.assert_array_equal(mesh.point_data["x1"], x[:, 1].numpy())

    def test_missing_pos(self):
        """Test error handling when graph.pos is missing."""
        graph = Data(face=self.face_2d)
        with self.assertRaises(ValueError) as context:
            convert_to_meshio_vtu(graph)
        self.assertIn(
            "Graph must have 'pos' attribute with node positions.",
            str(context.exception),
        )

    def test_missing_face(self):
        """Test error handling when graph.face is missing."""
        graph = Data(pos=self.pos_2d)
        with self.assertRaises(ValueError) as context:
            convert_to_meshio_vtu(graph)
        self.assertIn(
            "Graph must have 'face' attribute with face indices.",
            str(context.exception),
        )

    def test_unsupported_vertex_dimension(self):
        """Test error handling when pos has more than 3 dimensions."""
        pos = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        graph = Data(pos=pos, face=self.face_3d)
        with self.assertRaises(ValueError) as context:
            convert_to_meshio_vtu(graph)
        self.assertIn("Unsupported vertex dimension: 4", str(context.exception))

    def test_empty_graph(self):
        """Test conversion with an empty graph."""
        pos = torch.empty((0, 2))
        face = torch.empty((2, 0), dtype=torch.long)
        graph = Data(pos=pos, face=face)
        mesh = convert_to_meshio_vtu(graph)

        # Check that mesh has no points
        self.assertEqual(len(mesh.points), 0)

    def test_large_graph(self):
        """Test conversion with a large graph."""
        num_nodes = 1000
        pos = torch.randn((num_nodes, 2))
        face = torch.randint(0, num_nodes, (3, num_nodes), dtype=torch.long)

        graph = Data(pos=pos, face=face)
        mesh = convert_to_meshio_vtu(graph)

        # Check that mesh points and cells have the correct shape
        self.assertEqual(mesh.points.shape, (num_nodes, 3))
        self.assertEqual(mesh.cells_dict["triangle"].shape, (face.size(1), 3))


class TestVtuToXdmf(unittest.TestCase):
    def setUp(self):
        self.files_2d = [
            os.path.join(MOCK_VTU_FOLDER_PATH, f)
            for f in os.listdir(MOCK_VTU_FOLDER_PATH)
        ]
        self.files_3d = [
            os.path.join(MOCK_VTU_ANEURYSM_FOLDER_PATH, f)
            for f in os.listdir(MOCK_VTU_ANEURYSM_FOLDER_PATH)
        ]

        self.tmp_dir = "tests/mock_vtu_tmp"
        self.filename = os.path.join(self.tmp_dir, "test_xdmf_compression")
        shutil.copytree(MOCK_VTU_FOLDER_PATH, self.tmp_dir)
        self.tmp_files = [
            os.path.join(self.tmp_dir, f) for f in os.listdir(self.tmp_dir)
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_2d_vtus(self):
        """Test 2D vtus compression"""
        vtu_to_xdmf(self.filename, self.files_2d, remove_vtus=False)

        # Check that both archive files exist
        self.assertTrue(os.path.exists(f"{self.filename}.h5"))
        self.assertTrue(os.path.exists(f"{self.filename}.xdmf"))

        # Check the mesh structure
        vtu_meshes = [meshio.read(f) for f in self.files_2d]
        reader = meshio.xdmf.TimeSeriesReader(f"{self.filename}.xdmf")
        points, cells = reader.read_points_cells()
        self.assertEqual(len(points), len(vtu_meshes[0].points))
        self.assertEqual(reader.num_steps, len(vtu_meshes))

        # Compare xdmf and vtu data
        for i in range(reader.num_steps):
            time, point_data, cell_data = reader.read_data(i)
            self.assertEqual(point_data.keys(), vtu_meshes[i].point_data.keys())
            for key in point_data.keys():
                self.assertTrue(
                    np.array_equal(point_data[key], vtu_meshes[i].point_data[key])
                )

        os.remove(f"{self.filename}.h5")
        os.remove(f"{self.filename}.xdmf")

    def test_3d_vtus(self):
        """Test 3D vtus compression"""

        vtu_to_xdmf(self.filename, self.files_3d, remove_vtus=False)

        # Check that both archive files exist
        self.assertTrue(os.path.exists(f"{self.filename}.h5"))
        self.assertTrue(os.path.exists(f"{self.filename}.xdmf"))

        # Check the mesh structure
        vtu_meshes = [meshio.read(f) for f in self.files_3d]
        reader = meshio.xdmf.TimeSeriesReader(f"{self.filename}.xdmf")
        points, cells = reader.read_points_cells()
        self.assertEqual(len(points), len(vtu_meshes[0].points))
        self.assertEqual(reader.num_steps, len(vtu_meshes))

        # Compare xdmf and vtu data
        for i in range(reader.num_steps):
            time, point_data, cell_data = reader.read_data(i)
            self.assertEqual(point_data.keys(), vtu_meshes[i].point_data.keys())
            for key in point_data.keys():
                self.assertTrue(
                    np.array_equal(point_data[key], vtu_meshes[i].point_data[key])
                )

        os.remove(f"{self.filename}.h5")
        os.remove(f"{self.filename}.xdmf")

    def test_remove_vtus(self):
        """Test the VTUs removal after compression."""

        vtu_to_xdmf(self.filename, self.tmp_files, remove_vtus=True)

        # Check that all vtu files were removed.
        for file in self.tmp_files:
            self.assertFalse(os.path.exists(file))
        os.remove(f"{self.filename}.h5")
        os.remove(f"{self.filename}.xdmf")


if __name__ == "__main__":
    unittest.main()
