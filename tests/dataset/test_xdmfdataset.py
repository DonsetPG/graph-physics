import unittest
import math
import torch
import meshio
from graphphysics.dataset.preprocessing import (
    Random3DRotate,
    compute_min_distance_to_type,
)
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.utils.torch_graph import torch_graph_to_mesh
from graphphysics.utils.nodetype import NodeType
from graphphysics.external.aneurysm import aneurysm_node_type


class TestH5Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        assert len(self.dataset) == 5

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index is None


class TestH5DatasetDistance(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder="tests/mock_vtu_aneurysm",
            meta_path="tests/mock_h5/meta_aneurysm.json",
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 22535
        assert graph.edge_index is None

        node_type = aneurysm_node_type(graph)

        graph.x = torch.cat(
            (
                graph.x,
                node_type.unsqueeze(1),
            ),
            dim=1,
        )

        min_distance = compute_min_distance_to_type(
            graph=graph, target_type=NodeType.INFLOW, node_types=node_type
        )

        graph.x = torch.cat(
            (
                graph.x,
                min_distance.unsqueeze(1),
            ),
            dim=1,
        )

        assert graph.num_nodes == 22535
        assert graph.edge_index is None

        reader = meshio.xdmf.TimeSeriesReader(
            "tests/mock_vtu_aneurysm/AllFields_Resultats_MESH_11.xdmf"
        )
        points, cells = reader.read_points_cells()
        time, point_data, _ = reader.read_data(0)

        init_face = meshio.Mesh(points, cells, point_data=point_data).cells_dict[
            "tetra"
        ]
        graph.face = torch.Tensor(init_face)

        mesh = torch_graph_to_mesh(
            graph,
            node_features_mapping={
                "velocity_x": 0,
                "velocity_y": 1,
                "velocity_z": 2,
                "node_type": -2,
                "distance_to_inflow": -1,
            },
        )
        mesh.write("test_distance.vtu")


class TestH5DatasetRotating(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder="tests/mock_vtu_aneurysm",
            meta_path="tests/mock_h5/meta_aneurysm.json",
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 22535
        assert graph.edge_index is None
        feature_indices = [(0, 3)]  # Rotate x[:, 0:3]
        # Create the Random3DRotate instance
        rotate = Random3DRotate(feature_indices=feature_indices)

        def fixed_angles(self):
            return [math.pi / 2, 0.0, 0.0]  # alpha, beta, gamma

        # Monkey-patch the _get_random_angles method
        rotate._get_random_angles = fixed_angles.__get__(rotate, Random3DRotate)
        # Apply the transform
        rotated_graph = rotate(graph)

        assert rotated_graph.num_nodes == 22535
        assert rotated_graph.edge_index is None

        reader = meshio.xdmf.TimeSeriesReader(
            "tests/mock_vtu_aneurysm/AllFields_Resultats_MESH_11.xdmf"
        )
        points, cells = reader.read_points_cells()
        time, point_data, _ = reader.read_data(0)

        init_face = meshio.Mesh(points, cells, point_data=point_data).cells_dict[
            "tetra"
        ]
        rotated_graph.face = torch.Tensor(init_face)

        mesh = torch_graph_to_mesh(
            rotated_graph,
            node_features_mapping={"velocity_x": 0, "velocity_y": 1, "velocity_z": 2},
        )
        mesh.write("test_rotate.vtu")


class TestH5DatasetMasking(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            masking_ratio=0.4,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph, selected_index = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index is None
        assert selected_index is not None
        assert len(selected_index) == int((1 - 0.4) * 1923)


class TestH5DatasetPreprocessing(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr.shape == (11070, 4)
        assert graph.face is None


class TestH5DatasetPreprocessingNoEdgeFeatures(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr is None
        assert graph.face is None


class TestH5DatasetPreprocessingKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            khop=2,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 32638)
        assert graph.edge_attr.shape == (32638, 4)
        assert graph.face is None


class TestH5DatasetPreprocessingNoEdgeFeaturesKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            preprocessing=transform,
            khop=2,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 32638)
        assert graph.edge_attr is None
        assert graph.face is None
