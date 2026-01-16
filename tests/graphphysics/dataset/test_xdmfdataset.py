import unittest
import math
import torch
import meshio
from graphphysics.dataset.preprocessing import (
    Random3DRotate,
    compute_min_distance_to_type,
)
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import MOCK_XDMF_FOLDER, MOCK_H5_META10_SAVE_PATH, MOCK_XDMF_TARGETS
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.utils.torch_graph import torch_graph_to_mesh
from graphphysics.utils.nodetype import NodeType
from graphphysics.external.aneurysm import aneurysm_node_type


class TestXDMFDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        assert len(self.dataset) == 5

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index is None


class TestXDMFDatasetDistance(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder="tests/mock_xdmf_aneurysm",
            meta_path="tests/mock_h5/meta_aneurysm.json",
            targets=["Vitesse"],
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
            "tests/mock_xdmf_aneurysm/AllFields_Resultats_MESH_11.xdmf"
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


class TestXDMFDatasetRotating(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder="tests/mock_xdmf_aneurysm",
            meta_path="tests/mock_h5/meta_aneurysm.json",
            targets=["Vitesse"],
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
            return [0, math.pi / 2, 0]  # alpha, beta, gamma

        # Monkey-patch the _get_random_angles method
        rotate._get_random_angles = fixed_angles.__get__(rotate, Random3DRotate)
        # Apply the transform
        rotated_graph = rotate(graph)

        # assert rotated_graph.num_nodes == 22535
        # assert rotated_graph.edge_index is None

        reader = meshio.xdmf.TimeSeriesReader(
            "tests/mock_xdmf_aneurysm/AllFields_Resultats_MESH_11.xdmf"
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


class TestXDMFDataseTargets(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.y.shape[1] == 2


class TestXDMFDatasetMasking(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
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
            targets=MOCK_XDMF_TARGETS,
            preprocessing=transform,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr.shape == (11070, 4)


class TestXDMFDatasetPreprocessingNoEdgeFeatures(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=transform,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 11070)
        assert graph.edge_attr is None


class TestXDMFDatasetPreprocessingRE(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=transform,
            new_edges_ratio=0.5,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 16604)
        assert graph.edge_attr.shape == (16604, 4)
        assert graph.face is not None


class TestXDMFDatasetPreprocessingKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=transform,
            khop=2,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 32638)
        assert graph.edge_attr.shape == (32638, 4)
        assert graph.face is not None


class TestXDMFDatasetPreprocessingNoEdgeFeaturesRE(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=transform,
            new_edges_ratio=0.5,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1923
        assert graph.edge_index.shape == (2, 16604)
        assert graph.edge_attr is None


class TestXDMFDatasetPreprocessingNoEdgeFeaturesKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
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


class TestXDMFDatasetPartitioning(unittest.TestCase):
    def setUp(self):
        self.transform = build_preprocessing(add_edges_features=False)
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=self.transform,
        )
        self.dataset.trajectory_length += 1

    def test_no_partition_criteria_given(self):
        with self.assertRaises(ValueError):
            _ = XDMFDataset(
                xdmf_folder=MOCK_XDMF_FOLDER,
                meta_path=MOCK_H5_META10_SAVE_PATH,
                targets=MOCK_XDMF_TARGETS,
                use_partitioning=True,
            )

    def test_both_partition_criteria_given(self):
        with self.assertRaises(ValueError):
            _ = XDMFDataset(
                xdmf_folder=MOCK_XDMF_FOLDER,
                meta_path=MOCK_H5_META10_SAVE_PATH,
                targets=MOCK_XDMF_TARGETS,
                use_partitioning=True,
                num_partitions=4,
                max_nodes_per_partition=500,
            )

    def test_partitioning_with_num_partitions(self):
        partitioned_dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=self.transform,
            use_partitioning=True,
            num_partitions=4,
        )

        self.assertEqual(partitioned_dataset.size_dataset, self.dataset.size_dataset)
        self.assertEqual(
            len(partitioned_dataset), partitioned_dataset.num_partitions * len(self.dataset)
        )

        subgraph = partitioned_dataset[0]
        graph = self.dataset[0]
        self.assertLess(subgraph.num_nodes, graph.num_nodes)
        self.assertEqual(subgraph.x.shape[1], graph.x.shape[1])

    def test_partitioning_with_num_nodes(self):
        partitioned_dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
            preprocessing=self.transform,
            use_partitioning=True,
            max_nodes_per_partition=800,
        )

        self.assertEqual(partitioned_dataset.size_dataset, self.dataset.size_dataset)

        subgraph = partitioned_dataset[0]
        graph = self.dataset[0]

        self.assertEqual(
            len(partitioned_dataset),  math.ceil(graph.num_nodes / partitioned_dataset.max_nodes_per_partition) * len(self.dataset)
        )
        self.assertLess(subgraph.num_nodes, graph.num_nodes)
        self.assertEqual(subgraph.x.shape[1], graph.x.shape[1])
