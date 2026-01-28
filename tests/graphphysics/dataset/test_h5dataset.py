import unittest
import math

from graphphysics.dataset.h5_dataset import H5Dataset
from tests.mock import MOCK_H5_META_SAVE_PATH, MOCK_H5_SAVE_PATH, MOCK_H5_TARGETS
from graphphysics.dataset.preprocessing import build_preprocessing


class TestH5Dataset(unittest.TestCase):
    def setUp(self):
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        assert len(self.dataset) == 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index is None


class TestH5DatasetMasking(unittest.TestCase):
    def setUp(self):
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            masking_ratio=0.4,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        assert len(self.dataset) == 1

    def test_get(self):
        graph, selected_index = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index is None
        assert selected_index is not None
        assert len(selected_index) == int((1 - 0.4) * 1876)


class TestH5DatasetPreprocessing(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 10788)
        assert graph.edge_attr.shape == (10788, 3)


class TestH5DatasetPreprocessingNoEdgeFeatures(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 10788)
        assert graph.edge_attr is None


class TestH5DatasetPreprocessingRE(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
            new_edges_ratio=0.5,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 16182)
        assert graph.edge_attr.shape == (16182, 3)
        assert graph.face is not None


class TestH5DatasetPreprocessingKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
            khop=2,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 31746)
        assert graph.edge_attr.shape == (31746, 3)
        assert graph.face is not None


class TestH5DatasetPreprocessingNoEdgeFeaturesRE(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=True)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
            new_edges_ratio=0.5,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 16182)
        assert graph.edge_attr is None


class TestH5DatasetPreprocessingNoEdgeFeaturesKHOP(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=transform,
            khop=2,
            add_edge_features=False,
        )
        self.dataset.trajectory_length += 1

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1876
        assert graph.edge_index.shape == (2, 31746)
        assert graph.edge_attr is None


class TestXDMFDatasetPartitioning(unittest.TestCase):
    def setUp(self):
        self.transform = build_preprocessing(add_edges_features=False)
        self.dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=self.transform,
        )
        self.dataset.trajectory_length += 1

    def test_no_partition_criteria_given(self):
        with self.assertRaises(ValueError):
            _ = H5Dataset(
                h5_path=MOCK_H5_SAVE_PATH,
                meta_path=MOCK_H5_META_SAVE_PATH,
                targets=MOCK_H5_TARGETS,
                use_partitioning=True,
            )

    def test_both_partition_criteria_given(self):
        with self.assertRaises(ValueError):
            _ = H5Dataset(
                h5_path=MOCK_H5_SAVE_PATH,
                meta_path=MOCK_H5_META_SAVE_PATH,
                targets=MOCK_H5_TARGETS,
                use_partitioning=True,
                num_partitions=4,
                max_nodes_per_partition=500,
            )

    def test_partitioning_with_num_partitions(self):
        partitioned_dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
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
        partitioned_dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            preprocessing=self.transform,
            use_partitioning=True,
            max_nodes_per_partition=800,
        )

        self.assertEqual(partitioned_dataset.size_dataset, self.dataset.size_dataset)

        subgraph = partitioned_dataset[0]
        graph = self.dataset[0]

        self.assertEqual(
            len(partitioned_dataset),
            math.ceil(graph.num_nodes / partitioned_dataset.max_nodes_per_partition) * len(self.dataset)
        )
        self.assertLess(subgraph.num_nodes, graph.num_nodes)
        self.assertEqual(subgraph.x.shape[1], graph.x.shape[1])
