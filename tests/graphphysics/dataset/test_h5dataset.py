import unittest

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

from named_features import NamedData, make_x_layout

from graphphysics.dataset.h5_dataset import H5Dataset
from graphphysics.dataset.preprocessing import build_preprocessing
from tests.mock import MOCK_H5_META_SAVE_PATH, MOCK_H5_SAVE_PATH, MOCK_H5_TARGETS


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

    def test_get_with_layout_returns_named_data(self):
        layout = make_x_layout(
            ["velocity", "pressure", "node_type", "time"],
            {"velocity": 2, "pressure": 1, "node_type": 1, "time": 1},
        )
        dataset = H5Dataset(
            h5_path=MOCK_H5_SAVE_PATH,
            meta_path=MOCK_H5_META_SAVE_PATH,
            targets=MOCK_H5_TARGETS,
            x_layout=layout,
        )
        dataset.trajectory_length += 1
        graph = dataset[0]
        assert isinstance(graph, NamedData)
        assert graph.x_layout is layout
        assert graph.x.shape[1] == sum(layout.sizes().values())


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
