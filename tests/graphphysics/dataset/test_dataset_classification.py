import unittest

from graphphysics.dataset.dataset_classification import GraphClassificationDataset
from tests.mock import MOCK_CLASSIFICATION_META_SAVE_PATH, MOCK_CLASSIFICATION_SAVE_PATH
from graphphysics.dataset.preprocessing import build_preprocessing
from torch_geometric.data import DataLoader


class TestGraphClassificationDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = GraphClassificationDataset(
            root_folder=MOCK_CLASSIFICATION_SAVE_PATH,
            meta_path=MOCK_CLASSIFICATION_META_SAVE_PATH,
        )

    def test_length(self):
        assert len(self.dataset) == 2

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1024
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0
        assert graph.y.shape == (2,)


class TestGraphClassificationDatasetMasking(unittest.TestCase):
    def setUp(self):
        self.dataset = GraphClassificationDataset(
            root_folder=MOCK_CLASSIFICATION_SAVE_PATH,
            meta_path=MOCK_CLASSIFICATION_META_SAVE_PATH,
            masking_ratio=0.4,
        )

    def test_length(self):
        assert len(self.dataset) == 2

    def test_get(self):
        graph, selected_index = self.dataset[0]
        assert graph.num_nodes == 1024
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0
        assert selected_index is not None
        assert len(selected_index) == int((1 - 0.4) * 1024)


class TestGraphClassificationDatasetPreprocessing(unittest.TestCase):
    def setUp(self):
        transform = build_preprocessing(add_edges_features=False)
        self.dataset = GraphClassificationDataset(
            root_folder=MOCK_CLASSIFICATION_SAVE_PATH,
            meta_path=MOCK_CLASSIFICATION_META_SAVE_PATH,
            preprocessing=transform,
        )

    def test_get(self):
        graph = self.dataset[0]
        assert graph.num_nodes == 1024
        assert graph.edge_index.shape == (2, 5608)
        assert graph.face is None


if __name__ == "__main__":
    unittest.main()
