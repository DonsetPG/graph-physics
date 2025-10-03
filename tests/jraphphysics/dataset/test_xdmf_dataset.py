import unittest
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from torch.utils import data

from jraphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import MOCK_XDMF_FOLDER, MOCK_H5_META10_SAVE_PATH, MOCK_XDMF_TARGETS
from jraphphysics.utils.jax_graph import batched_meshdata_to_graph


class TestXDMFDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
        )
        self.dataset.trajectory_length += 1

    def test_length(self):
        self.assertEqual(len(self.dataset), 5)

    def test_get(self):
        input_graph = self.dataset[0]
        # Now you can perform assertions on input_graph and target_graph
        num_nodes = input_graph["points"].shape[0]
        self.assertGreater(num_nodes, 0)


class TestXDMFDataloader(unittest.TestCase):
    def setUp(self):
        self.dataset = XDMFDataset(
            xdmf_folder=MOCK_XDMF_FOLDER,
            meta_path=MOCK_H5_META10_SAVE_PATH,
            targets=MOCK_XDMF_TARGETS,
        )
        self.dataset.trajectory_length += 1
        self.dataloader = data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=4,
            num_workers=0,
        )

    def test_get(self):
        for batch_g in self.dataloader:
            batched_jraph = batched_meshdata_to_graph(
                points=batch_g["points"].numpy(),
                cells=batch_g["cells"].numpy(),
                point_data=batch_g["point_data"],
            )

            self.assertEqual(len(batched_jraph.n_node), 4)
            break
