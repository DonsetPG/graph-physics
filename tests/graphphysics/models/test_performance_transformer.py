from torch_geometric.loader import DataLoader
import unittest
import pytest

from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.models.layers import (
    Transformer,
)

ITEMS_TO_FETCH = 20
BATCH_SIZE = 4
NUM_WORKERS = 2
PREFETCH_FACTOR = 1
PERSISTENT_WORKERS = True

@pytest.mark.skip(reason="a bit long atm")
def process_items_xdmfdataloader():
    transform = build_preprocessing(add_edges_features=True)
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        preprocessing=transform,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    indx = 0
    input_dim = 3
    output_dim = 3
    num_heads = 1
    transformer = Transformer(input_dim, output_dim, num_heads)
    for batch_g in dataloader:
        transformer(batch_g.x, None)
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break

@pytest.mark.skip(reason="a bit long atm")
def test_process_items_xdmfdataloader(benchmark):
    benchmark(process_items_xdmfdataloader)
