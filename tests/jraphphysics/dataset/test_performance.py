from torch.utils import data

from jraphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import MOCK_XDMF_FOLDER, MOCK_H5_META10_SAVE_PATH, MOCK_XDMF_TARGETS
from jraphphysics.utils.jax_graph import batched_meshdata_to_graph

ITEMS_TO_FETCH = 20
BATCH_SIZE = 4
NUM_WORKERS = 2
PREFETCH_FACTOR = 1
PERSISTENT_WORKERS = True


def get_items_xdmf():
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        targets=MOCK_XDMF_TARGETS,
    )
    dataset.trajectory_length += 1
    for _ in range(ITEMS_TO_FETCH):
        dataset[0]


def get_items_xdmfdataloader():
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        targets=MOCK_XDMF_TARGETS,
    )
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    indx = 0
    for batch_g in dataloader:
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break


def get_items_xdmfdataloader_jraph():
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        targets=MOCK_XDMF_TARGETS,
    )
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    indx = 0
    for batch_g in dataloader:
        batched_jraph = batched_meshdata_to_graph(
            points=batch_g["points"].numpy(),
            cells=batch_g["cells"].numpy(),
            point_data=batch_g["point_data"],
        )
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break


def test_get_xdmfitems(benchmark):
    benchmark(get_items_xdmf)


def test_get_dataloader_xdmfitems(benchmark):
    benchmark(get_items_xdmfdataloader)


def test_get_dataloader_xdmfitems_jraph(benchmark):
    benchmark(get_items_xdmfdataloader_jraph)
