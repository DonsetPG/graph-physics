from torch_geometric.loader import DataLoader
import unittest
import pytest

from graphphysics.dataset.h5_dataset import H5Dataset
from tests.mock import (
    MOCK_H5_META_SAVE_PATH,
    MOCK_H5_SAVE_PATH,
)
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from tests.mock import (
    MOCK_XDMF_FOLDER,
    MOCK_H5_META10_SAVE_PATH,
)
from graphphysics.dataset.preprocessing import build_preprocessing

ITEMS_TO_FETCH = 20
BATCH_SIZE = 4
NUM_WORKERS = 2
PREFETCH_FACTOR = 1
PERSISTENT_WORKERS = True


def get_items_h5():
    transform = build_preprocessing(add_edges_features=True)
    dataset = H5Dataset(
        h5_path=MOCK_H5_SAVE_PATH,
        meta_path=MOCK_H5_META_SAVE_PATH,
        preprocessing=transform,
    )
    dataset.trajectory_length += 1
    for _ in range(ITEMS_TO_FETCH):
        dataset[0]


def get_items_xdmf():
    transform = build_preprocessing(add_edges_features=True)
    dataset = XDMFDataset(
        xdmf_folder=MOCK_XDMF_FOLDER,
        meta_path=MOCK_H5_META10_SAVE_PATH,
        preprocessing=transform,
    )
    dataset.trajectory_length += 1
    for _ in range(ITEMS_TO_FETCH):
        dataset[0]


def get_items_h5dataloader():
    transform = build_preprocessing(add_edges_features=True)
    dataset = H5Dataset(
        h5_path=MOCK_H5_SAVE_PATH,
        meta_path=MOCK_H5_META_SAVE_PATH,
        preprocessing=transform,
    )
    dataset.trajectory_length += 1
    dataloader = DataLoader(
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


def get_items_xdmfdataloader():
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
    for batch_g in dataloader:
        indx += 1
        if indx > ITEMS_TO_FETCH / BATCH_SIZE:
            break


def test_get_h5items(benchmark):
    benchmark(get_items_h5)


def test_get_xdmfitems(benchmark):
    benchmark(get_items_xdmf)


def test_get_dataloader_h5items(benchmark):
    benchmark(get_items_h5dataloader)


def test_get_dataloader_xdmfitems(benchmark):
    benchmark(get_items_xdmfdataloader)
