from jraphphysics.dataset.h5_dataset import H5Dataset
from tests.mock import MOCK_H5_META_SAVE_PATH, MOCK_H5_SAVE_PATH, MOCK_H5_TARGETS


def test_h5_dataset_len_and_getitem():
    dataset = H5Dataset(
        h5_path=MOCK_H5_SAVE_PATH,
        meta_path=MOCK_H5_META_SAVE_PATH,
        targets=MOCK_H5_TARGETS,
        use_previous_data=False,
    )
    dataset.trajectory_length += 1

    assert len(dataset) == 1
    item = dataset[0]
    assert "points" in item
    assert "cells" in item
    assert "point_data" in item
    assert "target_data" in item
    assert item["points"].shape[0] > 0


def test_h5_dataset_previous_data():
    dataset = H5Dataset(
        h5_path=MOCK_H5_SAVE_PATH,
        meta_path=MOCK_H5_META_SAVE_PATH,
        targets=MOCK_H5_TARGETS,
        use_previous_data=True,
    )
    dataset.trajectory_length += 1

    item = dataset[0]
    assert "previous_data" in item
