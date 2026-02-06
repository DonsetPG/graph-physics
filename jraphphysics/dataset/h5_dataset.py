from __future__ import annotations

from typing import Callable, Optional

import h5py
import numpy as np

from jraphphysics.dataset.dataset import BaseDataset
from jraphphysics.utils.hierarchical import get_frame_as_mesh, get_traj_as_meshes


class H5Dataset(BaseDataset):
    def __init__(
        self,
        h5_path: str,
        meta_path: str,
        targets: list[str] | None = None,
        preprocessing: Optional[Callable] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        add_edge_features: bool = False,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
    ):
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
        )
        if switch_to_val:
            h5_path = h5_path.replace("train", "test")

        self.h5_path = h5_path
        self.meta_path = meta_path
        self.dt = self.meta.get("dt", 1) or 1

        with h5py.File(self.h5_path, "r") as file_handle:
            self.datasets_index = list(file_handle.keys())
        self._size_dataset = len(self.datasets_index)

    @property
    def size_dataset(self) -> int:
        return self._size_dataset

    def __getitem__(self, index: int):
        traj_index, frame = self.get_traj_frame(index=index)
        traj_number = self.datasets_index[traj_index]

        with h5py.File(self.h5_path, "r") as file_handle:
            traj = get_traj_as_meshes(file_handle, traj_number, self.meta)

        points, cells, point_data, target_data, _ = get_frame_as_mesh(
            traj=traj,
            frame=frame,
            targets=self.targets,
            frame_target=frame + 1,
        )

        points = np.asarray(points, dtype=np.float32)
        cells = np.asarray(cells, dtype=np.int32)

        def _format_data(values: dict[str, np.ndarray] | None, use_targets: bool = False):
            if values is None:
                return None
            out = {}
            for key, value in values.items():
                if use_targets and key not in self.targets:
                    continue
                if key not in self.meta["features"]:
                    continue
                arr = np.asarray(value).astype(self.meta["features"][key]["dtype"])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                out[key] = arr
            return out

        point_data = _format_data(point_data, use_targets=False)
        target_data = _format_data(target_data, use_targets=True)

        item = {
            "points": points,
            "cells": cells,
            "point_data": point_data,
            "time": frame * self.dt,
            "target_data": target_data,
            "traj_index": traj_index,
        }

        if self.use_previous_data:
            _, _, previous_data, _, _ = get_frame_as_mesh(
                traj=traj,
                frame=frame - 1,
                targets=self.targets,
                frame_target=None,
            )
            dynamic_previous = {}
            for key, value in previous_data.items():
                feature_meta = self.meta["features"].get(key)
                if feature_meta is None:
                    continue
                if feature_meta.get("type") != "dynamic":
                    continue
                arr = np.asarray(value).astype(feature_meta["dtype"])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                dynamic_previous[key] = arr
            item["previous_data"] = dynamic_previous

        return item
