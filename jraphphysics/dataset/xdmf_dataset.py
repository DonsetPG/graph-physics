import os
from typing import Callable, List, Optional, Tuple

import meshio
import numpy as np
import jraph

from jraphphysics.dataset.dataset import BaseDataset


class XDMFDataset(BaseDataset):
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[
            Callable[[jraph.GraphsTuple], jraph.GraphsTuple]
        ] = None,
        khop: int = 1,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
    ):
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            khop=khop,
            use_previous_data=use_previous_data,
        )
        if switch_to_val:
            xdmf_folder = xdmf_folder.replace("train", "test")
        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._size_dataset: int = len(self.file_paths)

    @property
    def size_dataset(self) -> int:
        return self._size_dataset

    def __getitem__(self, index: int) -> Tuple[jraph.GraphsTuple, jraph.GraphsTuple]:
        traj_index, frame = self.get_traj_frame(index=index)

        xdmf_file = self.file_paths[traj_index]
        reader = meshio.xdmf.TimeSeriesReader(xdmf_file)

        num_steps = reader.num_steps
        if frame >= num_steps - 1:
            raise IndexError(
                f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
            )

        points, cells = reader.read_points_cells()
        time, point_data, _ = reader.read_data(frame)
        _, target_point_data, _ = reader.read_data(frame + 1)

        # Prepare the mesh data
        mesh = meshio.Mesh(points, cells, point_data=point_data)

        # Get faces or cells
        if "triangle" in mesh.cells_dict:
            faces = mesh.cells_dict["triangle"]
        elif "tetra" in mesh.cells_dict:
            face = mesh.cells_dict["tetra"].T
            faces = np.concatenate(
                [
                    face[0:3],
                    face[1:4],
                    np.stack([face[2], face[3], face[0]], axis=0),
                    np.stack([face[3], face[0], face[1]], axis=0),
                ],
                axis=1,
            ).T
        else:
            raise ValueError(
                "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
            )

        # Process point data and target data
        point_data = {
            k: np.array(v).astype(self.meta["features"][k]["dtype"])
            for k, v in point_data.items()
            if k in self.meta["features"]
        }

        target_data = {
            k: np.array(v).astype(self.meta["features"][k]["dtype"])
            for k, v in target_point_data.items()
            if k in self.targets
        }

        def _reshape_array(a: dict):
            for k, v in a.items():
                if v.ndim == 1:
                    a[k] = v.reshape(-1, 1)

        _reshape_array(point_data)
        _reshape_array(target_data)

        inputs_graph = {
            "points": points.astype(np.float32),
            "cells": faces,
            "point_data": point_data,
            "time": time,
            "target_data": target_data,
            "traj_index": traj_index,
        }

        if self.use_previous_data:
            _, previous_data, _ = reader.read_data(frame - 1)
            previous = {
                k: np.array(v).astype(self.meta["features"][k]["dtype"])
                for k, v in previous_data.items()
                if k in self.meta["features"]
                and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_array(previous)
            inputs_graph["previous_data"] = previous

        return inputs_graph
