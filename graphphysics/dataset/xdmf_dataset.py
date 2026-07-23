import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import meshio
import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data
from meshio._exceptions import ReadError

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.utils.torch_graph import meshdata_to_graph


class XDMFDataset(BaseDataset):
    DEFAULT_ALL_GRIDS = ("billet", "lower_die", "upper_die")

    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        targets: list[str] = None,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        previous_data_count: int = 1,
        use_partitioning: bool = False,
        num_partitions: Optional[int] = None,
        max_nodes_per_partition: Optional[int] = None,
    ):
        super().__init__(
            meta_path=meta_path,
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            previous_data_count=previous_data_count,
            use_partitioning=use_partitioning,
            num_partitions=num_partitions,
            max_nodes_per_partition=max_nodes_per_partition,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = "xdmf"

        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "The dataset has a timestep set to 0. Fallback to dt=1 to ensure xdmf can be saved."
            )

        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path
        self.xdmf_grid_names = self._resolve_xdmf_grid_names()
        self.xdmf_grid_name = ",".join(self.xdmf_grid_names)
        self._xdmf_sources: Dict[str, dict] = {}

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._size_dataset = len(self.file_paths)
        self._build_index_map()

    def _build_index_map(self):
        for traj_index, file_path in enumerate(self.file_paths):
            source = self._get_xdmf_source(file_path)
            if source["type"] == "meshio":
                with meshio.xdmf.TimeSeriesReader(file_path) as reader:
                    points, _ = reader.read_points_cells()
                    num_nodes = len(points)
                    trajectory_length = reader.num_steps
            else:
                first_frame = source["frames"][0]
                with h5py.File(source["h5_path"], "r") as h5_file:
                    num_nodes = sum(
                        h5_file[grid["points_path"]].shape[0]
                        for grid in first_frame["grids"]
                    )
                trajectory_length = len(source["frames"])
            self._add_traj_to_index_map(traj_index, num_nodes, trajectory_length)

    def _resolve_xdmf_grid_names(self) -> Tuple[str, ...]:
        configured_grids = self.meta.get(
            "xdmf_grids", self.meta.get("xdmf_grid", "billet")
        )
        if isinstance(configured_grids, str):
            if configured_grids.lower() == "all":
                return self.DEFAULT_ALL_GRIDS
            return (configured_grids,)

        grid_names = tuple(configured_grids)
        if not grid_names:
            raise ValueError("At least one XDMF grid must be configured.")
        return grid_names

    def _get_xdmf_source(self, file_path: str) -> dict:
        if file_path in self._xdmf_sources:
            return self._xdmf_sources[file_path]

        try:
            with meshio.xdmf.TimeSeriesReader(file_path):
                source = {"type": "meshio"}
        except ReadError:
            source = self._parse_nested_xdmf(file_path)

        self._xdmf_sources[file_path] = source
        return source

    def _parse_nested_xdmf(self, file_path: str) -> dict:
        tree = ET.parse(file_path)
        root = tree.getroot()
        frames = []
        h5_path = None

        frame_grids = [
            grid
            for grid in root.iter("Grid")
            if grid.attrib.get("FrameIndex") is not None
        ]
        frame_grids.sort(key=lambda grid: int(grid.attrib["FrameIndex"]))

        for frame_grid in frame_grids:
            grids = []
            frame_children = {
                child.attrib.get("Name"): child for child in frame_grid.findall("Grid")
            }

            for grid_name in self.xdmf_grid_names:
                mesh_grid = frame_children.get(grid_name)
                if mesh_grid is None:
                    raise ValueError(
                        f"Could not find XDMF grid '{grid_name}' in {file_path}."
                    )

                topology = mesh_grid.find("Topology")
                geometry = mesh_grid.find("Geometry")
                if topology is None or geometry is None:
                    raise ValueError(
                        f"XDMF grid '{grid_name}' is missing topology or geometry."
                    )

                topology_data = topology.find("DataItem")
                geometry_data = geometry.find("DataItem")
                if topology_data is None or geometry_data is None:
                    raise ValueError(
                        f"XDMF grid '{grid_name}' is missing HDF data items."
                    )

                h5_file, cells_path = self._split_hdf_reference(topology_data.text)
                points_h5_file, points_path = self._split_hdf_reference(
                    geometry_data.text
                )
                if h5_file != points_h5_file:
                    raise ValueError("Nested XDMF frame references more than one H5 file.")
                resolved_h5_path = os.path.join(os.path.dirname(file_path), h5_file)
                if h5_path is None:
                    h5_path = resolved_h5_path
                elif h5_path != resolved_h5_path:
                    raise ValueError(
                        "Nested XDMF trajectory references more than one H5 file."
                    )

                attributes = {}
                for attribute in mesh_grid.findall("Attribute"):
                    data_item = attribute.find("DataItem")
                    if data_item is None:
                        continue
                    attribute_h5_file, attribute_path = self._split_hdf_reference(
                        data_item.text
                    )
                    if attribute_h5_file != h5_file:
                        raise ValueError(
                            "Nested XDMF frame references attributes from another H5 file."
                        )
                    attributes[attribute.attrib["Name"]] = attribute_path

                grids.append(
                    {
                        "name": grid_name,
                        "cell_type": topology.attrib.get("TopologyType", "").lower(),
                        "cells_path": cells_path,
                        "points_path": points_path,
                        "attributes": attributes,
                    }
                )

            time_item = frame_grid.find("Time")
            time = (
                float(time_item.attrib["Value"])
                if time_item is not None and "Value" in time_item.attrib
                else len(frames) * self.dt
            )

            frames.append(
                {
                    "time": time,
                    "grids": grids,
                }
            )

        if not frames or h5_path is None:
            raise ValueError(f"No readable frames found in nested XDMF file {file_path}.")

        return {"type": "nested", "h5_path": h5_path, "frames": frames}

    def _split_hdf_reference(self, reference: str) -> Tuple[str, str]:
        if reference is None or ":" not in reference:
            raise ValueError(f"Invalid XDMF HDF reference: {reference}")
        h5_file, h5_dataset = reference.strip().split(":", 1)
        return h5_file, h5_dataset

    def _read_nested_xdmf_data(
        self, source: dict, frame: int
    ) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], float, dict]:
        frame_data = source["frames"][frame]
        with h5py.File(source["h5_path"], "r") as h5_file:
            points_parts = []
            cells = []
            feature_names = []
            for grid in frame_data["grids"]:
                for name in grid["attributes"]:
                    if name in self.meta["features"] and name not in feature_names:
                        feature_names.append(name)

            feature_widths = {}
            for name in feature_names:
                for grid in frame_data["grids"]:
                    path = grid["attributes"].get(name)
                    if path is None:
                        continue
                    value = h5_file[path]
                    feature_widths[name] = value.shape[1] if value.ndim > 1 else 1
                    break

            feature_parts: Dict[str, list[np.ndarray]] = {
                name: [] for name in feature_names
            }
            node_offset = 0

            for grid in frame_data["grids"]:
                points = h5_file[grid["points_path"]][()]
                cells_data = h5_file[grid["cells_path"]][()] + node_offset
                cell_type = grid["cell_type"]
                if cell_type == "tetrahedron":
                    cell_type = "tetra"

                points_parts.append(points)
                cells.append((cell_type, cells_data))

                for name in feature_names:
                    if name in grid["attributes"]:
                        value = h5_file[grid["attributes"][name]][()]
                        if value.ndim == 1:
                            value = value.reshape(-1, 1)
                    else:
                        value = np.zeros(
                            (points.shape[0], feature_widths[name]),
                            dtype=self.meta["features"][name]["dtype"],
                        )
                    feature_parts[name].append(value)

                node_offset += points.shape[0]

            points = np.concatenate(points_parts, axis=0)
            point_data = {
                name: np.concatenate(parts, axis=0)
                for name, parts in feature_parts.items()
            }

        return points, cells, frame_data["time"], point_data

    def __getitem__(self, index: int) -> Union[Data, Tuple[Data, torch.Tensor]]:
        """Retrieve a graph representation of a frame from a trajectory.

        This method extracts a single frame from a trajectory based on the index provided.
        It first determines the trajectory and frame number using `get_traj_frame` method.
        Then, it retrieves the trajectory data as meshes and converts the specified frame
        into a graph representation.

        Parameters:
            index (int): The index of the item in the dataset.

        Returns:
            Union[Data, Tuple[Data, torch.Tensor]]: A graph representation of the specified frame in the trajectory,
            optionally along with selected indices if masking is applied.
        """
        traj_index, frame, subgraph_idx = self._get_indices(index)
        xdmf_file = self.file_paths[traj_index]
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]
        source = self._get_xdmf_source(xdmf_file)

        if source["type"] == "meshio":
            with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
                num_steps = reader.num_steps
                if frame >= num_steps - 1:
                    raise IndexError(
                        f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
                    )

                points, cells = reader.read_points_cells()
                time, point_data, _ = reader.read_data(frame)
                _, target_point_data, _ = reader.read_data(frame + 1)

                if self.use_previous_data:
                    previous_data_sequence = []
                    for offset in range(1, self.previous_data_count + 1):
                        _, previous_data, _ = reader.read_data(frame - offset)
                        previous_data_sequence.append(previous_data)
        else:
            num_steps = len(source["frames"])
            if frame >= num_steps - 1:
                raise IndexError(
                    f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames."
                )

            points, cells, time, point_data = self._read_nested_xdmf_data(source, frame)
            _, _, _, target_point_data = self._read_nested_xdmf_data(source, frame + 1)

            if self.use_previous_data:
                previous_data_sequence = []
                for offset in range(1, self.previous_data_count + 1):
                    _, _, _, previous_data = self._read_nested_xdmf_data(
                        source,
                        frame - offset,
                    )
                    previous_data_sequence.append(previous_data)

        # Prepare the mesh data
        mesh = meshio.Mesh(points, cells, point_data=point_data)

        # Keep all supported cell blocks. Mixed nested XDMF collections can contain
        # tetra cells for the billet and triangle cells for rigid tool surfaces.
        cells = [
            (cell_block.type, cell_block.data)
            for cell_block in mesh.cells
            if cell_block.type in {"triangle", "tetra", "tetrahedron"}
        ]
        if not cells:
            raise ValueError(
                "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
            )

        # Process point data and target data
        point_data = {
            k: np.array(mesh.point_data[k]).astype(self.meta["features"][k]["dtype"])
            for k in self.meta["features"]
            if k in mesh.point_data.keys()
        }

        target_data = {}
        next_data = {}
        for k in self.meta["features"]:
            if k in self.targets:
                target_data[k] = np.array(target_point_data[k]).astype(
                    self.meta["features"][k]["dtype"]
                )
            else:
                if (
                    k in target_point_data.keys()
                    and self.meta["features"][k]["type"] == "dynamic"
                ):
                    next_data[k] = np.array(target_point_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )

        def _reshape_array(a: dict):
            for k, v in a.items():
                if v.ndim == 1:
                    a[k] = v.reshape(-1, 1)

        _reshape_array(point_data)
        _reshape_array(target_data)

        # Create graph from mesh data
        graph = meshdata_to_graph(
            points=points.astype(np.float32),
            cells=cells,
            point_data=point_data,
            time=time,
            target=target_data,
            id=mesh_id,
            next_data=next_data,
        )

        if self.use_previous_data:
            previous_sequence = []
            for previous_data in previous_data_sequence:
                previous = {
                    k: np.array(previous_data[k]).astype(
                        self.meta["features"][k]["dtype"]
                    )
                    for k in self.meta["features"]
                    if k in previous_data.keys()
                    and self.meta["features"][k]["type"] == "dynamic"
                }
                _reshape_array(previous)
                previous_sequence.append(previous)
            graph.previous_data = previous_sequence[0]
            graph.previous_data_sequence = previous_sequence

        graph = graph.to(self.device)

        graph = self._apply_preprocessing(graph)
        graph = self._apply_k_hop(graph, traj_index)
        graph = self._may_remove_edges_attr(graph)
        graph = self._add_random_edges(graph)
        selected_indices = self._get_masked_indexes(graph)
        graph.edge_index = (
            graph.edge_index.long() if graph.edge_index is not None else None
        )

        del graph.next_data
        del graph.previous_data
        del graph.previous_data_sequence
        graph.traj_index = traj_index

        # TODO: not working with masking and selected_indices yet
        if self.use_partitioning:
            graph = self._get_partition(graph, traj_index, subgraph_idx)

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
