import os
import random
from typing import Callable, List, Optional, Tuple, Union

import meshio
import numpy as np
import torch
from torch_cluster import knn
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.dataset import BaseDataset
from graphphysics.dataset.icp import iterative_closest_point
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.torch_graph import meshdata_to_graph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class XDMFDataset(BaseDataset):
    def __init__(
        self,
        xdmf_folder: str,
        meta_path: str,
        preprocessing: Optional[Callable[[Data], Data]] = None,
        masking_ratio: Optional[float] = None,
        khop: int = 1,
        new_edges_ratio: float = 0,
        add_edge_features: bool = True,
        use_previous_data: bool = False,
        switch_to_val: bool = False,
        target_same_frame: bool = False,
        random_prev: int = 1,  # If we use previous data, we will fetch one previous frame between [-1, -random_prev]
        random_next: int = 1,  # The target will be the frame : t + [1, random_next]
        node_type_index: int = 14,
        cache_dir: str = "cache",  # Directory for caching features
    ):
        super().__init__(
            meta_path=meta_path,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=add_edge_features,
            use_previous_data=use_previous_data,
            target_same_frame=target_same_frame,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = "xdmf"

        self.dt = self.meta["dt"]
        if self.dt == 0:
            self.dt = 1
            logger.warning(
                "The dataset has a timestep set to 0. Fallback to dt=1 to ensure xdmf can be saved."
            )
        self.random_next = random_next
        self.random_prev = random_prev

        if switch_to_val:
            xdmf_folder = xdmf_folder.replace("train", "test")
            self.random_next = 1
            self.random_prev = 1

        self.xdmf_folder = xdmf_folder
        self.meta_path = meta_path
        self.node_type_index = node_type_index
        self.target_same_frame = target_same_frame

        # Get list of XDMF files in the folder
        self.file_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".xdmf")
        ]
        self._size_dataset: int = len(self.file_paths)

        self.npzfile_paths: List[str] = [
            os.path.join(xdmf_folder, f)
            for f in os.listdir(xdmf_folder)
            if os.path.isfile(os.path.join(xdmf_folder, f)) and f.endswith(".npz")
        ]

        # Initialize caching mechanism
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def size_dataset(self) -> int:
        """Returns the number of trajectories in the dataset."""
        return self._size_dataset
    
    def get_encoding(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        file_name = file_name.replace(".xdmf", ".npz")

        data_npz = np.load(file_name)
        feats = data_npz["patchtokens"]
        coords = data_npz["indices"]

        return feats, coords

    def scale_pos(self, graph: Data, coords: np.ndarray):
        pos_graph = graph.pos.cpu().numpy()

        g_min, g_max = pos_graph.min(axis=0), pos_graph.max(axis=0)
        pc_min, pc_max = coords.min(axis=0), coords.max(axis=0)

        scale = (g_max - g_min) / (pc_max - pc_min)
        shift = g_min - pc_min * scale
        coords_scaled = coords * scale + shift

        return coords_scaled

    def plot_rescaled(self, graph: Data, index: int, file_name: str):
        feats, coords = self.get_encoding(file_name=file_name)
        coords[:, [1, 2]] = coords[:, [2, 1]]
        coords_scaled = self.scale_pos(graph, coords)
        coords_scaled = torch.tensor(
            coords_scaled, dtype=graph.pos.dtype, device=graph.pos.device
        )
        coords_scaled[:, 2] = -coords_scaled[:, 2]

        pos_graph = graph.pos

        graph_centroid = np.mean(pos_graph.cpu().numpy(), axis=0)
        coords_centroid = np.mean(coords_scaled.cpu().numpy(), axis=0)

        translation_vector = graph_centroid - coords_centroid
        coords_scaled[:, 2] += translation_vector[2]

        output_folder = "output2"
        os.makedirs(output_folder, exist_ok=True)

        # Save graph positions to a VTK file
        graph_vtk_path = os.path.join(output_folder, f"graph_{index}.vtk")
        graph_points = pos_graph.cpu().numpy()
        graph_cells = np.arange(graph_points.shape[0]).reshape(-1, 1)
        graph_mesh = meshio.Mesh(points=graph_points, cells={"vertex": graph_cells})
        meshio.write(graph_vtk_path, graph_mesh)

        # Save scaled coordinates to a VTK file
        coords_vtk_path = os.path.join(output_folder, f"coords_{index}.vtk")
        coords_points = coords_scaled.cpu().numpy()
        coords_cells = np.arange(coords_points.shape[0]).reshape(-1, 1)
        coords_mesh = meshio.Mesh(points=coords_points, cells={"vertex": coords_cells})
        meshio.write(coords_vtk_path, coords_mesh)

        print(f"Graph saved to {graph_vtk_path}")
        print(f"Coords saved to {coords_vtk_path}")

    def apply_icp(
        self, graph: Data, coords_scaled: np.ndarray, max_iterations=20, tolerance=0.001
    ):
        wall_mask = graph.x[:, self.node_type_index] == NodeType.WALL_BOUNDARY
        wall_indices = wall_mask.nonzero(as_tuple=True)[0]
        pos_graph = graph.pos[wall_indices]

        aligned_coords, _, _ = iterative_closest_point(
            coords_scaled, pos_graph, max_iterations=max_iterations, tolerance=tolerance
        )

        return aligned_coords

    def add_encoding(
        self, graph: Data, feats: torch.Tensor, coords: np.ndarray, K: int = 6
    ) -> torch.Tensor:
        wall_mask = graph.x[:, self.node_type_index] == NodeType.WALL_BOUNDARY
        wall_indices = wall_mask.nonzero(as_tuple=True)[0]
        wall_pos = graph.pos[wall_indices]

        edge_index = knn(coords, wall_pos, k=K)

        F = feats.shape[1]
        wall_features = torch.zeros(
            (wall_indices.shape[0], F), dtype=graph.x.dtype, device=graph.x.device
        )

        for i in range(wall_indices.shape[0]):
            neighbor_mask = edge_index[0] == i
            neighbor_indices = edge_index[1][neighbor_mask]
            if neighbor_indices.numel() > 0:
                wall_features[i] = feats[neighbor_indices].mean(dim=0)
            else:
                wall_features[i] = 0

        non_wall_mask = graph.x[:, self.node_type_index] != NodeType.WALL_BOUNDARY
        non_wall_indices = non_wall_mask.nonzero(as_tuple=True)[0]
        non_wall_pos = graph.pos[non_wall_indices]

        dists = torch.cdist(non_wall_pos, wall_pos)
        min_dists, min_idx = dists.min(dim=1)

        d_min = min_dists.min()
        d_max = min_dists.max()
        if d_max == d_min:
            weights = torch.ones_like(min_dists)
        else:
            weights = 1 - (min_dists - d_min) / (d_max - d_min)

        new_features = torch.zeros(
            (graph.x.shape[0], F), dtype=graph.x.dtype, device=graph.x.device
        )
        new_features[wall_indices] = wall_features
        new_features[non_wall_indices] = weights.unsqueeze(1) * wall_features[min_idx]

        return new_features

    def get_new_features(self, graph: Data):
        file_path = self.file_paths[graph.traj_index]
        feats, coords = self.get_encoding(file_name=file_path)
        feats = torch.tensor(feats, dtype=graph.x.dtype, device=graph.x.device)

        coords[:, [1, 2]] = coords[:, [2, 1]]
        coords_scaled = self.scale_pos(graph, coords)
        coords_scaled = torch.tensor(
            coords_scaled, dtype=graph.pos.dtype, device=graph.pos.device
        )
        coords_scaled[:, 2] = -coords_scaled[:, 2]

        pos_graph = graph.pos

        graph_centroid = np.mean(pos_graph.cpu().numpy(), axis=0)
        coords_centroid = np.mean(coords_scaled.cpu().numpy(), axis=0)

        translation_vector = graph_centroid - coords_centroid
        coords_scaled[:, 2] += translation_vector[2]

        # coords_scaled = self.apply_icp(graph, coords_scaled, max_iterations=20, tolerance=0.001)

        new_features = self.add_encoding(graph, feats, coords_scaled, K=6)

        return new_features

    def add_new_features(self, graph: Data, index: int):
        traj_index = graph.traj_index
        cache_file = os.path.join(self.cache_dir, f"features_{traj_index}.pt")

        if os.path.exists(cache_file):
            try:
                # Load features from cache
                features = torch.load(cache_file)
            except Exception as e:
                print(f"Error loading cached features: {e}. Recalculating features.")
                os.remove(cache_file)
                features = self.get_new_features(graph)
                torch.save(features, cache_file)
        else:
            # Calculate features and save to cache
            features = self.get_new_features(graph)
            torch.save(features, cache_file)

        if graph.x.shape[0] != features.shape[0]:
            features = self.get_new_features(graph)
            torch.save(features, cache_file)

        graph.x = torch.cat([graph.x, features], dim=1)
        return graph

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
        traj_index, frame = self.get_traj_frame(index=index)
        xdmf_file = self.file_paths[traj_index]
        mesh_id = os.path.splitext(os.path.basename(xdmf_file))[0].rsplit("_", 1)[-1]

        # Fetch index for previous_data and target
        _target_data_index = random.randint(1, self.random_next)
        _previous_data_index = random.randint(1, self.random_prev)

        # Read XDMF file
        # --- ADD ---
        # Option without time serie
        if self.target_same_frame:
            mesh = meshio.read(xdmf_file)
            points, cells = mesh.points, mesh.cells
            point_data = dict(mesh.point_data)
            # --- AJOUT : inclure mesh_pos et wall_mask ---
            point_data["mesh_pos"] = mesh.points.astype(
                self.meta["features"]["mesh_pos"]["dtype"]
            )
            if "wall_mask" in mesh.point_data.keys():
                point_data["wall_mask"] = np.array(
                    mesh.point_data["wall_mask"]
                ).astype(self.meta["features"]["wall_mask"]["dtype"])
            # --- FIN AJOUT ---
            target_point_data = point_data  # même frame = même cible
            previous_data = None
            num_steps = 1
            target_frame = 1

        # --- END ADD ---
        else:
            with meshio.xdmf.TimeSeriesReader(xdmf_file) as reader:
                num_steps = reader.num_steps

                if frame - _previous_data_index < 0:
                    _previous_data_index = 1
                if frame + _target_data_index > num_steps - 1:
                    _target_data_index = 1

                if frame >= num_steps - 1 and (not self.target_same_frame):
                    raise IndexError(
                        f"Frame index {frame} out of bounds for trajectory {traj_index} with {num_steps} frames.")

                points, cells = reader.read_points_cells()
                time, point_data, _ = reader.read_data(frame)
                target_frame = frame + 1
                if self.target_same_frame: target_frame = frame
                _, target_point_data, _ = reader.read_data(target_frame)

                if self.use_previous_data:
                    _, previous_data, _ = reader.read_data(frame - _previous_data_index)

        # Prepare the mesh data
        mesh = meshio.Mesh(points, cells, point_data=point_data)

        # Get faces or cells
        if "triangle" in mesh.cells_dict:
            cells = mesh.cells_dict["triangle"]
        elif "line" in mesh.cells_dict:
            cells = mesh.cells_dict["line"]
        elif "tetra" in mesh.cells_dict:
            cells = torch.tensor(mesh.cells_dict["tetra"], dtype=torch.long)
        else:
            raise ValueError(
                "Unsupported cell type. Only 'triangle' and 'tetra' cells are supported."
            )

        # Process point data and target data
        if self.target_same_frame:
            selected_features = ["cells", "mesh_pos", "TAWSS"]
            point_data = {
                k: np.array(mesh.point_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in mesh.point_data.keys() and k not in selected_features
            }

            target_data = {
                k: np.array(target_point_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in target_point_data.keys()
                and k == "TAWSS"
            }
        else:
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
        if self.target_same_frame:
            graph = meshdata_to_graph(
                points=points.astype(np.float32),
                cells=cells,
                point_data=point_data,
                time=0,
                target=target_data,
                id=mesh_id,
            )
        else:
            graph = meshdata_to_graph(
                points=points.astype(np.float32),
                cells=cells,
                point_data=point_data,
                time=time,
                target=target_data,
                id=mesh_id,
                next_data=next_data,
            )
        # TODO: add target_dt and previous_dt as features per node.
        graph.target_dt = _target_data_index * self.dt
        self.use_previous_data = False #### I force use_previous_data to be false

        if self.use_previous_data:
            previous = {
                k: np.array(previous_data[k]).astype(self.meta["features"][k]["dtype"])
                for k in self.meta["features"]
                if k in previous_data.keys()
                and self.meta["features"][k]["type"] == "dynamic"
            }
            _reshape_array(previous)
            graph.previous_data = previous
            graph.previous_dt = -_previous_data_index * self.dt

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
        graph.traj_index = traj_index

        graph = self.add_new_features(graph, index)

        if selected_indices is not None:
            return graph, selected_indices
        else:
            return graph
