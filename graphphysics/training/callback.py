import os
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import pyvista as pv
import torch
from lightning.pytorch.callbacks import Callback
from torch_geometric.data import Data, Dataset

import wandb
from graphphysics.training.lightning_module import build_mask
from graphphysics.models.hierarchical_pooling import DownSampler
from graphphysics.utils.pyvista_mesh import convert_to_pyvista_mesh


class LogPyVistaPredictionsCallback(Callback):
    """
    PyTorch Lightning Callback to log model predictions as images using PyVista.

    This callback fetches specified data samples from a dataset, makes predictions
    using the provided model, visualizes the predictions using PyVista, and logs
    the resulting images to wandb.

    Args:
        dataset (Dataset): The dataset to fetch data samples from.
        indices (List[int]): List of indices specifying which data samples to use.
        output_dir (str, optional): Directory to save the generated images. Defaults to 'predictions'.
        compare_downsampling (bool, optional): If True, log PyVista overlays of the original graph
            versus its downsampled counterpart. Defaults to False.
        downsample_ratio (Optional[float], optional): Override the downsampling ratio instead
            of reading it from the LightningModule configuration. Defaults to None.
        downsample_k (Optional[int], optional): Override the number of neighbors used while
            remeshing the downsampled graph. Defaults to None.
        downsample_traj_index (Optional[int], optional): If set, logs a video of the downsampled
            nodes for this trajectory index at the end of each validation epoch. Defaults to None.
        downsample_frame_step (int, optional): Frame stride when building the downsampled trajectory
            video. Defaults to 1 (every frame).
        downsample_max_frames (Optional[int], optional): Maximum number of frames to include in the
            downsampled trajectory video. Defaults to None (all frames following the stride).
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        output_dir: str = "predictions",
        compare_downsampling: bool = False,
        downsample_ratio: Optional[float] = None,
        downsample_k: Optional[int] = None,
        downsample_traj_index: Optional[int] = None,
        downsample_frame_step: int = 1,
        downsample_max_frames: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.output_dir = output_dir
        self.compare_downsampling = compare_downsampling
        self.downsample_ratio = downsample_ratio
        self.downsample_k = downsample_k
        self.downsample_traj_index = downsample_traj_index
        self.downsample_frame_step = max(1, downsample_frame_step)
        self.downsample_max_frames = downsample_max_frames
        self._downsampler: Optional[DownSampler] = None
        self._downsampler_device = torch.device("cpu")

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Called at the end of the validation epoch. Generates and logs the prediction images.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer.
            pl_module (pl.LightningModule): The LightningModule being trained.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        model = pl_module
        device = model.device

        images = []
        ground_truth = []
        score_images = []
        downsampled_images: List[wandb.Image] = []

        with torch.no_grad():
            for idx in self.indices:
                base_graph = self.dataset[idx]
                viz_graph = base_graph.clone()
                graph = base_graph.to(device)
                mask = build_mask(model.param, graph)
                _, _, predicted_outputs = model(graph)
                scores = self._compute_scores(model, graph, predicted_outputs, mask)

                pred_graph = base_graph.clone()
                pred_graph.x = predicted_outputs.detach().cpu()

                # Convert outputs to a PyVista mesh
                predicted_mesh = self._convert_to_pyvista_mesh(pred_graph)

                # Generate visualization
                img = self._generate_pyvista_image(predicted_mesh)
                images.append(
                    wandb.Image(img, caption=f"Prediction: Sample Index: {idx}")
                )

                if scores is not None:
                    predicted_mesh.point_data["score"] = scores.detach().cpu().numpy()
                    score_img = self._generate_pyvista_image(
                        predicted_mesh,
                        scalars="score",
                        label="Physical residual score",
                        cmap="viridis",
                    )
                    score_images.append(
                        wandb.Image(score_img, caption=f"Scores: Sample Index: {idx}")
                    )

                ### Same for Ground Truth:
                gt_graph = viz_graph.clone()
                gt_graph.x = gt_graph.y
                # Convert outputs to a PyVista mesh
                predicted_mesh = self._convert_to_pyvista_mesh(gt_graph)

                # Generate visualization
                img = self._generate_pyvista_image(predicted_mesh)
                ground_truth.append(
                    wandb.Image(img, caption=f"Ground Truth: Sample Index: {idx}"),
                )

                if self.compare_downsampling:
                    comparison_img = self._maybe_create_downsampling_image(
                        viz_graph, pl_module
                    )
                    if comparison_img is not None:
                        downsampled_images.append(
                            wandb.Image(
                                comparison_img,
                                caption=f"Downsampling: Sample Index: {idx}",
                            )
                        )

        wandb_logger = trainer.logger
        wandb_logger.experiment.log({"pyvista_predictions": images})
        wandb_logger.experiment.log({"pyvista_ground_truth": ground_truth})
        if score_images:
            wandb_logger.experiment.log({"pyvista_scores": score_images})
        if self.compare_downsampling and downsampled_images:
            wandb_logger.experiment.log(
                {"pyvista_downsampling": downsampled_images}
            )
        if self.compare_downsampling and self.downsample_traj_index is not None:
            traj_video = self._log_downsampled_trajectory_video(pl_module)
            if traj_video is not None:
                wandb_logger.experiment.log(
                    {"pyvista_downsampled_trajectory": traj_video}
                )

        frames_predictions = []
        frames_ground_truth = []
        predicted_outputs = None

        if len(self.indices) > 1:
            with torch.no_grad():
                last_traj_index = None
                for idx in range(self.indices[0], self.indices[-1]):
                    graph = self.dataset[idx]
                    if isinstance(graph, tuple):
                        graph = graph[0]
                    graph = graph.to(device)

                    traj_idx = getattr(graph, "traj_index", None)
                    node_count = graph.x.shape[0] if hasattr(graph, "x") and graph.x is not None else graph.num_nodes
                    # Reset autoregressive state if trajectory switches or node count changes.
                    if (
                        last_traj_index is not None
                        and traj_idx is not None
                        and traj_idx != last_traj_index
                    ) or (
                        predicted_outputs is not None
                        and predicted_outputs.shape[0] != node_count
                    ):
                        predicted_outputs = None

                    if predicted_outputs is not None:
                        # Update the graph with the last prediction (same size only)
                        if predicted_outputs.shape[0] != node_count:
                            predicted_outputs = None
                        else:
                            graph.x[
                                :,
                                model.model.output_index_start : model.model.output_index_end,
                            ] = predicted_outputs.detach()

                    mask = build_mask(model.param, graph)
                    target = graph.y

                    _, _, predicted_outputs = model(graph)
                    if target.shape[0] != node_count:
                        predicted_outputs = None
                        last_traj_index = traj_idx
                        continue
                    if predicted_outputs.shape[0] != node_count:
                        predicted_outputs = None
                        last_traj_index = traj_idx
                        continue
                    predicted_outputs[mask] = target[mask]

                    graph.x = predicted_outputs

                    # Convert outputs to a PyVista mesh
                    predicted_mesh = self._convert_to_pyvista_mesh(graph)

                    # Generate visualization
                    img = self._generate_pyvista_image(predicted_mesh)
                    # Ensure the image is a numpy array with dtype uint8
                    img_array = np.array(img).astype(np.uint8)
                    frames_predictions.append(img_array)

                    ### Same for Ground Truth:
                    graph.x = graph.y
                    # Convert outputs to a PyVista mesh
                    ground_truth_mesh = self._convert_to_pyvista_mesh(graph)

                    # Generate visualization
                    img = self._generate_pyvista_image(ground_truth_mesh)
                    img_array = np.array(img).astype(np.uint8)
                    frames_ground_truth.append(img_array)

                    last_traj_index = traj_idx

            # Convert frames to numpy arrays
            frames_predictions = np.stack(
                frames_predictions, axis=0
            )  # Shape: (time, height, width, channels)
            frames_ground_truth = np.stack(frames_ground_truth, axis=0)

            # Rearrange axes to (time, channels, height, width)
            frames_predictions = np.transpose(frames_predictions, (0, 3, 1, 2))
            frames_ground_truth = np.transpose(frames_ground_truth, (0, 3, 1, 2))

            # Create WandB Video objects
            video_predictions = wandb.Video(frames_predictions, fps=4)
            video_ground_truth = wandb.Video(frames_ground_truth, fps=4)

            # Log videos to WandB
            wandb_logger = trainer.logger
            wandb_logger.experiment.log(
                {"pyvista_predictions_video": video_predictions}
            )
            wandb_logger.experiment.log(
                {"pyvista_ground_truth_video": video_ground_truth}
            )

    def _convert_to_pyvista_mesh(self, graph: Data) -> pv.PolyData:
        """
        Converts model outputs to a PyVista mesh.

        Args:
            data (Any): The data to convert (model outputs or labels).

        Returns:
            pv.PolyData: The converted PyVista mesh.
        """
        mesh = convert_to_pyvista_mesh(graph=graph)

        # Add point data from graph.x if it exists
        if hasattr(graph, "x") and graph.x is not None:
            x_data = graph.x.cpu().numpy()
            if x_data.shape[1] >= 1:
                mesh.point_data["x0"] = x_data[:, 0]

        return mesh

    def _generate_pyvista_image(
        self,
        predicted_mesh: pv.PolyData,
        scalars: str = "x0",
        label: str = "graph.x[0]",
        cmap: str = "bwr",
    ):
        """
        Generates an image visualizing the predicted and ground truth meshes.

        Args:
            predicted_mesh (pv.PolyData): The predicted mesh.
            ground_truth_mesh (Optional[pv.PolyData]): The ground truth mesh, if available.

        Returns:
            Any: The generated image.
        """
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            predicted_mesh,
            scalars=scalars,
            label=label,
            opacity=0.8,
            cmap=cmap,
        )
        plotter.add_legend()
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

    def _compute_scores(
        self,
        pl_module: pl.LightningModule,
        graph: Data,
        predicted_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute per-node physical residual scores using the module's physical_loss.
        Returns None if scores cannot be computed.
        """
        physical_loss = getattr(pl_module, "physical_loss", None)
        if physical_loss is None:
            return None
        if predicted_outputs is None or predicted_outputs.shape[1] < 3:
            return None

        try:
            residual = physical_loss(graph, predicted_outputs[:, 0:3])
        except Exception:
            return None

        scores = residual.detach()
        if mask is not None:
            scores = scores.clone()
            scores[mask] = torch.nan
        return scores

    def _maybe_create_downsampling_image(
        self, graph: Data, pl_module: pl.LightningModule
    ):
        if not self.compare_downsampling:
            return None
        coarse_graph = self._downsample_graph(graph, pl_module)
        if coarse_graph is None:
            return None

        try:
            fine_mesh = self._convert_to_pyvista_mesh(graph)
            coarse_mesh = self._convert_to_pyvista_mesh(coarse_graph)
        except ValueError:
            return None
        return self._generate_downsampling_image(fine_mesh, coarse_mesh)

    def _downsample_graph(
        self, graph: Data, pl_module: pl.LightningModule
    ) -> Optional[Data]:
        if not hasattr(graph, "pos") or graph.pos is None:
            return None
        if not hasattr(graph, "edge_index") or graph.edge_index is None:
            return None

        self._ensure_downsampler(graph, pl_module)
        if self._downsampler is None:
            return None

        working_graph = graph.clone()
        working_graph = working_graph.to(self._downsampler_device)
        coarse_graph = self._downsampler(working_graph)
        coarse_graph = coarse_graph.to(torch.device("cpu"))
        return coarse_graph

    def _ensure_downsampler(
        self, graph: Data, pl_module: pl.LightningModule
    ) -> None:
        if not self.compare_downsampling or self._downsampler is not None:
            return

        model_cfg = {}
        training_cfg = {}
        if hasattr(pl_module, "param"):
            model_cfg = pl_module.param.get("model", {})
            training_cfg = pl_module.param.get("training", {})

        ratio = (
            self.downsample_ratio
            if self.downsample_ratio is not None
            else model_cfg.get("pool_ratio", 0.5)
        )
        k = (
            self.downsample_k
            if self.downsample_k is not None
            else model_cfg.get("pool_knn", 6)
        )
        method = training_cfg.get("sampling_method", "fps")
        sampling_temperature = training_cfg.get("sampling_temperature", 1.0)
        is_remeshing = model_cfg.get(
            "pool_is_remeshing", model_cfg.get("is_remeshing", True)
        )
        pool_node_mask = model_cfg.get("pool_node_mask", "normal")

        if not hasattr(graph, "x") or graph.x is None:
            return

        feature_dim = graph.x.shape[1]
        edge_dim = (
            graph.edge_attr.shape[1]
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None
            else 4
        )
        self._downsampler = DownSampler(
            d_in=feature_dim,
            d_out=feature_dim,
            edge_dim=edge_dim,
            ratio=ratio,
            k=k,
            method=method,
            sampling_temperature=sampling_temperature,
            is_remeshing=is_remeshing,
            node_mask=pool_node_mask,
        ).to(graph.x.device)
        with torch.no_grad():
            eye = torch.eye(feature_dim, device=self._downsampler.lin.weight.device)
            self._downsampler.lin.weight.copy_(eye)
            self._downsampler.lin.bias.zero_()
        self._downsampler_device = graph.x.device

    def _generate_downsampling_image(
        self, fine_mesh: pv.PolyData, coarse_mesh: pv.PolyData
    ):
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            fine_mesh,
            color="#008000",
            show_edges=True,
            opacity=0.35,
            label="Original Mesh",
        )
        plotter.add_mesh(
            coarse_mesh,
            style="points",
            color="#d62728",
            point_size=10,
            render_points_as_spheres=True,
            label="Downsampled Points",
        )
        plotter.add_legend()
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

    def _generate_downsample_only_image(self, coarse_mesh: pv.PolyData):
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            coarse_mesh,
            style="points",
            color="#d62728",
            point_size=10,
            render_points_as_spheres=True,
            label="Downsampled Points",
        )
        plotter.add_legend()
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

    def _log_downsampled_trajectory_video(
        self, pl_module: pl.LightningModule
    ) -> Optional[wandb.Video]:
        traj_len = getattr(self.dataset, "trajectory_length", None)
        if traj_len is None or traj_len < 2:
            return None

        traj_idx = self.downsample_traj_index
        frame_ids = list(range(0, traj_len - 1, self.downsample_frame_step))
        if self.downsample_max_frames is not None:
            frame_ids = frame_ids[: self.downsample_max_frames]

        frames: list[np.ndarray] = []
        with torch.no_grad():
            for frame in frame_ids:
                dataset_index = traj_idx * (traj_len - 1) + frame
                if dataset_index >= len(self.dataset):
                    break
                graph = self.dataset[dataset_index]
                coarse_graph = self._downsample_graph(graph, pl_module)
                if coarse_graph is None:
                    continue
                try:
                    coarse_mesh = self._convert_to_pyvista_mesh(coarse_graph)
                except ValueError:
                    continue
                img = self._generate_downsample_only_image(coarse_mesh)
                frames.append(np.array(img).astype(np.uint8))

        if not frames:
            return None

        frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
        frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)
        return wandb.Video(frames_np, fps=4)
