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
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: List[int],
        output_dir: str = "predictions",
        compare_downsampling: bool = False,
        downsample_ratio: Optional[float] = None,
        downsample_k: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.output_dir = output_dir
        self.compare_downsampling = compare_downsampling
        self.downsample_ratio = downsample_ratio
        self.downsample_k = downsample_k
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
        downsampled_images: List[wandb.Image] = []

        with torch.no_grad():
            for idx in self.indices:
                graph = self.dataset[idx]
                viz_graph = graph.clone()
                graph = graph.to(device)
                _, _, predicted_outputs = model(graph)

                graph.x = predicted_outputs

                # Convert outputs to a PyVista mesh
                predicted_mesh = self._convert_to_pyvista_mesh(graph)

                # Generate visualization
                img = self._generate_pyvista_image(predicted_mesh)
                images.append(
                    wandb.Image(img, caption=f"Prediction: Sample Index: {idx}")
                )

                ### Same for Ground Truth:
                graph.x = graph.y
                # Convert outputs to a PyVista mesh
                predicted_mesh = self._convert_to_pyvista_mesh(graph)

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
        if self.compare_downsampling and downsampled_images:
            wandb_logger.experiment.log(
                {"pyvista_downsampling": downsampled_images}
            )

        frames_predictions = []
        frames_ground_truth = []
        predicted_outputs = None

        if len(self.indices) > 1:
            with torch.no_grad():
                for idx in range(self.indices[0], self.indices[-1]):
                    graph = self.dataset[idx].to(device)

                    if predicted_outputs is not None:
                        # Update the graph with the last prediction
                        graph.x[
                            :,
                            model.model.output_index_start : model.model.output_index_end,
                        ] = predicted_outputs.detach()

                    mask = build_mask(model.param, graph)
                    target = graph.y

                    _, _, predicted_outputs = model(graph)
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

    def _generate_pyvista_image(self, predicted_mesh: pv.PolyData):
        """
        Generates an image visualizing the predicted and ground truth meshes.

        Args:
            predicted_mesh (pv.PolyData): The predicted mesh.
            ground_truth_mesh (Optional[pv.PolyData]): The ground truth mesh, if available.

        Returns:
            Any: The generated image.
        """
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(predicted_mesh, scalars="x0", label="graph.x[0]", opacity=0.8)
        plotter.add_legend()
        img = plotter.screenshot(return_img=True)
        plotter.close()
        return img

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
        if hasattr(pl_module, "param"):
            model_cfg = pl_module.param.get("model", {})

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
            color="#94c6ff",
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
