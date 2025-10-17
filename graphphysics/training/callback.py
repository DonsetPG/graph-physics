import os
from typing import List

import lightning.pytorch as pl
import numpy as np
import pyvista as pv
import torch
from lightning.pytorch.callbacks import Callback
from torch_geometric.data import Data, Dataset

import wandb
from graphphysics.training.lightning_module import build_mask
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
    """

    def __init__(
        self, dataset: Dataset, indices: List[int], output_dir: str = "predictions"
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.output_dir = output_dir

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

        with torch.no_grad():
            for idx in self.indices:
                graph = self.dataset[idx].to(device)
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

        wandb_logger = trainer.logger
        wandb_logger.experiment.log({"pyvista_predictions": images})
        wandb_logger.experiment.log({"pyvista_ground_truth": ground_truth})

        frames_predictions = []
        frames_ground_truth = []
        predicted_outputs = None

        if len(self.indices) > 1:
            with torch.no_grad():
                for idx in range(self.indices[0], self.indices[-1]):
                    graph = self.dataset[idx].to(device)

                    if predicted_outputs is not None:
                        # Update the graph with the last prediction
                        model.model.assign_targets_to_x(
                            graph, predicted_outputs.detach()
                        )

                    mask = build_mask(model.param, graph)
                    target = graph.y

                    _, _, predicted_outputs = model(graph)
                    predicted_outputs[mask] = target[mask]

                    model.model.assign_targets_to_x(graph, predicted_outputs)

                    # Convert outputs to a PyVista mesh
                    predicted_mesh = self._convert_to_pyvista_mesh(graph)

                    # Generate visualization
                    img = self._generate_pyvista_image(predicted_mesh)
                    # Ensure the image is a numpy array with dtype uint8
                    img_array = np.array(img).astype(np.uint8)
                    frames_predictions.append(img_array)

                    ### Same for Ground Truth:
                    model.model.assign_targets_to_x(graph, graph.y)
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
