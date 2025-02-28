import os

import lightning as L
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.training.parse_parameters import get_model, get_simulator
from graphphysics.utils.loss import DiagonalGaussianMixtureNLLLoss, L2Loss
from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.meshio_mesh import convert_to_meshio_vtu, vtu_to_xdmf
from graphphysics.utils.scheduler import CosineWarmupScheduler


def build_mask(param: dict, graph: Batch):
    if len(graph.x.shape) > 2:
        node_type = graph.x[:, 0, param["index"]["node_type_index"]]
    else:
        node_type = graph.x[:, param["index"]["node_type_index"]]
    mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
    mask = torch.logical_not(mask)

    return mask


class LightningModule(L.LightningModule):
    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
        trajectory_length: int = 599,
        only_processor: bool = False,
        masks: list[NodeType] = [NodeType.NORMAL, NodeType.OUTFLOW],
        use_previous_data: bool = False,
        previous_data_start: int = None,
        previous_data_end: int = None,
    ):
        """
        Initializes the LightningModule.

        Args:
            parameters (Dict[str, Any]): Configuration parameters for the model and simulator.
            learning_rate (float): Initial learning rate for the optimizer.
            num_steps (int): Total number of training steps.
            warmup (int): Number of warmup steps for the learning rate scheduler.
            only_processor (bool, optional): Whether to use only the processor part of the model.
                Defaults to False.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            use_previous_data (bool): If set to true, we also update autoregressively the
              features at previous_data_start : previous_data_end
        """
        super().__init__()
        self.save_hyperparameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters

        processor = get_model(param=parameters, only_processor=only_processor)

        print(processor)

        self.model = get_simulator(param=parameters, model=processor, device=device)
        self.K = processor.K

        if self.K == 0:
            self.loss = L2Loss()
        else:
            self.loss = DiagonalGaussianMixtureNLLLoss(
                d=processor.d,
                K=self.K,
                temperature=processor.temperature,
            )
        self.loss_masks = masks

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = []
        self.val_step_targets = []
        self.trajectory_length = trajectory_length
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

        self.use_previous_data = use_previous_data
        self.previous_data_start = previous_data_start
        self.previous_data_end = previous_data_end

        # For one trajectory vizualization
        self.trajectory_to_save: list[Batch] = []

    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        node_type = batch.x[:, self.model.node_type_index]
        network_output, target_delta_normalized, _ = self.model(batch)

        loss = self.loss(
            target_delta_normalized,
            network_output,
            node_type,
            masks=self.loss_masks,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        # Determine if we need to reset the trajectory
        if batch.traj_index > self.current_val_trajectory:
            self.current_val_trajectory += 1
            self.last_val_prediction = None
            self.last_previous_data_prediction = None

        # Prepare the batch for the current step
        batch = batch.clone()
        if self.last_val_prediction is not None:
            # Update the batch with the last prediction
            batch.x[:, self.model.output_index_start : self.model.output_index_end] = (
                self.last_val_prediction.detach()
            )
            if self.use_previous_data:
                batch.x[:, self.previous_data_start : self.previous_data_end] = (
                    self.last_previous_data_prediction.detach()
                )

        if self.current_val_trajectory == 0:
            self.trajectory_to_save.append(batch)

        mask = build_mask(self.param, batch)
        node_type = batch.x[:, self.model.node_type_index]
        target = batch.y

        current_output = batch.x[
            :, self.model.output_index_start : self.model.output_index_end
        ]

        with torch.no_grad():
            _, _, predicted_outputs = self.model(batch)

        # Apply mask to predicted outputs
        predicted_outputs[mask] = target[mask]
        self.val_step_outputs.append(predicted_outputs.cpu())
        self.val_step_targets.append(target.cpu())

        self.last_val_prediction = predicted_outputs

        if self.use_previous_data:
            self.last_previous_data_prediction = predicted_outputs - current_output

        if self.K == 0:
            val_loss = self.loss(
                target,
                predicted_outputs,
                node_type,
                masks=self.loss_masks,
            )
            self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Concatenate outputs and targets
        predicteds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        # Compute RMSE over all rollouts
        squared_diff = (predicteds - targets) ** 2
        all_rollout_rmse = torch.sqrt(squared_diff.mean()).item()

        self.log(
            "val_all_rollout_rmse",
            all_rollout_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Save trajectory graphs as .vtu files
        save_dir = os.path.join("meshes", f"epoch_{self.current_epoch}")
        os.makedirs(save_dir, exist_ok=True)
        for idx, graph in enumerate(self.trajectory_to_save):
            try:
                mesh = convert_to_meshio_vtu(graph, add_all_data=True)
                # Construct filename
                filename = os.path.join(save_dir, f"graph_{idx}.vtu")
                # Save the mesh
                mesh.write(filename)
            except Exception as e:
                logger.error(
                    f"Error saving graph {idx} at epoch {self.current_epoch}: {e}"
                )
        logger.info(f"Validation Trajectory saved at {save_dir}.")

        # Convert vtk files to XDMF/H5 file
        try:
            vtu_files = [
                os.path.join(save_dir, f"graph_{idx}.vtu")
                for idx in range(len(self.trajectory_to_save))
            ]
            print(f"FILE: {os.path.join(save_dir, 'graph')}")
            vtu_to_xdmf(
                os.path.join(save_dir, f"graph_epoch_{self.current_epoch}"), vtu_files
            )
        except Exception as e:
            logger.error(f"Error compressing vtus at epoch {self.current_epoch}: {e}")

        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None
        self.trajectory_to_save.clear()

    def configure_optimizers(self):
        """Initialize the optimizer"""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.95),
        )
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.num_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }
