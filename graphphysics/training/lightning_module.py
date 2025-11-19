import os
import shutil
from typing import List

import lightning as L
import meshio
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.training.parse_parameters import (
    get_gradient_method,
    get_loss,
    get_model,
    get_simulator,
)
from graphphysics.utils.loss import L2Loss, MultiLoss
from graphphysics.utils.meshio_mesh import convert_to_meshio_vtu
from graphphysics.utils.nodetype import NodeType
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
        timestep: float = 1.0,
        only_processor: bool = False,
        masks: list[NodeType] = [NodeType.NORMAL, NodeType.OUTFLOW],
        use_previous_data: bool = False,
        previous_data_start: int = None,
        previous_data_end: int = None,
        prediction_save_path: str = "predictions",
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
            prediction_save_path (str): Directory where predictions will be saved.
        """
        super().__init__()
        self.save_hyperparameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters
        self.wandb_run_id = None

        processor = get_model(param=parameters, only_processor=only_processor)

        print(processor)

        self.model = get_simulator(param=parameters, model=processor, device=device)

        for m in getattr(self.model, "processor_list", []):
            if hasattr(m, "use_activation_checkpointing"):
                m.use_activation_checkpointing = True

        self.loss, self.loss_name = get_loss(param=parameters)
        logger.info(f"Using loss {self.loss_name}")
        self.is_multiloss = False
        if isinstance(self.loss, MultiLoss):
            self.is_multiloss = True

        self.loss_masks = masks
        self.val_loss = L2Loss()
        self.gradient_method = get_gradient_method(
            param=parameters
        )  # finite_diff, least_squares

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.step_counter = 0
        self.first_step_losses: List[torch.Tensor] = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.trajectory_length = trajectory_length
        self.timestep = timestep
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

        self._last_val_num_nodes = None
        self._last_pred_num_nodes = None

        self.use_previous_data = use_previous_data
        self.previous_data_start = previous_data_start
        self.previous_data_end = previous_data_end

        # For one trajectory vizualization
        self.trajectory_to_save: list[Batch] = []

        # Prediction
        self.prediction_save_path: str = prediction_save_path
        self.current_pred_trajectory = 0
        self.prediction_trajectory: list[Batch] = []
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None

    # === DIAG: hooks concis pour voir où ça bloque ===
    def on_fit_start(self):
        if self.trainer.is_global_zero:
            print("[diag] fit_start", flush=True)

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"[diag] epoch_start={self.current_epoch}", flush=True)

    def on_train_batch_start(self, batch, batch_idx):
        # on log uniquement le tout 1er batch pour éviter le spam
        if batch_idx == 0 and self.trainer.is_global_zero:
            print("[diag] first_batch_start", flush=True)

    def on_after_backward(self):
        # valider que le 1er backward/allreduce passe
        if self.global_step == 0 and self.trainer.is_global_zero:
            print("[diag] first_backward_done", flush=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0 and self.trainer.is_global_zero:
            print("[diag] first_batch_end", flush=True)

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            print(f"[diag] epoch_end={self.current_epoch}", flush=True)
    ##########################################################

    def forward(self, graph: Batch):
        return self.model(graph)

    def training_step(self, batch: Batch):
        if self.global_step == 0 and self.trainer.is_global_zero:
            print(
                f"[diag] cuda reserved={torch.cuda.memory_reserved()/1e9:.2f} GB "
                f"allocated={torch.cuda.memory_allocated()/1e9:.2f} GB",
                flush=True,
            )
        batch = batch.to(self.device, non_blocking=True)
        node_type = batch.x[:, self.model.node_type_index]
        network_output, target_delta_normalized, _ = self.model(batch)

        if self.is_multiloss:
            network_output_physical = self.model.build_outputs(batch, network_output)
            target_physical = self.model.build_outputs(batch, target_delta_normalized)
            loss, train_losses = self.loss(
                graph=batch,
                target=target_delta_normalized,
                network_output=network_output,
                node_type=node_type,
                masks=self.loss_masks,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=self.gradient_method,
                return_all_losses=True,
            )
            for train_loss, loss_name in zip(train_losses, self.loss_name):
                self.log(
                    f"train_loss - {loss_name}",
                    train_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
            self.log(
                "train_multiloss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
            )

        else:  # Will raise an error if the single loss needs physical outputs.
            loss = self.loss(
                graph=batch,
                target=target_delta_normalized,
                network_output=network_output,
                node_type=node_type,
                masks=self.loss_masks,
                gradient_method=self.gradient_method,
            )

            self.log(
                f"train_{self.loss_name}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def _save_trajectory_to_xdmf(
        self,
        trajectory: list[Batch],
        save_dir: str,
        archive_filename: str,
        timestep: float = 1,
    ):
        os.makedirs(save_dir, exist_ok=True)
        archive_path = os.path.join(save_dir, archive_filename)
        xdmf_filename = f"{archive_path}.xdmf"
        init_mesh = convert_to_meshio_vtu(trajectory[0], add_all_data=True)
        points = init_mesh.points
        cells = init_mesh.cells

        # --- Option: without time series (single frame only) ---
        target_same_frame: bool = True
        if getattr(self, "target_same_frame", True):
            mesh = convert_to_meshio_vtu(trajectory[0], add_all_data=True)
            meshio.write(xdmf_filename, mesh)
            logger.info(
                f"[No Time Series] Single frame saved at {xdmf_filename}"
            )
            return


        try:
            with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
                # Write the mesh (points and cells) once
                writer.write_points_cells(points, cells)
                # Loop through time steps and write data
                t = timestep if not self.use_previous_data else 2 * timestep
                for idx, graph in enumerate(trajectory):
                    mesh = convert_to_meshio_vtu(graph, add_all_data=True)
                    point_data = mesh.point_data
                    cell_data = mesh.cell_data
                    writer.write_data(t, point_data=point_data, cell_data=cell_data)
                    t += timestep

        except Exception as e:
            logger.error(f"Error saving graph {idx} at epoch {self.current_epoch}: {e}")
        logger.info(
            f"Validation Trajectory {archive_filename.split('_')[-1]} saved at {save_dir}."
        )
        # The H5 archive is systematically created in cwd, we just need to move it
        shutil.move(
            src=os.path.join(
                os.getcwd(), os.path.split(f"{xdmf_filename.replace('xdmf', 'h5')}")[1]
            ),
            dst=f"{xdmf_filename.replace('xdmf', 'h5')}",
        )

    def _reset_validation_trajectory(self):
        self.current_val_trajectory += 1
        self.last_val_prediction = None
        self.last_previous_data_prediction = None

    def _make_prediction(self, batch, last_prediction, last_previous_data_prediction):
        batch = batch.clone()
        # Prepare the batch for the current step
        N = batch.x.shape[0]
        # reset history if graph size changed
        if last_prediction is not None and last_prediction.shape[0] != N:
            last_prediction = None
            last_previous_data_prediction = None
            
        if last_prediction is not None:
            # Update the batch with the last prediction
            batch.x[:, self.model.output_index_start : self.model.output_index_end] = (
                last_prediction.detach()
            )
            if self.use_previous_data:
                batch.x[:, self.previous_data_start : self.previous_data_end] = (
                    last_previous_data_prediction.detach()
                )
        mask = build_mask(self.param, batch)
        target = batch.y

        current_output = batch.x[
            :, self.model.output_index_start : self.model.output_index_end
        ]

        with torch.no_grad():
            _, _, predicted_outputs = self.model(batch)

        # Apply mask to predicted outputs and update the last prediction
        predicted_outputs[mask] = target[mask]
        last_prediction = predicted_outputs
        if self.use_previous_data:
            last_previous_data_prediction = predicted_outputs - current_output
        # add predic velocity to batch
        batch.x[:,0:3] = predicted_outputs
        return (
            batch,
            predicted_outputs,
            target,
            last_prediction,
            last_previous_data_prediction,
        )

    def validation_step(self, batch: Batch, batch_idx: int):
        batch = batch.to(self.device, non_blocking=True)
        # Determine if we need to reset the trajectory
        if batch.traj_index > self.current_val_trajectory:
            self._reset_validation_trajectory()
            self.step_counter = 0
        # Also reset the carry if num_nodes changes within a trajectory
        if self._last_val_num_nodes is not None and self._last_val_num_nodes != batch.x.shape[0]:
             self.last_val_prediction = None
             self.last_previous_data_prediction = None
 
        self._last_val_num_nodes = batch.x.shape[0]    
        (
            batch,
            predicted_outputs,
            target,
            self.last_val_prediction,
            self.last_previous_data_prediction,
        ) = self._make_prediction(
            batch, self.last_val_prediction, self.last_previous_data_prediction
        )

        if self.current_val_trajectory == 0:
            self.trajectory_to_save.append(batch)
        node_type = batch.x[:, self.model.node_type_index]

        #self.val_step_outputs.append(predicted_outputs.cpu())
        #self.val_step_targets.append(target.cpu())
        self.val_step_outputs.append(predicted_outputs.detach())  # rester sur GPU
        self.val_step_targets.append(target.detach())              # rester sur GPU
        val_loss = self.val_loss(
            target,
            predicted_outputs,
            node_type,
            masks=self.loss_masks,
        )
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # compute RMSE for the first step
        if self.step_counter == 0:
            squared_diff = (predicted_outputs - target) ** 2
            #rmse = torch.sqrt(squared_diff.mean()).detach().cpu()
            rmse = torch.sqrt(squared_diff.mean()).detach()  # rester sur GPU
            self.first_step_losses.append(rmse)
        self.step_counter += 1

    def _reset_validation_epoch_end(self):
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self._last_val_num_nodes = None
        self.current_val_trajectory = 0
        self.last_val_prediction = None
        self.last_previous_data_prediction = None
        self.trajectory_to_save.clear()
        self.step_counter = 0
        self.first_step_losses = []

    '''
    def on_validation_epoch_end(self):
        # n’exécuter la logique que sur le rank global 0
        if getattr(self.trainer, "is_global_zero", False) is False:
            return
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
            sync_dist=True,
        )

        # Compute RMSE for the first step
        if self.first_step_losses:
            mean_first_step_loss = torch.stack(self.first_step_losses).mean().item()
            self.log(
                "val_1step_rmse", mean_first_step_loss, on_epoch=True, prog_bar=True, sync_dist=True
            )

        # Save trajectory graphs
        save_dir = os.path.join("meshes", f"epoch_{self.current_epoch}")
        self._save_trajectory_to_xdmf(
            self.trajectory_to_save,
            save_dir,
            self._get_traj_savename(
                self.trajectory_to_save,
                self.current_val_trajectory,
                prefix=f"graph_epoch_{self.current_epoch}",
            ),
            timestep=self.timestep,
        )

        # Clear stored outputs
        self._reset_validation_epoch_end()
    '''
    def on_validation_epoch_end(self):
        # >>> Ne PAS sortir tôt sur les non-zero ranks si on utilise sync_dist=True <<<
        # (on veut que tous les ranks exécutent les self.log(..., sync_dist=True))
        device = self.device
        # 1) concat locales (par-rank)
        predicteds = torch.cat(self.val_step_outputs, dim=0) if self.val_step_outputs else None
        targets    = torch.cat(self.val_step_targets, dim=0)  if self.val_step_targets else None
        
        

        # 2) calc locales puis log avec sync_dist=True (Lightning fera la réduction)
        if predicteds is not None and targets is not None:
            # sécurité si jamais quelque chose est revenu sur CPU
            if predicteds.device.type != "cuda":
                predicteds = predicteds.to(device, non_blocking=True)
            if targets.device.type != "cuda":
                targets = targets.to(device, non_blocking=True)

            squared_diff = (predicteds - targets) ** 2
            all_rollout_rmse = torch.sqrt(squared_diff.mean())
            # ### IMPORTANT: laisser sync_dist=True mais appeler depuis TOUS les ranks
            self.log(
                "val_all_rollout_rmse",
                all_rollout_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        if self.first_step_losses:
            mean_first_step_loss = torch.stack(self.first_step_losses).mean()
            if mean_first_step_loss.device.type != "cuda":
                mean_first_step_loss= m.to(device, non_blocking=True)
            # ### idem, log sur tous les ranks
            self.log(
                "val_1step_rmse",
                mean_first_step_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # 3) Sauvegardes disque UNIQUEMENT sur rank 0 (I/O non distribuée)
        if getattr(self.trainer, "is_global_zero", False):
            save_dir = os.path.join("meshes", f"epoch_{self.current_epoch}")
            self._save_trajectory_to_xdmf(
                self.trajectory_to_save,
                save_dir,
                self._get_traj_savename(
                    self.trajectory_to_save,
                    self.current_val_trajectory,
                    prefix=f"graph_epoch_{self.current_epoch}",
                ),
                timestep=self.timestep,
            )

        # 4) reset buffers sur TOUS les ranks (pour éviter de trainer des tensors entre epochs)
        self._reset_validation_epoch_end()
        
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

    def _reset_prediction_trajectory(self):
        self.current_pred_trajectory += 1
        self.prediction_trajectory = []
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None

    def predict_step(self, batch: Batch):
        """
        Predict step: predict next step of the trajectory.
        If the next step is in the next trajectory, save the current trajectory
        to xdmf and reset the trajectory.
        """
        batch = batch.to(self.device, non_blocking=True)
        if batch.traj_index > self.current_pred_trajectory:
            # save
            self._save_trajectory_to_xdmf(
                self.prediction_trajectory,
                self.prediction_save_path,
                self._get_traj_savename(
                    self.prediction_trajectory, self.current_pred_trajectory
                ),
                timestep=self.timestep,
            )
            # reset
            self._reset_prediction_trajectory()
            self._last_pred_num_nodes = None

        # If graph size changed inside a trajectory, drop the carry
        if self._last_pred_num_nodes is not None and self._last_pred_num_nodes != batch.x.shape[0]:
            self.last_pred_prediction = None
            self.last_previous_data_pred_prediction = None

        # predict
        (
            batch,
            predicted_outputs,
            target,
            self.last_pred_prediction,
            self.last_previous_data_pred_prediction,
        ) = self._make_prediction(
            batch, self.last_pred_prediction, self.last_previous_data_pred_prediction
        )
        self._last_pred_num_nodes = batch.x.shape[0]
        self.prediction_trajectory.append(batch)

    def _reset_predict_epoch_end(self):
        self.prediction_trajectory.clear()
        self.last_pred_prediction = None
        self.last_previous_data_pred_prediction = None
        self.current_pred_trajectory = 0

    def on_predict_epoch_end(self):
        """
        Save last trajectory to xdmf and clear stored outputs.
        """
        self._save_trajectory_to_xdmf(
            self.prediction_trajectory,
            self.prediction_save_path,
            self._get_traj_savename(
                self.prediction_trajectory, self.current_pred_trajectory
            ),
            timestep=self.timestep,
        )

        # Clear stored outputs
        self._reset_predict_epoch_end()

    def on_save_checkpoint(self, checkpoint: dict):
        """
        Save the wandb run ID to the checkpoint.
        """
        if self.wandb_run_id is not None:
            checkpoint["wandb_run_id"] = self.wandb_run_id
        else:
            logger.warning("No wandb run ID found, skipping saving to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        """
        Load the wandb run ID from the checkpoint.
        """
        self.wandb_run_id = checkpoint.get("wandb_run_id", None)

    def _get_traj_savename(
        self, traj: list[Batch], traj_idx: int, prefix: str = "graph"
    ) -> str:
        """
        Get the name of the trajectory to save (id if provided in attributes, index otherwise).
        Args:
            traj (list[Batch]): List of Batch objects representing the trajectory.
            traj_idx (int): Index of the current trajectory.
            prefix (str): Prefix for the trajectory filename. (does not include trailing '_')
        Returns:
            str: The name of the trajectory to save (no extensions).
        """
        if hasattr(traj[0], "id") and traj[0].id[0] is not None:
            return f"{prefix}_{traj[0].id[0]}"
        else:
            return f"{prefix}_{traj_idx}"
