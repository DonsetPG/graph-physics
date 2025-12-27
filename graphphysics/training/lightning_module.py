import os
import shutil
from typing import List

import lightning as L
import meshio
import torch
from loguru import logger
from torch_geometric.data import Batch

from graphphysics.diagnostics.entropy_production.graph_ep import (
    EPEstimationConfig,
    estimate_graph_ep_all_orders,
)


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

        # Compute RMSE for the first step
        if self.first_step_losses:
            mean_first_step_loss = torch.stack(self.first_step_losses).mean().item()
            self.log(
                "val_1step_rmse", mean_first_step_loss, on_epoch=True, prog_bar=True
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

    
    def on_fit_end(self):
        """Compute and log entropy production + oversmoothing after training.

        Controlled by the training parameters JSON:

        parameters:
          diagnostics:
            entropy_production:
              enabled: true
              num_trajectories: 32
              proj_dim: 8
              orders: ["node", "edge", "1hop", "2hop"]
              max_nodes_per_traj: 1024
              max_edges_per_traj: 2048
              estimator: "MaxEnt"        # or "MTUR"
              optimizer: "Adam"
              max_iter: 500
              observable: "antisym"      # or "non_antisym"
              clip_objective: true
              share_trajectories_across_layers: true
              offload_to_cpu: null

        Notes:
        - Runs only on global rank 0.
        - Uses the *first* batch from the first val dataloader as the reference graph.
        - This is intended as a diagnostic; adjust sampling knobs to control runtime.
        """
        # --- rank guard (important for DDP) ---
        if not getattr(self.trainer, "is_global_zero", True):
            return

        params = getattr(self.hparams, "parameters", None)
        if not isinstance(params, dict):
            return
        ep_cfg = (
            params.get("diagnostics", {}).get("entropy_production", None)
            if isinstance(params.get("diagnostics", {}), dict)
            else None
        )
        if ep_cfg is None:
            ep_cfg = params.get("entropy_production", None)

        if not isinstance(ep_cfg, dict) or not ep_cfg.get("enabled", False):
            return

        # --- get a handle to the first val dataloader (we will sample from it below) ---
        try:
            # In Lightning, this is typically a list (one per dataloader)
            val_dls = getattr(self.trainer, "val_dataloaders", None)
            if isinstance(val_dls, list) and len(val_dls) > 0:
                dl0 = val_dls[0]
            else:
                dl0 = val_dls
        except Exception as e:
            logger.warning(f"[entropy_production] Could not access val_dataloader: {e}")
            return
        if dl0 is None:
            logger.warning("[entropy_production] No val_dataloader available; skipping diagnostics.")
            return

        # Underlying processor model (EncodeProcessDecode / EncodeTransformDecode / etc.)
        processor = getattr(self.model, "model", None)
        if processor is None:
            logger.warning("[entropy_production] Simulator has no attribute `.model` (processor). Skipping.")
            return

        # --- configure EP estimation ---
        cfg = EPEstimationConfig(
            num_trajectories=int(ep_cfg.get("num_trajectories", 32)),
            max_nodes_per_traj=ep_cfg.get("max_nodes_per_traj", 1024),
            max_edges_per_traj=ep_cfg.get("max_edges_per_traj", 2048),
            seed=int(ep_cfg.get("seed", 0)),
            proj_dim=int(ep_cfg.get("proj_dim", 8)),
            standardize=bool(ep_cfg.get("standardize", True)),
            observable=str(ep_cfg.get("observable", "antisym")),
            estimator=str(ep_cfg.get("estimator", "MaxEnt")),
            optimizer=str(ep_cfg.get("optimizer", "Adam")),
            max_iter=int(ep_cfg.get("max_iter", 500)),
            verbose=int(ep_cfg.get("verbose", 0)),
            clip_objective=bool(ep_cfg.get("clip_objective", True)),
            optimizer_kwargs=ep_cfg.get("optimizer_kwargs", None),
            val_fraction=float(ep_cfg.get("val_fraction", 0.1)),
            test_fraction=float(ep_cfg.get("test_fraction", 0.1)),
            offload_to_cpu=ep_cfg.get("offload_to_cpu", None),
            share_trajectories_across_layers=bool(ep_cfg.get("share_trajectories_across_layers", True)),
            noise_std=float(ep_cfg.get("noise_std", 0.0)),
            layer_modules_attr=ep_cfg.get("layer_modules_attr", None),
        )

        orders = tuple(ep_cfg.get("orders", ["node", "edge", "1hop", "2hop"]))
        # Safety: filter invalid order strings
        allowed = {"node", "edge", "1hop", "2hop"}
        orders = tuple([o for o in orders if o in allowed])

        num_batches = int(ep_cfg.get("num_batches", 1))
        if num_batches <= 0:
            num_batches = 1

        results_list = []
        it = iter(dl0)
        for bi in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            except Exception as e:
                logger.warning(f"[entropy_production] Failed to fetch batch {bi} from val_dataloader: {e}")
                break

            # Build normalized graph inputs as expected by the Simulator
            batch = batch.to(self.device, non_blocking=True)
            self.model = self.model.to(self.device, non_blocking=True)
            graph, _ = self.model._build_input_graph(inputs=batch, is_training=False)

            try:
                res_b = estimate_graph_ep_all_orders(
                    model=processor,
                    graph=graph,
                    cfg=cfg,
                    orders=orders,
                    device=self.device,
                )
                results_list.append(res_b)
            except Exception as e:
                logger.exception(f"[entropy_production] EP estimation failed on batch {bi}: {e}")
                break

        if len(results_list) == 0:
            return

        # Average results across batches (elementwise for per-layer arrays)
        if len(results_list) == 1:
            results = results_list[0]
        else:
            import numpy as _np

            results = {}
            base = results_list[0]
            for order, base_res in base.items():
                res_avg = dict(base_res)

                # Per-layer EP
                per_layer = _np.mean([r[order]["per_layer"] for r in results_list], axis=0).tolist()
                res_avg["per_layer"] = [float(v) for v in per_layer]
                res_avg["total"] = float(_np.sum(per_layer))

                # Oversmoothing arrays (may be missing if order config differs)
                if isinstance(base_res.get("V_per_layer", None), list):
                    V_per_layer = _np.mean([r[order]["V_per_layer"] for r in results_list], axis=0).tolist()
                    res_avg["V_per_layer"] = [float(v) for v in V_per_layer]
                    res_avg["V_final"] = float(res_avg["V_per_layer"][-1]) if len(res_avg["V_per_layer"]) else res_avg.get("V_final")
                if isinstance(base_res.get("V_det_per_layer", None), list):
                    V_det_per_layer = _np.mean([r[order]["V_det_per_layer"] for r in results_list], axis=0).tolist()
                    res_avg["V_det_per_layer"] = [float(v) for v in V_det_per_layer]
                    res_avg["V_det_final"] = float(res_avg["V_det_per_layer"][-1]) if len(res_avg["V_det_per_layer"]) else res_avg.get("V_det_final")

                # Scalar oversmoothing (average across graphs)
                for k in ["V0", "V_input", "V_final", "V_det_final"]:
                    if k in base_res and base_res[k] is not None:
                        try:
                            res_avg[k] = float(_np.mean([r[order][k] for r in results_list]))
                        except Exception:
                            pass

                res_avg["num_batches"] = int(len(results_list))
                results[order] = res_avg

        # --- persist raw results (arrays) to JSON for post-hoc analysis ---
        try:
            import json as _json
            import os as _os

            out_dir = None
            # Prefer logger log_dir if available
            if getattr(self.trainer, "logger", None) is not None and hasattr(self.trainer.logger, "log_dir"):
                out_dir = getattr(self.trainer.logger, "log_dir", None)
            if out_dir is None:
                out_dir = getattr(self.trainer, "default_root_dir", None)
            if out_dir is None:
                out_dir = "."
            _os.makedirs(out_dir, exist_ok=True)
            out_path = _os.path.join(out_dir, "entropy_production_results.json")
            with open(out_path, "w") as f:
                _json.dump({"config": ep_cfg, "results": results}, f, indent=2)
            logger.info(f"[entropy_production] Wrote diagnostics to {out_path}")
        except Exception as e:
            logger.warning(f"[entropy_production] Failed to write results JSON: {e}")

        # --- flatten metrics for logging ---
        metrics = {}
        # Oversmoothing metrics are order-independent; use the first order (if any)
        if len(results) > 0:
            any_order = next(iter(results.keys()))
            V0 = results[any_order].get("V0", None)
            V_input = results[any_order].get("V_input", V0)
            V_per_layer = results[any_order].get("V_per_layer", None)
            V_final = results[any_order].get("V_final", None)
            V_det_per_layer = results[any_order].get("V_det_per_layer", None)
            V_det_final = results[any_order].get("V_det_final", None)
            if V0 is not None:
                metrics["oversmoothing/V0"] = float(V0)
            if V_input is not None:
                metrics["oversmoothing/V_input"] = float(V_input)
            if V_final is not None:
                metrics["oversmoothing/V_final"] = float(V_final)
            if V_det_final is not None:
                metrics["oversmoothing_det/V_final"] = float(V_det_final)
            if isinstance(V_per_layer, list):
                for li, v in enumerate(V_per_layer):
                    metrics[f"oversmoothing/V_layer_{li}"] = float(v)
                    if V_input not in (None, 0.0):
                        metrics[f"oversmoothing/V_ratio_layer_{li}"] = float(v) / float(V_input)

            if isinstance(V_det_per_layer, list):
                for li, v in enumerate(V_det_per_layer):
                    metrics[f"oversmoothing_det/V_layer_{li}"] = float(v)
                    if V_input not in (None, 0.0):
                        metrics[f"oversmoothing_det/V_ratio_layer_{li}"] = float(v) / float(V_input)

        for order, res in results.items():
            total = res.get("total", None)
            per_layer = res.get("per_layer", None)
            if total is not None:
                metrics[f"entropy_production/{order}/total"] = float(total)
            if isinstance(per_layer, list):
                for li, ep_li in enumerate(per_layer):
                    metrics[f"entropy_production/{order}/layer_{li}"] = float(ep_li)

        # --- log to Lightning logger (wandb/tensorboard/etc) ---
        step = int(getattr(self.trainer, "global_step", 0))
        try:
            if getattr(self.trainer, "logger", None) is not None:
                self.trainer.logger.log_metrics(metrics, step=step)
                self.trainer.logger.save()
        except Exception as e:
            logger.warning(f"[entropy_production] Failed to log metrics via trainer.logger: {e}")

        # Also print a concise summary
        try:
            msg_parts = []
            for order in results.keys():
                msg_parts.append(f"{order}: total={results[order]['total']:.3f}")
            logger.info("[entropy_production] " + " | ".join(msg_parts))
        except Exception:
            pass

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
