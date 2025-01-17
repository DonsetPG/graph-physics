import os
import torch.nn as nn

import lightning as L
import torch
from loguru import logger
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix, f1_score

from graphphysics.utils.pyvista_mesh import convert_to_pyvista_mesh
from graphphysics.utils.scheduler import CosineWarmupScheduler
from graphphysics.models.classification_model import ClassificationModel
from graphphysics.models.classification_simulator import ClassificationSimulator

class LightningModuleClassification(L.LightningModule):
    def __init__(
        self,
        parameters: dict,
        learning_rate: float,
        num_steps: int,
        warmup: int,
    ):
        """
        Initializes the LightningModuleClassification.

        Args:
            parameters (Dict[str, Any]): Configuration parameters for the model and simulator.
            learning_rate (float): Initial learning rate for the optimizer.
            num_steps (int): Total number of training steps.
            warmup (int): Number of warmup steps for the learning rate scheduler.
            only_processor (bool, optional): Whether to use only the processor part of the model.
                Defaults to False.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
        """
        super().__init__()
        self.save_hyperparameters()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.param = parameters

        processor = ClassificationModel(
            message_passing_num=self.param["model"]["message_passing_num"],
            node_input_size=self.param["model"]["node_input_size"],
            edge_input_size=self.param["model"]["edge_input_size"],
            output_size=self.param["model"]["output_size"],
            hidden_size=self.param["model"]["hidden_size"],   
        )

        print(processor)

        self.model = ClassificationSimulator(
            node_input_size=self.param["model"]["node_input_size"],
            edge_input_size=self.param["model"]["edge_input_size"],
            output_size=self.param["model"]["output_size"],
            feature_index_start=self.param["index"]["feature_index_start"],
            feature_index_end=self.param["index"]["feature_index_end"],
            output_index_start=self.param["index"]["output_index_start"],
            output_index_end=self.param["index"]["output_index_end"],
            node_type_index=self.param["index"]["node_type_index"],
            batch_size=self.param["training"]["batch_size"],
            model=processor,
            device=device,
        )

        self.loss = nn.BCELoss() 

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.warmup = warmup

        self.val_step_outputs = []
        self.val_step_targets = []


    def forward(self, graph: Batch): 
        return self.model(graph)

    def training_step(self, batch: Batch): 
        pred = self.model(batch).reshape(-1)
        target = batch.y 
        loss = self.loss(pred, target) 
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):  

        with torch.no_grad():
            pred = self.model(batch).reshape(-1)
        target = batch.y    
        self.val_step_outputs.append(pred.cpu())
        self.val_step_targets.append(target.cpu())

        val_loss = self.loss(pred, batch.y)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Concatenate outputs and targets
        predicteds = torch.cat(self.val_step_outputs, dim=0)
        targets = torch.cat(self.val_step_targets, dim=0)

        # Compute confusion matrix and F1 score
        conf_matrix = confusion_matrix(targets.numpy(), predicteds.numpy())
        f1 = f1_score(targets.numpy(), predicteds.numpy(), average='weighted')

        self.log("val_conf_matrix", torch.tensor(conf_matrix), on_step=False ,on_epoch=True, prog_bar=True)
        self.log("val_f1_score", f1, on_step=False, on_epoch=True, prog_bar=True)

        # Clear stored outputs
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

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
