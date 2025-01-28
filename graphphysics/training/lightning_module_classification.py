import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch_geometric.data import Batch

import wandb
from graphphysics.models.classification_model import (
    ClassificationModel,
    ClassificationPointNetP2,
    ClassificationPointTransformer,
    PointNetClassifier,
)
from graphphysics.models.classification_simulator import ClassificationSimulator
from graphphysics.utils.scheduler import CosineWarmupScheduler


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

        model_type = parameters["model"]["type"]

        if model_type == "epd":
            processor = ClassificationModel(
                message_passing_num=parameters["model"]["message_passing_num"],
                node_input_size=parameters["model"]["node_input_size"],
                edge_input_size=parameters["model"]["edge_input_size"],
                output_size=parameters["model"]["output_size"],
                hidden_size=parameters["model"]["hidden_size"],
            )
        elif model_type == "pn":
            processor = PointNetClassifier(
                node_input_size=parameters["model"]["node_input_size"],
                hidden_layers=parameters["model"]["hidden_layers"],
                hidden_size=parameters["model"]["hidden_size"],
                output_size=parameters["model"]["output_size"],
            )
        elif model_type == "pn2":
            processor = ClassificationPointNetP2(
                node_input_size=parameters["model"]["node_input_size"],
                dim_model=parameters["model"]["dim_model"],
                output_size=parameters["model"]["output_size"],
                num_neighbors=parameters["dataset"]["number_of_connections"],
            )
        elif model_type == "pt":
            processor = ClassificationPointTransformer(
                in_channels=parameters["model"]["node_input_size"],
                dim_model=parameters["model"]["dim_model"],
                out_channels=parameters["model"]["output_size"],
                num_neighbors=parameters["dataset"]["number_of_connections"],
            )

        else:
            raise ValueError(f"Model type '{model_type}' not supported.")

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
            model=processor,
            model_type=self.param["model"]["type"],
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
        pred = self.model(batch).reshape(-1).to(torch.float32)
        target = batch.y.to(torch.float32)

        loss = self.loss(pred, target)
        self.log("Training Loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):

        with torch.no_grad():
            pred = self.model(batch).reshape(-1).to(torch.float32)
        target = batch.y.to(torch.float32)
        self.val_step_outputs.append(pred.cpu())
        self.val_step_targets.append(target.cpu())

        pred = pred.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        val_loss = self.loss(pred, target)

        self.log(
            "Validation Loss", val_loss, on_step=True, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        # Convert outputs to numpy arrays
        predicteds = np.array(self.val_step_outputs)
        targets = np.array(self.val_step_targets)

        # Get the predicted class by taking the argmax of the softmax output
        predicted_classes = np.argmax(predicteds, axis=1)
        target_classes = np.argmax(targets, axis=1)

        # Create a histogram with two colors for each class
        plt.figure(figsize=(10, 6))
        plt.hist(
            [
                predicteds[:, 0][targets[:, 0] == 0],
                predicteds[:, 0][targets[:, 0] == 1],
            ],
            bins=20,
            range=(0, 1),
            color=["blue", "orange"],
            alpha=0.7,
            label=["Class 0", "Class 1"],
        )
        plt.ylabel("Frequency")
        plt.xlabel("Predicted Values")
        plt.title("Validation Predictions Distribution")
        plt.legend(loc="upper right")

        # Save the plot as an image
        plt.savefig("validation_histogram_plot.png")

        # Log the image to wandb
        wandb.log(
            {"Validation Histogram Plot": wandb.Image("validation_histogram_plot.png")}
        )

        # Close the plot to avoid memory issues
        plt.close()

        # Compute confusion matrix and F1 score
        conf_matrix = confusion_matrix(target_classes, predicted_classes, labels=[0, 1])
        f1 = f1_score(target_classes, predicted_classes, average="weighted")

        self.log_dict(
            {
                "Aneurysm True ValCM[0,0]/pred": conf_matrix[0][0],
                "Aneurysm True ValCM[0,0]/Target": 66,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "Vessel False ValCM[0,1]/pred": conf_matrix[0][1],
                "Vessel False ValCM[0,1]/Target": 0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "Aneurysm False ValCM[1,0]/pred": conf_matrix[1][0],
                "Aneurysm False ValCM[1,0]/Target": 0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "Vessel True ValCM[1,1]/pred": conf_matrix[1][1],
                "Vessel True ValCM[1,1]/Target": 338,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("F1 Score Validation", f1, on_step=False, on_epoch=True, prog_bar=True)

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
