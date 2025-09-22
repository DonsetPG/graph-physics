import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data

from graphphysics.models.layers import Normalizer
from graphphysics.utils.nodetype import NodeType


class Simulator(nn.Module):
    """
    A simulator module that wraps a neural network model for graph data,
    handling normalization, forward passes, and checkpoint management.
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        feature_index_start: int,
        feature_index_end: int,
        output_index_start: int,
        output_index_end: int,
        node_type_index: int,
        model: nn.Module,
        model_type: str,
        device: torch.device,
        model_dir: str = "checkpoint/simulator.pth",
    ):
        """
        Initializes the Simulator module.

        Args:
            node_input_size (int): Size of node input features.
            edge_input_size (int): Size of edge input features.
            output_size (int): Size of the output/prediction from the network.
            feature_index_start (int): Start index of features in node features.
            feature_index_end (int): End index of features in node features.
            output_index_start (int): Start index of the target output in node features.
            output_index_end (int): End index of the target output in node features.
            node_type_index (int): Index of node type in node features.
            batch_size (int): Batch size for processing.
            model (nn.Module): The neural network model to be used.
            device (torch.device): The device to run the model on.
            model_dir (str, optional): Directory to save/load the model checkpoint.
            Defaults to "checkpoint/simulator.pth".
        """
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size if edge_input_size > 0 else None
        self.output_size = output_size

        self.feature_index_start = feature_index_start
        self.feature_index_end = feature_index_end
        self.node_type_index = node_type_index

        self.output_index_start = output_index_start
        self.output_index_end = output_index_end

        self.model_dir = model_dir
        self.model = model.to(device)
        self._output_normalizer = Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self._edge_normalizer = (
            Normalizer(size=edge_input_size, name="edge_normalizer", device=device)
            if self.edge_input_size is not None
            else None
        )

        self.model_type = model_type

        self.device = device

    def _get_pre_target(self, inputs: Data) -> torch.Tensor:
        """
        Extracts the previous target values from the input data.

        Args:
            inputs (Data): Input graph data containing node features.

        Returns:
            torch.Tensor: The previous target values extracted from node features.
        """
        return inputs.x[:, self.output_index_start : self.output_index_end]

    def _get_target_normalized(
        self, inputs: Data, is_training: bool = True
    ) -> torch.Tensor:
        """
        Computes the normalized target delta (difference between target and pre-target).

        Args:
            inputs (Data): Input graph data containing target values.
            is_training (bool, optional): Whether the model is in training mode. Defaults to True.

        Returns:
            torch.Tensor: The normalized target delta.
        """
        target = inputs.y
        pre_target = self._get_pre_target(inputs)
        target_delta = target - pre_target
        target_delta_normalized = self._output_normalizer(target_delta, is_training)

        return target_delta_normalized

    def _get_one_hot_type(self, inputs: Data) -> torch.Tensor:
        """
        Converts node types to one-hot encoded vectors.

        Args:
            inputs (Data): Input graph data containing node types.

        Returns:
            torch.Tensor: One-hot encoded node types.
        """
        node_type = inputs.x[:, self.node_type_index]
        return torch.nn.functional.one_hot(
            torch.squeeze(node_type.long()), NodeType.SIZE
        )

    def _build_node_features(
        self, inputs: Data, one_hot_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Builds the node features by concatenating selected features with one-hot encoded node types.

        Args:
            inputs (Data): Input graph data containing node features.
            one_hot_type (torch.Tensor): One-hot encoded node types.

        Returns:
            torch.Tensor: The concatenated node features.
        """
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]
        node_features = torch.cat([features, one_hot_type], dim=1)

        return node_features

    def _build_input_graph(
        self, inputs: Data, is_training: bool
    ) -> Tuple[Data, torch.Tensor]:
        """
        Builds the input graph for the model by normalizing features and target delta.

        Args:
            inputs (Data): Input graph data.
            is_training (bool): Whether the model is in training mode.

        Returns:
            Tuple[Data, torch.Tensor]: A tuple containing the processed input graph and normalized target delta.
        """
        target_delta_normalized = self._get_target_normalized(inputs, is_training)
        one_hot_type = self._get_one_hot_type(inputs)
        node_features = self._build_node_features(inputs, one_hot_type)

        node_features_normalized = self._node_normalizer(node_features, is_training)

        if self._edge_normalizer is not None:
            edge_attr = self._edge_normalizer(inputs.edge_attr, is_training)
        else:
            edge_attr = inputs.edge_attr

        graph = Data(
            x=node_features_normalized,
            pos=inputs.pos,
            edge_attr=edge_attr,
            edge_index=inputs.edge_index,
        )

        return graph, target_delta_normalized

    def build_outputs(self, inputs: Data, network_output: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the outputs by inverting normalization and adding the pre-target.

        Args:
            inputs (Data): Input graph data.
            network_output (torch.Tensor): The output from the network.

        Returns:
            torch.Tensor: The reconstructed outputs.
        """
        pre_target = self._get_pre_target(inputs)
        update = self._output_normalizer.inverse(network_output)
        return pre_target + update

    def forward(
        self, inputs: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Simulator module.

        Args:
            inputs (Data): Input graph data.

        Returns:
            Tuple containing:
                - network_output (torch.Tensor): The network's output.
                - target_delta_normalized (torch.Tensor): The normalized target delta.
                - outputs (torch.Tensor, optional): The reconstructed outputs (only during evaluation).
        """
        graph, target_delta_normalized = self._build_input_graph(
            inputs=inputs, is_training=self.training
        )
        if self.model_type == "bsms":
            network_output = self.model(graph, mesh_id=inputs.id[0])
        else:
            network_output = self.model(graph)

        if self.training:
            return network_output, target_delta_normalized, None
        else:
            outputs = self.build_outputs(inputs=inputs, network_output=network_output)
            return network_output, target_delta_normalized, outputs

    def freeze_all(self) -> None:
        """
        Freezes all parameters in the model to prevent them from being updated during training.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, ckpdir: Optional[str] = None) -> None:
        """
        Loads the model and normalizer states from a checkpoint file.

        Args:
            ckpdir (str, optional): Path to the checkpoint file. Defaults to self.model_dir.
        """
        if ckpdir is None:
            ckpdir = self.model_dir
        checkpoint = torch.load(ckpdir, map_location=self.device)
        self.load_state_dict(checkpoint["model"])

        normalizer_keys = ["_output_normalizer", "_node_normalizer", "_edge_normalizer"]
        for key in normalizer_keys:
            normalizer_state = checkpoint.get(key, {})
            normalizer = getattr(self, key, None)
            if normalizer and normalizer_state:
                for attr_name, value in normalizer_state.items():
                    setattr(normalizer, attr_name, value)

        logger.success(f"Simulator model loaded checkpoint {ckpdir}")

    def save_checkpoint(self, savedir: Optional[str] = None) -> None:
        """
        Saves the model and normalizer states to a checkpoint file.

        Args:
            savedir (str, optional): Path to save the checkpoint file. Defaults to self.model_dir.
        """
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model_state = self.state_dict()
        output_normalizer_state = self._output_normalizer.get_variable()
        node_normalizer_state = self._node_normalizer.get_variable()
        edge_normalizer_state = (
            self._edge_normalizer.get_variable() if self._edge_normalizer else None
        )

        to_save = {
            "model": model_state,
            "_output_normalizer": output_normalizer_state,
            "_node_normalizer": node_normalizer_state,
            "_edge_normalizer": edge_normalizer_state,
        }

        torch.save(to_save, savedir)
        logger.success(f"Simulator model saved checkpoint {savedir}")
