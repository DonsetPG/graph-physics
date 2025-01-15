import os
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data


from graphphysics.models.layers import Normalizer
from graphphysics.utils.nodetype import NodeType

class ClassificationSimulator(nn.Module):
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
        batch_size: int,
        model,
        device,
        model_dir="checkpoint/simulator.pth",
    ):
        super(ClassificationSimulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
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
        self._edge_normalizer = Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )

        self.device = device
        self.batch_size = batch_size



    def _build_input_graph(self, inputs: Data, is_training: bool) -> Data:
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]
        node_type = inputs.x[:, self.node_type_index]
        one_hot_type = torch.nn.functional.one_hot(
            torch.squeeze(node_type.long()), NodeType.SIZE
        )

        node_features = torch.cat([features, one_hot_type], dim=1)

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

        return graph

    def forward(self, inputs: Data):
        graph = self._build_input_graph(inputs=inputs, is_training=self.training)
        network_output = self.model(graph)
        return network_output
    
    def freeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, ckpdir: Optional[str] = None) -> None:
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