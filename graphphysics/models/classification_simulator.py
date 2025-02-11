import torch
import torch.nn as nn
from torch_geometric.data import Batch

from graphphysics.models.layers import Normalizer


class ClassificationSimulator(nn.Module):
    def __init__(
        self,
        node_input_size: int,
        output_size: int,
        feature_index_start: int,
        feature_index_end: int,
        output_index_start: int,
        output_index_end: int,
        node_type_index: int,
        model: nn.Module,
        device: torch.device,
        model_dir="checkpoint/simulator.pth",
    ):
        super(ClassificationSimulator, self).__init__()

        self.node_input_size = node_input_size
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

        self.device = device

    def _build_input_graph(self, inputs: Batch, is_training: bool) -> Batch:
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]
        node_features_list = [features]
        node_features = torch.cat(node_features_list, dim=1)
        node_features_normalized = self._node_normalizer(node_features, is_training)

        inputs.x = node_features_normalized

        return inputs

    def forward(self, inputs: Batch) -> torch.Tensor:
        graph = self._build_input_graph(inputs=inputs, is_training=self.training)
        network_output = self.model(graph)
        return network_output
