from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp

from graphphysics.utils.nodetype import NodeType
from jraphphysics.training.parse_parameters import (
    get_gradient_method,
    get_loss,
    get_model,
    get_simulator,
)


@dataclass
class LightningModule:
    """
    Lightweight compatibility shim for graphphysics-style API.
    """

    parameters: dict
    learning_rate: float
    num_steps: int
    warmup: int
    trajectory_length: int = 599
    timestep: float = 1.0
    only_processor: bool = False
    masks: list[NodeType] = (NodeType.NORMAL, NodeType.OUTFLOW)
    use_previous_data: bool = False
    previous_data_start: int | None = None
    previous_data_end: int | None = None
    prediction_save_path: str = "predictions"

    def __post_init__(self):
        self.processor = get_model(self.parameters)
        self.model = get_simulator(self.parameters, self.processor)
        self.loss, self.loss_name = get_loss(self.parameters)
        self.gradient_method = get_gradient_method(self.parameters)
        self.wandb_run_id = None

    def forward(self, graph):
        return self.model(graph)

    def training_step(self, graph):
        network_output, target_delta_normalized, _ = self.model(graph, is_training=True)
        node_type = graph.nodes["features"][:, self.model.node_type_index].astype(jnp.int32)

        network_output_physical = self.model._build_outputs(graph, network_output)
        target_physical = self.model._build_outputs(graph, target_delta_normalized)
        return self.loss(
            graph=graph,
            target=target_delta_normalized,
            network_output=network_output,
            node_type=node_type,
            masks=list(self.masks),
            network_output_physical=network_output_physical,
            target_physical=target_physical,
            gradient_method=self.gradient_method,
        )

    def validation_step(self, graph):
        return self.training_step(graph)

    def predict_step(self, graph):
        _, _, outputs = self.model(graph, is_training=False)
        return outputs
