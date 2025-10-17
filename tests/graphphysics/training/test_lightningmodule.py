from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("lightning")

import torch
from torch_geometric.data import Batch

from named_features import NamedData, XFeatureLayout

from graphphysics.training.lightning_module import LightningModule, build_mask
from graphphysics.utils.nodetype import NodeType


def _make_layout() -> XFeatureLayout:
    return XFeatureLayout(
        [
            ("velocity_x", 1),
            ("velocity_y", 1),
            ("pressure", 1),
            ("node_type", 1),
        ]
    )


def _make_named_graph(num_nodes: int = 4) -> NamedData:
    layout = _make_layout()
    x = torch.randn(num_nodes, layout.feature_dim())
    x[:, layout.slc("node_type")] = torch.randint(0, NodeType.SIZE, (num_nodes, 1))
    edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))
    data = NamedData(x=x, edge_index=edge_index, x_layout=layout)
    data.y = torch.randn(num_nodes, 2)
    data.traj_index = 0
    return data


def test_build_mask_prefers_named_features() -> None:
    layout = _make_layout()
    graph = NamedData.from_data_list([_make_named_graph(), _make_named_graph()])
    param = {
        "named_features": {"node_type": "node_type"},
        "index": {"node_type_index": 3},
    }
    mask = build_mask(param, graph)
    assert mask.shape[0] == graph.num_nodes


class _DummySimulator(torch.nn.Module):
    def __init__(self, layout: XFeatureLayout):
        super().__init__()
        self.layout = layout
        self.x_layout = layout
        self.feature_names = ["velocity_x", "velocity_y", "pressure"]
        self.target_names = ["velocity_x", "velocity_y"]
        self.node_type_name = "node_type"
        self.node_type_index = layout.slc("node_type").start

    def forward(self, graph: Batch):  # type: ignore[override]
        batch_size = graph.num_nodes
        zeros = torch.zeros(batch_size, 2)
        return zeros, zeros, None

    def build_outputs(self, graph: Batch, network_output: torch.Tensor) -> torch.Tensor:
        return network_output

    def get_node_type(self, graph: Batch) -> torch.Tensor:
        tensor = graph.x_sel(self.node_type_name)
        if tensor.dim() == graph.x.dim():
            tensor = tensor.squeeze(-1)
        return tensor.reshape(-1)


def _mock_parameters(layout: XFeatureLayout) -> dict:
    return {
        "model": {
            "type": "epd",
            "message_passing_num": 1,
            "hidden_size": 8,
            "node_input_size": 3,
            "output_size": 2,
            "edge_input_size": 0,
        },
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 3,
            "output_index_start": 0,
            "output_index_end": 2,
            "node_type_index": layout.slc("node_type").start,
        },
        "named_features": {
            "x_layout": layout,
            "node_type": "node_type",
            "targets": ["velocity_x", "velocity_y"],
        },
    }


def test_training_step_with_named_layout(monkeypatch) -> None:
    layout = _make_layout()
    params = _mock_parameters(layout)

    dummy_loss = MagicMock(return_value=torch.tensor(0.0))
    dummy_loss_name = "l2"

    monkeypatch.setattr(
        "graphphysics.training.lightning_module.get_model",
        lambda param, only_processor=False: MagicMock(),
    )
    monkeypatch.setattr(
        "graphphysics.training.lightning_module.get_simulator",
        lambda param, model, device: _DummySimulator(layout),
    )
    monkeypatch.setattr(
        "graphphysics.training.lightning_module.get_loss",
        lambda param: (dummy_loss, dummy_loss_name),
    )
    monkeypatch.setattr(
        "graphphysics.training.lightning_module.get_gradient_method",
        lambda param: "least_squares",
    )
    monkeypatch.setattr(
        "graphphysics.training.lightning_module.CosineWarmupScheduler",
        lambda optimizer, warmup, max_iters: MagicMock(),
    )
    monkeypatch.setattr(
        "graphphysics.training.lightning_module.L2Loss",
        lambda: MagicMock(return_value=torch.tensor(0.0)),
    )

    module = LightningModule(
        parameters=params,
        learning_rate=1e-3,
        num_steps=10,
        warmup=1,
    )
    module.device = torch.device("cpu")

    batch = NamedData.from_data_list([_make_named_graph()])
    loss = module.training_step(batch)
    assert torch.is_tensor(loss)
    dummy_loss.assert_called_once()
