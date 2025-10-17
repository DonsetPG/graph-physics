from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
import torch.nn as nn

from named_features import LegacyIndexAdapter, NamedData, XFeatureLayout

from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.models.simulator import Simulator
from graphphysics.utils.nodetype import NodeType


class DummyModel(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, graph: NamedData) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros(graph.num_nodes, self.output_size, device=graph.x.device)


def _make_layout() -> XFeatureLayout:
    return XFeatureLayout(
        [
            ("velocity_x", 1),
            ("velocity_y", 1),
            ("mesh_pos_x", 1),
            ("mesh_pos_y", 1),
            ("node_type", 1),
        ]
    )


def _make_graph(layout: XFeatureLayout) -> NamedData:
    num_nodes = 4
    x = torch.randn(num_nodes, layout.feature_dim())
    x[:, layout.slc("node_type")] = torch.randint(0, NodeType.SIZE, (num_nodes, 1))
    edge_index = torch.randint(0, num_nodes, (2, 6))
    graph = NamedData(x=x, edge_index=edge_index, x_layout=layout)
    graph.y = torch.randn(num_nodes, 2)
    graph.pos = torch.randn(num_nodes, 3)
    graph.traj_index = torch.zeros(num_nodes, dtype=torch.long)
    return graph


def test_end_to_end_named_pipeline():
    layout = _make_layout()
    graphs = [_make_graph(layout) for _ in range(2)]

    preprocess = build_preprocessing(
        noise_parameters={
            "noise_features": ["velocity_x", "velocity_y"],
            "noise_scale": 0.0,
            "node_type_feature": "node_type",
        },
        world_pos_parameters={"use": False},
        add_edges_features=False,
        x_layout=layout,
    )

    processed_graphs = [preprocess(graph.clone()) for graph in graphs]
    batch = NamedData.from_data_list(processed_graphs)

    adapter = LegacyIndexAdapter(
        layout, targets=["velocity_x", "velocity_y"], node_type_name="node_type"
    )
    indices = adapter.as_dict()

    simulator = Simulator(
        node_input_size=4 + NodeType.SIZE,
        edge_input_size=0,
        output_size=2,
        feature_index_start=indices["feature_index_start"],
        feature_index_end=indices["feature_index_end"],
        output_index_start=indices["output_index_start"],
        output_index_end=indices["output_index_end"],
        node_type_index=indices["node_type_index"],
        model=DummyModel(output_size=2),
        device=torch.device("cpu"),
        x_layout=layout,
        feature_names=["velocity_x", "velocity_y", "mesh_pos_x", "mesh_pos_y"],
        target_names=["velocity_x", "velocity_y"],
        node_type_name="node_type",
    )

    simulator.eval()
    output, target_delta_normalized, predictions = simulator(batch)
    assert output.shape == (batch.num_nodes, 2)
    assert target_delta_normalized.shape == (batch.num_nodes, 2)
    assert predictions is not None
    assert predictions.shape == (batch.num_nodes, 2)
