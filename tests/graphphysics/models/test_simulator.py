from __future__ import annotations

import pytest

from named_features import NamedData, XFeatureLayout

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
import torch.nn as nn
from torch_geometric.data import Data

from graphphysics.models.simulator import Simulator
from graphphysics.utils.nodetype import NodeType


class MockModel(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, input_graph: Data) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros(
            input_graph.x.shape[0], self.output_size, device=input_graph.x.device
        )


@pytest.fixture()
def layout() -> XFeatureLayout:
    # velocity_x, velocity_y, pressure, temperature, node_type
    return XFeatureLayout(
        [
            ("velocity_x", 1),
            ("velocity_y", 1),
            ("pressure", 1),
            ("temperature", 1),
            ("node_type", 1),
        ]
    )


@pytest.fixture()
def simulator(layout: XFeatureLayout) -> Simulator:
    output_size = 2
    feature_names = ["velocity_x", "velocity_y", "pressure", "temperature"]
    target_names = ["velocity_x", "velocity_y"]
    model = MockModel(output_size)

    return Simulator(
        node_input_size=len(feature_names) + NodeType.SIZE,
        edge_input_size=0,
        output_size=output_size,
        feature_index_start=0,
        feature_index_end=4,
        output_index_start=0,
        output_index_end=2,
        node_type_index=4,
        model=model,
        device=torch.device("cpu"),
        x_layout=layout,
        feature_names=feature_names,
        target_names=target_names,
        node_type_name="node_type",
    )


@pytest.fixture()
def sample_data(layout: XFeatureLayout) -> NamedData:
    num_nodes = 6
    x = torch.randn(num_nodes, layout.feature_dim())
    x[:, layout.slc("node_type")] = torch.randint(0, NodeType.SIZE, (num_nodes, 1))
    y = torch.randn(num_nodes, 2)
    edge_index = torch.randint(0, num_nodes, (2, 12))
    data = NamedData(x=x, edge_index=edge_index, y=y, x_layout=layout)
    data.pos = torch.randn(num_nodes, 3)
    return data


def test_selectors_use_named_layout(
    simulator: Simulator, sample_data: NamedData
) -> None:
    selected = simulator.select_targets_from_x(sample_data)
    torch.testing.assert_close(
        selected, sample_data.x_sel(["velocity_x", "velocity_y"])
    )


def test_assignment_updates_named_features(
    simulator: Simulator, sample_data: NamedData
) -> None:
    new_values = torch.ones(sample_data.num_nodes, 2)
    simulator.assign_targets_to_x(sample_data, new_values)
    torch.testing.assert_close(
        sample_data.x_sel(["velocity_x", "velocity_y"]), new_values
    )


def test_forward_training(simulator: Simulator, sample_data: NamedData) -> None:
    simulator.train()
    network_output, target_delta_normalized, outputs = simulator(sample_data)
    assert outputs is None
    assert network_output.shape == (sample_data.num_nodes, 2)
    assert target_delta_normalized.shape == (sample_data.num_nodes, 2)


def test_forward_evaluation(simulator: Simulator, sample_data: NamedData) -> None:
    simulator.eval()
    network_output, target_delta_normalized, outputs = simulator(sample_data)
    assert outputs is not None
    assert outputs.shape == (sample_data.num_nodes, 2)
    assert network_output.shape == (sample_data.num_nodes, 2)
    assert target_delta_normalized.shape == (sample_data.num_nodes, 2)


def test_one_hot_type_uses_named_lookup(
    simulator: Simulator, sample_data: NamedData
) -> None:
    node_type = simulator.get_node_type(sample_data)
    assert node_type.shape == (sample_data.num_nodes,)
    assert (node_type >= 0).all()


def test_build_outputs_preserves_named_targets(
    simulator: Simulator, sample_data: NamedData
) -> None:
    simulator.eval()
    network_output = torch.randn(sample_data.num_nodes, 2)
    outputs = simulator.build_outputs(sample_data, network_output)
    assert outputs.shape == (sample_data.num_nodes, 2)
