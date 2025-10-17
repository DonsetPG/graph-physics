from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("jraph")

import jax.numpy as jnp
from flax import nnx
import jraph
from jax import random

from named_features import LegacyIndexAdapter, XFeatureLayout

from graphphysics.utils.nodetype import NodeType
from jraphphysics.models.layers import Normalizer
from jraphphysics.models.simulator import Simulator


class DummyModel(nnx.Module):
    def __call__(self, graph):
        features = graph.nodes["features"]
        return features[:, 0:2]


@pytest.fixture()
def layout() -> XFeatureLayout:
    return XFeatureLayout(
        [
            ("velocity_x", 1),
            ("velocity_y", 1),
            ("pressure", 1),
            ("node_type", 1),
        ]
    )


@pytest.fixture()
def indices(layout: XFeatureLayout) -> dict[str, int]:
    adapter = LegacyIndexAdapter(layout, targets=["velocity_x", "velocity_y"], node_type_name="node_type")
    return adapter.as_dict()


@pytest.fixture()
def simulator(indices: dict[str, int]):
    rng = random.PRNGKey(0)
    model = DummyModel()
    return Simulator(
        node_input_size=3 + NodeType.SIZE,
        edge_input_size=0,
        output_size=2,
        feature_index_start=indices["feature_index_start"],
        feature_index_end=indices["feature_index_end"],
        output_index_start=indices["output_index_start"],
        output_index_end=indices["output_index_end"],
        node_type_index=indices["node_type_index"],
        model=model,
        rngs=nnx.Rngs({"params": rng}),
    )


@pytest.fixture()
def graph_tuple():
    num_nodes = 5
    node_features = jnp.array(
        [
            [1.0, 2.0, 3.0, 0],
            [4.0, 5.0, 6.0, 1],
            [7.0, 8.0, 9.0, 0],
            [10.0, 11.0, 12.0, 1],
            [13.0, 14.0, 15.0, 0],
        ],
        dtype=jnp.float32,
    )
    node_positions = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    nodes = {"features": node_features, "pos": node_positions}
    senders = jnp.array([0, 1, 2], dtype=jnp.int32)
    receivers = jnp.array([1, 2, 3], dtype=jnp.int32)
    globals_dict = {
        "target_features": jnp.array(
            [
                [0.1, 0.2],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.4, 0.5],
                [0.5, 0.6],
            ],
            dtype=jnp.float32,
        )
    }
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([num_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=globals_dict,
    )


def test_simulator_training(simulator: Simulator, graph_tuple: jraph.GraphsTuple):
    network_output, target_delta_normalized, outputs = simulator(graph_tuple, is_training=True)
    assert network_output.shape == (5, 2)
    assert target_delta_normalized.shape == (5, 2)
    assert outputs is None


def test_simulator_inference(simulator: Simulator, graph_tuple: jraph.GraphsTuple):
    network_output, target_delta_normalized, outputs = simulator(graph_tuple, is_training=False)
    assert network_output.shape == (5, 2)
    assert target_delta_normalized.shape == (5, 2)
    assert outputs is not None
    assert outputs.shape == (5, 2)


def test_one_hot_node_type(simulator: Simulator, graph_tuple: jraph.GraphsTuple):
    one_hot = simulator._get_one_hot_type(graph_tuple)
    assert one_hot.shape == (5, NodeType.SIZE)


def test_normalizers(simulator: Simulator):
    assert isinstance(simulator._node_normalizer, Normalizer)
    assert isinstance(simulator._output_normalizer, Normalizer)
