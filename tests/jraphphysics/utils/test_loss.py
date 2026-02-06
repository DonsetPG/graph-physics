import jax.numpy as jnp
import jraph

from graphphysics.utils.nodetype import NodeType
from jraphphysics.utils.loss import (
    ConvectionL2Loss,
    CosineLoss,
    DivergenceL1Loss,
    DivergenceL1SmoothLoss,
    DivergenceL2Loss,
    GradientL2Loss,
    L1SmoothLoss,
    L2Loss,
    MultiLoss,
)


def _build_graph():
    nodes = {
        "features": jnp.array(
            [
                [0.0, 0.0, float(NodeType.NORMAL)],
                [0.0, 0.0, float(NodeType.NORMAL)],
                [0.0, 0.0, float(NodeType.NORMAL)],
                [0.0, 0.0, float(NodeType.NORMAL)],
            ],
            dtype=jnp.float32,
        ),
        "pos": jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
    }
    senders = jnp.array([0, 1, 2, 3, 0, 2], dtype=jnp.int32)
    receivers = jnp.array([1, 3, 0, 2, 2, 3], dtype=jnp.int32)
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([senders.shape[0]], dtype=jnp.int32),
        globals={},
    )


def test_losses_return_scalar():
    graph = _build_graph()
    node_type = graph.nodes["features"][:, 2].astype(jnp.int32)
    masks = [NodeType.NORMAL]
    target = jnp.array(
        [[1.0, 1.0], [2.0, -2.0], [2.0, 3.0], [1.0, -4.0]],
        dtype=jnp.float32,
    )
    network_output = jnp.array(
        [[1.1, 0.9], [1.8, -2.1], [2.1, 2.8], [1.2, -3.9]],
        dtype=jnp.float32,
    )

    losses = [
        L2Loss(),
        CosineLoss(),
        L1SmoothLoss(),
        GradientL2Loss(),
        ConvectionL2Loss(),
        DivergenceL2Loss(),
        DivergenceL1Loss(),
        DivergenceL1SmoothLoss(),
    ]

    for loss_fn in losses:
        value = loss_fn(
            graph=graph,
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=masks,
            network_output_physical=network_output,
            target_physical=target,
            gradient_method="finite_diff",
        )
        assert value.shape == ()
        assert jnp.isfinite(value)


def test_multiloss_returns_scalar():
    graph = _build_graph()
    node_type = graph.nodes["features"][:, 2].astype(jnp.int32)
    masks = [NodeType.NORMAL]
    target = jnp.array(
        [[1.0, 1.0], [2.0, -2.0], [2.0, 3.0], [1.0, -4.0]],
        dtype=jnp.float32,
    )
    network_output = jnp.array(
        [[1.1, 0.9], [1.8, -2.1], [2.1, 2.8], [1.2, -3.9]],
        dtype=jnp.float32,
    )

    loss = MultiLoss(
        losses=[L2Loss(), GradientL2Loss(), DivergenceL2Loss()],
        weights=[0.5, 0.3, 0.2],
    )
    value = loss(
        graph=graph,
        target=target,
        network_output=network_output,
        node_type=node_type,
        masks=masks,
        network_output_physical=network_output,
        target_physical=target,
        gradient_method="finite_diff",
    )
    assert value.shape == ()
    assert jnp.isfinite(value)
