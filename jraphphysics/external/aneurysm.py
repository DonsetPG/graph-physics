import jax.numpy as jnp
import jraph

from jraphphysics.utils.nodetype import NodeType


def aneurysm_node_type(graph: jraph.GraphsTuple) -> jnp.ndarray:
    v_x = graph.nodes["features"][:, 0]
    wall_inputs = graph.nodes["features"][:, 3]
    node_type = jnp.zeros_like(v_x)

    wall_mask = wall_inputs == 1.0
    inflow_mask = jnp.logical_and(
        graph.nodes["pos"][:, 1] == 0.0, graph.nodes["pos"][:, 0] <= 0
    )
    outflow_mask = jnp.logical_and(
        graph.nodes["pos"][:, 1] == 0.0, graph.nodes["pos"][:, 0] >= 0
    )

    node_type = jnp.where(wall_mask, int(NodeType.WALL_BOUNDARY), node_type)
    node_type = jnp.where(inflow_mask, int(NodeType.INFLOW), node_type)
    node_type = jnp.where(outflow_mask, int(NodeType.OUTFLOW), node_type)
    return node_type


def build_features(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    node_type = aneurysm_node_type(graph)

    current_velocity = graph.nodes["features"][:, 0:3]
    target_velocity = graph.globals["target_features"][:, 0:3]

    previous_data = graph.globals.get("previous_data", None) if graph.globals else None
    if previous_data is not None and "Vitesse" in previous_data:
        previous_velocity = previous_data["Vitesse"]
    else:
        previous_velocity = current_velocity

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    not_inflow_mask = node_type != int(NodeType.INFLOW)
    next_acceleration = jnp.where(
        not_inflow_mask[:, None],
        jnp.zeros_like(next_acceleration),
        next_acceleration,
    )

    mean_next_accel = jnp.full(
        (node_type.shape[0], 1), jnp.mean(next_acceleration), dtype=current_velocity.dtype
    )
    min_next_accel = jnp.full(
        (node_type.shape[0], 1), jnp.min(next_acceleration), dtype=current_velocity.dtype
    )
    max_next_accel = jnp.full(
        (node_type.shape[0], 1), jnp.max(next_acceleration), dtype=current_velocity.dtype
    )

    features = jnp.concatenate(
        [
            graph.nodes["features"],
            acceleration,
            graph.nodes["pos"],
            mean_next_accel,
            min_next_accel,
            max_next_accel,
            node_type[:, None],
        ],
        axis=1,
    )
    nodes = dict(graph.nodes)
    nodes["features"] = features
    return graph._replace(nodes=nodes)
