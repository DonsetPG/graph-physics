import jax.numpy as jnp
import jraph

from jraphphysics.utils.nodetype import NodeType


def add_bezier_node_type(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    features = graph.nodes["features"]
    bn = features[:, 3]
    a1 = features[:, 4]
    a2 = features[:, 5]
    a3 = features[:, 6]
    a4 = features[:, 7]

    node_type = jnp.zeros_like(bn)
    wall_mask = jnp.logical_and(bn == 1.0, a1 == 0.0)
    wall_mask = jnp.logical_and(wall_mask, a2 == 0.0)
    wall_mask = jnp.logical_and(wall_mask, a3 == 0.0)
    wall_mask = jnp.logical_and(wall_mask, a4 == 0.0)

    inflow_mask = a1 == 1.0
    outflow_mask = a3 == 1.0

    node_type = jnp.where(wall_mask, int(NodeType.WALL_BOUNDARY), node_type)
    node_type = jnp.where(inflow_mask, int(NodeType.INFLOW), node_type)
    node_type = jnp.where(outflow_mask, int(NodeType.OUTFLOW), node_type)

    nodes = dict(graph.nodes)
    nodes["features"] = jnp.concatenate([features, node_type[:, None]], axis=1)
    return graph._replace(nodes=nodes)
