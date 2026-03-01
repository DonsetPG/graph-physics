import jax.numpy as jnp
import jraph


def build_features(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    features = graph.nodes["features"]
    current_velocity = features[:, 0:2]
    pressure = features[:, 3:4]
    levelset = features[:, 4:5]
    nodetype = features[:, 5:6]

    new_features = jnp.concatenate(
        [
            current_velocity,
            pressure,
            levelset,
            graph.nodes["pos"],
            nodetype,
        ],
        axis=1,
    )

    nodes = dict(graph.nodes)
    nodes["features"] = new_features

    globals_dict = dict(graph.globals) if graph.globals is not None else {}
    if "target_features" in globals_dict:
        globals_dict["target_features"] = globals_dict["target_features"][:, 0:2]

    return graph._replace(nodes=nodes, globals=globals_dict)
