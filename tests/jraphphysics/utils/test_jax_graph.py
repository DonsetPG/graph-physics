import numpy as np
import jax
import jax.numpy as jnp
import jraph
from meshio import Mesh
from jraphphysics.utils.jax_graph import (
    mesh_to_graph,
    meshdata_to_graph,
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    graph_to_mesh,
)
from tests.mock import get_meshs_from_vtu


def test_meshdata_to_graph():
    mesh = get_meshs_from_vtu()[0]
    graph = meshdata_to_graph(
        points=mesh.points,
        cells=mesh.cells_dict["triangle"],
        point_data=mesh.point_data,
    )
    assert graph.nodes["features"].shape[0] == 1923
    assert graph.nodes["pos"].shape[0] == 1923


def test_mesh_to_graph():
    mesh = get_meshs_from_vtu()[0]
    graph = mesh_to_graph(mesh=mesh)
    assert graph.nodes["features"].shape[0] == 1923
    assert graph.nodes["pos"].shape[0] == 1923


def test_graph_to_mesh():
    mesh = get_meshs_from_vtu()[0]
    graph = meshdata_to_graph(
        points=mesh.points,
        cells=mesh.cells_dict["triangle"],
        point_data=mesh.point_data,
    )

    node_features_mapping = {
        "velocity_x": 0,
        "velocity_y": 1,
    }

    new_mesh = graph_to_mesh(graph=graph, node_features_mapping=node_features_mapping)

    assert len(mesh.points) == len(new_mesh.points)
    for k in list(mesh.point_data.keys()):
        assert np.array_equal(mesh.point_data[k], new_mesh.point_data[k])
