import torch
import torch_geometric.transforms as T
import numpy as np
from graphphysics.utils.torch_graph import (
    mesh_to_graph,
    meshdata_to_graph,
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    torch_graph_to_mesh,
)
from tests.mock import get_meshs_from_vtu


def test_meshdata_to_graph():
    mesh = get_meshs_from_vtu()[0]
    graph = meshdata_to_graph(
        points=mesh.points,
        cells=mesh.cells_dict["triangle"],
        point_data=mesh.point_data,
    )
    assert graph.x.shape[0] == 1923
    assert graph.pos.shape[0] == 1923


def test_mesh_to_graph():
    mesh = get_meshs_from_vtu()[0]
    graph = mesh_to_graph(mesh=mesh)
    assert graph.x.shape[0] == 1923
    assert graph.pos.shape[0] == 1923


def test_khop_edges():
    mesh = get_meshs_from_vtu()[0]
    graph = mesh_to_graph(mesh=mesh)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    graph = T.FaceToEdge()(graph)
    khoped_edge_index = compute_k_hop_edge_index(graph.edge_index, 2, graph.num_nodes)
    assert khoped_edge_index[0].shape > graph.edge_index[0].shape


def test_khop_graph():
    mesh = get_meshs_from_vtu()[0]
    graph = mesh_to_graph(mesh=mesh)
    print(graph)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph.to(device)
    graph = T.FaceToEdge()(graph)
    print(graph)
    khoped_graph_no_edge_attribute = compute_k_hop_graph(graph, 2, False)
    assert (
        khoped_graph_no_edge_attribute.edge_index[0].shape > graph.edge_index[0].shape
    )
    assert khoped_graph_no_edge_attribute.edge_attr is None

    khoped_graph = compute_k_hop_graph(graph, 2, True)
    assert khoped_graph.edge_index[0].shape > graph.edge_index[0].shape
    assert khoped_graph.edge_attr is not None


def test_torch_graph_to_mesh():
    mesh = get_meshs_from_vtu()[0]
    graph = meshdata_to_graph(
        points=mesh.points,
        cells=mesh.cells_dict["triangle"],
        point_data=mesh.point_data,
    )

    new_mesh = torch_graph_to_mesh(
        graph=graph, node_features_mapping={"velocity_x": 0, "velocity_y": 1}
    )

    assert len(mesh.points) == len(new_mesh.points)
    for k in list(mesh.point_data.keys()):
        assert np.array_equal(mesh.point_data[k], new_mesh.point_data[k])
