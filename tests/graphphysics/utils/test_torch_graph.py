from collections import OrderedDict

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
np = pytest.importorskip("numpy")

import torch
import torch_geometric.transforms as T
from named_features import NamedData, make_x_layout

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


def test_meshdata_to_graph_3d():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    point_data = {"dummy_feature": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

    graph = meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
    )

    assert graph.x.shape[0] == 4


def test_meshdata_to_graph_with_layout():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cells = np.array([[0, 1, 2]], dtype=np.int32)
    point_data = OrderedDict(
        [
            ("velocity", np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32)),
            ("pressure", np.array([[0.1], [0.2], [0.3]], dtype=np.float32)),
            ("node_type", np.array([[0.0], [1.0], [2.0]], dtype=np.float32)),
            ("time", np.array([[0.5], [0.5], [0.5]], dtype=np.float32)),
        ]
    )
    layout = make_x_layout(
        ["velocity", "pressure", "node_type", "time"],
        {"velocity": 2, "pressure": 1, "node_type": 1, "time": 1},
    )

    graph = meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        x_layout=layout,
    )

    assert isinstance(graph, NamedData)
    assert graph.x_layout is layout
    assert graph.x.shape[1] == sum(layout.sizes().values())
    np.testing.assert_allclose(graph.x_sel("time").cpu().numpy(), point_data["time"])


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


def test_torch_graph_to_mesh_named():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    point_data = OrderedDict(
        [
            ("velocity", np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]], dtype=np.float32)),
            ("pressure", np.array([[0.1], [0.2], [0.3]], dtype=np.float32)),
        ]
    )
    layout = make_x_layout(["velocity", "pressure"], {"velocity": 2, "pressure": 1})
    graph = meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        x_layout=layout,
    )

    new_mesh = torch_graph_to_mesh(graph)
    np.testing.assert_allclose(new_mesh.point_data["velocity"], point_data["velocity"])
    np.testing.assert_allclose(new_mesh.point_data["pressure"], point_data["pressure"][:, 0])
