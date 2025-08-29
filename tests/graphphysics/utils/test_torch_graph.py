import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
from graphphysics.utils.torch_graph import (
    mesh_to_graph,
    meshdata_to_graph,
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    torch_graph_to_mesh,
    compute_gradient,
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


def test_compute_gradient_shapes():
    N = 5
    F = 2
    D = 3
    E = 4

    pos = torch.rand(N, D)
    edge_index = torch.randint(0, N, (2, E))
    field = torch.rand(N, F)

    graph = Data(pos=pos, edge_index=edge_index)

    gradients = compute_gradient(graph, field)

    assert gradients.shape == (N, F, D), (
        f"Expected gradient shape (N, F, D) = {(N, F, D)} " f"but got {gradients.shape}"
    )
    assert (
        gradients.device == pos.device
    ), "Gradient device does not match position device."


def test_compute_gradient_linear_2d():
    # 4 nodes in a square, for simplicity
    #   (0,1) --- (1,1)
    #     |         |
    #   (0,0) --- (1,0)
    pos = torch.tensor(
        [
            [0.0, 0.0],  # node 0
            [1.0, 0.0],  # node 1
            [0.0, 1.0],  # node 2
            [1.0, 1.0],  # node 3
        ],
        dtype=torch.float,
    )

    # Undirected edges of a square
    # edge_index = torch.tensor(
    #     [[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]], dtype=torch.long
    # )
    # Directed edges of a square, one outgoing edge per node
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]], dtype=torch.long)

    graph = Data(pos=pos, edge_index=edge_index)

    def vector_field(xy):
        x, y = xy
        return torch.tensor([x + y, 2 * x - y], dtype=torch.float)

    field = torch.stack([vector_field(p) for p in pos], dim=0)

    true_grad = torch.tensor([[0.5, 0.5], [1.0, -0.5]])  # shape (2,2)

    grad_methods = {
        "finite_differences": lambda: compute_gradient(
            graph, field, method="finite_diff"
        ),
        # "weighted_least_squares": lambda: compute_gradient(
        #     graph, field, method="least_squares"
        # ),
        # "green_gauss": lambda: compute_gradient(
        #     graph, field, method="green_gauss"
        # ),
    }

    for name, method in grad_methods.items():
        gradients = method()
        for node_grad in gradients:
            # node_grad shape => (2,2)
            # we test if each node’s gradient is "close" to the analytic solution
            assert torch.allclose(
                node_grad, true_grad, atol=1e-1
            ), f"Computed gradient (using {name} method) {node_grad} not close to expected {true_grad}"
