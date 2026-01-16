import unittest
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
from graphphysics.utils.torch_graph import (
    mesh_to_graph,
    meshdata_to_graph,
    compute_k_hop_edge_index,
    compute_k_hop_graph,
    torch_graph_to_mesh,
    create_subgraphs
)
from tests.mock import get_meshs_from_vtu


class TestTorchGraph(unittest.TestCase):

    def test_meshdata_to_graph(self):
        mesh = get_meshs_from_vtu()[0]
        graph = meshdata_to_graph(
            points=mesh.points,
            cells=mesh.cells_dict["triangle"],
            point_data=mesh.point_data,
        )
        self.assertEqual(graph.x.shape[0], 1923)
        self.assertEqual(graph.pos.shape[0], 1923)

    def test_meshdata_to_graph_3d(self):
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
        self.assertEqual(graph.x.shape[0], 4)

    def test_mesh_to_graph(self):
        mesh = get_meshs_from_vtu()[0]
        graph = mesh_to_graph(mesh=mesh)
        self.assertEqual(graph.x.shape[0], 1923)
        self.assertEqual(graph.pos.shape[0], 1923)

    def test_khop_edges(self):
        mesh = get_meshs_from_vtu()[0]
        graph = mesh_to_graph(mesh=mesh)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph.to(device)
        graph = T.FaceToEdge()(graph)
        khoped_edge_index = compute_k_hop_edge_index(graph.edge_index, 2, graph.num_nodes)
        self.assertGreater(khoped_edge_index[0].shape, graph.edge_index[0].shape)

    def test_khop_graph(self):
        mesh = get_meshs_from_vtu()[0]
        graph = mesh_to_graph(mesh=mesh)
        print(graph)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph.to(device)
        graph = T.FaceToEdge()(graph)
        print(graph)
        khoped_graph_no_edge_attribute = compute_k_hop_graph(graph, 2, False)
        self.assertGreater(
            khoped_graph_no_edge_attribute.edge_index[0].shape, graph.edge_index[0].shape
        )
        self.assertIsNone(khoped_graph_no_edge_attribute.edge_attr)

        khoped_graph = compute_k_hop_graph(graph, 2, True)
        self.assertGreater(khoped_graph.edge_index[0].shape, graph.edge_index[0].shape)
        self.assertIsNotNone(khoped_graph.edge_attr)

    def test_torch_graph_to_mesh(self):
        mesh = get_meshs_from_vtu()[0]
        graph = meshdata_to_graph(
            points=mesh.points,
            cells=mesh.cells_dict["triangle"],
            point_data=mesh.point_data,
        )

        new_mesh = torch_graph_to_mesh(
            graph=graph, node_features_mapping={"velocity_x": 0, "velocity_y": 1}
        )

        self.assertEqual(len(mesh.points), len(new_mesh.points))
        for k in list(mesh.point_data.keys()):
            assert np.array_equal(mesh.point_data[k], new_mesh.point_data[k])

    def test_create_subgraphs(self):
        # Create a sample graph
        num_nodes = 100
        edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)
        x = torch.randn(num_nodes, 3)
        graph = Data(x=x, edge_index=edge_index)

        num_partitions = 4
        loader, partitioned_node_ids = create_subgraphs(graph, num_partitions)

        # Check the number of partitions
        self.assertEqual(len(partitioned_node_ids), num_partitions)

        # Check that all nodes are assigned to a partition
        all_partitioned_nodes = torch.cat(partitioned_node_ids)
        self.assertEqual(all_partitioned_nodes.shape[0], num_nodes)

        # Check that all nodes are unique
        self.assertEqual(all_partitioned_nodes.unique().shape[0], num_nodes)

        # Check that node ids are within the valid range
        self.assertTrue(all_partitioned_nodes.min() >= 0)
        self.assertTrue(all_partitioned_nodes.max() < num_nodes)

        partition = loader[0]
        self.assertEqual(partition.x.shape[0], partitioned_node_ids[0].shape[0])


if __name__ == '__main__':
    unittest.main()
