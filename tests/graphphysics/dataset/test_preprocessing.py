import unittest
import math
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from graphphysics.utils.nodetype import NodeType
from graphphysics.dataset.preprocessing import (
    add_edge_features,
    add_obstacles_next_pos,
    add_world_edges,
    add_world_pos_features,
    add_noise,
    build_preprocessing,
    Random3DRotate,
)


class TestGraphPreprocessing(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 4
        self.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
        self.x = torch.tensor(
            [
                [0.0, 1.0, 2.0, NodeType.NORMAL],
                [1.0, 2.0, 3.0, NodeType.OBSTACLE],
                [2.0, 3.0, 4.0, NodeType.NORMAL],
                [3.0, 4.0, 5.0, NodeType.NORMAL],
            ],
            dtype=torch.float32,
        )
        self.pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.y = torch.tensor(
            [
                [0.1, 1.1, 2.1],
                [1.1, 2.1, 3.1],
                [2.1, 3.1, 4.1],
                [3.1, 4.1, 5.1],
            ],
            dtype=torch.float32,
        )
        self.graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)

    def test_add_edge_features(self):
        transform = Compose(add_edge_features())
        graph = transform(self.graph.clone())
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(graph.edge_attr.shape[1], 4)

    def test_add_obstacles_next_pos(self):
        graph = self.graph.clone()
        graph = add_obstacles_next_pos(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
        )
        self.assertEqual(graph.x.shape[1], self.x.shape[1] + 3)

    def test_add_world_edges(self):
        graph = self.graph.clone()
        graph = add_world_edges(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=5.0,  # Large radius to connect all nodes
        )
        self.assertGreater(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_add_world_pos_features(self):
        graph = self.graph.clone()
        # Assume edge_attr is initialized
        graph.edge_attr = torch.zeros((graph.edge_index.shape[1], 1))
        graph = add_world_pos_features(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
        )
        self.assertEqual(
            graph.edge_attr.shape[1], 5
        )  # Original 1 + 3 for relative position + 1 norm

    def test_add_noise(self):
        graph = self.graph.clone()
        graph = add_noise(
            graph,
            noise_index_start=0,
            noise_index_end=3,
            noise_scale=0.1,
            node_type_index=3,
        )
        # Check that node features have changed for NORMAL nodes
        normal_indices = (self.x[:, 3] == NodeType.NORMAL).nonzero(as_tuple=True)[0]
        self.assertFalse(
            torch.allclose(graph.x[normal_indices, 0:3], self.x[normal_indices, 0:3])
        )

        # Check that node features have not changed for non-NORMAL nodes
        non_normal_indices = (self.x[:, 3] != NodeType.NORMAL).nonzero(as_tuple=True)[0]
        self.assertTrue(
            torch.allclose(
                graph.x[non_normal_indices, 0:3], self.x[non_normal_indices, 0:3]
            )
        )

    def test_add_noise_with_t(self):
        """Test noise addition with curriculum parameter t."""
        # At t=0 => scale_ = 10 * scale * (1 + cos(0)) = 20 * scale
        # At t=1 => scale_ = 10 * scale * (1 + cos(pi)) = 0
        graph_base = self.graph.clone()
        normal_indices = (graph_base.x[:, 3] == NodeType.NORMAL).nonzero(as_tuple=True)[
            0
        ]

        # 1) t=0 -> Expect significant noise for NORMAL nodes
        graph_t0 = add_noise(
            graph_base.clone(),
            noise_index_start=0,
            noise_index_end=3,
            noise_scale=0.1,
            node_type_index=3,
            t=0,
        )
        # NORMAL nodes should change
        self.assertFalse(
            torch.allclose(
                graph_t0.x[normal_indices, 0:3], graph_base.x[normal_indices, 0:3]
            )
        )

        # 2) t=1 -> Expect NO noise for NORMAL nodes (scale_=0)
        graph_t1 = add_noise(
            graph_base.clone(),
            noise_index_start=0,
            noise_index_end=3,
            noise_scale=0.1,
            node_type_index=3,
            t=1,
        )
        # NORMAL nodes should remain the same
        self.assertTrue(
            torch.allclose(
                graph_t1.x[normal_indices, 0:3], graph_base.x[normal_indices, 0:3]
            )
        )

    def test_build_preprocessing(self):
        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(noise_params, world_pos_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(graph.x.shape[1], self.x.shape[1] + 3)

    def test_add_edge_features_without_world_pos(self):
        transform = build_preprocessing(add_edges_features=True)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            ).clone()
        )
        self.assertIsNotNone(graph.edge_attr)
        self.assertEqual(
            graph.edge_attr.shape[1], 4
        )  # Assuming 3 for Cartesian and 1 for Distance

    def test_build_preprocessing_without_noise(self):
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(world_pos_parameters=world_pos_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)

    def test_build_preprocessing_without_world_pos(self):
        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        transform = build_preprocessing(noise_parameters=noise_params)
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_attr)

    def test_add_noise_with_multiple_indices(self):
        graph = Data(
            x=self.x,
            face=torch.tensor([[0], [1], [2]], dtype=torch.int),
            y=self.y,
            pos=self.pos,
        )
        noise_params = {
            "noise_index_start": [0, 1],
            "noise_index_end": [1, 3],
            "noise_scale": [10, 20],
            "node_type_index": 3,
        }
        graph = add_noise(
            graph.clone(),
            noise_index_start=noise_params["noise_index_start"],
            noise_index_end=noise_params["noise_index_end"],
            noise_scale=noise_params["noise_scale"],
            node_type_index=noise_params["node_type_index"],
        )
        # Check that node features have changed for NORMAL nodes
        normal_indices = (self.x[:, 3] == NodeType.NORMAL).nonzero(as_tuple=True)[0]
        print(normal_indices)
        self.assertFalse(
            torch.allclose(graph.x[normal_indices, 0:3], self.x[normal_indices, 0:3])
        )

    def test_add_world_edges_with_small_radius(self):
        graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)
        graph = add_world_edges(
            graph,
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=0.1,  # Small radius to potentially not add new edges
        )
        # Edge index should be same or larger
        self.assertGreaterEqual(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_add_world_edges_with_large_radius(self):
        graph = Data(x=self.x, edge_index=self.edge_index, y=self.y, pos=self.pos)
        graph = add_world_edges(
            graph.clone(),
            world_pos_index_start=0,
            world_pos_index_end=3,
            node_type_index=3,
            radius=100.0,  # Large radius to connect all nodes
        )
        self.assertGreater(graph.edge_index.shape[1], self.edge_index.shape[1])

    def test_build_preprocessing_with_extra_features(self):
        # Define extra node and edge feature functions
        def extra_node_feature(graph):
            graph.x = torch.cat(
                (
                    graph.pos,
                    graph.x,
                ),
                dim=1,
            )
            return graph

        noise_params = {
            "noise_index_start": 0,
            "noise_index_end": 3,
            "noise_scale": 0.1,
            "node_type_index": 3,
        }
        world_pos_params = {
            "world_pos_index_start": 0,
            "world_pos_index_end": 3,
            "node_type_index": 3,
            "radius": 5.0,
        }
        transform = build_preprocessing(
            noise_params,
            world_pos_params,
            extra_node_features=extra_node_feature,
        )
        graph = transform(
            Data(
                x=self.x,
                face=torch.tensor([[0], [1], [2]], dtype=torch.int),
                y=self.y,
                pos=self.pos,
            )
        )
        assert graph.x.shape[1] == 10

    def test_random_3d_rotate(self):
        # Clone the graph
        graph = self.graph.clone()
        expected_graph = self.graph.clone()

        # Define feature indices to rotate (assuming x, y, z are at indices 0,1,2)
        feature_indices = [(0, 3)]  # Rotate x[:, 0:3]

        # Create the Random3DRotate instance
        rotate = Random3DRotate(feature_indices=feature_indices)

        # Define fixed angles for testing (rotate 90 degrees around the z-axis)
        def fixed_angles(self):
            return [math.pi / 2, 0.0, 0.0]  # alpha, beta, gamma

        # Monkey-patch the _get_random_angles method
        rotate._get_random_angles = fixed_angles.__get__(rotate, Random3DRotate)

        # Apply the transform
        rotated_graph = rotate(graph)

        # Define the expected rotation matrix around the z-axis
        alpha = math.pi / 2  # 90 degrees in radians
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        rotation_matrix = torch.tensor(
            [
                [cos_a, sin_a, 0],
                [-sin_a, cos_a, 0],
                [0, 0, 1],
            ],
            dtype=graph.pos.dtype,
        )
        # Rotate the positions
        expected_pos = expected_graph.pos @ rotation_matrix

        # Compare rotated positions
        self.assertTrue(torch.allclose(rotated_graph.pos, expected_pos, atol=1e-6))

        # Rotate the features x[:, 0:3]
        expected_features = expected_graph.x[:, 0:3] @ rotation_matrix

        print(rotated_graph.x[:, 0:3])
        print(expected_features)

        # Compare rotated features
        self.assertTrue(
            torch.allclose(rotated_graph.x[:, 0:3], expected_features, atol=1e-6)
        )

        # Rotate y[:, 0:3]
        expected_y = expected_graph.y[:, 0:3] @ rotation_matrix

        # Compare rotated targets
        self.assertTrue(torch.allclose(rotated_graph.y, expected_y, atol=1e-6))

    def test_fixed_3d_rotate(self):
        # Clone the graph
        graph = self.graph.clone()

        # Define feature indices to rotate (assuming x, y, z are at indices 0,1,2)
        feature_indices = [(0, 3)]  # Rotate x[:, 0:3]

        # Create the Random3DRotate instance
        rotate = Random3DRotate(feature_indices=feature_indices)

        # Define fixed angles for testing (rotate 90 degrees around the z-axis)
        def fixed_angles(self):
            return [math.pi] * 3  # alpha, beta, gamma

        # Monkey-patch the _get_random_angles method
        rotate._get_random_angles = fixed_angles.__get__(rotate, Random3DRotate)

        # Apply the transform
        rotated_graph = rotate(graph)
        # Rotate the positions
        expected_pos = graph.pos
        # Compare rotated positions
        self.assertTrue(torch.allclose(rotated_graph.pos, expected_pos, atol=1e-6))
        # Rotate the features x[:, 0:3]
        expected_features = graph.x[:, 0:3]
        # Compare rotated features
        self.assertTrue(
            torch.allclose(rotated_graph.x[:, 0:3], expected_features, atol=1e-6)
        )
        # Rotate y[:, 0:3]
        expected_y = graph.y[:, 0:3]
        # Compare rotated targets
        self.assertTrue(torch.allclose(rotated_graph.y, expected_y, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
