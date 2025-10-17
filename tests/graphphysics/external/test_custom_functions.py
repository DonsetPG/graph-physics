import unittest

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from named_features import NamedData, make_x_layout
from graphphysics.utils.nodetype import NodeType
from graphphysics.external.bezier import add_bezier_node_type
from graphphysics.external.aneurysm import build_features


class TestCustomFeatures(unittest.TestCase):
    def setUp(self):
        # Set up data for testing add_bezier_node_type
        self.graph_bezier = Data()
        # Create x with columns indices 0 to 7
        # Columns 3 to 7 correspond to bn, a1, a2, a3, a4
        # Set up data to test different node types

        # Node 0: Wall node (bn == 1.0, a1 == a2 == a3 == a4 == 0.0)
        # Node 1: Inflow node (a1 == 1.0)
        # Node 2: Outflow node (a3 == 1.0)
        # Node 3: Default node (no specific conditions met)
        # Node 4: Not a wall node (bn == 1.0 but a2 == 1.0)

        self.graph_bezier.x = torch.tensor(
            [
                [0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Node 0
                [0, 0, 0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Node 1
                [0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Node 2
                [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Node 3
                [0, 0, 0, 1.0, 0.0, 1.0, 0.0, 0.0],  # Node 4
            ],
            dtype=torch.float32,
        )

        self.graph_build = Data()
        # Columns: [v_x, v_y, v_z, wall_inputs]
        self.graph_build.x = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],  # Node 0: Wall node
                [0.0, 0.0, 0.0, 0.0],  # Node 1: Inflow node
                [0.0, 0.0, 0.0, 0.0],  # Node 2: Outflow node
                [0.0, 0.0, 0.0, 0.0],  # Node 3: Regular node
            ],
            dtype=torch.float32,
        )

        # Set positions (graph.pos) for aneurysm_node_type
        # Inflow: y == 0.0, x <= 0
        # Outflow: y == 0.0, x >= 0
        self.graph_build.pos = torch.tensor(
            [
                [0.0, 1.0, 0.0],  # Node 0
                [-1.0, 0.0, 0.0],  # Node 1
                [1.0, 0.0, 0.0],  # Node 2
                [0.0, 1.0, 0.0],  # Node 3
            ],
            dtype=torch.float32,
        )

        # Set graph.y (target velocities)
        self.graph_build.y = torch.tensor(
            [
                [1.1, 0.0, 0.0],  # Node 0
                [0.1, 0.0, 0.0],  # Node 1
                [0.2, 0.0, 0.0],  # Node 2
                [0.3, 0.1, 0.1],  # Node 3
            ],
            dtype=torch.float32,
        )

        # Set graph.previous_data
        self.graph_build.previous_data = {
            "Vitesse": [
                [0.9, 0.0, 0.0],  # Node 0
                [0.0, 0.0, 0.0],  # Node 1
                [0.0, 0.0, 0.0],  # Node 2
                [0.0, 0.0, 0.0],  # Node 3
            ]
        }

        # NamedData variants
        bezier_layout = make_x_layout(
            ["unused", "bn", "a1", "a2", "a3", "a4"],
            {"unused": 3, "bn": 1, "a1": 1, "a2": 1, "a3": 1, "a4": 1},
        )
        self.graph_bezier_named = NamedData(
            x=self.graph_bezier.x.clone(), x_layout=bezier_layout
        )

        aneurysm_layout = make_x_layout(
            ["Vitesse", "wall_mask"], {"Vitesse": 3, "wall_mask": 1}
        )
        self.graph_build_named = NamedData(
            x=self.graph_build.x.clone(),
            pos=self.graph_build.pos.clone(),
            y=self.graph_build.y.clone(),
            previous_data={"Vitesse": list(self.graph_build.previous_data["Vitesse"])},
            x_layout=aneurysm_layout,
        )

    def test_add_bezier_node_type(self):
        # Apply the function
        graph = add_bezier_node_type(self.graph_bezier)

        # The node_type is appended as the last column in graph.x
        node_type_column = graph.x[:, -1]

        # Node 0 should be WALL_BOUNDARY
        self.assertEqual(node_type_column[0].item(), NodeType.WALL_BOUNDARY)

        # Node 1 should be INFLOW
        self.assertEqual(node_type_column[1].item(), NodeType.INFLOW)

        # Node 2 should be OUTFLOW
        self.assertEqual(node_type_column[2].item(), NodeType.OUTFLOW)

        # Node 3 should be default (0)
        self.assertEqual(node_type_column[3].item(), NodeType.NORMAL)

        # Node 4 should be default (0) because a2 == 1.0, so wall_mask should be False
        self.assertEqual(node_type_column[4].item(), NodeType.NORMAL)

    def test_add_bezier_node_type_named(self):
        graph = add_bezier_node_type(self.graph_bezier_named)
        node_names = graph.x_layout.names()
        self.assertTrue(any(name.startswith("node_type") for name in node_names))
        appended = graph.x[:, -1]
        self.assertTrue(torch.all(torch.isfinite(appended)))

    def test_build_features(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.graph_build = self.graph_build.to(device)

        graph = build_features(self.graph_build)

        # Check that node_type is correctly computed
        node_type_column = graph.x[:, -1]

        # Node 0: Wall node
        self.assertEqual(node_type_column[0].item(), NodeType.WALL_BOUNDARY)

        # Node 1: Inflow node
        self.assertEqual(node_type_column[1].item(), NodeType.INFLOW)

        # Node 2: Outflow node
        self.assertEqual(node_type_column[2].item(), NodeType.OUTFLOW)

        # Node 3: Regular node
        self.assertEqual(node_type_column[3].item(), NodeType.NORMAL)

        # Check that acceleration is computed correctly
        acceleration = graph.x[:, 4:7]
        current_velocity = self.graph_build.x[:, 0:3]
        previous_velocity = torch.tensor(
            self.graph_build.previous_data["Vitesse"], device=device
        )
        expected_acceleration = current_velocity - previous_velocity

        self.assertTrue(torch.allclose(acceleration, expected_acceleration))

        # Check that mean_next_accel, min_next_accel, max_next_accel are computed and added
        # They should be the same for all nodes since we use unique values
        mean_next_accel = graph.x[:, 10]
        min_next_accel = graph.x[:, 11]
        max_next_accel = graph.x[:, 12]

        # Compute expected next_acceleration
        target_velocity = self.graph_build.y[:, 0:3]
        next_acceleration = target_velocity - current_velocity
        node_type = node_type_column
        not_inflow_mask = node_type != NodeType.INFLOW
        next_acceleration[not_inflow_mask] = 0
        next_acceleration_unique = next_acceleration.unique()

        expected_mean = next_acceleration_unique.mean()
        expected_min = next_acceleration_unique.min()
        expected_max = next_acceleration_unique.max()

        self.assertTrue(torch.all(mean_next_accel == expected_mean))
        self.assertTrue(torch.all(min_next_accel == expected_min))
        self.assertTrue(torch.all(max_next_accel == expected_max))

    def test_build_features_named(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        graph_named = self.graph_build_named.clone()
        graph_named.x = graph_named.x.to(device)
        graph_named.pos = graph_named.pos.to(device)
        graph_named.y = graph_named.y.to(device)
        graph_named = build_features(graph_named)
        names = graph_named.x_layout.names()
        expected_suffixes = {
            "acceleration",
            "mesh_pos",
            "mean_next_accel",
            "min_next_accel",
            "max_next_accel",
        }
        self.assertTrue(any(name.startswith("node_type") for name in names))
        for suffix in expected_suffixes:
            self.assertTrue(
                any(name.startswith(suffix) for name in names),
                msg=f"Missing feature starting with {suffix}",
            )

    def test_build_features_with_missing_previous_data(self):
        graph = self.graph_build.clone()
        del graph.previous_data
        with self.assertRaises(AttributeError):
            build_features(graph)


if __name__ == "__main__":
    unittest.main()
