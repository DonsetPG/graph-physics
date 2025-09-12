import unittest
import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.loss import (
    L2Loss,
    L1SmoothLoss,
    GradientL2Loss,
    ConvectionL2Loss,
    DivergenceLoss,
    DivL1Loss,
    DivL1SmoothLoss,
    MultiLoss,
)


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.loss_functions = [
            L2Loss(),
            L1SmoothLoss(),
            GradientL2Loss(),
            ConvectionL2Loss(),
            DivergenceLoss(),
            DivL1Loss(),
            DivL1SmoothLoss(),
        ]
        # Mock 2D graph
        pos = torch.tensor(
            [
                [0.0, 0.0],  # node 0
                [1.0, 0.0],  # node 1
                [0.0, 1.0],  # node 2
                [1.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )
        face = torch.tensor([[0, 0], [1, 2], [3, 3]])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]], dtype=torch.long)

        self.graph = Data(pos=pos, edge_index=edge_index, face=face)
        self.field = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, -2.0],
                [2.0, 3.0],
                [1.0, -4.0],
            ],
            dtype=torch.float,
        )
        self.node_type = torch.full((4,), NodeType.NORMAL)
        self.network_output = torch.randn(4, 2)
        self.target = torch.randn(4, 2)

    def test_forward_basic(self):
        for loss_fn in self.loss_functions:
            loss_name = loss_fn.__class__.__name__
            with self.subTest(loss=loss_name):
                loss_val = loss_fn(
                    graph=self.graph,
                    target=self.target,
                    network_output=self.network_output,
                    node_type=self.node_type,
                    masks=[NodeType.NORMAL],
                    network_output_physical=self.network_output,
                    target_physical=self.target,
                    gradient_method="finite_diff",
                )
                # Result should be a scalar
                self.assertTrue(
                    loss_val.dim() == 0, f"Loss dim is not 0 for {loss_name}."
                )
                self.assertFalse(
                    torch.isnan(loss_val).any(), f"Loss value is NaN for {loss_name}."
                )
                self.assertFalse(
                    torch.isinf(loss_val).any(), f"Loss value is Inf for {loss_name}."
                )

    def test_masked_nodes(self):
        masked_node_type = self.node_type
        masked_node_type[2] = NodeType.OUTFLOW
        for loss_fn in self.loss_functions:
            loss_name = loss_fn.__class__.__name__
            with self.subTest(loss=loss_name):
                loss_val = loss_fn(
                    graph=self.graph,
                    target=self.target,
                    network_output=self.network_output,
                    node_type=masked_node_type,
                    masks=[NodeType.NORMAL],
                    network_output_physical=self.network_output,
                    target_physical=self.target,
                    gradient_method="finite_diff",
                )
                # Result should be a scalar
                self.assertTrue(
                    loss_val.dim() == 0, f"Loss dim is not 0 for {loss_name}."
                )
                self.assertFalse(
                    torch.isnan(loss_val).any(), f"Loss value is NaN for {loss_name}."
                )
                self.assertFalse(
                    torch.isinf(loss_val).any(), f"Loss value is Inf for {loss_name}."
                )


class TestMultiLoss(unittest.TestCase):
    def setUp(self):
        self.losses = [L2Loss(), GradientL2Loss(), DivergenceLoss()]
        self.weights = [0.5, 0.3, 0.1]
        self.multiloss = MultiLoss(losses=self.losses, weights=self.weights)

        # Mock 2D graph
        pos = torch.tensor(
            [
                [0.0, 0.0],  # node 0
                [1.0, 0.0],  # node 1
                [0.0, 1.0],  # node 2
                [1.0, 1.0],  # node 3
            ],
            dtype=torch.float,
        )
        face = torch.tensor([[0, 0], [1, 2], [3, 3]])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]], dtype=torch.long)

        self.graph = Data(pos=pos, edge_index=edge_index, face=face)
        self.field = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, -2.0],
                [2.0, 3.0],
                [1.0, -4.0],
            ],
            dtype=torch.float,
        )
        self.node_type = torch.full((4,), NodeType.NORMAL)
        self.network_output = torch.randn(4, 2)
        self.target = torch.randn(4, 2)

    def test_forward_basic(self):
        loss_val = self.multiloss(
            graph=self.graph,
            target=self.target,
            network_output=self.network_output,
            node_type=self.node_type,
            masks=[NodeType.NORMAL],
            network_output_physical=self.network_output,
            target_physical=self.target,
            gradient_method="finite_diff",
        )
        self.assertTrue(loss_val.dim() == 0, "MultiLoss dim is not 0")
        self.assertFalse(torch.isnan(loss_val).any(), "MultiLoss value is NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "MultiLoss value is Inf.")

    def test_masked_nodes(self):
        masked_node_type = self.node_type
        masked_node_type[2] = NodeType.OUTFLOW
        loss_val = self.multiloss(
            graph=self.graph,
            target=self.target,
            network_output=self.network_output,
            node_type=masked_node_type,
            masks=[NodeType.NORMAL],
            network_output_physical=self.network_output,
            target_physical=self.target,
            gradient_method="finite_diff",
        )
        self.assertTrue(loss_val.dim() == 0, "MultiLoss dim is not 0")
        self.assertFalse(torch.isnan(loss_val).any(), "MultiLoss value is NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "MultiLoss value is Inf.")


if __name__ == "__main__":
    unittest.main()
