import unittest
import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.loss import (
    GaussianMixtureNLLLoss,
    DiagonalGaussianMixtureNLLLoss,
    L2Loss,
    L1SmoothLoss,
    GradientL2Loss,
    ConvectionL2Loss,
    DivergenceLoss,
    DivL1Loss,
    DivL1SmoothLoss,
    DivL2Loss,
    MultiLoss,
)

# TODO: test all loss functions


class TestGaussianMixtureNLLLossDiagonal(unittest.TestCase):
    def setUp(self):
        # Suppose velocity dimension
        self.d = 3
        # Suppose we have K mixture components
        self.K = 2
        # We'll keep a temperature factor for scaling
        self.temperature = 1.0
        # Our diagonal GMM loss
        self.loss_fn = DiagonalGaussianMixtureNLLLoss(
            d=self.d, K=self.K, temperature=self.temperature
        )
        # The shape is 2d + 1 per component => 2*3 + 1 = 7
        # with K=2 => 2*7=14 output features per node

    def test_forward_basic(self):
        """
        Basic test: random N nodes, ensure we get a scalar loss and no NaNs/inf.
        """
        N = 5
        # GMM output shape: [N, K*(2*d + 1)] => [N, 14]
        network_output = torch.randn(N, self.K * (2 * self.d + 1))
        # target shape: [N, d] => [5,3]
        target = torch.randn(N, self.d)

        # node_type => let's say all are NORMAL
        node_type = torch.full((N,), NodeType.NORMAL)

        # run the loss
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],  # we include these nodes
        )
        self.assertTrue(loss_val.dim() == 0, "Loss should be a scalar.")
        self.assertFalse(torch.isnan(loss_val).any(), "Loss returned NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "Loss returned Inf.")

    def test_masked_nodes(self):
        """
        Check that only masked node types are included in the GMM NLL.
        """
        N = 6
        net_out_dim = self.K * (2 * self.d + 1)
        network_output = torch.randn(N, net_out_dim)
        target = torch.randn(N, self.d)

        # half normal, half outflow
        node_type = torch.zeros(N, dtype=torch.long)  # Normal=0
        node_type[3:] = NodeType.OUTFLOW  # 1 => outflow

        # only compute for normal => first 3 nodes
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],
        )
        self.assertTrue(loss_val.dim() == 0)


class TestGaussianMixtureNLLLoss(unittest.TestCase):
    def setUp(self):
        # Set up some mock parameters
        self.d = 4  # dimension of velocity
        self.K = 3  # number of mixture components
        self.temperature = 1.0
        self.loss_fn = GaussianMixtureNLLLoss(
            d=self.d, K=self.K, temperature=self.temperature
        )

    def test_forward_basic(self):
        """
        Test the forward pass of GaussianMixtureNLLLoss under normal conditions.
        """
        # Suppose we have 5 nodes
        N = 5
        # network_output shape: [N, K * per_comp]
        # per_comp = d + d(d+1)//2 + 1 => with d=4 => 4 + (4*5)//2 + 1 = 4 + 10 + 1 = 15
        per_comp = self.d + (self.d * (self.d + 1)) // 2 + 1
        net_out_dim = self.K * per_comp  # => 3 * 15 = 45
        network_output = torch.randn(N, net_out_dim)

        # target shape: [N, d]
        target = torch.randn(N, self.d)

        # node_type shape: [N]; We'll say all nodes are normal
        node_type = torch.full((N,), NodeType.NORMAL)

        # we want to compute the NLL
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],  # so we include all nodes
        )

        # The result should be a single scalar
        self.assertTrue(loss_val.dim() == 0)
        self.assertFalse(torch.isnan(loss_val).any(), "Loss value is NaN.")
        self.assertFalse(torch.isinf(loss_val).any(), "Loss value is Inf.")

    def test_masked_nodes(self):
        """
        Test that nodes not in masks are ignored properly.
        """
        N = 6
        per_comp = self.d + (self.d * (self.d + 1)) // 2 + 1
        net_out_dim = self.K * per_comp
        network_output = torch.randn(N, net_out_dim)
        target = torch.randn(N, self.d)

        # node_type with half normal, half outflow
        node_type = torch.zeros(N, dtype=torch.long)
        # Make first 3 normal, last 3 outflow
        node_type[3:] = NodeType.OUTFLOW

        # We'll only keep mask for NodeType.NORMAL => that is 3 nodes
        loss_val = self.loss_fn(
            target=target,
            network_output=network_output,
            node_type=node_type,
            masks=[NodeType.NORMAL],
        )
        # still a scalar
        self.assertTrue(loss_val.dim() == 0)


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
            DivL2Loss(),
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
