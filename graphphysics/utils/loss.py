import enum

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.vectorial_operators import (
    compute_divergence,
    compute_gradient,
    compute_vector_gradient_product,
)
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_mask_for_loss(
    network_output: torch.Tensor,
    node_type: torch.Tensor,
    masks: list[NodeType],
    selected_indexes: torch.Tensor = None,
):
    mask = node_type == masks[0]
    for i in range(1, len(masks)):
        mask = torch.logical_or(mask, node_type == masks[i])

    if selected_indexes is not None:
        n, _ = network_output.shape
        nodes_mask = ~torch.isin(torch.arange(n), selected_indexes).to(device)
        mask = torch.logical_and(nodes_mask, mask)

    return mask


class L2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.

        Note:
            This method calculates the L2 loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]
        return torch.mean(errors)

class CosineLoss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cos = nn.CosineEmbeddingLoss(reduction="none")

    @property
    def __name__(self):
        return "Cosine"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes a cosine similarity loss for nodes of specific types.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean cosine embedding loss for the specified node types.

        Note:
            This method calculates the cosine loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        target_tensor = torch.ones(
            target.shape[0], device=mask.device, dtype=target.dtype
        )
        errors = self.cos(network_output, target, target_tensor)[mask]
        return torch.mean(errors)
class L1SmoothLoss(_Loss):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    @property
    def __name__(self):
        return "L1Smooth"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L1Smooth loss for nodes of specific types.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The L1Smooth loss for the specified node types.

        Note:
            This method calculates the L1Smooth loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = F.smooth_l1_loss(
            network_output, target, reduction="none", beta=self.beta
        )[mask]
        return torch.mean(errors)


class GradientL2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "GradientL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types.
        The loss is computed between the spatial gradient of target and network_output.

        Args:
            target_physical (torch.Tensor): The physical target values.
            network_output (torch.Tensor): The predicted physical values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            gradient_method (str): Method to compute the gradient ("finite_diff","least_square")
            target_gradient (torch.Tensor, optional): Gradient of the physical target.
            network_output_gradient (torch.Tensor, optional): Gradient of the physical network output.

        Returns:
            torch.Tensor: The L2 loss for the specified node types.
        """
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )

        if network_output_gradient is None:
            network_output_gradient = compute_gradient(
                graph, network_output_physical, method=gradient_method, device=device
            )
        if target_gradient is None:
            target_gradient = compute_gradient(
                graph, target_physical, method=gradient_method, device=device
            )
        errors = ((network_output_gradient - target_gradient) ** 2)[mask]
        return torch.mean(errors)


class ConvectionL2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "ConvectionL2Loss"

    def forward(
        self,
        graph: Data,
        target_physical: torch.Tensor,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        target_gradient: torch.Tensor = None,
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types.
        The loss is computed between the NS convection term ((u.grad)u) of target and network_output.

        Args:
            target_physical (torch.Tensor): The physical target values.
            network_output_physical (torch.Tensor): The predicted physical values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            gradient_method (str): Method to compute the gradient ("finite_diff","least_square").
            target_gradient (torch.Tensor, optional): Gradient of the physical target.
            network_output_gradient (torch.Tensor, optional): Gradient of the physical network output.

        Returns:
            torch.Tensor: The L2 loss for the specified node types.
        """
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        network_output_convection = compute_vector_gradient_product(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
            device=device,
        )
        target_convection = compute_vector_gradient_product(
            graph,
            target_physical,
            gradient=target_gradient,
            method=gradient_method,
            device=device,
        )
        errors = ((network_output_convection - target_convection) ** 2)[mask]
        return torch.mean(errors)


class DivergenceL2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "DivergenceL2Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes L2 norm of the divergence of the network physical output.

        Args:
            network_output_physical (torch.Tensor): The predicted physical values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            gradient_method (str): Method to compute the gradient ("finite_diff","least_square").
            network_output_gradient (torch.Tensor, optional): Gradient of the physical network output.

        Returns:
            torch.Tensor: The L2 loss for the specified node types.
        """
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
            device=device,
        )
        errors = (divergence**2)[mask]
        return torch.mean(errors)


class DivergenceL1Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "DivergenceL1Loss"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes divergence L1 loss for nodes of specific types.

        Args:
            network_output_physical (torch.Tensor): The predicted physical values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            gradient_method (str): Method to compute the gradient ("finite_diff","least_square").
            network_output_gradient (torch.Tensor, optional): Gradient of the physical network output.

        Returns:
            torch.Tensor: The L1 loss for the specified node types.
        """
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
            device=device,
        )
        errors = torch.abs(divergence)[mask]
        return torch.mean(errors)


class DivergenceL1SmoothLoss(_Loss):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    @property
    def __name__(self):
        return "DivergenceL1Smooth"

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Computes Divergence L1Smooth loss for nodes of specific types.

        Args:
            network_output_physical (torch.Tensor): The predicted physical values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.
            gradient_method (str): Method to compute the gradient ("finite_diff","least_square").
            network_output_gradient (torch.Tensor, optional): Gradient of the physical network output.

        Returns:
            torch.Tensor: The L1Smooth loss of the Divergence (using method provided) for the specified node types.

        Note:
            This method calculates the L1Smooth loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph,
            network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
            device=device,
        )
        zeros = torch.zeros_like(divergence)
        errors = F.smooth_l1_loss(divergence, zeros, reduction="none", beta=self.beta)[
            mask
        ]
        return torch.mean(errors)


class MultiLoss(_Loss):
    def __init__(self, losses, weights, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        self.weights = weights

    @property
    def __name__(self):
        return "MultiLoss"

    def forward(
        self,
        graph: Data = None,
        network_output_physical: torch.Tensor = None,
        target_physical: torch.Tensor = None,
        gradient_method: str = None,
        return_all_losses: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Combines multiple loss, weighted with fixed weights.
        """
        network_output_gradient = None
        target_gradient = None

        losses = [
            w
            * loss(
                graph=graph,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=gradient_method,
                network_output_gradient=network_output_gradient,
                target_gradient=target_gradient,
                **kwargs
            )
            for w, loss in zip(self.weights, self.losses)
        ]
        errors = sum(losses)
        if return_all_losses:
            return errors, losses
        else:
            return errors


class LossType(enum.Enum):
    L2LOSS = L2Loss
    COSINEL2LOSS = CosineLoss
    L1SMOOTHLOSS = L1SmoothLoss
    GRADIENTL2LOSS = GradientL2Loss
    CONVECTIONL2LOSS = ConvectionL2Loss
    DIVERGENCEL2LOSS = DivergenceL2Loss
    DIVERGENCEL1LOSS = DivergenceL1Loss
    DIVERGENCEL1SMOOTHLOSS = DivergenceL1SmoothLoss
