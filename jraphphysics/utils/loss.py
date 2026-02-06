from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Optional

import jax.numpy as jnp
import jraph

from graphphysics.utils.nodetype import NodeType
from jraphphysics.utils.vectorial_operators import (
    compute_divergence,
    compute_gradient,
    compute_vector_gradient_product,
)


def _prepare_mask_for_loss(
    network_output: jnp.ndarray,
    node_type: jnp.ndarray,
    masks: list[NodeType],
    selected_indexes: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    del network_output
    mask = node_type == int(masks[0])
    for node_type_value in masks[1:]:
        mask = jnp.logical_or(mask, node_type == int(node_type_value))

    if selected_indexes is not None:
        n = node_type.shape[0]
        nodes = jnp.arange(n)
        selected = jnp.isin(nodes, selected_indexes)
        mask = jnp.logical_and(mask, jnp.logical_not(selected))
    return mask


@dataclass
class L2Loss:
    @property
    def __name__(self):
        return "MSE"

    def __call__(
        self,
        target: jnp.ndarray,
        network_output: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        errors = ((network_output - target) ** 2)[mask]
        return jnp.mean(errors)


@dataclass
class CosineLoss:
    @property
    def __name__(self):
        return "Cosine"

    def __call__(
        self,
        target: jnp.ndarray,
        network_output: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        target_norm = jnp.linalg.norm(target, axis=1) + 1e-8
        output_norm = jnp.linalg.norm(network_output, axis=1) + 1e-8
        cosine = jnp.sum(network_output * target, axis=1) / (target_norm * output_norm)
        errors = 1.0 - cosine
        return jnp.mean(errors[mask])


@dataclass
class L1SmoothLoss:
    beta: float = 1.0

    @property
    def __name__(self):
        return "L1Smooth"

    def __call__(
        self,
        target: jnp.ndarray,
        network_output: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output, node_type, masks, selected_indexes
        )
        diff = jnp.abs(network_output - target)
        errors = jnp.where(
            diff < self.beta,
            0.5 * (diff**2) / self.beta,
            diff - 0.5 * self.beta,
        )
        return jnp.mean(errors[mask])


@dataclass
class GradientL2Loss:
    @property
    def __name__(self):
        return "GradientL2Loss"

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        target_physical: jnp.ndarray,
        network_output_physical: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        gradient_method: str = "finite_diff",
        target_gradient: Optional[jnp.ndarray] = None,
        network_output_gradient: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        if network_output_gradient is None:
            network_output_gradient = compute_gradient(
                graph=graph,
                field=network_output_physical,
                method=gradient_method,
            )
        if target_gradient is None:
            target_gradient = compute_gradient(
                graph=graph,
                field=target_physical,
                method=gradient_method,
            )
        errors = ((network_output_gradient - target_gradient) ** 2)[mask]
        return jnp.mean(errors)


@dataclass
class ConvectionL2Loss:
    @property
    def __name__(self):
        return "ConvectionL2Loss"

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        target_physical: jnp.ndarray,
        network_output_physical: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        gradient_method: str = "finite_diff",
        target_gradient: Optional[jnp.ndarray] = None,
        network_output_gradient: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        target_convection = compute_vector_gradient_product(
            graph=graph,
            field=target_physical,
            gradient=target_gradient,
            method=gradient_method,
        )
        network_output_convection = compute_vector_gradient_product(
            graph=graph,
            field=network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        errors = ((network_output_convection - target_convection) ** 2)[mask]
        return jnp.mean(errors)


@dataclass
class DivergenceL2Loss:
    @property
    def __name__(self):
        return "DivergenceL2Loss"

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        network_output_physical: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph=graph,
            field=network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        return jnp.mean((divergence**2)[mask])


@dataclass
class DivergenceL1Loss:
    @property
    def __name__(self):
        return "DivergenceL1Loss"

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        network_output_physical: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph=graph,
            field=network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        return jnp.mean(jnp.abs(divergence)[mask])


@dataclass
class DivergenceL1SmoothLoss:
    beta: float = 1.0

    @property
    def __name__(self):
        return "DivergenceL1Smooth"

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        network_output_physical: jnp.ndarray,
        node_type: jnp.ndarray,
        masks: list[NodeType],
        selected_indexes: Optional[jnp.ndarray] = None,
        gradient_method: str = "finite_diff",
        network_output_gradient: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> jnp.ndarray:
        del kwargs
        mask = _prepare_mask_for_loss(
            network_output_physical, node_type, masks, selected_indexes
        )
        divergence = compute_divergence(
            graph=graph,
            field=network_output_physical,
            gradient=network_output_gradient,
            method=gradient_method,
        )
        diff = jnp.abs(divergence)
        errors = jnp.where(
            diff < self.beta,
            0.5 * (diff**2) / self.beta,
            diff - 0.5 * self.beta,
        )
        return jnp.mean(errors[mask])


@dataclass
class MultiLoss:
    losses: Iterable
    weights: Iterable[float]

    @property
    def __name__(self):
        return "MultiLoss"

    def __call__(
        self,
        graph: Optional[jraph.GraphsTuple] = None,
        network_output_physical: Optional[jnp.ndarray] = None,
        target_physical: Optional[jnp.ndarray] = None,
        gradient_method: Optional[str] = None,
        return_all_losses: bool = False,
        **kwargs,
    ):
        network_output_gradient = None
        target_gradient = None
        if gradient_method is not None and graph is not None:
            network_output_gradient = compute_gradient(
                graph=graph,
                field=network_output_physical,
                method=gradient_method,
            )
            target_gradient = compute_gradient(
                graph=graph,
                field=target_physical,
                method=gradient_method,
            )

        weighted_losses = []
        for weight, loss in zip(self.weights, self.losses):
            value = loss(
                graph=graph,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=gradient_method,
                network_output_gradient=network_output_gradient,
                target_gradient=target_gradient,
                **kwargs,
            )
            weighted_losses.append(weight * value)

        total = sum(weighted_losses)
        if return_all_losses:
            return total, weighted_losses
        return total


class LossType(enum.Enum):
    L2LOSS = L2Loss
    COSINEL2LOSS = CosineLoss
    L1SMOOTHLOSS = L1SmoothLoss
    GRADIENTL2LOSS = GradientL2Loss
    CONVECTIONL2LOSS = ConvectionL2Loss
    DIVERGENCEL2LOSS = DivergenceL2Loss
    DIVERGENCEL1LOSS = DivergenceL1Loss
    DIVERGENCEL1SMOOTHLOSS = DivergenceL1SmoothLoss
