from typing import Optional, Tuple
import jax.numpy as jnp
from flax import nnx
import jraph
import jax.nn as nn
from jraphphysics.models.layers import Normalizer
from jraphphysics.models.processors import EncodeTransformDecode
from graphphysics.utils.nodetype import NodeType
from jaxtyping import ArrayLike


class Simulator(nnx.Module):
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        feature_index_start: int,
        feature_index_end: int,
        output_index_start: int,
        output_index_end: int,
        node_type_index: int,
        model: EncodeTransformDecode,
        *,
        rngs: nnx.Rngs,
    ):
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size if edge_input_size > 0 else None
        self.output_size = output_size

        self.feature_index_start = feature_index_start
        self.feature_index_end = feature_index_end
        self.node_type_index = node_type_index

        self.output_index_start = output_index_start
        self.output_index_end = output_index_end

        self.model = model

        self._output_normalizer = Normalizer(
            size=output_size,
        )
        self._node_normalizer = Normalizer(
            size=node_input_size,
        )

    def _get_pre_target(self, inputs: jraph.GraphsTuple) -> ArrayLike:
        return inputs.nodes["features"][
            :, self.output_index_start : self.output_index_end
        ]

    def _get_target_normalized(
        self, inputs: jraph.GraphsTuple, is_training: bool = True
    ) -> ArrayLike:

        target = inputs.globals["target_features"]
        pre_target = self._get_pre_target(inputs)
        target_delta = target - pre_target
        target_delta_normalized = self._output_normalizer(target_delta, is_training)

        return target_delta_normalized

    def _get_one_hot_type(self, inputs: jraph.GraphsTuple) -> ArrayLike:
        node_type = inputs.nodes["features"][:, self.node_type_index]
        node_type = node_type.astype(jnp.int32)  # Ensure integer type
        return nn.one_hot(node_type, NodeType.SIZE)

    def _build_node_features(
        self, inputs: jraph.GraphsTuple, one_hot_type: ArrayLike
    ) -> ArrayLike:
        features = inputs.nodes["features"][
            :, self.feature_index_start : self.feature_index_end
        ]
        pos = inputs.nodes["pos"]
        node_features = jnp.concatenate([pos, features, one_hot_type], axis=1)

        return node_features

    def _build_input_graph(
        self, inputs: jraph.GraphsTuple, is_training: bool
    ) -> Tuple[jraph.GraphsTuple, ArrayLike]:
        target_delta_normalized = self._get_target_normalized(inputs, is_training)
        one_hot_type = self._get_one_hot_type(inputs)
        node_features = self._build_node_features(inputs, one_hot_type)

        node_features_normalized = self._node_normalizer(node_features, is_training)

        """graph = jraph.GraphsTuple(
            nodes={"features": node_features_normalized},
            edges=None,
            senders=inputs.senders,
            receivers=inputs.receivers,
            n_node=inputs.n_node,
            n_edge=inputs.n_edge,
            globals=inputs.globals,
        )"""

        graph = inputs._replace(
            nodes={"features": node_features_normalized},
        )

        return graph, target_delta_normalized

    def _build_outputs(
        self, inputs: jraph.GraphsTuple, network_output: ArrayLike
    ) -> ArrayLike:
        pre_target = self._get_pre_target(inputs)
        update = self._output_normalizer.inverse(network_output)
        return pre_target + update

    def __call__(
        self, inputs: jraph.GraphsTuple, is_training: bool = True
    ) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
        graph, target_delta_normalized = self._build_input_graph(
            inputs=inputs, is_training=is_training
        )
        network_output = self.model(graph)

        if is_training:
            return network_output, target_delta_normalized, None
        else:
            outputs = self._build_outputs(inputs=inputs, network_output=network_output)
            return network_output, target_delta_normalized, outputs
