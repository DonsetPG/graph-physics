import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

import graphphysics.models.transolver as Transolver
from graphphysics.models.layers import (
    GraphNetBlock,
    TemporalAttention,
    Transformer,
    build_mlp,
)

try:
    import dgl.sparse as dglsp

    HAS_DGL_SPARSE = True
except ImportError as e:
    HAS_DGL_SPARSE = False
    dglsp = None
    logger.critical(
        f"Failed to import DGL. Transformer architecture will default to torch_geometric.TransformerConv. Reason: {e}"
    )


class EncodeProcessDecode(nn.Module):
    """
    An Encode-Process-Decode model for graph neural networks.

    This model architecture is designed for processing graph-structured data. It consists of three main components:
    an encoder, a processor, and a decoder. The encoder maps input graph features to a latent space, the processor
    performs message passing and updates node and edge representations, and the decoder generates the final output from the
    processed graph.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        only_processor: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        use_gated_mlp: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        """
        Initializes the EncodeProcessDecode model.

        Args:
            message_passing_num (int): Number of message passing steps.
            node_input_size (int): Size of the node input features.
            edge_input_size (int): Size of the edge input features.
            output_size (int): Size of the output features.
            hidden_size (int, optional): Size of the hidden representations. Defaults to 128.
            only_processor (bool, optional): If True, only the processor is used (no encoding or decoding). Defaults to False.
            use_rope_embeddings (bool, optional): Apply relative RoPE inside each GraphNetBlock.
                Requires node coordinates (`graph.pos`) during the forward pass. Defaults to False.
            use_gated_attention (bool, optional): Enable query-conditioned aggregation gates
                inside each GraphNetBlock. Defaults to False.
            use_gated_mlp (bool, optional): Replace GraphNetBlock MLPs with gated variants.
                Defaults to False.
            rope_pos_dimension (int, optional): Number of spatial axes (2 or 3) used for RoPE
                rotations when `use_rope_embeddings=True`. Defaults to 3.
            rope_base (float, optional): Base frequency for RoPE rotations. Defaults to 10000.0.
            use_temporal_block (bool, optional): Whether to enable the temporal attention block. Defaults to False.
        """
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.d = output_size
        self.use_temporal_block = use_temporal_block
        self.use_gated_mlp = use_gated_mlp
        self.use_rope = use_rope_embeddings
        self.use_gate = use_gated_attention
        self.rope_axes = rope_pos_dimension
        self.rope_base = rope_base
        if self.use_rope and self.rope_axes not in (2, 3):
            raise ValueError(
                "rope_pos_dimension must be 2 or 3 when use_rope_embeddings=True."
            )
        if self.use_temporal_block and not HAS_DGL_SPARSE:
            logger.warning(
                "use_temporal_block=True but DGL sparse backend is unavailable. "
                "Temporal attention will run without sparse adjacency."
            )
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size)
            if self.use_temporal_block
            else None
        )

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )

            self.edges_encoder = build_mlp(
                in_size=edge_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )

            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
            )

        self.processor_list = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_size=hidden_size,
                    use_gated_mlp=use_gated_mlp,
                    use_rope=use_rope_embeddings,
                    rope_axes=rope_pos_dimension,
                    rope_base=rope_base,
                    use_gate=use_gated_attention,
                )
                for _ in range(message_passing_num)
            ]
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass of the EncodeProcessDecode model.

        Args:
            graph (Data): Input graph data containing 'x' (node features), 'edge_index', and 'edge_attr'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node features and edge features.
                If 'only_processor' is False, the node features are passed through the decoder before returning.
        """
        edge_index = graph.edge_index
        adj = None

        if self.only_processor:
            x, edge_attr = graph.x, graph.edge_attr
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(graph.edge_attr)

        if self.use_temporal_block and HAS_DGL_SPARSE:
            adj = dglsp.spmatrix(indices=edge_index, shape=(x.size(0), x.size(0)))

        prev_x = x
        last_x = x
        pos = getattr(graph, "pos", None) if self.use_rope else None
        if self.use_rope and pos is None:
            raise ValueError(
                "Graph data must contain `pos` when use_rope_embeddings=True."
            )
        phi = getattr(graph, "phi", None) if self.use_gate else None
        for block in self.processor_list:
            prev_x = x
            x, edge_attr = block(
                x,
                edge_index,
                edge_attr,
                pos=pos,
                phi=phi,
            )
            last_x = x

        if self.use_temporal_block and self.temporal_block is not None:
            x = self.temporal_block(
                prev_x,
                last_x,
                adj if HAS_DGL_SPARSE else None,
            )

        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded


class EncodeTransformDecode(nn.Module):
    """
    An Encode-Process-Decode model using Transformer blocks for graph neural networks.

    This model architecture is designed for processing graph-structured data. It consists of three main components:
    an encoder, a processor using Transformer blocks, and a decoder. The encoder maps input node features to a latent space,
    the processor performs message passing and updates node representations using Transformer blocks, and the decoder generates
    the final output from the processed node features.
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        only_processor: bool = False,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        """
        Initializes the EncodeTransformDecode model.

        Args:
            message_passing_num (int): Number of Transformer blocks (message passing steps).
            node_input_size (int): Size of the node input features.
            output_size (int): Size of the output features.
            hidden_size (int, optional): Size of the hidden representations. Defaults to 128.
            num_heads (int, optional): Number of attention heads in the Transformer blocks. Defaults to 4.
            only_processor (bool, optional): If True, only the processor is used (no encoding or decoding). Defaults to False.
            use_proj_bias (bool, optional): Whether to use bias in the projection layers of the Transformer blocks. Defaults to True.
            use_separate_proj_weight (bool, optional): Whether to use separate weights for Q, K, V projections in the Transformer blocks.
                If False, weights are shared. Defaults to True.
            use_rope_embeddings (bool, optional): Whether to enable rotary positional embeddings. Defaults to False.
            use_gated_attention (bool, optional): Whether to apply gated attention. Defaults to False.
            rope_pos_dimension (int, optional): Dimensionality of positional inputs for RoPE. Defaults to 3.
            rope_base (float, optional): Base used in RoPE inverse frequency computation. Defaults to 10000.0.
        """

        super(EncodeTransformDecode, self).__init__()
        self.hidden_size = hidden_size
        self.only_processor = only_processor
        self.d = output_size
        self.use_rope_embeddings = use_rope_embeddings and HAS_DGL_SPARSE
        self.use_gated_attention = use_gated_attention
        self._requested_rope = use_rope_embeddings
        self.use_temporal_block = use_temporal_block

        if not self.only_processor:
            self.nodes_encoder = build_mlp(
                in_size=node_input_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )

            self.decode_module = build_mlp(
                in_size=hidden_size,
                hidden_size=hidden_size,
                out_size=output_size,
                layer_norm=False,
            )

        self.processor_list = (
            nn.ModuleList(
                [
                    Transformer(
                        input_dim=hidden_size,
                        output_dim=hidden_size,
                        num_heads=num_heads,
                        use_proj_bias=use_proj_bias,
                        use_separate_proj_weight=use_separate_proj_weight,
                        use_rope_embeddings=self.use_rope_embeddings,
                        use_gated_attention=use_gated_attention,
                        pos_dimension=rope_pos_dimension,
                        rope_base=rope_base,
                    )
                    for _ in range(message_passing_num)
                ]
            )
            if HAS_DGL_SPARSE
            else nn.ModuleList(
                [
                    TransformerConv(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        heads=num_heads,
                        concat=False,
                        beta=True,
                    )
                    for _ in range(message_passing_num)
                ]
            )
        )
        if self._requested_rope and not HAS_DGL_SPARSE:
            logger.warning(
                "use_rope_embeddings=True but DGL sparse backend is unavailable. "
                "RoPE will be ignored."
            )
        if use_gated_attention and not HAS_DGL_SPARSE:
            logger.warning(
                "use_gated_attention=True but DGL sparse backend is unavailable. "
                "Gated attention will be ignored."
            )
        if use_temporal_block and not HAS_DGL_SPARSE:
            logger.warning(
                "use_temporal_block=True but DGL sparse backend is unavailable. "
                "Temporal attention will run without sparse adjacency."
            )
        self.temporal_block = (
            TemporalAttention(hidden_size=hidden_size, num_heads=num_heads)
            if self.use_temporal_block
            else None
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Forward pass of the EncodeTransformDecode model.

        Args:
            graph (Data): Input graph data containing 'x' (node features) and 'edge_index'.

        Returns:
            torch.Tensor: Output node features after processing and decoding (if 'only_processor' is False).
        """
        edge_index = graph.edge_index

        if self.only_processor:
            x = graph.x
        else:
            x = self.nodes_encoder(graph.x)

        pos = getattr(graph, "pos", None)
        if self.use_rope_embeddings and pos is None:
            raise ValueError(
                "use_rope_embeddings=True requires 'pos' attribute in the input graph."
            )

        prev_x = x
        last_x = x
        adj = None

        if HAS_DGL_SPARSE:
            adj = dglsp.spmatrix(indices=edge_index, shape=(x.shape[0], x.shape[0]))
            for block in self.processor_list:
                prev_x = x
                last_x = block(prev_x, adj, pos=pos)
                x = last_x
        else:
            for block in self.processor_list:
                prev_x = x
                last_x = block(prev_x, edge_index)
                x = last_x

        if self.use_temporal_block and self.temporal_block is not None:
            x = self.temporal_block(prev_x, last_x, adj)

        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded


class TransolverProcessor(nn.Module):
    """
    Wrapper that adapts Transolver++ Model.
    Usage: instantiate with node_input_size etc. Then call forward(graph: torch_geometric.data.Data)
    graph.x: node features (num_nodes, in_dim).
    If graph.pos exists, it will be used as 'pos' (num_nodes, 3).
    If graph.u or graph.condition exists, it will be used as the 'condition' (global vector).
    """

    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        dropout: float = 0.0,
        mlp_ratio: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_rope_embeddings = use_rope_embeddings

        n_layers = message_passing_num
        out_dim = output_size

        self.model = Transolver.Model(
            space_dim=0,
            n_layers=n_layers,
            n_hidden=hidden_size,
            dropout=dropout,
            n_head=num_heads,
            act="gelu",
            mlp_ratio=mlp_ratio,
            fun_dim=node_input_size,
            out_dim=out_dim,
            slice_num=slice_num,
            ref=ref,
            unified_pos=unified_pos,
            use_rope_embeddings=use_rope_embeddings,
            use_gated_attention=use_gated_attention,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_temporal_block=use_temporal_block,
        )

    def forward(self, graph: Data) -> torch.Tensor:
        """
        graph.x: node features (num_nodes, in_dim)
        graph.pos (optional): (num_nodes, 3) positions
        returns: tensor of shape (num_nodes, output_size)
        """
        # Transolver expects B dimension:
        x_batched = graph.x.unsqueeze(0)  # (1, N, C)
        pos_batched = (
            graph.pos.unsqueeze(0) if graph.pos is not None else None
        )  # (1, N, 3)
        condition = None  # Condition / global features (optional)
        if self.use_rope_embeddings and pos_batched is None:
            raise ValueError(
                "use_rope_embeddings=True requires 'pos' attribute in the input graph."
            )

        out = self.model.forward(x_batched, pos_batched, condition)
        out = out.squeeze(0)  # (N, out_dim)
        return out
