import os
import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from graphphysics.models.layers import (
    GraphNetBlock,
    Transformer,
    build_mlp,
)
from graphphysics.models.bsms import MLP, BSGMP
from graphphysics.utils.bsms_wrappers import load_multi_mesh

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
        """
        super().__init__()
        self.only_processor = only_processor
        self.hidden_size = hidden_size
        self.d = output_size

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
            [GraphNetBlock(hidden_size=hidden_size) for _ in range(message_passing_num)]
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

        if self.only_processor:
            x, edge_attr = graph.x, graph.edge_attr
        else:
            x = self.nodes_encoder(graph.x)
            edge_attr = self.edges_encoder(graph.edge_attr)

        for block in self.processor_list:
            x, edge_attr = block(x, edge_index, edge_attr)

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
        """

        super(EncodeTransformDecode, self).__init__()
        self.hidden_size = hidden_size
        self.only_processor = only_processor
        self.d = output_size

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

        if HAS_DGL_SPARSE:
            adj = dglsp.spmatrix(indices=edge_index, shape=(x.shape[0], x.shape[0]))
            for block in self.processor_list:
                x = block(x, adj)
        else:
            for block in self.processor_list:
                x = block(x, edge_index)

        if self.only_processor:
            return x
        else:
            x_decoded = self.decode_module(x)
            return x_decoded


class BSMSProcessor(torch.nn.Module):
    def __init__(
        self,
        message_passing_num: int,
        node_input_size: int,
        output_size: int,
        hidden_size: int = 64,
        hidden_layer: int = 2,
        pos_dim: int = 3,
        device: torch.device = None,
        data_dir: str = "",
    ):
        super(BSMSProcessor, self).__init__()
        self.encode = MLP(node_input_size, hidden_size, hidden_size, hidden_layer, True)
        self.process = BSGMP(message_passing_num, hidden_size, hidden_layer, pos_dim)
        self.decode = MLP(hidden_size, hidden_size, output_size, hidden_layer, False)
        self.pos_dim = pos_dim
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.message_passing_num = message_passing_num
        self.data_dir = data_dir
        self.mm_layer_dir = os.path.join(
            self.data_dir, f"mm_layers_{self.message_passing_num}"
        )

    def forward(self, graph: Data, mesh_id: str = None) -> torch.Tensor:
        if mesh_id is not None:
            m_gs, m_ids = load_multi_mesh(
                mm_layer_dir=self.mm_layer_dir,
                mesh_id=mesh_id,
                device=self.device,
            )
        else:
            raise ValueError("mesh_id must be provided.")
        x = graph.x.to(self.device)
        pos = graph.pos.to(self.device)
        x = self.encode(x)
        x = self.process(x, m_ids, m_gs, pos)
        x = self.decode(x)
        return x
