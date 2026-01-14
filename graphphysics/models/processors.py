from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric.nn import TransformerConv
import numpy as np
from scipy.spatial import Delaunay

import graphphysics.models.transolver as Transolver
from graphphysics.models.layers import (
    GraphNetBlock,
    TemporalAttention,
    Transformer,
    build_mlp,
)
from graphphysics.models.hierarchical_pooling import DownSampler, UpSampler
from graphphysics.utils.physical_loss import HyperelasticResidual

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

class HierarchicalPooler(nn.Module):
    """
    V-cycle style hierarchical GNN:

      - 4 message-passing steps on the fine graph (pre-down).
      - Downscale (coarsen) the graph.
      - 10 message-passing steps on the coarse graph.
      - Upscale (interpolate) back to the fine graph.
      - 1 message-passing step on the fine graph (post-up).
      - Residual connection from pre-down fine features to post-up fine features.
      - Final MLP decoder on the fine graph node features.
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        use_gated_mlp: bool = False,
        message_passing_num_coarse: int = 5,
        ratio: float = 0.5,
        k: int = 6,
        method: str = "fps",
        is_remeshing: bool = True,
        pool_node_mask: str = "normal",
    ):
        super().__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.method = method
        self.is_remeshing = is_remeshing
        self.pool_node_mask = pool_node_mask
        self.residual = HyperelasticResidual()

        # Pour Snapshot
        self._capture_residual = True
        self._residual_snapshot = None
        self._residual_snapshot_downsampled = None

        # Encoder : nodes (node_input_size -> hidden_size), edges (edge_input_size -> hidden_size)
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

        # Fine-level message passing before downscaling: 5 steps
        self.down_processors = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_size=hidden_size,
                    use_gated_mlp=use_gated_mlp,
                    use_rope=False,
                    rope_axes=3,
                    rope_base=10000.0,
                    use_gate=False,
                )
                for _ in range(5)
            ]
        )

        # # Decoder after first processing used for downsampling (not used currently because we are trying the same decoder as at the end.)
        # self.downsample_decoder = build_mlp(
        #     in_size=hidden_size,
        #     hidden_size=hidden_size,
        #     out_size=3,
        #     layer_norm=False,
        # )

        # Downsampler: fine -> coarse
        self.downsampler = DownSampler(
            d_in=hidden_size,
            d_out=hidden_size,
            edge_dim=hidden_size,
            ratio=ratio,
            k=k,
            method=self.method,
            is_remeshing=self.is_remeshing,
            node_mask=self.pool_node_mask,
        )

        # Coarse-level message passing: 5 steps
        self.coarse_processors = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_size=hidden_size,
                    use_gated_mlp=use_gated_mlp,
                    use_rope=False,
                    rope_axes=3,
                    rope_base=10000.0,
                    use_gate=False,
                )
                for _ in range(5)
            ]
        )

        # Upsampler: coarse -> fine
        self.up_sampler = UpSampler(
            d_in=hidden_size,
            d_out=hidden_size,
            k=k,
        )


        # Fine-level message passing after upscaling: 5 steps
        self.up_processor = nn.ModuleList(
            [
                GraphNetBlock(
                    hidden_size=hidden_size,
                    use_gated_mlp=use_gated_mlp,
                    use_rope=False,
                    rope_axes=3,
                    rope_base=10000.0,
                    use_gate=False,
                )
                for _ in range(5)
            ]
        )

        # Final decoder
        self.decode_module = build_mlp(
            in_size=hidden_size,
            hidden_size=hidden_size,
            out_size=output_size,
            layer_norm=False,
        )

    def enable_residual_capture(self) -> None:
        self._capture_residual = True

    def pop_residual_snapshot(self) -> Optional[Data]:
        snapshot = self._residual_snapshot
        self._residual_snapshot = None
        return snapshot

    def pop_residual_snapshot_downsampled(self) -> Optional[Data]:
        snapshot = self._residual_snapshot_downsampled
        self._residual_snapshot_downsampled = None
        return snapshot

    def forward(self, graph: Data) -> torch.Tensor:
        fine_edge_index = graph.edge_index
        fine_edge_attr_raw = graph.edge_attr
        fine_node_type = graph.node_type
        fine_pos = graph.pos
        face_elements = graph.face
        fine_batch = graph.batch
        fine_ptr = graph.ptr

        # 1) Encode raw features -> latent space
        x = self.nodes_encoder(graph.x)            # [N_fine, hidden]
        e = self.edges_encoder(fine_edge_attr_raw) # [E_fine, hidden]

        # 2) 5 MP steps on fine graph before downscaling
        for block in self.down_processors:
            x, e = block(
                x,
                fine_edge_index,
                e,
                pos=fine_pos,
                phi=None,
            )

        # Save fine latent features for residual connection
        x_fine_skip = x  # [N_fine, hidden]
        e_fine_skip = e # [E_fine, hidden]

        # # Decode to 3D for downsampling if method=residu
        if self.method == "residu":
            fine_graph_for_down = Data(
                x=self.decode_module(x),
                edge_index=fine_edge_index,
                edge_attr=e,
                pos=fine_pos,
                face=face_elements,
                node_type=fine_node_type,
                batch=fine_batch,
                ptr=fine_ptr
                )

        # Build latent fine graph for the downsampler
        fine_graph_latent = Data(
            x=x,
            edge_index=fine_edge_index,
            edge_attr=e,
            pos=fine_pos,
            node_type=fine_node_type,
            batch=fine_batch,
            ptr=fine_ptr
        )

        scores = HyperelasticResidual()(fine_graph_for_down, displacement=fine_graph_for_down.x[:, 0:3]) if self.method == "residu" else None

        # 3) Downscale: fine -> coarse
        coarse_graph = self.downsampler(
            fine_graph_latent,
            attn=scores if self.method == "residu" else None,
        )
        # coarse_graph has x=[N_coarse, hidden], pos=[N_coarse,dim], edge_attr geom

        # Save residual snapshot with fine graph+scores and for coarse downsampled graph
        if self._capture_residual:
            n0 = fine_graph_latent.num_nodes
            if getattr(fine_graph_latent, "batch", None) is not None:
                if getattr(fine_graph_latent, "ptr", None) is not None:
                    n0 = int(fine_graph_latent.ptr[1])  # first graph size
                else:
                    n0 = int((fine_graph_latent.batch == 0).sum())

            if self.method == "residu":
                snapshot = fine_graph_for_down.clone()

                # Keep only batch 0
                if getattr(snapshot, "batch", None) is not None:
                    snapshot.x = snapshot.x[:n0]
                    snapshot.pos = snapshot.pos[:n0]
                    snapshot.node_type = snapshot.node_type[:n0]
                    snapshot.batch = None
                    snapshot.ptr = None

                    # Filter faces to batch 0
                    if getattr(snapshot, "face", None) is not None:
                        face = snapshot.face
                        if face.dim() == 2 and face.shape[0] not in (3, 4):
                            face = face.T
                        mask = (face < n0).all(dim=0)
                        snapshot.face = face[:, mask]

                snapshot.hyper_residual = scores.detach()[:n0] if scores is not None else None
                self._residual_snapshot = snapshot
                self._capture_residual = False

            pool_perm = getattr(coarse_graph, "pool_perm", None)
            if pool_perm is None:
                logger.warning(
                    "Downsampled residual snapshot missing pool_perm stooooop"
                )
            else:
                coarse_mask = pool_perm < n0
                snapshot_downsampled = coarse_graph.clone()

                edge_index = None
                edge_attr = None
                if getattr(snapshot_downsampled, "edge_index", None) is not None:
                    edge_index, edge_attr = subgraph(
                        coarse_mask,
                        snapshot_downsampled.edge_index,
                        snapshot_downsampled.edge_attr,
                        relabel_nodes=True,
                        num_nodes=snapshot_downsampled.num_nodes,
                    )

                snapshot_downsampled.x = snapshot_downsampled.x[coarse_mask]
                snapshot_downsampled.pos = snapshot_downsampled.pos[coarse_mask]
                if getattr(snapshot_downsampled, "node_type", None) is not None:
                    snapshot_downsampled.node_type = snapshot_downsampled.node_type[
                        coarse_mask
                    ]
                if getattr(snapshot_downsampled, "batch", None) is not None:
                    snapshot_downsampled.batch = None
                    snapshot_downsampled.ptr = None

                snapshot_downsampled.edge_index = edge_index
                snapshot_downsampled.edge_attr = edge_attr

                if self.method == "residu" and scores is not None:
                    snapshot_downsampled.hyper_residual = scores.detach()[
                        pool_perm[coarse_mask]
                    ]
                else:
                    snapshot_downsampled.hyper_residual = None
                self._residual_snapshot_downsampled = snapshot_downsampled

        # Re-encode coarse edges into hidden space
        e_c = self.edges_encoder(coarse_graph.edge_attr)  # [E_coarse, hidden]
        x_c = coarse_graph.x                              # [N_coarse, hidden]
        pos_xc = coarse_graph.pos

        # 4) 5 MP steps on coarse graph
        for block in self.coarse_processors:
            x_c, e_c = block(
                x_c,
                coarse_graph.edge_index,
                e_c,
                pos=pos_xc,
                phi=None,
            )
        coarse_graph.x, coarse_graph.edge_attr = x_c, e_c

        # 5) Upscaling coarse -> fine
        upsampled_x = self.up_sampler(
            x_coarse=coarse_graph.x,
            pos_coarse=coarse_graph.pos,
            pos_fine=fine_pos,
            batch_coarse=coarse_graph.batch,
            batch_fine=fine_batch,
        )

        # 6) Residual connection from pre-down fine features
        x_f = upsampled_x + x_fine_skip   # [N_fine, hidden]

        # 7) 5 MP steps on fine graph after upscaling
        for block in self.up_processor:
            x_f, e_f = block(
                x_f,
                fine_edge_index,
                e_fine_skip,
                pos=fine_pos,
                phi=None,
        )

        # 8) Final MLP decoder
        output = self.decode_module(x_f)  # [N_fine, output_size]
        return output
