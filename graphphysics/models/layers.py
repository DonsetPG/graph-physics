from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

try:
    import dgl.sparse as dglsp
    from dgl.sparse import SparseMatrix

    HAS_DGL_SPARSE = True
except ImportError:
    HAS_DGL_SPARSE = False
    dglsp = None
    SparseMatrix = Any  # Use Any as a placeholder for SparseMatrix


# Rien à changer de tout le fichier. On va peut-être devoir rajouter le pooling pour la dernièe couche du modèle
# A voir si c'est nécessaire ou si on a juste une fonction déjà existante à utiliser.

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This module applies RMS normalization over the last dimension of the input tensor.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        """
        Initializes the RMSNorm module.

        Args:
            d (int): The dimension of the input tensor.
            p (float, optional): Partial RMSNorm. Valid values are in [0, 1].
                Default is -1.0 (disabled).
            eps (float, optional): A small value to avoid division by zero.
                Default is 1e-8.
            bias (bool, optional): Whether to include a bias term. Default is False.
        """
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-0.5)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
) -> nn.Module:
    """
    Builds a Multilayer Perceptron.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layers.
        out_size (int): Size of the output features.
        nb_of_layers (int, optional): Total number of linear layers in the MLP.
            Must be at least 2. Defaults to 4.
        layer_norm (bool, optional): Whether to apply RMS normalization to the
            output layer. Defaults to True.

    Returns:
        nn.Module: The constructed MLP model.
    """
    assert nb_of_layers >= 2, "The MLP must have at least 2 layers (input and output)."

    layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]

    # Add hidden layers
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

    # Add output layer
    layers.append(nn.Linear(hidden_size, out_size))

    if layer_norm:
        layers.append(RMSNorm(out_size))

    return nn.Sequential(*layers)


class GatedMLP(nn.Module):
    """
    A Gated Multilayer Perceptron.

    This layer applies a gated activation to the input features.
    """

    def __init__(self, in_size: int, hidden_size: int, expansion_factor: int):
        """
        Initializes the GatedMLP layer.

        Args:
            in_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            expansion_factor (int): Expansion factor for the hidden layer size.
        """
        super().__init__()

        self.linear1 = nn.Linear(in_size, expansion_factor * hidden_size)
        self.linear2 = nn.Linear(in_size, expansion_factor * hidden_size)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedMLP layer.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_size).

        Returns:
            torch.Tensor: Output tensor of shape (..., expansion_factor * hidden_size).
        """
        left = self.activation(self.linear1(x))
        right = self.linear2(x)
        return left * right


def build_gated_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    expansion_factor: int = 3,
) -> nn.Module:
    """
    Builds a Gated MLP.

    Args:
        in_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        out_size (int): Size of the output features.
        expansion_factor (int, optional): Expansion factor for the hidden layer size.
            Defaults to 3.

    Returns:
        nn.Module: The constructed Gated MLP model.
    """
    layers = [
        RMSNorm(in_size),
        GatedMLP(
            in_size=in_size, hidden_size=hidden_size, expansion_factor=expansion_factor
        ),
        nn.Linear(hidden_size * expansion_factor, out_size),
    ]
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """
    A module for normalizing data during training.

    This module maintains running statistics to normalize input data.
    """

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: Optional[Union[str, torch.device]] = "cuda",
    ):
        """
        Initializes the Normalizer module.

        Args:
            size (int): Size of the input data.
            max_accumulations (int, optional): Maximum number of accumulations allowed.
                Defaults to 1e5.
            std_epsilon (float, optional): Epsilon value to avoid division by zero in
                standard deviation. Defaults to 1e-8.
            name (str, optional): Name of the Normalizer. Defaults to "Normalizer".
            device (str or torch.device, optional): Device to run the Normalizer on.
                Defaults to "cuda".
        """
        super().__init__()
        self.name = name
        self.device = device
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(
            std_epsilon, dtype=torch.float32, requires_grad=False, device=device
        )
        self.register_buffer("_acc_count", torch.tensor(0.0, device=device))
        self.register_buffer("_num_accumulations", torch.tensor(0.0, device=device))
        self.register_buffer(
            "_acc_sum",
            torch.zeros(
                (1, size), dtype=torch.float32, requires_grad=False, device=device
            ),
        )
        self.register_buffer(
            "_acc_sum_squared",
            torch.zeros(
                (1, size), dtype=torch.float32, requires_grad=False, device=device
            ),
        )

    def forward(
        self, batched_data: torch.Tensor, accumulate: bool = True
    ) -> torch.Tensor:
        """
        Normalizes input data and accumulates statistics.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
            accumulate (bool, optional): Whether to accumulate statistics.
                Defaults to True.

        Returns:
            torch.Tensor: Normalized data of the same shape as input.
        """
        if accumulate:
            # Stop accumulating after reaching max_accumulations to prevent numerical issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation of the normalizer.

        Args:
            normalized_batch_data (torch.Tensor): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        """
        Accumulates the statistics of the batched data.

        Args:
            batched_data (torch.Tensor): Input data of shape (batch_size, size).
        """
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0, keepdim=True)
        squared_data_sum = torch.sum(batched_data**2, dim=0, keepdim=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self) -> torch.Tensor:
        safe_count = torch.max(
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        safe_count = torch.max(
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
        )
        variance = self._acc_sum_squared / safe_count - self._mean() ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        return torch.max(std, self._std_epsilon)

    def get_variable(self) -> Dict[str, Any]:
        """
        Returns the internal variables of the normalizer.

        Returns:
            Dict[str, Any]: A dictionary containing the normalizer's variables.
        """
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask,
) -> torch.Tensor:
    """
    Computes the scaled query-key softmax for attention.

    Args:
        q (torch.Tensor): Query tensor of shape (N, d_k).
        k (torch.Tensor): Key tensor of shape (N, d_k).
        att_mask (Optional[SparseMatrix]): Optional attention mask.

    Returns:
        torch.Tensor: Attention scores.
    """
    scaling_factor = k.size(-1) ** 0.5
    q = q / scaling_factor

    if att_mask is not None and HAS_DGL_SPARSE:
        attn = dglsp.bsddmm(att_mask, q, k.transpose(1, 0))
        attn = attn.softmax()
    else:
        attn = q @ k.transpose(-2, -1)
        attn = torch.softmax(attn, dim=-1)

    return attn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask=None,
    return_attention: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor of shape (N, d_k).
        k (torch.Tensor): Key tensor of shape (N, d_k).
        v (torch.Tensor): Value tensor of shape (N, d_v).
        att_mask (Optional[SparseMatrix], optional): Optional attention mask.
        return_attention (bool, optional): Whether to return attention weights.
            Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            The output tensor, and optionally the attention weights.
    """
    attn = scaled_query_key_softmax(q, k, att_mask=att_mask)

    # Compute the output
    if att_mask is not None and HAS_DGL_SPARSE:
        y = dglsp.bspmm(attn, v)
    else:
        y = attn @ v

    if return_attention:
        return y, attn
    else:
        return y


class Attention(nn.Module):

    def __init__(
        self,
        input_dim=512,
        output_dim=512,
        num_heads=4,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
    ):
        """
        Initializes the Attention module.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_heads (int): Number of attention heads.
            use_proj_bias (bool, optional): Whether to use bias in projection layers.
                Defaults to True.
            use_separate_proj_weight (bool, optional): Whether to use separate weights
                for Q, K, V projections. If False, weights are shared. Defaults to True.
        """
        super().__init__()

        assert (
            output_dim % num_heads == 0
        ), "Output dimension must be divisible by number of heads."

        self.hidden_size = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.q_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.v_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.proj = nn.Linear(output_dim, output_dim, bias=use_proj_bias)

        if not use_separate_proj_weight:
            # Compute optimization used at times, share the parameters in between Q/K/V
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

    def forward(
        self,
        x: torch.Tensor,
        adj,
        return_attention: bool = False,
    ):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
            adj (Optional[SparseMatrix]): Optional adjacency matrix for sparse attention.
            return_attention (bool, optional): Whether to return attention weights.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                The output tensor, and optionally the attention weights.
        """
        N = x.size(0)
        query, key, value = x, x, x

        q, k, v = map(
            lambda fn, t: fn(t),
            [self.q_proj, self.k_proj, self.v_proj],
            [query, key, value],
        )

        q = q.reshape(N, self.head_dim, self.num_heads)
        k = k.reshape(N, self.head_dim, self.num_heads)
        v = v.reshape(N, self.head_dim, self.num_heads)

        if return_attention:
            y, attn = scaled_dot_product_attention(q, k, v, adj, return_attention=True)
        else:
            y = scaled_dot_product_attention(q, k, v, adj)

        out = self.proj(y.reshape(N, -1))

        if return_attention:
            return out, attn
        else:
            return out


class Transformer(nn.Module):
    """
    A single transformer block for graph neural networks.

    This module implements a transformer block with optional sparse attention.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation_layer: torch.nn.Module = nn.ReLU,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
    ):
        """
        Initializes the Transformer module.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_heads (int): Number of attention heads.
            activation_layer (Callable[[], nn.Module], optional): Activation function
                applied after the attention layer. Defaults to nn.ReLU.
            use_proj_bias (bool, optional): Whether to use bias in projection layers.
                Defaults to True.
            use_separate_proj_weight (bool, optional): Whether to use separate weights
                for Q, K, V projections. If False, weights are shared. Defaults to True.
        """
        super().__init__()

        self.attention = Attention(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            use_proj_bias=use_proj_bias,
            use_separate_proj_weight=use_separate_proj_weight,
        )

        # initialize mlp
        self.activation = activation_layer()
        self.norm1, self.norm2 = RMSNorm(output_dim), RMSNorm(output_dim)
        self.gated_mlp = build_gated_mlp(
            in_size=output_dim, hidden_size=output_dim, out_size=output_dim
        )

        self.use_adjacency = HAS_DGL_SPARSE

    def forward(
        self, x: torch.Tensor, adj, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
            adj (Optional[SparseMatrix]): Optional adjacency matrix for sparse attention.
            return_attention (bool, optional): Whether to return attention weights.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                The output tensor, and optionally the attention weights.
        """
        if not self.use_adjacency:
            adj = None

        if return_attention:
            x_, attn = self.attention(x, adj, return_attention)
            x = x + x_
        else:
            x = x + self.attention(x, adj, return_attention)

        x = self.norm1(x)

        x = x + self.gated_mlp(x)
        x = self.norm2(x)

        if return_attention:
            return x, attn
        else:
            return x


class GraphNetBlock(MessagePassing):
    """
    Graph Network Block implementing the message passing mechanism.
    This block updates both node and edge features.
    """

    def __init__(
        self, hidden_size: int, nb_of_layers: int = 4, layer_norm: bool = True
    ):
        """
        Initializes the GraphNetBlock.

        Args:
            hidden_size (int): The size of the hidden representations.
            nb_of_layers (int, optional): The number of layers in the MLPs.
                Defaults to 4.
            layer_norm (bool, optional): Whether to use layer normalization in the MLPs.
                Defaults to True.
        """
        super().__init__(aggr="add", flow="source_to_target")
        edge_input_dim = 3 * hidden_size
        node_input_dim = 2 * hidden_size
        self.edge_block = build_mlp(
            in_size=edge_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )
        self.node_block = build_mlp(
            in_size=node_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers,
            layer_norm=layer_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphNetBlock.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_size].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, hidden_size].
            size (Size, optional): The size of the source and target nodes.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node features and edge features.
        """
        # Update edge attributes
        row, col = edge_index
        x_i = x[col]  # Target node features
        x_j = x[row]  # Source node features
        edge_attr = self.edge_update(edge_attr, x_i, x_j)

        # Perform message passing and update node features
        x = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        return x, edge_attr

    def edge_update(
        self, edge_attr: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates edge features.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].
            x_i (torch.Tensor): Target node features [num_edges, hidden_size].
            x_j (torch.Tensor): Source node features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Updated edge features [num_edges, hidden_size].
        """
        edge_input = torch.cat([edge_attr, x_i, x_j], dim=-1)
        edge_attr = self.edge_block(edge_input)
        return edge_attr

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages to be aggregated.

        Args:
            edge_attr (torch.Tensor): Edge features [num_edges, hidden_size].

        Returns:
            torch.Tensor: Messages [num_edges, hidden_size].
        """
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Updates node features after aggregation.

        Args:
            aggr_out (torch.Tensor): Aggregated messages [num_nodes, hidden_size].
            x (torch.Tensor): Node features [num_nodes, hidden_size].

        Returns:
            torch.Tensor: Updated node features [num_nodes, hidden_size].
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        x = self.node_block(node_input)
        return x
