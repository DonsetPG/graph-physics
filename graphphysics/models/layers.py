import math
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

        rms_x = norm_x / math.sqrt(d_x)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


_USE_SILU_ACTIVATION: bool = False


def set_use_silu_activation(use_silu: bool) -> None:
    """
    Toggles whether SiLU should be used as the default activation across MLP utilities.
    """
    global _USE_SILU_ACTIVATION
    _USE_SILU_ACTIVATION = use_silu


def use_silu_activation() -> bool:
    """
    Returns True if SiLU activations are globally enabled.
    """
    return _USE_SILU_ACTIVATION


def _resolve_activation(act: Optional[str]) -> str:
    if act is None:
        return "silu" if _USE_SILU_ACTIVATION else "relu"
    return act


ACTIVATION = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: Optional[str] = None,
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
        act (str, optional): Activation function to use. Defaults to 'relu',
            unless SiLU has been globally enabled.

    Returns:
        nn.Module: The constructed MLP model.
    """
    assert nb_of_layers >= 2, "The MLP must have at least 2 layers (input and output)."

    act_key = _resolve_activation(act)

    if act_key not in ACTIVATION:
        raise NotImplementedError(
            f"Activation '{act_key}' not supported. Available: {list(ACTIVATION)}."
        )
    activation = ACTIVATION[act_key]

    layers = [nn.Linear(in_size, hidden_size), activation()]

    # Add hidden layers
    for _ in range(nb_of_layers - 2):
        layers.extend([nn.Linear(hidden_size, hidden_size), activation()])

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

        activation_cls = nn.SiLU if use_silu_activation() else nn.GELU
        self.activation = activation_cls()

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


def _make_inv_freq(m: int, base: float, device: torch.device) -> torch.Tensor:
    """
    Precomputes inverse frequencies for rotary positional embeddings.
    """
    if m <= 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    step = math.log(base) / max(m, 1)
    return torch.exp(-torch.arange(m, device=device, dtype=torch.float32) * step)


def _apply_rope_with_inv(
    q: torch.Tensor,
    k: torch.Tensor,
    pos: torch.Tensor,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (N, D, H).
        k (torch.Tensor): Key tensor of shape (N, D, H).
        pos (torch.Tensor): Positional tensor of shape (N, pos_dim).
        inv_freq (torch.Tensor): Precomputed inverse frequencies of shape (m,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    N, D, H = q.shape
    pos_dimension = pos.shape[1]
    m = D // (pos_dimension * 2)
    if m == 0 or inv_freq.numel() == 0:
        return q, k

    d_rope = pos_dimension * 2 * m
    q_dtype = q.dtype

    pos_f32 = pos[:, :pos_dimension].to(torch.float32)
    inv_freq_f32 = inv_freq.to(pos.device, dtype=torch.float32)
    angles = pos_f32.unsqueeze(-1) * inv_freq_f32.view(1, 1, m)

    if hasattr(torch, "sincos"):
        sin_f32, cos_f32 = torch.sincos(angles)
    else:
        cos_f32, sin_f32 = torch.cos(angles), torch.sin(angles)

    sin = sin_f32.to(dtype=q_dtype, device=q.device)
    cos = cos_f32.to(dtype=q_dtype, device=q.device)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        part = (
            x[:, :d_rope, :]
            .contiguous()
            .view(N, pos_dimension, 2 * m, H)
            .view(N, pos_dimension, m, 2, H)
        )
        rest = x[:, d_rope:, :]

        even = part[..., 0, :]
        odd = part[..., 1, :]

        cos_b = cos.unsqueeze(-1)
        sin_b = sin.unsqueeze(-1)

        rot_even = even * cos_b - odd * sin_b
        rot_odd = even * sin_b + odd * cos_b

        rot = (
            torch.stack((rot_even, rot_odd), dim=3)
            .reshape(N, pos_dimension, 2 * m, H)
            .reshape(N, d_rope, H)
        )

        out = torch.empty_like(x)
        out[:, :d_rope, :] = rot
        if D > d_rope:
            out[:, d_rope:, :] = rest
        return out

    return _apply(q), _apply(k)


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
    scaling_factor = math.sqrt(k.size(1))
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
        pos_dimension: int = 3,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_base: float = 10000.0,
    ):
        """
        Initializes the Attention module.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_heads (int): Number of attention heads.
            pos_dimension (int): Spatial dimensionality used for RoPE.
            use_proj_bias (bool, optional): Whether to use bias in projection layers.
                Defaults to True.
            use_separate_proj_weight (bool, optional): Whether to use separate weights
                for Q, K, V projections. If False, weights are shared. Defaults to True.
            use_rope_embeddings (bool, optional): Whether to enable rotary positional embeddings.
                Defaults to False.
            use_gated_attention (bool, optional): Whether to apply a learnable gate on the attention output.
                Defaults to False.
            rope_base (float, optional): Base used for inverse frequency calculation in RoPE.
                Defaults to 10000.0.
        """
        super().__init__()

        assert (
            output_dim % num_heads == 0
        ), "Output dimension must be divisible by number of heads."

        self.hidden_size = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.pos_dimension = pos_dimension
        self.rope_base = rope_base

        self.q_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.v_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        self.proj = nn.Linear(output_dim, output_dim, bias=use_proj_bias)

        if self.use_rope_embeddings:
            self.m = self.head_dim // max(self.pos_dimension * 2, 1)
            inv = _make_inv_freq(self.m, self.rope_base, torch.device("cpu"))
            self.register_buffer("rope_inv_freq", inv, persistent=True)
        else:
            self.m = 0
            self.register_buffer(
                "rope_inv_freq", torch.empty(0, dtype=torch.float32), persistent=False
            )

        if self.use_gated_attention:
            self.gate_proj = nn.Linear(input_dim, output_dim, bias=use_proj_bias)
        else:
            self.gate_proj = None

        if not use_separate_proj_weight:
            # Compute optimization used at times, share the parameters in between Q/K/V
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

    def forward(
        self,
        x: torch.Tensor,
        adj,
        pos: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
            adj (Optional[SparseMatrix]): Optional adjacency matrix for sparse attention.
            pos (Optional[torch.Tensor]): Positional tensor of shape (N, pos_dimension) used for RoPE.
            return_attention (bool, optional): Whether to return attention weights.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                The output tensor, and optionally the attention weights.
        """
        if self.use_rope_embeddings:
            if pos is None:
                raise ValueError(
                    "RoPE embeddings require positional information when enabled."
                )

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

        if self.use_rope_embeddings and self.rope_inv_freq.numel() > 0:
            q, k = _apply_rope_with_inv(q, k, pos, self.rope_inv_freq)

        if return_attention:
            y, attn = scaled_dot_product_attention(q, k, v, adj, return_attention=True)
        else:
            y = scaled_dot_product_attention(q, k, v, adj)

        if self.use_gated_attention and self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(x)).reshape(
                N, self.head_dim, self.num_heads
            )
            gate = gate.to(dtype=y.dtype, device=y.device)
            y = y * gate

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
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        pos_dimension: int = 3,
        rope_base: float = 10000.0,
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
            use_rope_embeddings (bool, optional): Whether to enable rotary positional embeddings.
            use_gated_attention (bool, optional): Whether to apply learned gating on attention outputs.
            pos_dimension (int, optional): Dimensionality of positional information for RoPE.
            rope_base (float, optional): Base value for RoPE frequency computation.
        """
        super().__init__()

        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.pos_dimension = pos_dimension

        self.attention = Attention(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            pos_dimension=pos_dimension,
            use_proj_bias=use_proj_bias,
            use_separate_proj_weight=use_separate_proj_weight,
            use_rope_embeddings=use_rope_embeddings,
            use_gated_attention=use_gated_attention,
            rope_base=rope_base,
        )

        # initialize mlp
        self.activation = activation_layer()
        self.norm1, self.norm2 = RMSNorm(output_dim), RMSNorm(output_dim)
        self.gated_mlp = build_gated_mlp(
            in_size=output_dim, hidden_size=output_dim, out_size=output_dim
        )

        self.use_adjacency = HAS_DGL_SPARSE

    def forward(
        self,
        x: torch.Tensor,
        adj,
        pos: Optional[torch.Tensor] = None,
        return_attention: bool = False,
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

        if self.use_rope_embeddings:
            if pos is None:
                raise ValueError(
                    "Transformer blocks require node positions when use_rope_embeddings=True."
                )

        if return_attention:
            x_, attn = self.attention(
                self.norm1(x), adj, pos=pos, return_attention=True
            )
            x = x + x_
        else:
            x = x + self.attention(self.norm1(x), adj, pos=pos)

        x = x + self.gated_mlp(self.norm2(x))

        if return_attention:
            return x, attn
        else:
            return x


class TemporalAttention(nn.Module):
    """
    Temporal corrector as sparse cross-attention.
    Queries/Values: predicted state
    Keys:           previous state
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, use_gate: bool = True):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.h = hidden_size
        self.H = num_heads
        self.d = hidden_size // num_heads
        self.use_gate = use_gate

        # Per-node linear projections
        self.q_proj = nn.Linear(self.h, self.h, bias=True)
        self.k_proj = nn.Linear(self.h, self.h, bias=True)
        self.v_proj = nn.Linear(self.h, self.h, bias=True)
        self.out_proj = nn.Linear(self.h, self.h, bias=True)

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * self.h, self.h),
                nn.SiLU(),
                nn.Linear(self.h, self.h),
                nn.Sigmoid(),
            )

        self.mixer = nn.Sequential(
            nn.Linear(2 * self.h, self.h),
            nn.SiLU(),
            nn.Linear(self.h, self.h),
        )

    def forward(
        self,
        h_prev: torch.Tensor,  # [N, H]
        h_pred: torch.Tensor,  # [N, H]
        adj: "SparseMatrix" = None,
    ) -> torch.Tensor:

        N = h_prev.size(0)

        # Project and split heads
        q = self.q_proj(h_pred)
        k = self.k_proj(h_prev)
        v = self.v_proj(h_pred)

        q = q.reshape(N, self.d, self.H)
        k = k.reshape(N, self.d, self.H)
        v = v.reshape(N, self.d, self.H)

        y = scaled_dot_product_attention(q, k, v, adj)

        out = self.out_proj(y.reshape(N, self.h))

        if self.use_gate:
            g = self.gate(torch.cat([h_pred, h_prev], dim=-1))
            out = g * out
        h_corr = h_prev + out

        fused = h_corr + self.mixer(torch.cat([h_corr, h_prev], dim=-1))
        return fused


class GraphNetBlock(MessagePassing):
    """
    Graph Network Block implementing the message passing mechanism.
    This block updates both node and edge features.
    """

    def __init__(
        self,
        hidden_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = True,
        use_rope: bool = False,
        rope_axes: int = 3,
        rope_base: float = 10000.0,
        use_gated_mlp: bool = False,
        use_gate: bool = False,
    ):
        """
        Initializes the GraphNetBlock.

        Args:
            hidden_size (int): The size of the hidden representations.
            nb_of_layers (int, optional): The number of layers in the MLPs.
                Defaults to 4.
            layer_norm (bool, optional): Whether to use layer normalization in the MLPs.
                Defaults to True.
            use_rope (bool, optional): Apply rotary position embeddings to source node
                features before message construction. Defaults to False.
            rope_axes (int, optional): Number of spatial axes (2 or 3) to use for RoPE.
                Defaults to 3.
            rope_base (float, optional): Frequency base for RoPE. Defaults to 10000.0.
            use_gated_mlp (bool, optional): Replace edge/node MLPs with gated variants.
                Defaults to False.
            use_gate (bool, optional): Enable query-conditioned multiplicative gating on
                aggregated messages. Defaults to False.
        """
        super().__init__(aggr="add", flow="source_to_target")
        edge_input_dim = 3 * hidden_size
        node_input_dim = 2 * hidden_size
        self.hidden_size = hidden_size
        self.use_gated_mlp = use_gated_mlp

        if self.use_gated_mlp:
            self.edge_block = build_gated_mlp(
                in_size=edge_input_dim,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )
            self.node_block = build_gated_mlp(
                in_size=node_input_dim,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )
        else:
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

        # RoPE configuration
        self.use_rope = use_rope
        self.rope_axes = rope_axes
        self.rope_base = rope_base

        if self.use_rope:
            if rope_axes not in (2, 3):
                raise ValueError("rope_axes must be 2 or 3 when use_rope=True.")
            self._pair_count = hidden_size // (2 * rope_axes)
            self._rope_dim = self._pair_count * 2 * rope_axes
            if self._pair_count == 0:
                raise ValueError(
                    f"hidden_size={hidden_size} too small for rope_axes={rope_axes}; "
                    "need at least 2 * rope_axes channels."
                )
            inv = torch.arange(self._pair_count, dtype=torch.float32)
            denom = max(float(self._pair_count), 1.0)
            inv = torch.pow(self.rope_base, -inv / denom)
            self.register_buffer("_rope_inv_freq", inv, persistent=False)
        else:
            self._pair_count = 0
            self._rope_dim = 0
            self.register_buffer("_rope_inv_freq", torch.zeros(0), persistent=False)

        # Gated aggregation configuration
        self.use_gate = use_gate
        if self.use_gate:
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.gate_pos = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: int = None,
        pos: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphNetBlock.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, hidden_size].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, hidden_size].
            size (Size, optional): The size of the source and target nodes.
                Defaults to None.
            pos (torch.Tensor, optional): Node positions of shape [num_nodes, rope_axes].
                Required when use_rope is True. Defaults to None.
            phi (torch.Tensor, optional): Optional per-node scalar used for the gate.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node features and edge features.
        """
        # Update edge attributes
        row, col = edge_index
        x_i = x[col]  # Target node features
        x_j = x[row]  # Source node features

        if self.use_rope:
            if pos is None:
                raise ValueError(
                    "Node positions `pos` must be provided when use_rope=True."
                )
            delta_pos = pos[row, : self.rope_axes] - pos[col, : self.rope_axes]
            x_j = self._apply_rope_rel(x_j, delta_pos)

        edge_attr_ = self.edge_update(edge_attr, x_i, x_j)

        # Perform message passing and update node features
        x_ = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr_,
            size=(x.size(0), x.size(0)),
            phi=phi,
        )

        edge_attr = edge_attr + edge_attr_
        x = x + x_

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

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Updates node features after aggregation.

        Args:
            aggr_out (torch.Tensor): Aggregated messages [num_nodes, hidden_size].
            x (torch.Tensor): Node features [num_nodes, hidden_size].
            phi (torch.Tensor, optional): Optional per-node scalar used for gating.

        Returns:
            torch.Tensor: Updated node features [num_nodes, hidden_size].
        """
        if self.use_gate:
            gate_logits = self.gate_proj(x)
            if phi is not None:
                phi = phi.view(-1, 1).to(device=gate_logits.device, dtype=gate_logits.dtype)
                gate_logits = gate_logits + phi * self.gate_pos.view(1, -1)
            gate_logits = gate_logits.to(dtype=aggr_out.dtype, device=aggr_out.device)
            gate = torch.sigmoid(gate_logits)
            aggr_out = aggr_out * gate

        node_input = torch.cat([x, aggr_out], dim=-1)
        x = self.node_block(node_input)
        return x

    def _apply_rope_rel(
        self, x_src: torch.Tensor, delta_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply relative 2D/3D RoPE rotations to the source node features.

        Args:
            x_src (torch.Tensor): Source node features [num_edges, hidden_size].
            delta_pos (torch.Tensor): Relative offsets [num_edges, rope_axes].

        Returns:
            torch.Tensor: Rotated source node features [num_edges, hidden_size].
        """
        if self._pair_count == 0:
            return x_src

        num_edges, hidden_dim = x_src.shape
        rope_dim = self._rope_dim

        x_rot = x_src[:, :rope_dim]
        x_rest = x_src[:, rope_dim:]

        parts = []
        start = 0
        inv_freq = self._rope_inv_freq
        delta = delta_pos.to(device=x_src.device, dtype=inv_freq.dtype)

        for axis in range(self.rope_axes):
            seg = x_rot[:, start : start + 2 * self._pair_count].reshape(
                num_edges, self._pair_count, 2
            )
            theta = delta[:, axis].unsqueeze(1) * inv_freq.unsqueeze(0)
            cos_theta = torch.cos(theta).to(dtype=x_src.dtype)
            sin_theta = torch.sin(theta).to(dtype=x_src.dtype)
            even = seg[..., 0]
            odd = seg[..., 1]
            rot_even = even * cos_theta - odd * sin_theta
            rot_odd = even * sin_theta + odd * cos_theta
            seg_rot = torch.stack([rot_even, rot_odd], dim=-1).reshape(
                num_edges, 2 * self._pair_count
            )
            parts.append(seg_rot)
            start += 2 * self._pair_count

        x_rotated = torch.cat(parts, dim=-1)
        return torch.cat([x_rotated, x_rest], dim=-1)
