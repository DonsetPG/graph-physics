import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TransformerConv

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


class StableInjection(nn.Module):
    """Stable diagonal recurrence used by the looped transformer core."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.full((dim,), 0.1))

    def get_A(self) -> torch.Tensor:
        coeff = torch.clamp(self.log_A + self.log_dt, min=-20.0, max=20.0)
        return torch.exp(-torch.exp(coeff))

    def forward(
        self,
        h: torch.Tensor,
        encoded_input: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        a = self.get_A().view(1, -1)
        b = self.B.view(1, -1)
        return a * h + b * encoded_input + update


class LoopIndexEmbedding(nn.Module):
    """Broadcast sinusoidal loop-depth embeddings over node states."""

    def __init__(self, hidden_size: int, loop_dim: int, theta: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.loop_dim = max(0, min(loop_dim, hidden_size))
        self.theta = theta
        if self.loop_dim > 0:
            inv_freq = theta ** (
                -torch.arange(0, self.loop_dim, 2, dtype=torch.float32)
                / max(self.loop_dim, 1)
            )
        else:
            inv_freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        if self.loop_dim == 0:
            return x
        freq = self.inv_freq.to(device=x.device, dtype=x.dtype)
        angles = loop_index * freq
        emb = torch.cat((angles.sin(), angles.cos()), dim=0)[: self.loop_dim]
        full = torch.zeros(self.hidden_size, dtype=x.dtype, device=x.device)
        full[: self.loop_dim] = emb
        return x + full.view(1, -1)


class NodeMoEFFN(nn.Module):
    """Node-wise top-k MoE feed-forward layer for recurrent blocks."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        expert_ctor: Optional[Callable[[], nn.Module]] = None,
        n_shared: int = 0,
    ):
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = max(1, min(top_k, n_experts))
        self.router = nn.Linear(dim, n_experts, bias=False)
        expert_ctor = expert_ctor or (
            lambda: build_gated_mlp(
                in_size=dim,
                hidden_size=dim,
                out_size=dim,
            )
        )
        self.experts = nn.ModuleList([expert_ctor() for _ in range(n_experts)])
        self.shared = nn.ModuleList([expert_ctor() for _ in range(n_shared)])
        self.last_stats: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.router(x), dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x)
        expert_counts = torch.zeros(
            self.n_experts,
            device=x.device,
            dtype=x.dtype,
        )

        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_mask = indices == expert_id
            routed_weight = (weights * expert_mask.float()).sum(dim=-1, keepdim=True)
            out = out + routed_weight * expert_out
            expert_counts[expert_id] = expert_mask.any(dim=-1).float().sum()

        for expert in self.shared:
            out = out + expert(x)

        router_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        utilization = expert_counts / max(float(x.size(0)), 1.0)
        self.last_stats = {
            "moe/router_entropy": router_entropy.detach(),
        }
        for expert_id in range(self.n_experts):
            self.last_stats[f"moe/expert_{expert_id}_utilization"] = utilization[
                expert_id
            ].detach()
        return out


class NodeACTHalting(nn.Module):
    """Online ACT-style halting for per-node recurrent depth."""

    def __init__(self, dim: int, threshold: float = 0.99):
        super().__init__()
        self.threshold = threshold
        self.halt_head = nn.Linear(dim, 1)

    def initial_state(
        self,
        num_nodes: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        return {
            "halted": torch.zeros(num_nodes, dtype=torch.bool, device=device),
            "cum_prob": torch.zeros(num_nodes, dtype=dtype, device=device),
            "output": torch.zeros(num_nodes, hidden_size, dtype=dtype, device=device),
            "exit_depth_sum": torch.zeros(num_nodes, dtype=dtype, device=device),
            "exit_count": torch.zeros(num_nodes, dtype=dtype, device=device),
        }

    def detach_state(
        self, state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in state.items()
        }

    def step(
        self,
        candidate: torch.Tensor,
        previous: torch.Tensor,
        state: Dict[str, torch.Tensor],
        loop_active: torch.Tensor,
        loop_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        halted = state["halted"]
        effective = loop_active & (~halted)
        if not effective.any():
            return previous, state

        probs = torch.sigmoid(self.halt_head(candidate)).squeeze(-1)
        probs = probs * effective.float()
        remainder = (1.0 - state["cum_prob"]).clamp(min=0.0)
        will_halt = effective & ((state["cum_prob"] + probs) >= self.threshold)
        weights = torch.where(will_halt, remainder, probs)

        updated_state = dict(state)
        updated_state["output"] = state["output"] + weights.unsqueeze(-1) * candidate
        updated_state["cum_prob"] = state["cum_prob"] + weights
        updated_state["exit_depth_sum"] = state["exit_depth_sum"] + will_halt.float() * (
            loop_index + 1
        )
        updated_state["exit_count"] = state["exit_count"] + will_halt.float()
        updated_state["halted"] = halted | will_halt

        new_hidden = torch.where(halted.unsqueeze(-1), previous, candidate)
        return new_hidden, updated_state

    def finalize(
        self,
        current_hidden: torch.Tensor,
        state: Dict[str, torch.Tensor],
        final_depths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        remaining = ~state["halted"]
        remainder = (1.0 - state["cum_prob"]).clamp(min=0.0)
        output = state["output"] + remainder.unsqueeze(-1) * current_hidden * remaining.unsqueeze(
            -1
        )
        exit_depth_sum = state["exit_depth_sum"] + remaining.float() * final_depths.float()
        exit_count = state["exit_count"] + remaining.float()
        mean_exit_depth = exit_depth_sum.sum() / exit_count.sum().clamp_min(1.0)
        return output, {"act/mean_exit_depth": mean_exit_depth.detach()}


class AdaptiveAdjacencyPolicy(nn.Module):
    """Target-active masking and learned edge pruning over the base adjacency."""

    def __init__(
        self,
        hidden_size: int,
        mode: str = "off",
        pos_dim: int = 3,
        top_k_edges: int = 8,
        edge_gate_threshold: float = 0.5,
    ):
        super().__init__()
        valid_modes = {"off", "target_active", "edge_gate", "topk_edge_gate"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported adaptive adjacency mode: {mode}")
        self.mode = mode
        self.pos_dim = max(0, pos_dim)
        self.top_k_edges = max(1, top_k_edges)
        self.edge_gate_threshold = edge_gate_threshold
        gate_input_dim = 2 * hidden_size + self.pos_dim
        self.edge_gate = (
            build_mlp(
                in_size=gate_input_dim,
                hidden_size=hidden_size,
                out_size=1,
                nb_of_layers=3,
                layer_norm=False,
                act="silu",
            )
            if mode in {"edge_gate", "topk_edge_gate"}
            else None
        )

    def _edge_features(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        senders, receivers = edge_index
        feats = [hidden[receivers], hidden[senders]]
        if self.pos_dim > 0:
            if pos is None:
                delta = hidden.new_zeros(edge_index.size(1), self.pos_dim)
            else:
                delta = pos[receivers, : self.pos_dim] - pos[senders, : self.pos_dim]
            feats.append(delta)
        return torch.cat(feats, dim=-1)

    def _topk_mask(
        self,
        receivers: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        keep = torch.zeros_like(scores, dtype=torch.bool)
        for receiver in receivers.unique(sorted=False):
            receiver_idx = torch.nonzero(receivers == receiver, as_tuple=False).view(-1)
            top_k = min(self.top_k_edges, receiver_idx.numel())
            if top_k <= 0:
                continue
            local_scores = scores.index_select(0, receiver_idx)
            local_topk = torch.topk(local_scores, k=top_k, dim=0).indices
            keep.index_fill_(0, receiver_idx.index_select(0, local_topk), True)
        return keep

    def forward(
        self,
        base_edge_index: torch.Tensor,
        hidden: torch.Tensor,
        active_targets: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if active_targets is None:
            active_targets = torch.ones(
                hidden.size(0),
                dtype=torch.bool,
                device=hidden.device,
            )
        base_edges = max(base_edge_index.size(1), 1)
        receiver_mask = active_targets[base_edge_index[1]]

        if self.mode == "off":
            filtered_edge_index = base_edge_index
        elif self.mode == "target_active":
            filtered_edge_index = base_edge_index[:, receiver_mask]
        else:
            candidate_edges = base_edge_index[:, receiver_mask]
            if candidate_edges.numel() == 0:
                filtered_edge_index = candidate_edges
            else:
                gate_inputs = self._edge_features(hidden, candidate_edges, pos)
                scores = torch.sigmoid(self.edge_gate(gate_inputs)).squeeze(-1)
                if self.mode == "edge_gate":
                    keep = scores >= self.edge_gate_threshold
                    if not keep.any():
                        keep = scores >= scores.max()
                else:
                    keep = self._topk_mask(candidate_edges[1], scores)
                filtered_edge_index = candidate_edges[:, keep]

        stats = {
            "adj/active_target_ratio": active_targets.float().mean().detach(),
            "adj/retained_edge_ratio": hidden.new_tensor(
                filtered_edge_index.size(1) / base_edges
            ).detach(),
        }
        return filtered_edge_index, stats


class LoopedProcessorBlock(nn.Module):
    """Transformer-style block that supports custom recurrent FFNs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        use_proj_bias: bool = True,
        use_separate_proj_weight: bool = True,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        pos_dimension: int = 3,
        rope_base: float = 10000.0,
        feedforward: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_sparse_adjacency = HAS_DGL_SPARSE
        self.use_rope_embeddings = use_rope_embeddings and HAS_DGL_SPARSE
        self.norm1 = RMSNorm(output_dim)
        self.norm2 = RMSNorm(output_dim)
        self.feedforward = feedforward or build_gated_mlp(
            in_size=output_dim,
            hidden_size=output_dim,
            out_size=output_dim,
        )
        self.last_ffn_metrics: Dict[str, torch.Tensor] = {}

        if self.use_sparse_adjacency:
            self.processor = Attention(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                pos_dimension=pos_dimension,
                use_proj_bias=use_proj_bias,
                use_separate_proj_weight=use_separate_proj_weight,
                use_rope_embeddings=self.use_rope_embeddings,
                use_gated_attention=use_gated_attention,
                rope_base=rope_base,
            )
        else:
            self.processor = TransformerConv(
                in_channels=input_dim,
                out_channels=output_dim,
                heads=num_heads,
                concat=False,
                beta=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        structure,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.norm1(x)
        if self.use_sparse_adjacency:
            if self.use_rope_embeddings and pos is None:
                raise ValueError(
                    "LoopedProcessorBlock requires node positions when use_rope_embeddings=True."
                )
            x = x + self.processor(residual, structure, pos=pos)
        else:
            x = x + self.processor(residual, structure)

        ff_out = self.feedforward(self.norm2(x))
        self.last_ffn_metrics = getattr(self.feedforward, "last_stats", {})
        return x + ff_out


class RecurrentGraphCore(nn.Module):
    """Shared recurrent stack with stable injection, loop sampling, and ACT."""

    def __init__(
        self,
        recurrent_blocks: nn.ModuleList,
        hidden_size: int,
        max_loops: int,
        eval_loops: int,
        loop_embedding: Optional[LoopIndexEmbedding] = None,
        stable_injection: Optional[StableInjection] = None,
        halting: Optional[NodeACTHalting] = None,
        adjacency_policy: Optional[AdaptiveAdjacencyPolicy] = None,
        loop_sampling: str = "off",
        mu_rec: int = 8,
        mu_bwd: int = 4,
    ):
        super().__init__()
        self.recurrent_blocks = recurrent_blocks
        self.hidden_size = hidden_size
        self.max_loops = max(1, max_loops)
        self.eval_loops = max(1, eval_loops)
        self.loop_embedding = loop_embedding
        self.stable_injection = stable_injection
        self.halting = halting
        self.adjacency_policy = adjacency_policy
        self.loop_sampling = loop_sampling
        self.mu_rec = mu_rec
        self.mu_bwd = max(1, mu_bwd)
        self.input_norm = RMSNorm(hidden_size)
        self.last_metrics: Dict[str, torch.Tensor] = {}

    def _build_structure(self, edge_index: torch.Tensor, num_nodes: int):
        if HAS_DGL_SPARSE:
            return dglsp.spmatrix(indices=edge_index, shape=(num_nodes, num_nodes))
        return edge_index

    def _sample_loops(
        self,
        batch_index: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if batch_index is None:
            num_graphs = 1
        else:
            num_graphs = int(batch_index.max().item()) + 1

        if not self.training:
            return torch.full(
                (num_graphs,),
                self.eval_loops,
                device=device,
                dtype=torch.long,
            )
        if self.loop_sampling != "truncated_poisson":
            return torch.full(
                (num_graphs,),
                self.max_loops,
                device=device,
                dtype=torch.long,
            )

        sampled = torch.poisson(
            torch.full((num_graphs,), float(self.mu_rec), device=device)
        ).long()
        return sampled.clamp(min=1, max=self.max_loops)

    def _collect_ffn_metrics(self) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        for block in self.recurrent_blocks:
            for key, value in getattr(block, "last_ffn_metrics", {}).items():
                if key not in metrics:
                    metrics[key] = value.detach().float()
                else:
                    metrics[key] = metrics[key] + value.detach().float()
        if metrics:
            for key in list(metrics.keys()):
                metrics[key] = metrics[key] / max(len(self.recurrent_blocks), 1)
        return metrics

    def _detach_state(
        self,
        hidden: torch.Tensor,
        halting_state: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        hidden = hidden.detach()
        if self.halting is not None and halting_state is not None:
            halting_state = self.halting.detach_state(halting_state)
        return hidden, halting_state

    def forward(
        self,
        encoded_input: torch.Tensor,
        base_edge_index: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = encoded_input
        sampled_loops = self._sample_loops(batch_index, hidden.device)
        total_loops = int(sampled_loops.max().item())
        node_loop_limits = (
            sampled_loops[batch_index]
            if batch_index is not None
            else sampled_loops.new_full((hidden.size(0),), sampled_loops.item())
        )

        halting_state = (
            self.halting.initial_state(
                num_nodes=hidden.size(0),
                hidden_size=self.hidden_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if self.halting is not None
            else None
        )

        scalar_accumulators: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        last_residual_jump = hidden.new_tensor(0.0)

        for loop_idx in range(total_loops):
            active_nodes = node_loop_limits > loop_idx
            if halting_state is not None:
                active_nodes = active_nodes & (~halting_state["halted"])
            if not active_nodes.any():
                break

            loop_hidden = hidden
            if self.loop_embedding is not None:
                loop_hidden = self.loop_embedding(loop_hidden, loop_idx)
            recurrent_input = self.input_norm(loop_hidden + encoded_input)

            edge_index = base_edge_index
            if self.adjacency_policy is not None:
                edge_index, adj_stats = self.adjacency_policy(
                    base_edge_index,
                    hidden=hidden,
                    active_targets=active_nodes,
                    pos=pos,
                )
                for key, value in adj_stats.items():
                    scalar_accumulators[key] = scalar_accumulators.get(
                        key, hidden.new_tensor(0.0)
                    ) + value.float()
                    counts[key] = counts.get(key, 0) + 1

            structure = self._build_structure(edge_index, hidden.size(0))
            update = recurrent_input
            for block in self.recurrent_blocks:
                update = block(update, structure, pos=pos)

            if self.stable_injection is not None:
                candidate = self.stable_injection(hidden, encoded_input, update)
            else:
                candidate = hidden + update

            candidate = torch.where(active_nodes.unsqueeze(-1), candidate, hidden)
            last_residual_jump = (candidate - hidden).norm(dim=-1).mean().detach()

            if halting_state is not None:
                hidden, halting_state = self.halting.step(
                    candidate=candidate,
                    previous=hidden,
                    state=halting_state,
                    loop_active=active_nodes,
                    loop_index=loop_idx,
                )
            else:
                hidden = candidate

            loop_metrics = self._collect_ffn_metrics()
            for key, value in loop_metrics.items():
                scalar_accumulators[key] = scalar_accumulators.get(
                    key, hidden.new_tensor(0.0)
                ) + value.float()
                counts[key] = counts.get(key, 0) + 1

            if self.stable_injection is not None:
                spectral_radius = self.stable_injection.get_A().max().detach()
                scalar_accumulators["loop/spectral_radius_max"] = scalar_accumulators.get(
                    "loop/spectral_radius_max", hidden.new_tensor(0.0)
                ) + spectral_radius.float()
                counts["loop/spectral_radius_max"] = (
                    counts.get("loop/spectral_radius_max", 0) + 1
                )

            state_norm = hidden.norm(dim=-1).mean().detach()
            scalar_accumulators["loop/state_norm"] = scalar_accumulators.get(
                "loop/state_norm", hidden.new_tensor(0.0)
            ) + state_norm.float()
            counts["loop/state_norm"] = counts.get("loop/state_norm", 0) + 1

            if self.training and (loop_idx + 1) % self.mu_bwd == 0 and (loop_idx + 1) < total_loops:
                hidden, halting_state = self._detach_state(hidden, halting_state)

        metrics: Dict[str, torch.Tensor] = {}
        for key, value in scalar_accumulators.items():
            metrics[key] = value / max(counts.get(key, 1), 1)
        metrics["loop/mean_sampled_loops"] = sampled_loops.float().mean().detach()
        metrics["loop/residual_jump"] = last_residual_jump

        if halting_state is not None:
            hidden, act_metrics = self.halting.finalize(
                current_hidden=hidden,
                state=halting_state,
                final_depths=node_loop_limits.float(),
            )
            metrics.update(act_metrics)

        self.last_metrics = metrics
        return hidden, metrics


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
