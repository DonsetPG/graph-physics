import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# --- Helper SpMM: force l'op en FP32 pour éviter mismatch sous BF16 ---
import torch
from contextlib import contextmanager

# Activation checkpoint 
from torch.utils.checkpoint import checkpoint

@contextmanager
def _no_autocast_cuda():
    # évite que autocast BF16 ré-intercepte SpMM
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield

def spmm_fp32(adj, x):
    """
    Effectue Y = adj @ x en FP32 (hors autocast), puis cast Y vers x.dtype.
    Pourquoi: DGL SpMM exige même dtype entre valeurs(A) et X; BF16 n'est pas toujours supporté.
    """
    x32 = x.float()
    # Certaines versions DGL ont .astype ; sinon, le backend convertira côté C.
    try:
        adj32 = adj.astype(torch.float32)  # DGL SparseMatrix
    except AttributeError:
        adj32 = adj  # fallback: beaucoup de builds acceptent A en fp32 par défaut

    with _no_autocast_cuda():
        y32 = adj32 @ x32  # déclenche torch.ops.dgl_sparse.spmm(...)
    return y32.to(x.dtype)

# --- Helpers DGL sparse en FP32 (BF16-safe) ---
def bsddmm_fp32(mask, q, kT):
    q32, kT32 = q.float(), kT.float()
    try:
        mask32 = mask.astype(torch.float32)
    except AttributeError:
        mask32 = mask
    with _no_autocast_cuda():
        out = dglsp.bsddmm(mask32, q32, kT32)
    return out  # SparseMatrix (valeurs fp32)

def bspmm_fp32(attn, v, out_dtype):
    v32 = v.float()
    try:
        attn32 = attn.astype(torch.float32)
    except AttributeError:
        attn32 = attn
    with _no_autocast_cuda():
        y32 = dglsp.bspmm(attn32, v32)
    return y32.to(out_dtype)

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


ACTIVATION = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

_USE_SILU_ACTIVATION: bool = True


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
        act (str, optional): Activation function to use ('relu' or 'gelu'). Defaults to 'relu'.

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
        #attn = dglsp.bsddmm(att_mask, q, k.transpose(1, 0))
        attn = bsddmm_fp32(att_mask, q, k.transpose(1, 0))
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
        #y = dglsp.bspmm(attn, v)
        y = bspmm_fp32(attn, v, v.dtype)
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
            in_size=output_dim, hidden_size=output_dim, out_size=output_dim,
            expansion_factor=2   # <-- change clé : 3 -> 2
        )

        self.use_adjacency = HAS_DGL_SPARSE
        self.use_activation_checkpointing = False  # togglé depuis l'extérieur

    '''def forward(
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
            x_, attn = self.attention(self.norm1(x), adj, return_attention)
            x = x + x_
        else:
            x = x + self.attention(self.norm1(x), adj, return_attention)

        x = x + self.gated_mlp(self.norm2(x))

        if return_attention:
            return x, attn
        else:
            return x
    '''
    def forward(
            self, 
            x: torch.Tensor, 
            adj, 
            pos: Optional[torch.Tensor] = None,
            return_attention: bool = False
        ) -> torch.Tensor:
        if not self.use_adjacency:
            adj = None

        if self.use_rope_embeddings:
            if pos is None:
                raise ValueError(
                    "Transformer blocks require node positions when use_rope_embeddings=True."
                )

        if return_attention:
            # En validation/diag, on évite le checkpoint pour récupérer les poids d’attention
            x_, attn = self.attention(self.norm1(x), adj, pos=pos, return_attention=True)
            x = x + x_
            x = x + self.gated_mlp(self.norm2(x))
            return x, attn

        # --- En TRAIN: checkpoint fin pour réduire les activations gardées en mémoire ---

        # 1) Attention
        def _attn_only(tensor_x):
            # adj est capturé par la closure (non Tensor) => OK pour checkpoint
            return self.attention(self.norm1(tensor_x), adj, pos=pos, return_attention=False)

        # 2) MLP
        def _mlp_only(tensor_x):
            return self.gated_mlp(self.norm2(tensor_x))

        if self.training:
            # use_reentrant=False consomme moins de mémoire avec PyTorch ≥1.12/2.0
            x = x + checkpoint(_attn_only, x, use_reentrant=False)
            x = x + checkpoint(_mlp_only,  x, use_reentrant=False)
        else:
            x = x + self.attention(self.norm1(x), adj, pos=pos, return_attention=False)
            x = x + self.gated_mlp(self.norm2(x))

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
        edge_attr_ = self.edge_update(edge_attr, x_i, x_j)

        # Perform message passing and update node features
        x_ = self.propagate(
            edge_index, x=x, edge_attr=edge_attr_, size=(x.size(0), x.size(0))
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
