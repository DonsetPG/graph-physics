import math
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from typing import Any, Optional, Union, Tuple
from collections.abc import Sequence

from jax.experimental import sparse as jsparse
from jaxtyping import Array, ArrayLike
import jax

Shape = Sequence[Union[int, Any]]


class Einsum(nnx.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(self, shape: Shape, *, rngs: nnx.Rngs):
        self.w = nnx.Param(nn.initializers.normal()(rngs.params(), shape))

    def __call__(self, eqn: str, x: ArrayLike) -> Array:
        return jnp.einsum(eqn, x, self.w.value)

    @property
    def shape(self) -> Shape:
        return self.w.value.shape


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(nn.initializers.zeros_init()(rngs.params(), dim))

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs


class LinearWithValue(nnx.Module):
    """Linear wrapper exposing a `.value` proxy to match existing tests."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )

    @property
    def value(self) -> jnp.ndarray:
        return self.linear.kernel.value

    @value.setter
    def value(self, value: jnp.ndarray) -> None:
        self.linear.kernel.value = value

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.linear(x)


class FeedForward(nnx.Module):
    """Feed forward module."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):

        self.linear1 = LinearWithValue(
            in_features=features,
            out_features=hidden_dim,
            rngs=nnx.Rngs(0),
        )
        self.linear2 = LinearWithValue(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=nnx.Rngs(0),
        )
        self.norm = RMSNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: ArrayLike) -> ArrayLike:

        ff = self.linear1(x)
        ff = nnx.relu(ff)
        ff = self.linear2(ff)
        outputs = self.norm(ff)
        return outputs


class GatedMLP(nnx.Module):
    """Feed forward module."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.gating_einsum = nnx.Param(
            nn.initializers.zeros_init()(
                rngs.params(),
                ((2, features, hidden_dim)),
            )
        )
        self.linear = LinearWithValue(
            in_features=hidden_dim,
            out_features=features,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        ff_gate = jnp.dot(x, self.gating_einsum.value[0])
        gate_value = nnx.gelu(ff_gate)

        ff1 = jnp.dot(x, self.gating_einsum.value[1])
        activations = gate_value * ff1

        outputs = self.linear(activations)
        return outputs


class Normalizer(nnx.Module):

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**5,
        std_epsilon: float = 1e-8,
    ):
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon
        self._acc_count = nnx.Variable(jnp.zeros(()))
        self._num_accumulations = nnx.Variable(jnp.zeros(()))
        self._acc_sum = nnx.Variable(jnp.zeros((1, size)))
        self._acc_sum_squared = nnx.Variable(jnp.zeros((1, size)))

    def __call__(
        self, batched_data: jnp.ndarray, accumulate: bool = True
    ) -> jnp.ndarray:
        if accumulate:
            if self._num_accumulations.value < self.max_accumulations:
                self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: jnp.ndarray) -> jnp.ndarray:
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: jnp.ndarray):
        count = batched_data.shape[0]
        data_sum = jnp.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = jnp.sum(batched_data**2, axis=0, keepdims=True)

        self._acc_sum.value += data_sum
        self._acc_sum_squared.value += squared_data_sum
        self._acc_count.value += count
        self._num_accumulations.value += 1

    def _mean(self) -> jnp.ndarray:
        safe_count = jnp.maximum(self._acc_count.value, 1.0)
        return self._acc_sum.value / safe_count

    def _std_with_epsilon(self) -> jnp.ndarray:
        safe_count = jnp.maximum(self._acc_count.value, 1.0)
        variance = self._acc_sum_squared.value / safe_count - self._mean() ** 2
        std = jnp.sqrt(jnp.clip(variance, a_min=0.0))
        return jnp.maximum(std, self.std_epsilon)


def scaled_query_key_softmax(
    q: jnp.ndarray,
    k: jnp.ndarray,
    att_mask: Optional[jsparse.BCOO] = None,
) -> jnp.ndarray:
    scaling_factor = jnp.sqrt(k.shape[-1])
    q = q / scaling_factor

    if att_mask is not None:
        attn = jsparse.bcoo_dot_general(
            att_mask,
            jnp.einsum("TNH,SNH->TNS", q, k),
            dimension_numbers=(([1], [0]), ([], [])),
        )
        attn = nn.softmax(attn)
    else:
        attn = jnp.einsum("TNH,SNH->TNS", q, k)
        attn = nn.softmax(attn, axis=-1)

    return attn


def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    att_mask: Optional[jsparse.BCOO] = None,
    return_attention: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    attn = scaled_query_key_softmax(q, k, att_mask=att_mask)
    y = jnp.einsum("TNS,BNH->BNH", attn, v)

    if return_attention:
        return y, attn
    else:
        return y


class Attention(nnx.Module):

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        assert (
            output_dim % num_heads == 0
        ), "Output dimension must be divisible by number of heads."
        head_dim = output_dim // num_heads
        self.qkv_einsum = Einsum(
            shape=(3, num_heads, input_dim, head_dim),
            rngs=rngs,
        )

        self.proj = nnx.Linear(output_dim, output_dim, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        adj: Optional[jsparse.BCOO] = None,
        return_attention: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        N = x.shape[0]
        q, k, v = self.qkv_einsum("BD,SNDH->SBNH", x)

        if return_attention:
            y, attn = scaled_dot_product_attention(q, k, v, adj, return_attention=True)
        else:
            y = scaled_dot_product_attention(q, k, v, adj)

        y = y.reshape(N, -1)
        out = self.proj(y)

        if return_attention:
            return out, attn
        else:
            return out


class Transformer(nnx.Module):

    def __init__(
        self, input_dim: int, output_dim: int, num_heads: int, *, rngs: nnx.Rngs
    ):
        self.attention = Attention(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            rngs=rngs,
        )
        self.norm1 = RMSNorm(output_dim, rngs=rngs)
        self.norm2 = RMSNorm(output_dim, rngs=rngs)
        self.gated_mlp = GatedMLP(
            features=output_dim,
            hidden_dim=3 * output_dim,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        adj: Optional[jsparse.BCOO] = None,
        return_attention: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
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


_USE_SILU_ACTIVATION = False


def set_use_silu_activation(use_silu: bool) -> None:
    global _USE_SILU_ACTIVATION
    _USE_SILU_ACTIVATION = bool(use_silu)


def use_silu_activation() -> bool:
    return _USE_SILU_ACTIVATION


def _activate(x: jnp.ndarray, act: Optional[str] = None) -> jnp.ndarray:
    act_name = act or ("silu" if _USE_SILU_ACTIVATION else "relu")
    if act_name == "relu":
        return nnx.relu(x)
    if act_name == "gelu":
        return nnx.gelu(x)
    if act_name == "silu":
        return nnx.silu(x)
    raise NotImplementedError(f"Activation '{act_name}' is not supported.")


class MLP(nnx.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        nb_of_layers: int = 4,
        layer_norm: bool = True,
        act: Optional[str] = None,
        *,
        rngs: nnx.Rngs,
    ):
        if nb_of_layers < 2:
            raise ValueError("nb_of_layers must be >= 2.")
        self.act = act
        layers = [
            nnx.Linear(in_features=in_size, out_features=hidden_size, rngs=rngs)
        ]
        for _ in range(nb_of_layers - 2):
            layers.append(
                nnx.Linear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    rngs=rngs,
                )
            )
        layers.append(
            nnx.Linear(in_features=hidden_size, out_features=out_size, rngs=rngs)
        )
        self.layers = nnx.List(layers)
        self.norm = RMSNorm(out_size, rngs=rngs) if layer_norm else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = _activate(layer(x), self.act)
        x = self.layers[-1](x)
        if self.norm is not None:
            x = self.norm(x)
        return x


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    layer_norm: bool = True,
    act: Optional[str] = None,
    *,
    rngs: nnx.Rngs,
) -> MLP:
    return MLP(
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        nb_of_layers=nb_of_layers,
        layer_norm=layer_norm,
        act=act,
        rngs=rngs,
    )


class GatedMLPBlock(nnx.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        expansion_factor: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        expanded = hidden_size * expansion_factor
        self.norm = RMSNorm(in_size, rngs=rngs)
        self.linear_gate = nnx.Linear(in_features=in_size, out_features=expanded, rngs=rngs)
        self.linear_value = nnx.Linear(in_features=in_size, out_features=expanded, rngs=rngs)
        self.linear_out = nnx.Linear(in_features=expanded, out_features=out_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.norm(x)
        gate = _activate(self.linear_gate(x), "silu" if _USE_SILU_ACTIVATION else "gelu")
        value = self.linear_value(x)
        return self.linear_out(gate * value)


def build_gated_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    expansion_factor: int = 3,
    *,
    rngs: nnx.Rngs,
) -> GatedMLPBlock:
    return GatedMLPBlock(
        in_size=in_size,
        hidden_size=hidden_size,
        out_size=out_size,
        expansion_factor=expansion_factor,
        rngs=rngs,
    )


class TemporalAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        use_gate: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_gate = use_gate

        self.q_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.mixer = build_mlp(
            in_size=2 * hidden_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=2,
            layer_norm=False,
            rngs=rngs,
        )

        if use_gate:
            self.gate = build_mlp(
                in_size=2 * hidden_size,
                hidden_size=hidden_size,
                out_size=hidden_size,
                nb_of_layers=2,
                layer_norm=False,
                act="silu",
                rngs=rngs,
            )
        else:
            self.gate = None

    def __call__(
        self,
        h_prev: jnp.ndarray,
        h_pred: jnp.ndarray,
        adj: Optional[jsparse.BCOO] = None,
    ) -> jnp.ndarray:
        n = h_prev.shape[0]
        q = self.q_proj(h_pred).reshape(n, self.head_dim, self.num_heads)
        k = self.k_proj(h_prev).reshape(n, self.head_dim, self.num_heads)
        v = self.v_proj(h_pred).reshape(n, self.head_dim, self.num_heads)

        y = scaled_dot_product_attention(q, k, v, att_mask=adj)
        out = self.out_proj(y.reshape(n, self.hidden_size))

        if self.gate is not None:
            gate = nnx.sigmoid(self.gate(jnp.concatenate([h_pred, h_prev], axis=-1)))
            out = gate * out

        h_corr = h_prev + out
        mixed = h_corr + self.mixer(jnp.concatenate([h_corr, h_prev], axis=-1))
        return mixed


class GraphNetBlock(nnx.Module):
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
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.rope_axes = rope_axes
        self.rope_base = rope_base
        self.use_gate = use_gate

        edge_in = 3 * hidden_size
        node_in = 2 * hidden_size

        if use_gated_mlp:
            self.edge_block = build_gated_mlp(
                in_size=edge_in,
                hidden_size=hidden_size,
                out_size=hidden_size,
                rngs=rngs,
            )
            self.node_block = build_gated_mlp(
                in_size=node_in,
                hidden_size=hidden_size,
                out_size=hidden_size,
                rngs=rngs,
            )
        else:
            self.edge_block = build_mlp(
                in_size=edge_in,
                hidden_size=hidden_size,
                out_size=hidden_size,
                nb_of_layers=nb_of_layers,
                layer_norm=layer_norm,
                rngs=rngs,
            )
            self.node_block = build_mlp(
                in_size=node_in,
                hidden_size=hidden_size,
                out_size=hidden_size,
                nb_of_layers=nb_of_layers,
                layer_norm=layer_norm,
                rngs=rngs,
            )

        if use_gate:
            self.gate_proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
            self.gate_pos = nnx.Param(jnp.zeros((hidden_size,)))
        else:
            self.gate_proj = None
            self.gate_pos = None

        if self.use_rope:
            if rope_axes not in (2, 3):
                raise ValueError("rope_axes must be 2 or 3 when use_rope=True.")
            self._pair_count = hidden_size // (2 * rope_axes)
            self._rope_dim = self._pair_count * 2 * rope_axes
            if self._pair_count == 0:
                raise ValueError(
                    f"hidden_size={hidden_size} too small for rope_axes={rope_axes}."
                )
            inv = jnp.arange(self._pair_count, dtype=jnp.float32)
            self._rope_inv_freq = jnp.power(
                self.rope_base, -inv / max(float(self._pair_count), 1.0)
            )
        else:
            self._pair_count = 0
            self._rope_dim = 0
            self._rope_inv_freq = jnp.zeros((0,), dtype=jnp.float32)

    def __call__(
        self,
        x: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_attr: jnp.ndarray,
        pos: Optional[jnp.ndarray] = None,
        phi: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_i = x[receivers]
        x_j = x[senders]

        if self.use_rope:
            if pos is None:
                raise ValueError("`pos` is required when use_rope=True.")
            delta_pos = pos[senders, : self.rope_axes] - pos[receivers, : self.rope_axes]
            x_j = self._apply_rope_rel(x_j, delta_pos)

        edge_input = jnp.concatenate([edge_attr, x_i, x_j], axis=-1)
        edge_update = self.edge_block(edge_input)

        num_nodes = x.shape[0]
        aggr = jax.ops.segment_sum(edge_update, receivers, num_segments=num_nodes)

        if self.use_gate and self.gate_proj is not None and self.gate_pos is not None:
            gate_logits = self.gate_proj(x)
            if phi is not None:
                gate_logits = gate_logits + phi.reshape(-1, 1) * self.gate_pos.value.reshape(1, -1)
            aggr = aggr * nnx.sigmoid(gate_logits)

        node_input = jnp.concatenate([x, aggr], axis=-1)
        x_update = self.node_block(node_input)

        return x + x_update, edge_attr + edge_update

    def _apply_rope_rel(
        self,
        x_src: jnp.ndarray,
        delta_pos: jnp.ndarray,
    ) -> jnp.ndarray:
        if self._pair_count == 0:
            return x_src

        num_edges = x_src.shape[0]
        rope_dim = self._rope_dim
        x_rot = x_src[:, :rope_dim]
        x_rest = x_src[:, rope_dim:]

        parts = []
        start = 0
        for axis in range(self.rope_axes):
            seg = x_rot[:, start : start + 2 * self._pair_count].reshape(
                num_edges, self._pair_count, 2
            )
            theta = delta_pos[:, axis][:, None] * self._rope_inv_freq[None, :]
            cos_theta = jnp.cos(theta).astype(x_src.dtype)
            sin_theta = jnp.sin(theta).astype(x_src.dtype)
            even = seg[..., 0]
            odd = seg[..., 1]
            rot_even = even * cos_theta - odd * sin_theta
            rot_odd = even * sin_theta + odd * cos_theta
            seg_rot = jnp.stack([rot_even, rot_odd], axis=-1).reshape(
                num_edges, 2 * self._pair_count
            )
            parts.append(seg_rot)
            start += 2 * self._pair_count

        x_rotated = jnp.concatenate(parts, axis=-1)
        return jnp.concatenate([x_rotated, x_rest], axis=-1)
