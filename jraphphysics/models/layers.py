import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from typing import Any, Optional, Union, Tuple
from collections.abc import Sequence

from jax.experimental import sparse as jsparse
from jaxtyping import Array, ArrayLike

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


class FeedForward(nnx.Module):
    """Feed forward module."""

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):

        self.linear1 = nnx.Linear(
            in_features=features, out_features=hidden_dim, rngs=nnx.Rngs(0)
        )
        self.linear2 = nnx.Linear(
            in_features=hidden_dim, out_features=hidden_dim, rngs=nnx.Rngs(0)
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
        self.linear = nnx.Linear(
            in_features=hidden_dim, out_features=features, rngs=nnx.Rngs(0)
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
