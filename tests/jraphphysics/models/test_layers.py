import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import sparse as jsparse
import flax.nnx as nnx
import pytest
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

# Import the modules to be tested
from jraphphysics.models.layers import (
    scaled_dot_product_attention,
    Attention,
    Transformer,
    Normalizer,
    FeedForward,
    GatedMLP,
    Einsum,
    RMSNorm,
)


class EinsumTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(
            inputs_shape=(1, 4),
            params_shape=(3, 2, 4, 3),
            eqn="TD,SNDH->STNH",
            expected_shape=(3, 1, 2, 3),
        ),
        dict(
            inputs_shape=(1, 2, 4),
            params_shape=(2, 4, 8),
            eqn="ANH,NHD->AD",
            expected_shape=(1, 8),
        ),
    )
    def test_einsum(self, inputs_shape, params_shape, eqn, expected_shape):
        einsum = Einsum(params_shape, rngs=nnx.Rngs(params=0))
        output = einsum(
            eqn,
            jnp.ones(inputs_shape),
        )
        self.assertEqual(output.shape, expected_shape)

    @parameterized.parameters(
        dict(
            shape=(1, 4),
        ),
        dict(
            shape=(2, 5, 4, 7),
        ),
    )
    def test_shape(self, shape):
        einsum = Einsum(shape, rngs=nnx.Rngs(params=0))
        self.assertEqual(einsum.shape, shape)


class RMSNormTest(parameterized.TestCase):
    @parameterized.parameters(dict(x=[0.1, 0.2], expected=[0.6324429, 1.2648858]))
    def test_rmsnorm(self, x, expected):
        x = jnp.array([x])
        rmsnorm = RMSNorm(x.shape[-1], rngs=nnx.Rngs(params=0))
        output = rmsnorm(x)
        np.testing.assert_array_equal(output, jnp.array([expected]))


class FeedForwardTest(parameterized.TestCase):

    @parameterized.parameters(
        dict(
            features=2,
            hidden_dim=3,
            batch_size=2,
            expected_val=[-0.4697958, -0.46979585],
            expected_shape=(2, 1, 3),
        ),
    )
    def test_ffw(self, features, hidden_dim, batch_size, expected_val, expected_shape):
        inputs = jnp.arange(1, batch_size + 1)[:, None, None]
        inputs = jnp.repeat(inputs, features, axis=-1)
        ffw = FeedForward(
            features=features,
            hidden_dim=hidden_dim,
            rngs=nnx.Rngs(params=0),
        )
        ffw.linear1.value = jnp.ones((features, hidden_dim))
        ffw.linear2.value = jnp.ones((hidden_dim, hidden_dim))

        with jax.default_matmul_precision("float32"):
            outputs = ffw(inputs)

        np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
        self.assertEqual(outputs.shape, expected_shape)


class GatedMLPTest(parameterized.TestCase):

    @parameterized.parameters(
        dict(
            features=2,
            hidden_dim=3,
            batch_size=2,
            expected_val=[4.1905556, 17.151281],
            expected_shape=(2, 1, 2),
        ),
    )
    def test_ffw(self, features, hidden_dim, batch_size, expected_val, expected_shape):
        inputs = jnp.arange(1, batch_size + 1)[:, None, None]
        inputs = jnp.repeat(inputs, features, axis=-1)
        ffw = GatedMLP(
            features=features,
            hidden_dim=hidden_dim,
            rngs=nnx.Rngs(params=0),
        )
        ffw.gating_einsum.value = jnp.ones((2, features, hidden_dim))
        ffw.linear.value = jnp.ones((hidden_dim, features))

        with jax.default_matmul_precision("float32"):
            outputs = ffw(inputs)

        np.testing.assert_array_almost_equal(outputs[:, 0, 0], expected_val)
        self.assertEqual(outputs.shape, expected_shape)


def test_scaled_dot_product_attention():
    key = random.key(0)
    q = random.normal(key, (10, 4, 16))
    k = random.normal(key, (10, 4, 16))
    v = random.normal(key, (10, 4, 16))

    # Test without adjacency matrix
    output = scaled_dot_product_attention(q, k, v)
    assert output.shape == (10, 4, 16)

    # Test with adjacency matrix
    adj = jsparse.BCOO(
        (
            jnp.array([1, 1, 0, 0]),  # data
            jnp.array([[0, 1], [1, 2], [2, 0], [3, 3]]),  # indices
        ),
        shape=(10, 10),
    )

    output_masked = scaled_dot_product_attention(q, k, v, adj)
    assert output_masked.shape == (10, 4, 16)

    # Test with attention return
    output_with_attn, attn = scaled_dot_product_attention(
        q, k, v, return_attention=True
    )
    assert output_with_attn.shape == (10, 4, 16)
    assert attn.shape == (10, 4, 10)


def test_attention():
    key = random.key(0)
    rngs = nnx.Rngs(key)

    # Create an attention module
    attention = Attention(input_dim=64, output_dim=64, num_heads=4, rngs=rngs)

    # Prepare input
    x = random.normal(key, (10, 64))

    # Test without adjacency matrix
    output = attention(x)
    assert output.shape == (10, 64)

    # Test with adjacency matrix
    adj = jsparse.BCOO(
        (
            jnp.array([1, 1, 0, 0]),  # data
            jnp.array([[0, 1], [1, 2], [2, 0], [3, 3]]),  # indices
        ),
        shape=(10, 10),
    )

    output_masked = attention(x, adj)
    assert output_masked.shape == (10, 64)

    # Test with attention return
    output_with_attn, attn = attention(x, return_attention=True)
    assert output_with_attn.shape == (10, 64)
    assert attn.shape == (10, 4, 10)


def test_transformer():
    key = random.key(0)
    rngs = nnx.Rngs(key)

    # Create a transformer module
    transformer = Transformer(input_dim=64, output_dim=64, num_heads=4, rngs=rngs)

    # Prepare input
    x = random.normal(key, (10, 64))

    # Test without adjacency matrix
    output = transformer(x)
    assert output.shape == (10, 64)

    # Test with adjacency matrix
    adj = jsparse.BCOO(
        (
            jnp.array([1, 1, 0, 0]),  # data
            jnp.array([[0, 1], [1, 2], [2, 0], [3, 3]]),  # indices
        ),
        shape=(10, 10),
    )

    output_masked = transformer(x, adj)
    assert output_masked.shape == (10, 64)

    # Test with attention return
    output_with_attn, attn = transformer(x, return_attention=True)
    assert output_with_attn.shape == (10, 64)
    assert attn.shape == (10, 4, 10)


def test_normalizer():
    # Create a normalizer
    key = random.key(0)
    normalizer = Normalizer(size=64)

    # Prepare input
    x = random.normal(key, (10, 64))

    # Test normalization
    normalized_x = normalizer(x)
    assert normalized_x.shape == x.shape

    # Check properties of normalized data
    assert jnp.allclose(normalized_x.mean(axis=0), 0.0, atol=1e-5)
    assert jnp.allclose(normalized_x.std(axis=0), 1.0, atol=1e-5)

    # Test inverse normalization
    reconstructed_x = normalizer.inverse(normalized_x)
    assert jnp.allclose(reconstructed_x, x, atol=1e-5)
