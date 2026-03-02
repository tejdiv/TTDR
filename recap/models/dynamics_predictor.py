"""Dynamics predictor f_ψ: predicts next-step state in contrastive space Z'.

Given current projected state z'_t, predicts the anchor ẑ'_{t+m} — where the
agent should be m steps into the future.

Architecture: 3-hidden-layer MLP (width 1024), LayerNorm + GELU activations,
L2-normalized output. Language conditioning is already encoded in z'_t via
Octo's cross-attention between T5 language tokens and image tokens.

~2.6M parameters.
"""

import flax.linen as nn
import jax.numpy as jnp


class DynamicsPredictor(nn.Module):
    """Predicts next-step anchor in contrastive space Z'.

    Input:
        z_prime: projected state h(φ(o_t)),  shape (batch, z_dim)     [256]

    Output:
        anchor: predicted ẑ'_{t+m},          shape (batch, z_dim)     [256, L2-normalized]
    """

    hidden_dim: int = 1024
    num_layers: int = 3

    @nn.compact
    def __call__(self, z_prime: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        """
        Args:
            z_prime: Projected state, shape (batch, z_dim).
            train: Whether in training mode (unused, kept for API consistency).
        Returns:
            anchor: L2-normalized predicted next state, shape (batch, z_dim).
        """
        z_dim = z_prime.shape[-1]
        x = z_prime

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        x = nn.Dense(z_dim)(x)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x
