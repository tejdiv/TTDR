"""Projection head h: maps Octo encoder outputs φ(o) into contrastive dynamics space Z'.

Architecture: MLP 768 → 512 → 256, with LayerNorm + GELU, L2-normalized output.
The contrastive space Z' is shaped by InfoNCE loss so that L2 distances reflect
one-step dynamics similarity, not semantic similarity.

~1.2M parameters.
"""

import flax.linen as nn
import jax.numpy as jnp


class ProjectionHead(nn.Module):
    """Projects Octo encoder readout tokens into contrastive dynamics space.

    Input:  z_t = φ(o_t), shape (batch, encoder_dim)    [768 for Octo-Base]
    Output: z'_t ∈ Z',    shape (batch, output_dim)      [256, L2-normalized]
    """

    hidden_dim: int = 512
    output_dim: int = 256

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: Encoder output, shape (batch, encoder_dim).
        Returns:
            z_prime: L2-normalized projection, shape (batch, output_dim).
        """
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        # L2 normalize onto unit sphere — distances in Z' are meaningful
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x
