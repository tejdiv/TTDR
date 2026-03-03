"""Dynamics predictor f_ψ: predicts next-step state in contrastive space Z'.

Given projected state(s), predicts the anchor ẑ'_{t+m} — where the
agent should be m steps into the future.

Architecture: MLP with LayerNorm + GELU activations, L2-normalized output.
Hidden dim, num layers, and output dim are configurable. Language conditioning
is already encoded in the input via Octo's cross-attention.
"""

import flax.linen as nn
import jax.numpy as jnp


class DynamicsPredictor(nn.Module):
    """Predicts next-step anchor in contrastive space Z'.

    Input:
        z_prime: concatenated projected states [h(read[t-1]), h(read[t,t-1])],
                 shape (batch, 2 * proj_dim)

    Output:
        anchor: predicted ẑ'_{t+m}, shape (batch, output_dim), L2-normalized.
    """

    hidden_dim: int = 1024
    num_layers: int = 3
    output_dim: int = None  # If None, infer from input (backward compat)

    @nn.compact
    def __call__(self, z_prime: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        """
        Args:
            z_prime: Projected state (or concatenated states), shape (batch, input_dim).
            train: Whether in training mode (unused, kept for API consistency).
        Returns:
            anchor: L2-normalized predicted next state, shape (batch, out_dim).
        """
        out_dim = self.output_dim if self.output_dim is not None else z_prime.shape[-1]
        x = z_prime

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        x = nn.Dense(out_dim)(x)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x
