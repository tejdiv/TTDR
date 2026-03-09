"""Value head V: predicts expected tracking reward from projected states.

Input: concatenation of [z'_t, z'_t1, z_wmpred] (all L2-normalized, in contrastive space Z').
Output: scalar predicted tracking reward in [-1, 1].

Architecture: small MLP with LayerNorm + GELU, tanh output (reward is cosine sim in [-1,1]).
Shares the projection head's output space -- no extra encoder computation.
"""

import flax.linen as nn
import jax.numpy as jnp


class ValueHead(nn.Module):
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, z_prime_t, z_prime_t1, z_wmpred, *, train=False):
        x = jnp.concatenate([z_prime_t, z_prime_t1, z_wmpred], axis=-1)  # (B, 3*proj_dim)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return jnp.tanh(x).squeeze(-1)  # (B,) in [-1, 1]
