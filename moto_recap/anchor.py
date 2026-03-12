"""Anchor projection and value head (Flax/JAX).

AnchorProjection: maps Moto-GPT hidden state (moto_hidden_dim,) to Octo's
readout space (768,). Added to readout tokens alongside the I embedding for
classifier-free guidance conditioning.

ValueHead: predicts expected surprisal given (obs_encoding, anchor). Used
for advantage computation A = r - V(o, g). Retrained from V_0 each RECAP cycle.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn


class AnchorProjection(nn.Module):
    """Project Moto-GPT hidden state into Octo readout token space.

    The projected anchor is added to readout tokens (same mechanism as the
    indicator embedding), giving the action head goal-conditioned input.

    This module is trained during Stage 2 (policy pretraining with hindsight
    relabeling) and frozen during Stage 3 (test-time adaptation).
    """
    moto_hidden_dim: int = 768
    octo_token_dim: int = 768
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, moto_hidden):
        """
        Args:
            moto_hidden: (B, moto_hidden_dim) — Moto-GPT hidden state.

        Returns:
            anchor_embed: (B, octo_token_dim) — additive conditioning for readouts.
        """
        x = nn.Dense(self.hidden_dim)(moto_hidden)
        x = nn.gelu(x)
        x = nn.Dense(self.octo_token_dim)(x)
        return x


class ValueHead(nn.Module):
    """Predict expected surprisal reward given (obs_encoding, anchor).

    V(o, g) = MLP(concat(z_obs, g_proj))

    Trained during Stage 2 on hindsight-relabeled data. At test time,
    reset to V_0 and fine-tuned on the adaptation buffer each RECAP cycle.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs_encoding, anchor_embed):
        """
        Args:
            obs_encoding: (B, 768) — Octo encoder readout.
            anchor_embed: (B, 768) — projected anchor from AnchorProjection.

        Returns:
            v: (B,) — predicted expected surprisal.
        """
        x = jnp.concatenate([obs_encoding, anchor_embed], axis=-1)  # (B, 1536)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


def init_anchor_params(rng, moto_hidden_dim=768, octo_token_dim=768):
    """Initialize anchor projection + value head params.

    Returns:
        anchor_params: dict with 'projection' and 'value_head' sub-dicts.
    """
    rng_proj, rng_v = jax.random.split(rng)

    proj = AnchorProjection(moto_hidden_dim=moto_hidden_dim,
                            octo_token_dim=octo_token_dim)
    proj_params = proj.init(rng_proj, jnp.zeros((1, moto_hidden_dim)))

    vh = ValueHead()
    vh_params = vh.init(rng_v, jnp.zeros((1, octo_token_dim)),
                        jnp.zeros((1, octo_token_dim)))

    return {
        "projection": proj_params,
        "value_head": vh_params,
    }
