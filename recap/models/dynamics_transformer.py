"""Dynamics transformer: predicts future frame tokens from context frame tokens.

Architecture:
  Input: language embedding (projected + prepended as L tokens)
       + z_t (256 tokens) + z_{t+1} (256 tokens) from frozen VQ-VAE encoder
  Output: logits (256 × K) predicting z_{t+m} at each spatial position
  Loss: per-position cross-entropy

The transformer uses bidirectional attention over language + context tokens.
Language tokens are prepended so frame tokens attend to them, conditioning
predictions on the task instruction.

Output predictions are independent per spatial position — each position predicts
which codebook entry will appear there in m steps, conditioned on the full context.

Co-trains a value head V(z_t, z_{t+1}, lang) that predicts the expected tracking
reward (mean negative cross-entropy). Used as advantage baseline during RECAP
adaptation.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional


class DynamicsTransformerBlock(nn.Module):
    """Pre-norm transformer block for dynamics model."""
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, *, mask=None, train=False):
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=not train,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = residual + x

        # MLP
        residual = x
        x = nn.LayerNorm()(x)
        mlp_dim = int(self.embed_dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.embed_dim)(x)
        x = residual + x
        return x


class DynamicsTransformer(nn.Module):
    """Predicts future frame tokens given two context frames + language.

    Language conditioning: a (B, lang_dim) embedding from a frozen text encoder
    (e.g. CLIP) is projected to embed_dim and prepended as L tokens before the
    frame tokens. All tokens attend to each other bidirectionally.

    Input tokens from VQ-VAE codebook (looked up by index).
    Parallel output: each spatial position independently predicts
    which codebook entry will appear at that position in m steps.

    Also includes a value head that predicts expected tracking reward
    from a pooled representation of the context.
    """
    num_codes: int = 512         # VQ codebook size K
    num_positions: int = 256     # Spatial tokens per frame (16×16)
    embed_dim: int = 512         # Transformer hidden dim
    num_heads: int = 8
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    value_hidden_dim: int = 256
    value_num_layers: int = 2
    lang_dim: int = 512          # Input language embedding dim (CLIP = 512)
    num_lang_tokens: int = 4     # Number of tokens to project language into

    @nn.compact
    def __call__(self, z_t_indices, z_t1_indices, *, lang_embed=None, train=False):
        """Forward pass: language + context tokens → predicted logits + value.

        Args:
            z_t_indices: (B, N) integer indices for frame t
            z_t1_indices: (B, N) integer indices for frame t+1
            lang_embed: (B, lang_dim) language embedding from frozen text encoder.
                        If None, uses a learned null embedding (unconditional).

        Returns:
            logits: (B, N, K) predicted logits for each spatial position of z_{t+m}
            v_pred: (B,) predicted tracking reward (value baseline)
        """
        B = z_t_indices.shape[0]
        N = self.num_positions
        L = self.num_lang_tokens

        # --- Language projection → L tokens of dim D ---
        lang_proj = nn.Dense(L * self.embed_dim, name="lang_proj")
        null_embed = self.param(
            "null_lang_embed",
            nn.initializers.normal(0.02),
            (1, self.lang_dim),
        )
        if lang_embed is None:
            lang_embed = jnp.broadcast_to(null_embed, (B, self.lang_dim))
        lang_tokens = lang_proj(lang_embed).reshape(B, L, self.embed_dim)  # (B, L, D)

        # Token embeddings (shared codebook embedding for both frames)
        token_embed = nn.Embed(self.num_codes, self.embed_dim, name="token_embed")
        z_t_emb = token_embed(z_t_indices)      # (B, N, D)
        z_t1_emb = token_embed(z_t1_indices)     # (B, N, D)

        # Learned positional embeddings — separate for lang, frame0, frame1
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(0.02),
            (1, L + 2 * N, self.embed_dim),
        )

        # Concatenate: [lang_tokens, z_t, z_{t+1}] → (B, L + 2N, D)
        x = jnp.concatenate([lang_tokens, z_t_emb, z_t1_emb], axis=1)
        x = x + pos_embed

        # Transformer: bidirectional attention over all L + 2N tokens
        for _ in range(self.num_layers):
            x = DynamicsTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = nn.LayerNorm()(x)

        # --- Value head: predict expected tracking reward from context ---
        # Pool over all tokens (language + frame)
        context_pooled = jnp.mean(x, axis=1)  # (B, D)
        v = context_pooled
        for _ in range(self.value_num_layers):
            v = nn.Dense(self.value_hidden_dim)(v)
            v = nn.LayerNorm()(v)
            v = nn.gelu(v)
        v = nn.Dense(1)(v)
        v_pred = jnp.tanh(v.squeeze(-1))  # (B,) in [-1, 1]

        # --- Prediction head: per-position logits for z_{t+m} ---
        # Use the second frame's token representations (positions L+N .. L+2N-1)
        x_t1 = x[:, L + N:, :]  # (B, N, D)

        logits = nn.Dense(self.num_codes, name="output_head")(x_t1)  # (B, N, K)

        return logits, v_pred

    def score(self, z_t_indices, z_t1_indices, z_target_indices, *, lang_embed=None):
        """Compute tracking reward: mean negative CE of actual target tokens.

        Args:
            z_t_indices: (B, N) frame t tokens
            z_t1_indices: (B, N) frame t+1 tokens
            z_target_indices: (B, N) actual frame t+m tokens
            lang_embed: (B, lang_dim) language embedding, or None for unconditional.

        Returns:
            r_aux: (B,) tracking reward (higher = better prediction match)
            v_pred: (B,) value baseline
        """
        logits, v_pred = self(z_t_indices, z_t1_indices,
                              lang_embed=lang_embed, train=False)
        # Per-position log probabilities
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, N, K)
        # Gather log prob of actual target token at each position
        target_log_probs = jnp.take_along_axis(
            log_probs, z_target_indices[:, :, None], axis=-1
        ).squeeze(-1)  # (B, N)
        # Mean over spatial positions → scalar reward per batch element
        r_aux = jnp.mean(target_log_probs, axis=-1)  # (B,)
        return r_aux, v_pred
