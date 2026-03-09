"""VQ-VAE tokenizer: encodes 256x256 RGB frames into discrete spatial tokens.

Architecture (Genie-style, scaled down):
  Encoder: ViT — patchify (16x16 patches) → transformer → quantize → discrete tokens
  Decoder: CNN — codebook embeddings → ConvTranspose upsample → reconstructed pixels

Each frame is encoded independently (no temporal conditioning).
16x16 patches on 256x256 → 16×16 = 256 spatial positions, each picking from K codebook entries.

Training losses:
  L_recon: MSE(x̂_t, x_t) + optional perceptual (LPIPS)
  L_vq: commitment loss ||e - sg[z_q]||² + codebook loss ||sg[e] - z_q||²
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class PatchEmbed(nn.Module):
    """Patchify image and project to embedding dim."""
    patch_size: int = 16
    embed_dim: int = 512

    @nn.compact
    def __call__(self, x):
        """x: (B, H, W, 3) → (B, n_patches, embed_dim)"""
        B, H, W, C = x.shape
        p = self.patch_size
        # Conv2d with kernel=stride=patch_size is equivalent to linear patch projection
        x = nn.Conv(self.embed_dim, kernel_size=(p, p), strides=(p, p),
                     padding="VALID", name="patch_proj")(x)
        # (B, H//p, W//p, embed_dim) → (B, n_patches, embed_dim)
        return x.reshape(B, -1, self.embed_dim)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, *, train=False):
        # Self-attention (bidirectional)
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=not train,
        )(x, x)
        x = residual + x

        # MLP
        residual = x
        x = nn.LayerNorm()(x)
        mlp_dim = int(self.embed_dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = residual + x
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator.

    Maintains a codebook of K vectors of dimension D.
    Maps continuous encoder outputs to nearest codebook entry.
    """
    num_codes: int = 512
    code_dim: int = 512
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, z_e):
        """
        Args:
            z_e: Encoder output, (B, N, D) continuous vectors.

        Returns:
            z_q: Quantized vectors (B, N, D) — straight-through gradient.
            indices: Codebook indices (B, N) integers.
            vq_loss: Scalar VQ loss (commitment + codebook).
        """
        codebook = self.param(
            "codebook",
            nn.initializers.uniform(scale=1.0 / self.num_codes),
            (self.num_codes, self.code_dim),
        )

        # Distances: (B, N, K)
        # ||z_e - c_k||² = ||z_e||² - 2*z_e·c_k + ||c_k||²
        z_e_sq = jnp.sum(z_e ** 2, axis=-1, keepdims=True)        # (B, N, 1)
        c_sq = jnp.sum(codebook ** 2, axis=-1, keepdims=False)    # (K,)
        dot = jnp.einsum("bnd,kd->bnk", z_e, codebook)            # (B, N, K)
        distances = z_e_sq - 2 * dot + c_sq[None, None, :]        # (B, N, K)

        # Nearest codebook entry
        indices = jnp.argmin(distances, axis=-1)  # (B, N)
        z_q = codebook[indices]                    # (B, N, D)

        # VQ losses
        commitment_loss = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
        codebook_loss = jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward uses z_q, backward passes to z_e
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        return z_q_st, indices, vq_loss


class Encoder(nn.Module):
    """ViT encoder: image → continuous patch embeddings."""
    patch_size: int = 16
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, *, train=False):
        """x: (B, 256, 256, 3) float in [0, 1] → (B, N, embed_dim)"""
        # Patchify
        x = PatchEmbed(self.patch_size, self.embed_dim)(x)  # (B, N, D)
        B, N, D = x.shape

        # Learned positional embedding
        pos_embed = self.param("pos_embed", nn.initializers.normal(0.02), (1, N, D))
        x = x + pos_embed

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x, train=train)

        x = nn.LayerNorm()(x)
        return x  # (B, N, D)


class Decoder(nn.Module):
    """CNN decoder: codebook embeddings → reconstructed image.

    Reshapes (B, N, D) → (B, H', W', D) spatial grid,
    then upsamples with ConvTranspose + ResBlocks to (B, 256, 256, 3).
    """
    patch_size: int = 16
    embed_dim: int = 512

    @nn.compact
    def __call__(self, z_q, *, train=False):
        """z_q: (B, N, D) quantized embeddings → (B, 256, 256, 3) reconstructed image."""
        B, N, D = z_q.shape
        grid_size = int(N ** 0.5)  # 16 for N=256

        # Reshape to spatial grid
        x = z_q.reshape(B, grid_size, grid_size, D)  # (B, 16, 16, 512)

        # Upsample: 16→32→64→128→256 (4 stages, each 2x)
        channels = [256, 128, 64, 32]
        for ch in channels:
            x = nn.ConvTranspose(ch, kernel_size=(4, 4), strides=(2, 2),
                                  padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(32, ch))(x)
            x = nn.gelu(x)
            # ResBlock: conv → norm → gelu → conv → add
            residual = x
            x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(32, ch))(x)
            x = nn.gelu(x)
            x = nn.Conv(ch, kernel_size=(3, 3), padding="SAME")(x)
            x = residual + x

        # Final projection to RGB
        x = nn.Conv(3, kernel_size=(1, 1))(x)
        x = nn.sigmoid(x)  # Output in [0, 1]
        return x  # (B, 256, 256, 3)


class VQVAETokenizer(nn.Module):
    """Complete VQ-VAE: encode frames to discrete tokens, decode back to pixels.

    Architecture:
        Encoder (ViT): (B, 256, 256, 3) → (B, 256, 512) continuous
        VQ: (B, 256, 512) → (B, 256, 512) quantized + (B, 256) indices
        Decoder (CNN): (B, 256, 512) → (B, 256, 256, 3) reconstructed

    Training: call with train=True, returns (x_recon, indices, vq_loss)
    Inference: use encode() for tokenization, decode() for reconstruction
    """
    patch_size: int = 16
    embed_dim: int = 512
    num_codes: int = 512
    encoder_layers: int = 6
    encoder_heads: int = 8
    commitment_cost: float = 0.25

    def setup(self):
        self.encoder = Encoder(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.encoder_heads,
            num_layers=self.encoder_layers,
        )
        self.vq = VectorQuantizer(
            num_codes=self.num_codes,
            code_dim=self.embed_dim,
            commitment_cost=self.commitment_cost,
        )
        self.decoder = Decoder(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )

    def encode(self, x, *, train=False):
        """Encode image to discrete token indices.

        Args:
            x: (B, 256, 256, 3) float in [0, 1]
        Returns:
            indices: (B, 256) integer codebook indices
        """
        z_e = self.encoder(x, train=train)
        _, indices, _ = self.vq(z_e)
        return indices

    def decode_from_indices(self, indices):
        """Decode discrete token indices back to image.

        Args:
            indices: (B, 256) integer codebook indices
        Returns:
            x_recon: (B, 256, 256, 3) float in [0, 1]
        """
        codebook = self.vq.variables["params"]["codebook"]
        z_q = codebook[indices]  # (B, 256, embed_dim)
        return self.decoder(z_q)

    def __call__(self, x, *, train=False):
        """Full forward pass for training.

        Args:
            x: (B, 256, 256, 3) float in [0, 1]
        Returns:
            x_recon: (B, 256, 256, 3) reconstructed image
            indices: (B, 256) codebook indices
            vq_loss: scalar VQ loss
        """
        z_e = self.encoder(x, train=train)         # (B, 256, D)
        z_q, indices, vq_loss = self.vq(z_e)        # (B, 256, D), (B, 256), scalar
        x_recon = self.decoder(z_q, train=train)     # (B, 256, 256, 3)
        return x_recon, indices, vq_loss
