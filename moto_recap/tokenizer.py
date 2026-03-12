"""Wrapper around Moto's frozen Latent Motion Tokenizer (PyTorch).

The tokenizer encodes a pair of frames (f_t, f_{t+m}) into 8 discrete motion
tokens from a codebook of size 128 (dim 32). These tokens capture structured
visual changes (arm position, object displacement, gripper state) between frames.

Architecture (from Moto config):
  - Image encoder: facebook/vit-mae-large (frozen, mask_ratio=0)
  - M-Former: 4-layer ViT with 8 learned query tokens (query_num=8)
  - VQ: codebook n_e=128, e_dim=32, beta=0.25
  - Decoder: 12-layer ViT, patch_size=16, image_size=224

Real Moto API:
  - tokenizer.tokenize(cond_pv, target_pv) → (quant, indices, commit_loss)
  - tokenizer.decode_image(cond_pv, token_ids) → {"recons_pixel_values": tensor}
  - tokenizer.forward(..., return_motion_token_ids_only=True) → indices
  - Input tensors: (B, 3, 224, 224) normalized with ImageNet stats

The tokenizer is always frozen — it was pretrained on OXE and defines the
motion vocabulary. Only the GPT prior over these tokens gets fine-tuned.

HuggingFace checkpoint: TencentARC/Moto (subdir: latent_motion_tokenizer_trained_on_oxe/)
"""

import numpy as np
import torch
import torchvision.transforms as T


# ImageNet normalization (ViT-MAE-Large expects this)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _preprocess_frames(frames_np, device):
    """Convert (B, H, W, 3) float32 [0,1] numpy → (B, 3, 224, 224) normalized torch.

    Moto's ViT-MAE-Large expects 224x224 images normalized with ImageNet stats.
    """
    # (B, H, W, 3) → (B, 3, H, W)
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device)
    return _preprocess(t)


def load_tokenizer(checkpoint="TencentARC/Moto", device="cuda"):
    """Load frozen Moto latent motion tokenizer from HuggingFace.

    Downloads the OXE-pretrained tokenizer weights from HuggingFace and
    builds the full LatentMotionTokenizer (image_encoder + m_former +
    vector_quantizer + decoder) using hydra/OmegaConf config.

    Args:
        checkpoint: HuggingFace repo ID or local path to checkpoint dir.
        device: torch device.

    Returns:
        tokenizer: LatentMotionTokenizer (eval mode, all params frozen).
    """
    import os
    from huggingface_hub import snapshot_download

    # Download full repo (or use local path)
    if os.path.isdir(checkpoint):
        local_path = checkpoint
    else:
        local_path = snapshot_download(checkpoint)

    # The OXE tokenizer checkpoint is in a subdirectory
    tokenizer_dir = os.path.join(local_path, "latent_motion_tokenizer_trained_on_oxe")
    if not os.path.isdir(tokenizer_dir):
        # Maybe the user pointed directly at the tokenizer dir
        tokenizer_dir = local_path

    # Load checkpoint — Moto saves as a .pt or .ckpt file
    ckpt_files = [f for f in os.listdir(tokenizer_dir)
                  if f.endswith((".pt", ".ckpt", ".bin"))]
    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoint found in {tokenizer_dir}. "
            f"Contents: {os.listdir(tokenizer_dir)}"
        )
    ckpt_path = os.path.join(tokenizer_dir, ckpt_files[0])

    # Build model from Moto's source
    from latent_motion_tokenizer.src.models.latent_motion_tokenizer import LatentMotionTokenizer
    from latent_motion_tokenizer.src.models.m_former import MFormer
    from latent_motion_tokenizer.src.models.vector_quantizer import VectorQuantizer2
    from latent_motion_tokenizer.src.models.latent_motion_decoder import LatentMotionDecoder
    from transformers import ViTMAEModel, ViTConfig

    # Image encoder: frozen ViT-MAE-Large
    image_encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-large")

    # M-Former: 4-layer ViT with 8 query tokens
    mformer_config = ViTConfig(
        hidden_size=768, num_hidden_layers=4, num_attention_heads=12,
        intermediate_size=3072, hidden_act="gelu",
        attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
        initializer_range=0.02, layer_norm_eps=1e-12, qkv_bias=True,
    )
    mformer_config.query_num = 8
    mformer_config.input_hidden_size = 1024  # ViT-MAE-Large output dim
    mformer_config.num_patches = 197
    m_former = MFormer(mformer_config)

    # Vector quantizer: 128 codes, dim 32
    vector_quantizer = VectorQuantizer2(
        n_e=128, e_dim=32, beta=0.25, sane_index_shape=True,
    )

    # Decoder: 12-layer ViT decoder
    decoder_config = ViTConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, hidden_act="gelu",
        image_size=224, patch_size=16, num_channels=3,
        attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
        initializer_range=0.02, layer_norm_eps=1e-12, qkv_bias=True,
        encoder_stride=16,
    )
    decoder_config.query_num = 8
    decoder_config.num_patches = 196
    decoder = LatentMotionDecoder(decoder_config)

    # Assemble tokenizer
    tokenizer = LatentMotionTokenizer(
        image_encoder=image_encoder,
        m_former=m_former,
        vector_quantizer=vector_quantizer,
        decoder=decoder,
        codebook_dim=32,
    )

    # Load trained weights — checkpoint contains m_former, vq, decoder weights
    # but NOT image_encoder (ViT-MAE is loaded separately from HuggingFace).
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Keys are already bare (e.g. "m_former.embeddings.latent_motion_token")
    tokenizer.load_state_dict(state_dict, strict=False)

    tokenizer = tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    return tokenizer


def tokenize_frames(tokenizer, frame_t, frame_t_plus_m, device="cuda"):
    """Encode a pair of frames into discrete motion tokens.

    Uses tokenizer.tokenize() which returns (quant, indices, commit_loss).
    We only need the indices (discrete token IDs).

    Args:
        tokenizer: Frozen Moto LatentMotionTokenizer.
        frame_t: (B, H, W, 3) float32 numpy array in [0, 1].
        frame_t_plus_m: (B, H, W, 3) float32 numpy array in [0, 1].
        device: torch device.

    Returns:
        tokens: (B, 8) int64 numpy array — discrete motion token IDs from codebook.
    """
    cond_pv = _preprocess_frames(frame_t, device)      # (B, 3, 224, 224)
    target_pv = _preprocess_frames(frame_t_plus_m, device)  # (B, 3, 224, 224)

    with torch.no_grad():
        _quant, indices, _commit_loss = tokenizer.tokenize(cond_pv, target_pv)
        # indices: (B, 8) int64 — codebook indices for the 8 motion tokens

    return indices.cpu().numpy()


def decode_tokens(tokenizer, tokens, frame_t, device="cuda"):
    """Decode motion tokens back to a predicted future frame (for visualization).

    Uses tokenizer.decode_image() which takes cond_pixel_values and token IDs,
    returns {"recons_pixel_values": (B, 3, 224, 224)}.

    Args:
        tokenizer: Frozen Moto LatentMotionTokenizer.
        tokens: (B, 8) int64 numpy array — motion token IDs.
        frame_t: (B, H, W, 3) float32 numpy array in [0, 1].
        device: torch device.

    Returns:
        predicted_frame: (B, H, W, 3) float32 numpy array in [0, 1].
    """
    token_ids = torch.from_numpy(tokens).long().to(device)
    cond_pv = _preprocess_frames(frame_t, device)

    with torch.no_grad():
        result = tokenizer.decode_image(cond_pv, token_ids)
        recon = result["recons_pixel_values"]  # (B, 3, 224, 224)

    # Undo ImageNet normalization
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    recon = recon * std + mean
    recon = torch.clamp(recon, 0, 1)

    # (B, 3, H, W) → (B, H, W, 3)
    return recon.permute(0, 2, 3, 1).cpu().numpy()


def embed_frames(tokenizer, frame_t, frame_t_plus_m, pool=False, device="cuda"):
    """Get continuous motion embeddings (before VQ discretization).

    Useful for analysis — shows what the motion representation looks like
    before quantization.

    Args:
        tokenizer: Frozen Moto LatentMotionTokenizer.
        frame_t: (B, H, W, 3) float32 numpy in [0, 1].
        frame_t_plus_m: (B, H, W, 3) float32 numpy in [0, 1].
        pool: if True, return mean-pooled (B, dim) vector.
        device: torch device.

    Returns:
        embeddings: (B, 8, 768) or (B, 768) if pooled — continuous motion repr.
    """
    cond_pv = _preprocess_frames(frame_t, device)
    target_pv = _preprocess_frames(frame_t_plus_m, device)

    with torch.no_grad():
        emb = tokenizer.embed(cond_pv, target_pv, pool=pool, before_vq=True)

    return emb.cpu().numpy()
