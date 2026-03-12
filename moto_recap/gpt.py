"""Wrapper around Moto-GPT (PyTorch).

Moto-GPT is an autoregressive model over motion token sequences, conditioned
on a language instruction (T5-base encoder) and visual context (ViT-MAE-Large).

Architecture (from Moto config):
  - Language model: T5-base encoder (frozen)
  - Vision model: shared ViT-MAE-Large (frozen) — same as tokenizer
  - Causal transformer: GPT-2, 12 layers, 768 hidden, 12 heads
  - Prediction head: Linear(768, 128) — logits over codebook
  - Sequence layout: [lang_tokens, patch_tokens, obs_token, motion_tokens...]
  - per_latent_motion_len: 8 tokens per transition
  - latent_motion_codebook_size: 128

Real Moto API:
  - gpt.forward(rgb, language, attention_mask, latent_motion_ids, latent_mask,
                train=True, lang_attention_mask=None)
  - gpt.greedy_decode(...) / gpt.beam_search_decode(...)
  - Language is T5 token IDs, not raw strings
  - RGB is (B, 1, 3, H, W) — single frame with time dim

This wrapper provides:
  1. encode()        → hidden state before prediction head (anchor g)
  2. score_tokens()  → log p(target_tokens | instruction, frame) (reward)
  3. predict_tokens() → greedy decoded motion tokens (visualization)

HuggingFace: TencentARC/Moto (subdir: moto_gpt_pretrained_on_oxe/)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import T5Tokenizer


# ImageNet normalization (ViT-MAE-Large expects this)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Cache the T5 tokenizer (loaded once)
_t5_tokenizer = None


def _get_t5_tokenizer():
    global _t5_tokenizer
    if _t5_tokenizer is None:
        _t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return _t5_tokenizer


def _preprocess_frames(frames_np, device):
    """(B, H, W, 3) float32 [0,1] numpy → (B, 1, 3, 224, 224) torch.

    Moto-GPT expects rgb with shape (B, 1, C, H, W) — batch x time x channels.
    """
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device)
    t = _preprocess(t)
    return t.unsqueeze(1)  # (B, 1, 3, 224, 224)


def _tokenize_instruction(instruction, device):
    """Convert instruction string(s) to T5 token IDs + attention mask.

    Args:
        instruction: str or list of str.
        device: torch device.

    Returns:
        input_ids: (B, max_len) int64 tensor.
        attention_mask: (B, max_len) int64 tensor.
    """
    if isinstance(instruction, str):
        instruction = [instruction]

    tok = _get_t5_tokenizer()
    encoded = tok(
        instruction, return_tensors="pt", padding=True, truncation=True,
        max_length=64,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def load_gpt(checkpoint="TencentARC/Moto", device="cuda"):
    """Load Moto-GPT from HuggingFace or local checkpoint.

    Loads the GPT-2 based causal transformer with T5 language encoder
    and ViT-MAE-Large vision encoder.

    Args:
        checkpoint: HuggingFace repo ID, local path to repo, or path to
                    a fine-tuned .pt state_dict file.
        device: torch device.

    Returns:
        gpt: MotoGPT model (eval mode).
    """
    from huggingface_hub import snapshot_download

    # Determine checkpoint path
    if os.path.isfile(checkpoint) and checkpoint.endswith(".pt"):
        # Direct path to a fine-tuned state_dict
        state_dict_path = checkpoint
        # Still need the base model structure — download from HF
        repo_path = snapshot_download("TencentARC/Moto")
    elif os.path.isdir(checkpoint):
        repo_path = checkpoint
        state_dict_path = None
    else:
        repo_path = snapshot_download(checkpoint)
        state_dict_path = None

    # Find GPT checkpoint in repo
    if state_dict_path is None:
        # Try OXE-pretrained GPT first
        gpt_dir = os.path.join(repo_path, "moto_gpt_pretrained_on_oxe")
        if not os.path.isdir(gpt_dir):
            gpt_dir = repo_path
        ckpt_files = [f for f in os.listdir(gpt_dir)
                      if f.endswith((".pt", ".ckpt", ".bin"))]
        if not ckpt_files:
            raise FileNotFoundError(
                f"No GPT checkpoint in {gpt_dir}. "
                f"Contents: {os.listdir(gpt_dir)}"
            )
        state_dict_path = os.path.join(gpt_dir, ckpt_files[0])

    # Build model from Moto's source
    from moto_gpt.src.models.moto_gpt import MotoGPT
    from moto_gpt.src.models.mae_model import MaeEncoder
    from moto_gpt.src.models.trajectory_gpt2 import GPT2Model, GPT2Config
    from transformers import T5EncoderModel

    # Language encoder: T5-base (frozen)
    model_lang = T5EncoderModel.from_pretrained("t5-base")

    # Vision encoder: MaeEncoder wrapping ViT-MAE-Large (frozen)
    model_vision = MaeEncoder(
        use_obs_feature=True,
        pretrained_model_name_or_path="facebook/vit-mae-large",
    )

    # Causal transformer: GPT-2 (Moto's custom version)
    gpt2_config = GPT2Config(
        vocab_size=1,
        n_layer=12, n_head=12, n_embd=768,
        n_positions=1024,
        activation_function="relu",
    )
    gpt2_config.dropout = 0.1
    model_causal_transformer = GPT2Model(gpt2_config)

    # Assemble MotoGPT
    gpt = MotoGPT(
        model_lang=model_lang,
        model_vision=model_vision,
        model_causal_transformer=model_causal_transformer,
        act_dim=7,
        hidden_size=768,
        sequence_length=2,
        chunk_size=3,
        per_latent_motion_len=8,
        latent_motion_codebook_size=128,
        latent_motion_pred=True,
        act_pred=False,
        img_feat_dim=1024,
        patch_feat_dim=1024,
        lang_feat_dim=768,
        mask_latent_motion_probability=0.0,
        freeze_lang=True,
        freeze_vision=True,
    )

    # Load trained weights — checkpoint contains model_causal_transformer,
    # embed_* projections, and pred_* heads, but NOT lang/vision encoders
    # (those are loaded separately from HuggingFace).
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Keys are already correct (e.g. "model_causal_transformer.h.0.ln_1.weight")
    gpt.load_state_dict(state_dict, strict=False)

    gpt = gpt.to(device).eval()
    return gpt


def encode(gpt, instruction, visual_context, device="cuda"):
    """Get Moto-GPT's hidden state — the anchor g.

    Runs the conditioning path (language + vision) through the GPT and
    extracts the hidden state at the position just before motion token
    prediction. This dense vector summarizes "what motion should happen next"
    and serves as the anchor for goal-conditioned RECAP.

    Args:
        gpt: MotoGPT model.
        instruction: str or list of str.
        visual_context: (B, H, W, 3) float32 numpy in [0, 1].
        device: torch device.

    Returns:
        hidden: (B, hidden_dim=768) numpy float32 — the anchor g.
    """
    if isinstance(instruction, str):
        instruction = [instruction]
    B = len(instruction)

    lang_ids, lang_mask = _tokenize_instruction(instruction, device)
    rgb = _preprocess_frames(visual_context, device)  # (B, 1, 3, 224, 224)

    with torch.no_grad():
        # Run language encoder
        lang_emb = gpt.model_lang(
            input_ids=lang_ids, attention_mask=lang_mask
        ).last_hidden_state  # (B, n_lang, 768)
        lang_emb = gpt.embed_lang(lang_emb.float())

        # Run vision encoder — MaeEncoder returns (B, 1024), (B, 196, 1024)
        obs_emb, patch_emb = gpt.model_vision(
            rgb.reshape(B, 3, 224, 224)
        )
        obs_emb = obs_emb.view(B, 1, -1)  # (B, 1, 1024)
        obs_emb = gpt.embed_img(obs_emb.float())     # (B, 1, 768)
        patch_emb = gpt.embed_patch(patch_emb.float())  # (B, 196, 768)

        # Add condition embedding (matches Moto forward)
        cond_emb = gpt.embed_condition.weight.view(1, 1, gpt.hidden_size)
        lang_emb = lang_emb + cond_emb
        patch_emb = patch_emb + cond_emb
        obs_emb = obs_emb + cond_emb

        # Concatenate conditioning tokens
        cond = torch.cat([lang_emb, patch_emb, obs_emb], dim=1)  # (B, n_cond, 768)
        cond = gpt.embed_ln(cond)

        # Run through GPT to get contextualized hidden states
        cond_mask = torch.ones(cond.shape[:2], device=device, dtype=torch.long)
        transformer_out = gpt.model_causal_transformer(
            inputs_embeds=cond, attention_mask=cond_mask
        )
        hidden = transformer_out["last_hidden_state"]  # (B, n_cond, 768)

        # Take the last conditioning token's hidden state as the anchor
        # This position is where motion prediction begins
        anchor = hidden[:, -1, :]  # (B, 768)

    return anchor.cpu().numpy()


def score_tokens(gpt, instruction, visual_context, target_tokens, device="cuda"):
    """Score actual motion tokens under the GPT's predicted distribution.

    Runs the full forward pass with the target tokens provided, extracts
    logits at each of the 8 motion token positions, computes log p(m*).

    This is the core reward computation:
      r_t = sum_i log p(m_t*[i] | instruction, f_{t-1})

    Args:
        gpt: MotoGPT model.
        instruction: str or list of str.
        visual_context: (B, H, W, 3) float32 numpy in [0, 1].
        target_tokens: (B, 8) int64 numpy — actual observed motion tokens.
        device: torch device.

    Returns:
        log_probs: (B, 8) numpy float32 — per-token log probabilities.
        total_log_prob: (B,) numpy float32 — sum of per-token log probs.
    """
    if isinstance(instruction, str):
        instruction = [instruction]
    B = len(instruction)

    lang_ids, lang_mask = _tokenize_instruction(instruction, device)
    rgb = _preprocess_frames(visual_context, device)  # (B, 1, 3, 224, 224)
    tgt = torch.from_numpy(target_tokens).long().to(device)  # (B, 8)

    # Moto expects sequence_length=2 timesteps. Pad with a dummy second timestep.
    # (B, 8) → (B, 2, 8): real tokens at t=0, zeros at t=1 (masked out)
    tgt_seq = torch.stack([tgt, torch.zeros_like(tgt)], dim=1)  # (B, 2, 8)
    attention_mask = torch.ones(B, 2, device=device, dtype=torch.long)
    latent_mask = torch.tensor([[1, 0]] * B, device=device, dtype=torch.long)  # only t=0 valid

    # Use train=True to get logits directly (train=False does autoregressive decode)
    with torch.no_grad():
        outputs = gpt.forward(
            rgb=rgb,
            language=lang_ids,
            attention_mask=attention_mask,
            latent_motion_ids=tgt_seq,
            latent_mask=latent_mask,
            train=True,
            lang_attention_mask=lang_mask,
        )

    # Extract logits — train=True returns {"latent_motion_preds": (B, T, 8, 128), ...}
    logits = outputs["latent_motion_preds"]

    # logits shape: (B, 1, 8, 128)
    logits = logits[:, 0]  # (B, 8, 128) — take first (only) timestep

    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, 8, 128)

    # Gather log probs for the actual tokens
    log_probs = log_probs_all.gather(
        2, tgt.unsqueeze(-1)
    ).squeeze(-1)  # (B, 8)

    lp = log_probs.cpu().numpy()
    return lp, lp.sum(axis=-1)


def predict_tokens(gpt, instruction, visual_context, device="cuda"):
    """Predict motion tokens via greedy decoding.

    Uses Moto's forward(train=False) which runs decode_latent_motion()
    for autoregressive prediction of 8 motion tokens.

    Args:
        gpt: MotoGPT model.
        instruction: str or list of str.
        visual_context: (B, H, W, 3) float32 numpy in [0, 1].
        device: torch device.

    Returns:
        tokens: (B, 8) int64 numpy — predicted motion token IDs.
    """
    if isinstance(instruction, str):
        instruction = [instruction]
    B = len(instruction)

    lang_ids, lang_mask = _tokenize_instruction(instruction, device)
    rgb = _preprocess_frames(visual_context, device)

    # Moto expects sequence_length=2. Dummy tokens — forward(train=False) decodes.
    dummy_tokens = torch.zeros(B, 2, 8, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, 2, device=device, dtype=torch.long)
    latent_mask = torch.tensor([[1, 0]] * B, device=device, dtype=torch.long)

    with torch.no_grad():
        outputs = gpt.forward(
            rgb=rgb,
            language=lang_ids,
            attention_mask=attention_mask,
            latent_motion_ids=dummy_tokens,
            latent_mask=latent_mask,
            train=False,
            lang_attention_mask=lang_mask,
        )

    # forward(train=False) returns "latent_motion_id_preds": (B, seq_len*8)
    # We padded to seq_len=2 but only the first 8 tokens are from our real input
    tokens = outputs["latent_motion_id_preds"]
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    # Reshape and take first timestep only
    if tokens.ndim == 3:
        tokens = tokens[:, 0]  # (B, 2, 8) → (B, 8)
    elif tokens.shape[-1] == 16:
        tokens = tokens[:, :8]  # (B, 16) → (B, 8)
    return tokens
