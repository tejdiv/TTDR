"""Surprisal reward from Moto-GPT world model.

r_t = sum_i log p(m_t*[i] | instruction, f_{t-1})

where m_t* = Tokenizer(f_{t-1}, f_t) are the actual motion tokens and
p is Moto-GPT's predicted distribution. Higher = less surprising = better
alignment with what the world model expected.

Advantage: A_t = r_t - V(o_t, g_t).  Exact, deterministic, zero variance.
"""

import numpy as np

from moto_recap.tokenizer import tokenize_frames
from moto_recap.gpt import encode as gpt_encode, score_tokens


def compute_surprisal_reward(gpt, tokenizer, instruction, frame_t, frame_t_plus_m,
                              device="cuda"):
    """Compute tracking reward as motion token surprisal.

    Args:
        gpt: Frozen fine-tuned Moto-GPT.
        tokenizer: Frozen Moto tokenizer.
        instruction: str or list of str.
        frame_t: (B, H, W, 3) float32 numpy in [0, 1].
        frame_t_plus_m: (B, H, W, 3) float32 numpy in [0, 1].
        device: torch device.

    Returns:
        reward: (B,) numpy float32 — total log probability (higher = better).
        per_token_log_probs: (B, 8) numpy float32.
        actual_tokens: (B, 8) int64 numpy — the observed motion tokens.
    """
    actual_tokens = tokenize_frames(tokenizer, frame_t, frame_t_plus_m, device)
    per_token_lp, reward = score_tokens(
        gpt, instruction, frame_t, actual_tokens, device
    )
    return reward, per_token_lp, actual_tokens


def compute_advantage(reward, v_pred):
    """Exact advantage: A = r - V(o, g). Deterministic, zero variance.

    Args:
        reward: (B,) surprisal reward.
        v_pred: (B,) value head prediction.

    Returns:
        advantage: (B,) numpy float32.
    """
    return reward - v_pred


def compute_indicator(advantage, threshold=0.0):
    """Assign improvement indicator I based on advantage sign.

    I = 1 when A > threshold (action beat V's expectation).
    This is the label for classifier-free guidance conditioning.

    Args:
        advantage: (B,) or scalar.
        threshold: advantage threshold for I=1.

    Returns:
        indicator: same shape as advantage, bool.
    """
    return np.asarray(advantage) > threshold
