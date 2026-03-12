"""Hindsight anchor relabeling for KM data efficiency.

From K real transitions, generate K*M training samples by re-scoring the
same (o_t, a_t, o_{t+m}) against M different anchors. The action is real,
the outcome is real, only the anchor and its reward/I label change.

This is what makes I pretraining non-trivial: the same action gets I=1 for
anchors whose expected motion matches what happened, and I=0 for anchors it
didn't match. The policy learns the structural relationship between g, I,
and the action — not just "try harder."

Preserves RECAP guarantees: all actions come from pi_ref, all outcomes are
real, rewards are deterministic given (real outcome, anchor).
"""

import numpy as np

from moto_recap.gpt import encode as gpt_encode, score_tokens


def relabel_transition(transition, gpt, tokenizer, anchor_proj_fn,
                       value_fn, instruction_bank, buffer,
                       M=50, threshold=0.0, device="cuda"):
    """Generate M relabeled (anchor, reward, I) tuples from one real transition.

    The action is always real (from pi_ref). The outcome is always real.
    Only the anchor g' and its associated reward/advantage/I change.

    Args:
        transition: dict with 'frame_t', 'frame_t_plus_m', 'action',
                    'actual_tokens', 'obs_encoding'.
        gpt: Frozen Moto-GPT.
        tokenizer: Frozen Moto tokenizer (only needed if actual_tokens missing).
        anchor_proj_fn: callable (B, moto_hidden_dim) → (B, 768).
        value_fn: callable (obs_encoding, anchor_embed) → (B,) scalar.
        instruction_bank: list of instruction strings for alt-instruction strategy.
        buffer: list of past transitions for buffer-outcome strategy.
        M: number of relabeled anchors per transition.
        threshold: advantage threshold for I=1.
        device: torch device.

    Returns:
        list of M dicts, each with 'anchor_embed', 'reward', 'advantage', 'I'.
    """
    frame_t = transition["frame_t"]
    actual_tokens = transition["actual_tokens"]
    obs_enc = transition["obs_encoding"]  # (1, 768) JAX array

    relabeled = []
    strategies = ["alt_instruction", "buffer_outcome", "perturbed"]

    for i in range(M):
        strategy = strategies[i % len(strategies)]

        if strategy == "alt_instruction" and len(instruction_bank) > 1:
            # Different task → different expected motion → different anchor
            instr = instruction_bank[np.random.randint(len(instruction_bank))]
            g_hidden = gpt_encode(gpt, instr, frame_t, device)  # (1, hidden_dim)

        elif strategy == "buffer_outcome" and len(buffer) > 0:
            # Use another transition's context as anchor source
            other = buffer[np.random.randint(len(buffer))]
            other_frame = other["frame_t"]
            instr = transition.get("instruction", instruction_bank[0])
            g_hidden = gpt_encode(gpt, instr, other_frame, device)

        else:
            # Perturb the actual anchor with Gaussian noise
            instr = transition.get("instruction", instruction_bank[0])
            g_hidden = gpt_encode(gpt, instr, frame_t, device)
            g_hidden = g_hidden + np.random.randn(*g_hidden.shape).astype(
                np.float32
            ) * 0.1

        # Score actual tokens against this different anchor's instruction context
        _, reward = score_tokens(gpt, instr, frame_t, actual_tokens, device)

        # Project anchor to Octo space and compute value
        import jax.numpy as jnp
        g_jax = jnp.array(g_hidden)
        anchor_embed = anchor_proj_fn(g_jax)  # (1, 768)
        v_pred = float(value_fn(obs_enc, anchor_embed))
        r = float(reward[0])
        advantage = r - v_pred
        I_label = advantage > threshold

        relabeled.append({
            "anchor_embed": np.array(anchor_embed),  # (1, 768)
            "anchor_hidden": g_hidden,                # (1, moto_hidden_dim)
            "reward": r,
            "advantage": advantage,
            "I": bool(I_label),
        })

    return relabeled


def build_instruction_bank(dataset_instructions, max_size=100):
    """Build a bank of diverse instructions for alt-instruction relabeling.

    Args:
        dataset_instructions: iterable of instruction strings from Bridge V2.
        max_size: max bank size.

    Returns:
        list of unique instruction strings.
    """
    unique = list(set(dataset_instructions))
    if len(unique) > max_size:
        np.random.shuffle(unique)
        unique = unique[:max_size]
    return unique
