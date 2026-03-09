"""Dense per-chunk tracking reward from the frozen world model.

Supports two world model types:
  1. Contrastive (legacy): cosine similarity in projected space
  2. Discrete (VQ-VAE + dynamics transformer): per-token cross-entropy,
     conditioned on language via frozen CLIP embedding.

Also provides V-based advantage computation and value head fine-tuning.
"""

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Discrete VQ-VAE + dynamics transformer reward
# ---------------------------------------------------------------------------

def discrete_tracking_reward(x_t, x_t1, x_t_plus_m,
                              tokenizer_apply, tokenizer_params,
                              dynamics_apply, dynamics_params,
                              lang_embed=None):
    """Compute tracking reward using frozen VQ-VAE tokenizer + dynamics transformer.

    Tokenizes raw frames, runs language-conditioned dynamics prediction,
    scores actual future tokens via cross-entropy.

    Args:
        x_t: Raw frame at time t, (B, 256, 256, 3) float [0, 1].
        x_t1: Raw frame at time t+1, (B, 256, 256, 3) float [0, 1].
        x_t_plus_m: Raw frame at time t+m, (B, 256, 256, 3) float [0, 1].
        tokenizer_apply: VQVAETokenizer.apply method.
        tokenizer_params: Frozen tokenizer params.
        dynamics_apply: DynamicsTransformer.apply method.
        dynamics_params: Frozen dynamics params.
        lang_embed: (B, lang_dim) language embedding from frozen text encoder.

    Returns:
        r_aux: (B,) tracking reward (mean negative CE). Higher = better.
        v_pred: (B,) value baseline from dynamics model.
    """
    # Tokenize all three frames
    z_t = tokenizer_apply({"params": tokenizer_params}, x_t,
                           method="encode", train=False)
    z_t1 = tokenizer_apply({"params": tokenizer_params}, x_t1,
                            method="encode", train=False)
    z_target = tokenizer_apply({"params": tokenizer_params}, x_t_plus_m,
                                method="encode", train=False)

    # Dynamics prediction + value (language-conditioned)
    logits, v_pred = dynamics_apply(
        {"params": dynamics_params}, z_t, z_t1,
        lang_embed=lang_embed, train=False,
    )

    # Per-position cross-entropy
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, z_target)  # (B, N)
    r_aux = -jnp.mean(ce, axis=-1)  # (B,) — higher = better prediction match

    return r_aux, v_pred


def discrete_tracking_reward_from_tokens(z_t, z_t1, z_target,
                                          dynamics_apply, dynamics_params,
                                          lang_embed=None):
    """Same as discrete_tracking_reward but with pre-tokenized inputs.

    Use when frames have already been tokenized (avoids redundant encoding).
    """
    logits, v_pred = dynamics_apply(
        {"params": dynamics_params}, z_t, z_t1,
        lang_embed=lang_embed, train=False,
    )
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, z_target)
    r_aux = -jnp.mean(ce, axis=-1)
    return r_aux, v_pred


# ---------------------------------------------------------------------------
# Contrastive world model reward (legacy)
# ---------------------------------------------------------------------------

def tracking_reward(z_t, z_t1, z_t_plus_m, wm_params, model_apply_fn, eps=1e-6):
    """Compute the normalized auxiliary tracking reward.

    Args:
        z_t: Octo encoder readout at time t, 1-frame context (batch, 768).
        z_t1: Octo encoder readout at time t, 2-frame context (batch, 768).
        z_t_plus_m: Octo encoder readout m steps later (batch, 768).
        wm_params: Frozen world model params dict.
        model_apply_fn: WorldModel.apply.

    Returns:
        r_aux: (batch,) tracking reward in [-1, 1]. Higher = better tracking.
    """
    result = model_apply_fn(
        {"params": wm_params}, z_t, z_t1, z_t_plus_m, train=False
    )
    anchor, actual = result[0], result[1]
    tracking_sim = jnp.sum(anchor * actual, axis=-1)
    return tracking_sim


def tracking_reward_with_value(z_t, z_t1, z_t_plus_m, wm_params, model_apply_fn):
    """Compute tracking reward and V baseline in one WM forward pass.

    Returns:
        tracking_sim: (batch,) cosine similarity reward.
        v_pred: (batch,) value head prediction, or None if no value head.
    """
    result = model_apply_fn(
        {"params": wm_params}, z_t, z_t1, z_t_plus_m, train=False
    )
    anchor, actual = result[0], result[1]
    tracking_sim = jnp.sum(anchor * actual, axis=-1)
    v_pred = result[2] if len(result) == 3 else None
    return tracking_sim, v_pred


def finetune_value_head(wm_model, wm_params_v0, buffer_entries,
                        num_steps=50, lr=1e-3, batch_size=32, rng=None):
    """Retrain V_φ from V_0 on buffer, keeping h and f_ψ frozen.

    Matches Algorithm 1 line 9: "Retrain V_φ from V_0 on B".
    Uses optax.masked to only update ValueHead params; stop_gradient
    inside WorldModel ensures h and f_ψ receive zero gradients anyway.

    Args:
        wm_model: WorldModel instance.
        wm_params_v0: Initial (pretrained) WM params — V resets to this.
        buffer_entries: List of dicts with 'z_t', 'z_t1', 'z_target', 'r_aux'.
        num_steps: SGD steps for V fine-tuning.
        lr: Learning rate.
        batch_size: Mini-batch size.
        rng: PRNG key.

    Returns:
        Updated wm_params with fine-tuned V (h and f_ψ unchanged).
    """
    wm_params = jax.tree.map(lambda x: x, wm_params_v0)

    mask = jax.tree_util.tree_map_with_path(
        lambda path, _: any('ValueHead' in str(p) for p in path),
        wm_params,
    )
    tx = optax.masked(optax.adam(lr), mask)
    opt_state = tx.init(wm_params)

    z_t_all = jnp.concatenate([e["z_t"] for e in buffer_entries])
    z_t1_all = jnp.concatenate([e["z_t1"] for e in buffer_entries])
    z_target_all = jnp.concatenate([e["z_target"] for e in buffer_entries])
    r_actual_all = jnp.array([float(e["r_aux"]) for e in buffer_entries])
    N = z_t_all.shape[0]

    @jax.jit
    def v_step(params, opt_st, z_t, z_t1, z_tgt, r_act):
        def loss_fn(p):
            result = wm_model.apply({"params": p}, z_t, z_t1, z_tgt, train=False)
            return jnp.mean((result[2] - r_act) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_st = tx.update(grads, opt_st, params)
        return optax.apply_updates(params, updates), new_opt_st, loss

    for step in range(num_steps):
        rng, sample_rng = jax.random.split(rng)
        bs = min(batch_size, N)
        idx = jax.random.choice(sample_rng, N, shape=(bs,), replace=False)
        wm_params, opt_state, loss = v_step(
            wm_params, opt_state,
            z_t_all[idx], z_t1_all[idx], z_target_all[idx], r_actual_all[idx],
        )

    return wm_params


def recompute_advantages(wm_model, wm_params, buffer_entries):
    """Recompute A_t = r_aux - V_φ(o_t, g_{t+m}) for all buffer entries.

    Runs one batched WM forward pass, extracts V predictions, updates
    each entry's 'advantage' field in-place.
    """
    z_t_all = jnp.concatenate([e["z_t"] for e in buffer_entries])
    z_t1_all = jnp.concatenate([e["z_t1"] for e in buffer_entries])
    z_target_all = jnp.concatenate([e["z_target"] for e in buffer_entries])

    result = wm_model.apply(
        {"params": wm_params}, z_t_all, z_t1_all, z_target_all, train=False,
    )
    v_preds = result[2]
    r_actuals = jnp.array([float(e["r_aux"]) for e in buffer_entries])
    advantages = r_actuals - v_preds

    for i, entry in enumerate(buffer_entries):
        entry["advantage"] = float(advantages[i])
