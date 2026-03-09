"""RECAP test-time adaptation loop.

Rolls out the policy, computes tracking rewards via the frozen world model,
uses the learned value head V for advantage estimation (A = r - V), and
updates LoRA weights using classifier-free guidance with a language-token
improvement indicator.

Key design choices:
  - Single action sample per step (V provides the baseline, no K-way forking needed).
  - Language indicator I appended as last token: "... World Model Advantage: positive"
  - I=1 when A_t > 0 (the action beat V's expectation).
  - Loss (Eq. 2): L = -log π(a|o,ℓ) - α·log π(a|I,o,ℓ)
    Both terms are BC on ALL data. I varies per sample.
  - At test time: always append indicator → classifier-free guidance extracts improved policy.

Per-step efficiency (2 transformer passes + 1 diffusion call):
  Pass 1 — Batched CFG: [obs, obs] × [task, task_improved] in one forward pass.
    Split → original-task readouts for WM, improved-task readouts for action head.
    ONE diffusion call → action sample.
  Execute action, observe next state.
  Pass 2 — Encode next obs → readout for tracking reward + V baseline.

Every M chunks:
  1. Retrain V_φ from V_0 on buffer (MSE on actual tracking rewards).
  2. Recompute A_t = r_aux - V_φ for all buffer entries.
  3. I_t = 1 if A_t > 0.
  4. Retrain LoRA from scratch on buffer with RECAP loss.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax.training import checkpoints

from recap.models.lora_adapter import init_lora_params, apply_lora
from recap.losses.tracking_reward import tracking_reward_with_value, finetune_value_head, recompute_advantages
from recap.training.train_world_model import WorldModel
from recap.envs.perturbations import postprocess_octo_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMPROVEMENT_SUFFIX = " World Model Advantage: positive"


def load_world_model(config):
    """Load frozen world model params and build apply function.

    Supports both local Flax checkpoint dirs and HuggingFace repos:
      - Local: "checkpoints/world_model" (standard flax.training.checkpoints format)
      - HuggingFace: "hf://4manifold/ttdr-world-model" (downloads then loads)

    Returns:
        wm_model: WorldModel instance (for .apply).
        wm_params: Params dict (flat, without "params" wrapper).
    """
    checkpoint_path = config.wm_checkpoint

    if checkpoint_path.startswith("hf://"):
        from huggingface_hub import snapshot_download
        repo_id = checkpoint_path.removeprefix("hf://")
        checkpoint_path = snapshot_download(repo_id)

    state = checkpoints.restore_checkpoint(checkpoint_path, target=None)

    wm_model = WorldModel(
        projection_head_kwargs=config.wm_projection_head_kwargs,
        dynamics_predictor_kwargs=config.wm_dynamics_predictor_kwargs,
        value_head_kwargs=getattr(config, "wm_value_head_kwargs", None),
    )

    return wm_model, state["params"]


def _concat_batch(*dicts):
    """Concatenate dicts of arrays along batch dim 0."""
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *dicts)


# ---------------------------------------------------------------------------
# Batched forward passes
# ---------------------------------------------------------------------------

def _make_jit_transformer(module):
    """Create a JIT-wrapped transformer call. Captures module in closure.

    Needed because non-JIT cuDNN can fail on certain CUDA versions
    (e.g. cuDNN 8.9 + CUDA 12.8). OctoModule is not hashable so
    it can't be a static JIT arg — we close over it instead.
    """
    @jax.jit
    def run(params, obs, task, pad_mask):
        return module.apply(
            {"params": params}, obs, task, pad_mask,
            train=False, method="octo_transformer",
        )
    return run


# Cache per module instance to avoid retracing
_jit_transformers = {}


def _run_transformer(module, params, obs, task, pad_mask):
    key = id(module)
    if key not in _jit_transformers:
        _jit_transformers[key] = _make_jit_transformer(module)
    return _jit_transformers[key](params, obs, task, pad_mask)


def encode_obs(octo_model, octo_params, obs, task):
    """Single transformer pass → WM readouts + transformer output for action head.

    Uses plain task (no suffix). The indicator embedding is injected separately
    into readout tokens before the action head, not through the language model.

    Returns:
        z_t: (B, 768) 1-frame readout (for WM).
        z_t1: (B, 768) 2-frame readout (for WM).
        trans_out: transformer output dict → action head (after indicator injection).
    """
    pad_mask = obs["pad_mask"]
    trans_out = _run_transformer(
        octo_model.module, octo_params, obs, task, pad_mask
    )

    tokens = trans_out["readout_action"].tokens  # (B, T, N, 768)
    z_t = jnp.mean(tokens[:, 0, :, :], axis=1)    # (B, 768)
    z_t1 = jnp.mean(tokens[:, -1, :, :], axis=1)  # (B, 768)

    return z_t, z_t1, trans_out


def inject_indicator(trans_out, indicator_embed):
    """Add learned indicator embedding to readout tokens.

    The indicator is a (768,) vector added to every readout token position.
    This shifts the action head's conditioning to produce "improved" actions,
    analogous to the language suffix but in the action head's native space.
    """
    tokens = trans_out["readout_action"].tokens  # (B, T, N, 768)
    shifted_tokens = tokens + indicator_embed[None, None, None, :]
    # Replace tokens in the TokenGroup
    new_token_group = trans_out["readout_action"].replace(tokens=shifted_tokens)
    return {**trans_out, "readout_action": new_token_group}


def _make_jit_action_head(module):
    """Create a JIT-wrapped action head predict call. Captures module in closure.

    Without JIT, every call re-traces the full 10-step diffusion denoising loop
    from Python. With JIT, it compiles once and reuses the GPU kernel.
    """
    @jax.jit
    def run(params, transformer_out, rng):
        bound = module.bind({"params": params})
        return bound.heads["action"].predict_action(
            transformer_out, rng=rng, train=False
        )
    return run


_jit_action_heads = {}


def sample_actions_from_readouts(octo_model, merged_params, transformer_out, rng):
    """Run ONLY the action head on pre-computed transformer readouts (JIT'd).

    Args:
        octo_model: OctoModel instance.
        merged_params: Params with LoRA merged (action head weights modified).
        transformer_out: Output of run_transformer (dict with readout tokens).
        rng: PRNG key for diffusion sampling.

    Returns:
        actions: (B, action_horizon, action_dim).
    """
    key = id(octo_model.module)
    if key not in _jit_action_heads:
        _jit_action_heads[key] = _make_jit_action_head(octo_model.module)
    return _jit_action_heads[key](merged_params, transformer_out, rng)


def sample_K_actions(octo_model, merged_params, trans_out_improved, K, B, rng):
    """Sample K action sets with K independent RNG keys.

    Separate diffusion calls per K ensure truly independent denoising
    trajectories. Batching with tiled inputs + one RNG causes the score
    network to converge all K samples to the same mode.

    Returns:
        list of K arrays, each (B, action_horizon, action_dim).
    """
    rngs = jax.random.split(rng, K)
    return [
        sample_actions_from_readouts(
            octo_model, merged_params, trans_out_improved, rngs[k]
        )
        for k in range(K)
    ]


def batched_encode_next(octo_model, octo_params, next_obs_list, task):
    """Encode K next observations in ONE transformer pass.

    Stacks K obs dicts along batch dim, runs transformer once with original
    task (no indicator — WM needs in-distribution readouts), splits back.

    Returns:
        list of K arrays, each (B, 768).
    """
    K = len(next_obs_list)
    B = jax.tree.leaves(next_obs_list[0])[0].shape[0]

    obs_Kx = _concat_batch(*next_obs_list)
    task_Kx = jax.tree.map(
        lambda x: jnp.tile(x, (K,) + (1,) * (x.ndim - 1)), task
    )
    pad_mask = obs_Kx["pad_mask"]

    out = _run_transformer(
        octo_model.module, octo_params, obs_Kx, task_Kx, pad_mask
    )

    tokens = out["readout_action"].tokens  # (K*B, T, N, 768)
    z_all = jnp.mean(tokens[:, 0, :, :], axis=1)  # (K*B, 768)
    return [z_all[k * B : (k + 1) * B] for k in range(K)]


# ---------------------------------------------------------------------------
# Main adaptation loop
# ---------------------------------------------------------------------------

def recap_adapt(octo_model, octo_params, env, instruction, config, rng):
    """Run RECAP adaptation with V-based advantage and learned indicator embedding.

    Per step: 2 transformer passes + 1 diffusion call.
      Pass 1: encode_obs → WM readouts + transformer output.
        Inject indicator embedding into readout tokens → action head input.
      1 action_head.predict_action on indicator-shifted readouts.
      Execute action, observe next state.
      Pass 2: encode next obs → readout for tracking reward.
      Tracking reward + V baseline → advantage A = r - V.

    Every M chunks: fine-tune V from V_0 on buffer, recompute advantages,
    assign I=1 for A>0, retrain LoRA + indicator embedding.

    Args:
        octo_model: OctoModel instance.
        octo_params: Frozen Octo parameters (from octo_model.params).
        env: SimplerEnv with .reset() and .step(action) interface.
        instruction: Language instruction string.
        config: Adaptation config.
        rng: JAX PRNG key.

    Returns:
        adapted_params: Octo params with trained LoRA merged in.
    """
    # --- Setup ---
    wm_model, wm_params = load_world_model(config)
    rng, lora_rng = jax.random.split(rng)
    backbone_lora = getattr(config, "backbone_lora", False)
    lora_params = init_lora_params(
        lora_rng, octo_params, rank=config.rank, backbone_lora=backbone_lora
    )
    num_lora_layers = len(lora_params["layers"])
    logging.info(f"  LoRA layers: {num_lora_layers} (backbone_lora={backbone_lora})")
    optimizer = optax.adam(config.lr)
    # Optimize both LoRA layers and indicator embedding jointly
    trainable = {"layers": lora_params["layers"], "indicator_embed": lora_params["indicator_embed"]}
    opt_state = optimizer.init(trainable)
    buffer = []

    task = octo_model.create_tasks(texts=[instruction])

    # Action denormalization (Bridge V2 dataset stats)
    action_mean = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    # --- Episode loop ---
    total_steps = 0
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        done = False
        chunk_count = 0
        ep_start = time.time()

        logging.info(f"  Episode {episode+1}/{config.num_episodes}")

        while not done:
            step_start = time.time()
            merged_params = apply_lora(octo_params, lora_params)

            # ── Pass 1: encode obs (frozen backbone, plain task) ─────
            z_t, z_t1, trans_out = encode_obs(
                octo_model, octo_params, obs, task
            )

            # Inject indicator embedding → shifted readouts for action head
            trans_out_shifted = inject_indicator(trans_out, lora_params["indicator_embed"])

            # ── Sample single action ─────────────────────────────────
            rng, act_rng = jax.random.split(rng)
            norm_actions = sample_actions_from_readouts(
                octo_model, merged_params, trans_out_shifted, act_rng
            )

            # Denormalize + postprocess for env
            actions_denorm = norm_actions * action_std[None] + action_mean[None]
            action_env = postprocess_octo_action(np.array(actions_denorm[0, 0]))

            # Execute
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated
            task_success = info.get("success", False)
            if hasattr(task_success, "item"):
                task_success = task_success.item()

            # ── Pass 2: encode next observation ──────────────────────
            pad_mask_next = next_obs["pad_mask"]
            trans_out_next = _run_transformer(
                octo_model.module, octo_params, next_obs, task, pad_mask_next
            )
            tokens_next = trans_out_next["readout_action"].tokens
            z_t_plus_m = jnp.mean(tokens_next[:, 0, :, :], axis=1)  # (B, 768)

            # ── Tracking reward + V baseline ─────────────────────────
            r_aux, v_pred = tracking_reward_with_value(
                z_t, z_t1, z_t_plus_m, wm_params, wm_model.apply
            )

            # Store in buffer (skip chunk 0 — padded obs window)
            if chunk_count > 0:
                advantage = float(r_aux - v_pred) if v_pred is not None else 0.0
                buffer.append({
                    "obs": next_obs,
                    "actions": norm_actions,
                    "r_aux": float(r_aux),
                    "advantage": advantage,
                    "z_t": z_t,
                    "z_t1": z_t1,
                    "z_target": z_t_plus_m,
                })

            obs = next_obs
            chunk_count += 1
            total_steps += 1
            if len(buffer) > config.buffer_size:
                buffer = buffer[-config.buffer_size:]

            step_time = time.time() - step_start
            r_val = float(r_aux)
            v_val = float(v_pred) if v_pred is not None else 0.0
            suc_str = " SUCCESS" if task_success else ""
            logging.info(
                f"    step {chunk_count} | {step_time:.1f}s | "
                f"r={r_val:.4f} V={v_val:.4f} A={r_val - v_val:.4f} | "
                f"buf={len(buffer)} | total={total_steps}{suc_str}"
            )

            # RECAP update every M chunks
            if (chunk_count % config.update_freq == 0
                    and len(buffer) >= config.min_buffer):
                update_start = time.time()
                rng, lora_rng = jax.random.split(rng)
                lora_params = init_lora_params(
                    lora_rng, octo_params, rank=config.rank, backbone_lora=backbone_lora
                )
                trainable = {"layers": lora_params["layers"], "indicator_embed": lora_params["indicator_embed"]}
                opt_state = optimizer.init(trainable)
                lora_params, opt_state = _update_lora(
                    lora_params, opt_state, optimizer,
                    octo_model, octo_params, buffer,
                    task, config, rng,
                    wm_model=wm_model, wm_params_v0=wm_params,
                )
                logging.info(
                    f"    LoRA update ({config.num_bc_steps} steps) "
                    f"in {time.time() - update_start:.1f}s"
                )

    # --- Return adapted policy ---
    logging.info(f"  Adaptation complete: {total_steps} total steps, {len(buffer)} buffer entries")
    indicator_norm = float(jnp.linalg.norm(lora_params["indicator_embed"]))
    logging.info(f"  Indicator embedding norm: {indicator_norm:.4f}")
    return apply_lora(octo_params, lora_params), lora_params["indicator_embed"]


# ---------------------------------------------------------------------------
# JIT'd gradient step for LoRA + indicator embedding update
# ---------------------------------------------------------------------------

def _make_jit_indicator_grad_step(module, optimizer):
    """Create a JIT-wrapped gradient step that trains both LoRA and indicator.

    Loss: L = -log π(a|o,ℓ) + α · -log π(a|o⊕I,ℓ)

    The unconditional term (bc_loss) uses plain readout tokens.
    The conditioned term (cond_loss) uses readout tokens + indicator embedding
    for I=1 samples, or plain tokens for I=0 samples.

    Gradients flow through both the LoRA weight matrices and the indicator
    embedding vector.
    """
    @jax.jit
    def step(octo_params, lora_params,
             trans_out, batch_indicators, actions_expanded,
             timestep_pad_mask, action_pad_mask,
             recap_alpha, step_rng, opt_state, trainable):

        def loss_fn(trainable):
            lp = {**lora_params, "layers": trainable["layers"],
                   "indicator_embed": trainable["indicator_embed"]}
            merged = apply_lora(octo_params, lp)
            bound = module.bind(
                {"params": merged}, rngs={"dropout": step_rng}
            )

            # Unconditional BC loss on plain readouts
            bc_loss, _ = bound.heads["action"].loss(
                trans_out, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            # Conditioned loss: inject indicator for I=1 samples
            # indicator_embed is (768,), broadcast to (B, T, N, 768) readout tokens
            tokens = trans_out["readout_action"].tokens
            embed = trainable["indicator_embed"]
            # Per-sample masking: add embed only where indicator=True
            indicator_mask = batch_indicators[:, None, None, None]  # (B, 1, 1, 1)
            shifted_tokens = tokens + indicator_mask * embed[None, None, None, :]
            new_tg = trans_out["readout_action"].replace(tokens=shifted_tokens)
            trans_cond = {**trans_out, "readout_action": new_tg}

            cond_loss, _ = bound.heads["action"].loss(
                trans_cond, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            return bc_loss + recap_alpha * cond_loss

        grads = jax.grad(loss_fn)(trainable)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        return new_trainable, new_opt_state

    return step


_jit_indicator_grad_steps = {}


def _jit_indicator_grad_step(module, octo_params, lora_params,
                              trans_out, batch_indicators, actions_expanded,
                              timestep_pad_mask, action_pad_mask,
                              recap_alpha, step_rng, opt_state, optimizer,
                              trainable):
    """Dispatch to cached JIT'd indicator gradient step."""
    key = id(module)
    if key not in _jit_indicator_grad_steps:
        _jit_indicator_grad_steps[key] = _make_jit_indicator_grad_step(module, optimizer)
    new_trainable, opt_state = _jit_indicator_grad_steps[key](
        octo_params, lora_params,
        trans_out, batch_indicators, actions_expanded,
        timestep_pad_mask, action_pad_mask,
        recap_alpha, step_rng, opt_state, trainable,
    )
    return new_trainable, opt_state


# ---------------------------------------------------------------------------
# LoRA update with classifier-free guidance loss (batched)
# ---------------------------------------------------------------------------

def _update_lora(lora_params, opt_state, optimizer,
                 octo_model, octo_params, buffer,
                 task, config, rng,
                 wm_model=None, wm_params_v0=None):
    """LoRA + indicator update: L = -log π(a|o,ℓ) - α·log π(a|I⊕o,ℓ).

    Instead of two language prompts (task vs task_improved), uses a single task
    and injects a learned indicator embedding into readout tokens for I=1 samples.

    Both terms are BC on ALL data. The indicator I varies per sample:
      I=1 when A_t > 0, where A_t = r_aux - V_φ(o_t, g_{t+m}).

    Gradients flow through both the LoRA layers AND the indicator embedding.
    """
    N = len(buffer)

    # --- Fine-tune V from V_0 on buffer, recompute advantages ---
    if wm_model is not None and wm_params_v0 is not None:
        rng, v_rng = jax.random.split(rng)
        v_steps = getattr(config, 'v_finetune_steps', 50)
        v_lr = getattr(config, 'v_finetune_lr', 1e-3)
        wm_params_ft = finetune_value_head(
            wm_model, wm_params_v0, buffer,
            num_steps=v_steps, lr=v_lr, rng=v_rng,
        )
        recompute_advantages(wm_model, wm_params_ft, buffer)
        all_indicators = jnp.array([entry["advantage"] > 0 for entry in buffer])
        n_pos = int(all_indicators.sum())
        mean_adv = float(jnp.mean(jnp.array([e["advantage"] for e in buffer])))
        logging.info(
            f"    V-advantage: {n_pos}/{N} entries I=1 (A>0), "
            f"mean_adv={mean_adv:.4f}"
        )
    else:
        all_indicators = jnp.ones(N, dtype=bool)
        logging.info(f"    No value head — all {N} entries get I=1")

    # Pre-stack entire buffer into batched tensors
    all_obs = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *[entry["obs"] for entry in buffer]
    )
    all_actions = jnp.concatenate([entry["actions"] for entry in buffer], axis=0)
    recap_alpha = config.recap_alpha

    # Scale gradient steps with buffer size: enough passes to converge from
    # a fresh LoRA init, but cap to avoid overfitting on tiny buffers.
    # ~10 passes over the buffer, with a floor of 20 and ceiling of num_bc_steps.
    max_passes = 10
    effective_steps = min(
        config.num_bc_steps,
        max(20, max_passes * N // max(config.bc_batch_size, 1)),
    )
    logging.info(f"    LoRA grad steps: {effective_steps} (buffer={N}, max={config.num_bc_steps})")

    for step in range(effective_steps):
        rng, sample_rng, step_rng = jax.random.split(rng, 3)
        batch_size = min(config.bc_batch_size, N)
        indices = jax.random.choice(
            sample_rng, N, shape=(batch_size,), replace=False
        )

        batch_obs = jax.tree.map(lambda x: x[indices], all_obs)
        batch_actions = all_actions[indices]
        batch_indicators = all_indicators[indices]

        B = batch_size
        task_B = jax.tree.map(
            lambda x: jnp.tile(x, (B,) + (1,) * (x.ndim - 1)), task
        )

        # Single transformer pass with plain task (frozen)
        pad_mask = batch_obs["pad_mask"]
        trans_out = _run_transformer(
            octo_model.module, octo_params, batch_obs, task_B, pad_mask
        )

        # Prepare action targets
        window_size = pad_mask.shape[1]
        actions_expanded = jnp.broadcast_to(
            batch_actions[:, None, :, :],
            (B, window_size, batch_actions.shape[-2], batch_actions.shape[-1]),
        )
        action_pad_mask = jnp.ones_like(actions_expanded, dtype=bool)
        timestep_pad_mask = pad_mask.astype(bool)

        # Gradient step over both LoRA layers and indicator embedding
        trainable = {"layers": lora_params["layers"], "indicator_embed": lora_params["indicator_embed"]}
        trainable, opt_state = _jit_indicator_grad_step(
            octo_model.module, octo_params, lora_params,
            trans_out, batch_indicators, actions_expanded,
            timestep_pad_mask, action_pad_mask,
            recap_alpha, step_rng, opt_state, optimizer,
            trainable,
        )
        lora_params["layers"] = trainable["layers"]
        lora_params["indicator_embed"] = trainable["indicator_embed"]

    return lora_params, opt_state
