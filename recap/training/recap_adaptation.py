"""RECAP test-time adaptation loop.

Rolls out the policy in a perturbed environment, computes tracking rewards
per chunk using the frozen world model, and updates LoRA weights on the
action head via advantage-conditioned behavioral cloning.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

from recap.models.lora_adapter import init_lora_params, apply_lora
from recap.losses.tracking_reward import tracking_reward
from recap.training.train_world_model import WorldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_world_model(checkpoint_path):
    """Load frozen world model params from checkpoint.

    Returns:
        wm_params: dict with "projection_head" and "dynamics_predictor" keys.
    """
    state = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    return {
        "projection_head": state["params"]["ProjectionHead_0"],
        "dynamics_predictor": state["params"]["DynamicsPredictor_0"],
    }


def recap_adapt(octo_model, octo_params, env, task, config, rng):
    """Run RECAP adaptation in a perturbed environment.

    Args:
        octo_model: Octo model (architecture, for forward passes).
        octo_params: Frozen Octo parameters.
        env: Environment with .reset() and .step(action) interface.
        task: Task specification (language instruction).
        config: Adaptation config (rank, alpha, lr, update_freq, num_bc_steps,
                wm_checkpoint, etc.).
        rng: JAX PRNG key.

    Returns:
        adapted_params: Octo params with trained LoRA merged in.
    """
    # --- Setup ---
    wm_params = load_world_model(config.wm_checkpoint)
    rng, lora_rng = jax.random.split(rng)
    lora_params = init_lora_params(lora_rng, octo_params, rank=config.rank, alpha=config.alpha)
    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(lora_params["layers"])

    # Buffer stores encoder outputs (not raw obs) per the spec
    # Each entry: {z_t, z_t1, z_t_plus_m, actions, r_aux}
    buffer = []
    chunk_count = 0

    # --- Episode loop ---
    for episode in range(config.num_episodes):
        obs = env.reset()
        done = False
        prev_z_t = None   # 1-frame readout from previous chunk
        prev_z_t1 = None  # 2-frame readout from previous chunk

        while not done:
            # 1. Encode current observation (frozen)
            rng, act_rng = jax.random.split(rng)
            pad_mask = obs["timestep_pad_mask"]
            transformer_out = octo_model.run_transformer(obs, task, pad_mask)
            tokens = transformer_out["readout_action"].tokens
            z_t = jnp.mean(tokens[:, 0, :, :], axis=1)    # (batch, 768) 1-frame
            z_t1 = jnp.mean(tokens[:, -1, :, :], axis=1)  # (batch, 768) 2-frame

            # 2. Act: run policy with current LoRA weights for m steps, essentially run the m chunk
            merged_params = apply_lora(octo_params, lora_params)
            actions = octo_model.sample_actions(merged_params, obs, task, rng=act_rng)
            next_obs, done = env.step(actions)  # execute chunk of m actions

            # 3. Encode result (frozen)
            # we reached the observation o_{t+m} so we want to compute it and compute reward for every m step
            next_pad_mask = next_obs["timestep_pad_mask"]
            next_out = octo_model.run_transformer(next_obs, task, next_pad_mask)
            next_tokens = next_out["readout_action"].tokens
            z_t_plus_m = jnp.mean(next_tokens[:, 0, :, :], axis=1)  # (batch, 768)

            # 4. Compute tracking reward (needs previous chunk's encodings)
            if prev_z_t is not None:
                r_aux = tracking_reward(prev_z_t, prev_z_t1, z_t_plus_m, wm_params)
                buffer.append({
                    "z_t": prev_z_t,
                    "z_t1": prev_z_t1,
                    "z_t_plus_m": z_t_plus_m,
                    "actions": actions,
                    "r_aux": r_aux,
                })
                chunk_count += 1

                # Evict oldest if buffer is full
                if len(buffer) > config.buffer_size:
                    buffer.pop(0)

            prev_z_t = z_t
            prev_z_t1 = z_t1
            obs = next_obs

            # 5. Update LoRA every M chunks (reset to pretrained, retrain)
            if chunk_count > 0 and chunk_count % config.update_freq == 0 and len(buffer) >= config.update_freq:
                rng, lora_rng = jax.random.split(rng)
                lora_params = init_lora_params(lora_rng, octo_params, rank=config.rank, alpha=config.alpha)
                opt_state = optimizer.init(lora_params["layers"])
                lora_params, opt_state = _update_lora(
                    lora_params, opt_state, optimizer,
                    octo_model, octo_params, buffer, config, rng,
                )

    # --- Return adapted policy ---
    return apply_lora(octo_params, lora_params)


def _update_lora(lora_params, opt_state, optimizer,
                 octo_model, octo_params, buffer, config, rng):
    """LoRA update per the RECAP algorithm (Eq. 3).

    Resets LoRA to pretrained each call, then trains on the full buffer using:
      - Advantages computed from all buffer rewards (no value function)
      - Binary indicator I_t = 1(A_t > advantage_eps) to filter which chunks
        get the improvement-conditioned loss term
      - Two-term loss: standard BC + alpha * improvement-conditioned BC
    """
    # Compute advantages over the full buffer
    all_rewards = jnp.array([entry["r_aux"] for entry in buffer])
    reward_mean = jnp.mean(all_rewards)
    reward_std = jnp.std(all_rewards)
    all_advantages = (all_rewards - reward_mean) / (reward_std + config.advantage_eps)

    # Binary indicator: I_t = 1(A_t > advantage_eps)
    all_indicators = (all_advantages > config.advantage_eps).astype(jnp.float32)

    for step in range(config.num_bc_steps):
        rng, sample_rng = jax.random.split(rng)
        batch_size = min(config.bc_batch_size, len(buffer))
        indices = jax.random.choice(
            sample_rng, len(buffer), shape=(batch_size,), replace=False
        )
        batch = [buffer[int(i)] for i in indices]
        batch_indicators = all_indicators[indices]

        def loss_fn(lora_layers):
            lp = {**lora_params, "layers": lora_layers}
            merged_params = apply_lora(octo_params, lp)
            total_loss = 0.0
            for i, entry in enumerate(batch):
                # Term 1: standard BC loss (unconditional)
                bc_loss = octo_model.action_head_loss(
                    merged_params, entry["z_t"], entry["actions"]
                )
                # Term 2: improvement-conditioned BC (only when I_t = 1)
                improvement_loss = batch_indicators[i] * octo_model.action_head_loss(
                    merged_params, entry["z_t"], entry["actions"]
                )
                total_loss += bc_loss + config.alpha * improvement_loss
            return total_loss / batch_size

        grads = jax.grad(loss_fn)(lora_params["layers"])
        updates, opt_state = optimizer.update(grads, opt_state)
        lora_params["layers"] = optax.apply_updates(lora_params["layers"], updates)

    return lora_params, opt_state
