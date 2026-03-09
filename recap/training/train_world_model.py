"""World model training loop.

Trains projection head h and dynamics predictor f_ψ on pre-computed
Bridge V2 encoder outputs using InfoNCE contrastive loss.

Usage:
    python -m recap.training.train_world_model --config configs/train_wm.yaml
"""

import os
import time
from typing import Any, Dict, Tuple

from absl import app, flags, logging
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorflow as tf
import yaml
from flax.training import train_state, checkpoints

from recap.models.projection_head import ProjectionHead
from recap.models.dynamics_predictor import DynamicsPredictor
from recap.models.value_head import ValueHead
from recap.losses.contrastive import infonce_loss
from recap.data.oxe_contrastive import CachedContrastiveDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/train_wm.yaml", "Path to config file.",
                    allow_override=True)


class WorldModel(nn.Module):
    """Combined projection head + dynamics predictor for joint training."""

    projection_head_kwargs: Dict[str, Any]
    dynamics_predictor_kwargs: Dict[str, Any]
    value_head_kwargs: Dict[str, Any] = None  # None = no value head (backward compat)

    @nn.compact
    def __call__(self, z_t, z_t1, z_target, *, train=False):
        """Forward pass: project, predict, and return projected target.

        Inputs come from a single Octo forward pass at time t:
            z_t:     read[t-1]      — position 0 readout (1-frame context)
            z_t1:    read[t, t-1]   — position 1 readout (2-frame context)
            z_target: read[t+m, t+m-1] — position 1 readout m steps later

        Returns:
            If value_head_kwargs is None:
                (predicted, target_proj)
            Otherwise:
                (predicted, target_proj, v_pred)
        """
        h = ProjectionHead(**self.projection_head_kwargs)
        f_psi = DynamicsPredictor(**self.dynamics_predictor_kwargs)

        # Shared projection head for all readouts
        z_prime_t = h(z_t)            # (batch, proj_dim)
        z_prime_t1 = h(z_t1)          # (batch, proj_dim)
        z_prime_target = h(z_target)  # (batch, proj_dim)

        # Dynamics predictor: concat projected [read[t-1], read[t,t-1]] → predict target
        dynamics_input = jnp.concatenate([z_prime_t, z_prime_t1], axis=-1)  # (batch, 2*proj_dim)
        predicted = f_psi(dynamics_input, train=train)  # (batch, proj_dim)

        if self.value_head_kwargs is not None:
            v_head = ValueHead(**self.value_head_kwargs)
            # stop_gradient: V is a passive observer, shouldn't affect projection/dynamics learning
            v_pred = v_head(
                jax.lax.stop_gradient(z_prime_t),
                jax.lax.stop_gradient(z_prime_t1),
                jax.lax.stop_gradient(predicted),
                train=train,
            )
            return predicted, z_prime_target, v_pred

        return predicted, z_prime_target


def shard_batch(batch, dp_sharding):
    """Place batch arrays on devices with data-parallel sharding."""
    return jax.tree.map(lambda x: jax.device_put(x, dp_sharding), batch)


def create_train_state(config: dict, rng: jax.Array) -> train_state.TrainState:
    """Initialize model, optimizer, and training state."""
    value_head_cfg = config["model"].get("value_head", None)
    model = WorldModel(
        projection_head_kwargs=config["model"]["projection_head"],
        dynamics_predictor_kwargs=config["model"]["dynamics_predictor"],
        value_head_kwargs=value_head_cfg,
    )

    # Dummy inputs for initialization
    encoder_dim = 768  # Octo-Base
    dummy_z = jnp.zeros((2, encoder_dim))
    dummy_z1 = jnp.zeros((2, encoder_dim))
    dummy_target = jnp.zeros((2, encoder_dim))

    params = model.init(rng, dummy_z, dummy_z1, dummy_target, train=False)["params"]

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"World model parameters: {param_count:,}")

    # Optimizer: AdamW with warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["training"]["lr"],
        warmup_steps=config["training"]["warmup_steps"],
        decay_steps=config["training"]["total_steps"],
        end_value=config["training"]["lr"] * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config["training"]["grad_clip"]),
        optax.adamw(learning_rate=schedule, weight_decay=config["training"]["weight_decay"]),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def make_train_step(replicated_sharding, dp_sharding, temperature: float,
                    intra_weight: float = 0.0, value_loss_weight: float = 0.0):
    """Create a sharded train_step function.

    Args:
        replicated_sharding: NamedSharding for replicated params/metrics.
        dp_sharding: NamedSharding for data-parallel batch sharding.
        temperature: InfoNCE temperature (static).
        intra_weight: Weight for intra-trajectory loss (0.0 = cross-only).
        value_loss_weight: Weight for value head MSE loss (0.0 = no value head).

    Returns:
        JIT-compiled train_step function with correct shardings.
    """

    def train_step(
        state: train_state.TrainState,
        batch: dict,
    ) -> Tuple[train_state.TrainState, dict]:
        def loss_fn(params):
            result = state.apply_fn(
                {"params": params},
                batch["z_t"],
                batch["z_t1"],
                batch["z_target"],
                train=True,
            )

            if len(result) == 3:
                predicted, target_proj, v_pred = result
                # Tracking reward target: cosine similarity (what V should predict)
                tracking_target = jnp.sum(predicted * target_proj, axis=-1)  # (B,)
                tracking_target = jax.lax.stop_gradient(tracking_target)
                value_loss = jnp.mean((v_pred - tracking_target) ** 2)
            else:
                predicted, target_proj = result
                value_loss = 0.0

            traj_ids = batch["traj_id"]
            contrastive_loss = infonce_loss(
                predicted, target_proj,
                temperature=temperature,
                traj_ids=traj_ids,
                intra_weight=intra_weight,
            )
            total_loss = contrastive_loss + value_loss_weight * value_loss

            aux = {
                "loss": contrastive_loss,
                "value_loss": value_loss,
                "predicted": predicted,
                "target_proj": target_proj,
            }
            if len(result) == 3:
                aux["v_pred"] = v_pred
                aux["tracking_target"] = tracking_target
            return total_loss, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        # Compute retrieval accuracy (is the nearest neighbor the correct target?)
        predicted = aux["predicted"]
        target_proj = aux["target_proj"]
        # Pairwise distances — XLA inserts all-gather across devices automatically
        diff = predicted[:, None, :] - target_proj[None, :, :]
        distances = jnp.sum(diff ** 2, axis=-1)  # (B, B)
        nearest = jnp.argmin(distances, axis=-1)  # (B,)
        accuracy = jnp.mean(nearest == jnp.arange(predicted.shape[0]))

        # --- Representation diagnostics ---
        # Effective dimension: how many singular values carry meaningful variance.
        # Low eff_dim = collapsed representations. Healthy = 50-200+ for proj_dim=512.
        # Uses participation ratio: (sum σ_i)^2 / sum σ_i^2
        pred_centered = predicted - jnp.mean(predicted, axis=0, keepdims=True)
        _, svs, _ = jnp.linalg.svd(pred_centered, full_matrices=False)
        eff_dim = jnp.square(jnp.sum(svs)) / (jnp.sum(jnp.square(svs)) + 1e-8)

        # Embedding std: should be O(1) if not collapsed
        embed_std = jnp.std(predicted)

        # Cosine similarity between predicted and target (tracking reward distribution)
        cosine_sims = jnp.sum(predicted * target_proj, axis=-1)  # (B,)

        metrics = {
            "loss": aux["loss"],
            "value_loss": aux["value_loss"],
            "retrieval_accuracy": accuracy,
            "grad_norm": optax.global_norm(grads),
            # Representation health
            "eff_dim": eff_dim,
            "embed_std": embed_std,
            "cosine_sim_mean": jnp.mean(cosine_sims),
            "cosine_sim_std": jnp.std(cosine_sims),
        }

        # Value head diagnostics (only meaningful when V is active)
        if "v_pred" in aux:
            v_pred = aux["v_pred"]
            tracking_target = aux["tracking_target"]
            metrics["v_pred_mean"] = jnp.mean(v_pred)
            metrics["v_pred_std"] = jnp.std(v_pred)
            metrics["v_target_mean"] = jnp.mean(tracking_target)
            metrics["v_target_std"] = jnp.std(tracking_target)
            # Correlation: does V track the target, or is it constant?
            v_centered = v_pred - jnp.mean(v_pred)
            t_centered = tracking_target - jnp.mean(tracking_target)
            v_corr = jnp.sum(v_centered * t_centered) / (
                jnp.sqrt(jnp.sum(v_centered ** 2) * jnp.sum(t_centered ** 2)) + 1e-8
            )
            metrics["v_correlation"] = v_corr

        return state, metrics

    # JAX 0.4.20: jax.jit requires fun as first positional arg
    train_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        donate_argnums=0,
    )
    return train_step


def main(_):
    # Prevent TensorFlow from grabbing GPU memory
    tf.config.set_visible_devices([], "GPU")

    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

    # Multi-GPU mesh setup (works transparently on 1 GPU)
    devices = jax.devices()
    logging.info(f"JAX devices: {len(devices)} ({[d.platform for d in devices]})")
    assert config["data"]["batch_size"] % len(devices) == 0, (
        f"batch_size {config['data']['batch_size']} must be divisible by "
        f"device_count {len(devices)}"
    )

    mesh = Mesh(jax.devices(), axis_names="batch")
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    # Initialize
    rng = jax.random.PRNGKey(config["data"]["seed"])
    state = create_train_state(config, rng)

    # Load cached encodings
    logging.info(f"Loading cached encodings from {config['data']['cache_dir']}...")
    dataset = CachedContrastiveDataset(
        cache_dir=config["data"]["cache_dir"],
        batch_size=config["data"]["batch_size"],
        seed=config["data"]["seed"],
    )
    logging.info(f"Loaded {dataset.num_samples:,} transitions, {len(dataset)} batches/epoch")

    temperature = config["loss"]["temperature"]
    intra_weight = config["loss"].get("intra_weight", 0.0)
    total_steps = config["training"]["total_steps"]
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    value_loss_weight = config["loss"].get("value_loss_weight", 0.0)

    # Build sharded train_step
    train_step = make_train_step(replicated_sharding, dp_sharding, temperature,
                                 intra_weight=intra_weight, value_loss_weight=value_loss_weight)

    # Training loop
    step = 0
    epoch = 0
    t0 = time.time()

    while step < total_steps:
        epoch += 1
        for batch in dataset.get_iterator():
            if step >= total_steps:
                break

            batch = shard_batch(batch, dp_sharding)
            state, metrics = train_step(state, batch)
            step += 1

            if step % config["training"]["log_every"] == 0:
                metrics = jax.device_get(metrics)
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                v_loss_str = f" | v_loss={metrics['value_loss']:.4f}" if metrics['value_loss'] != 0.0 else ""
                logging.info(
                    f"Step {step}/{total_steps} | "
                    f"loss={metrics['loss']:.4f}{v_loss_str} | "
                    f"retrieval_acc={metrics['retrieval_accuracy']:.3f} | "
                    f"grad_norm={metrics['grad_norm']:.3f} | "
                    f"{steps_per_sec:.1f} steps/sec"
                )

            # Detailed diagnostics at eval_every intervals
            if step % config["training"]["eval_every"] == 0:
                if not isinstance(metrics, dict):
                    metrics = jax.device_get(metrics)
                elif "eff_dim" in metrics and hasattr(metrics["eff_dim"], "device"):
                    metrics = jax.device_get(metrics)
                logging.info(
                    f"  [diag] eff_dim={metrics['eff_dim']:.1f} | "
                    f"embed_std={metrics['embed_std']:.4f} | "
                    f"cosine_sim={metrics['cosine_sim_mean']:.4f}±{metrics['cosine_sim_std']:.4f}"
                )
                if "v_correlation" in metrics:
                    logging.info(
                        f"  [diag] V: pred={metrics['v_pred_mean']:.4f}±{metrics['v_pred_std']:.4f} | "
                        f"target={metrics['v_target_mean']:.4f}±{metrics['v_target_std']:.4f} | "
                        f"corr={metrics['v_correlation']:.4f}"
                    )

            if step % config["training"]["save_every"] == 0:
                checkpoints.save_checkpoint(
                    config["output"]["checkpoint_dir"],
                    state,
                    step,
                    keep=3,
                )
                logging.info(f"Saved checkpoint at step {step}")

    # Final save (skip if already saved by save_every)
    if step % config["training"]["save_every"] != 0:
        checkpoints.save_checkpoint(
            config["output"]["checkpoint_dir"],
            state,
            step,
            keep=3,
        )
    logging.info(f"Training complete. Final step {step}, final loss {metrics['loss']:.4f}")


if __name__ == "__main__":
    app.run(main)
