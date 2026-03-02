"""World model training loop.

Trains projection head h and dynamics predictor f_ψ on pre-computed
Bridge V2 encoder outputs using InfoNCE contrastive loss.

Usage:
    python -m recap.training.train_world_model --config configs/train_wm.yaml
"""

import os
import time
from functools import partial
from typing import Any, Dict, Tuple

from absl import app, flags, logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import yaml
from flax.training import train_state, checkpoints

from recap.models.projection_head import ProjectionHead
from recap.models.dynamics_predictor import DynamicsPredictor
from recap.losses.contrastive import infonce_loss
from recap.data.oxe_contrastive import CachedContrastiveDataset

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/train_wm.yaml", "Path to config file.")


class WorldModel(nn.Module):
    """Combined projection head + dynamics predictor for joint training."""

    projection_head_kwargs: Dict[str, Any]
    dynamics_predictor_kwargs: Dict[str, Any]

    @nn.compact
    def __call__(self, z_t, z_target, *, train=False):
        """Forward pass: project, predict, and return projected target.

        Args:
            z_t: Encoder output at time t, shape (batch, 768).
            z_target: Encoder output at time t+m, shape (batch, 768).
            train: Training mode flag.

        Returns:
            predicted: Predicted anchor ẑ'_{t+m}, shape (batch, 256).
            target_proj: Projected actual z'_{t+m}, shape (batch, 256).
        """
        h = ProjectionHead(**self.projection_head_kwargs)
        f_psi = DynamicsPredictor(**self.dynamics_predictor_kwargs)

        # Project current and target states into contrastive space
        z_prime_t = h(z_t)            # (batch, 256)
        z_prime_target = h(z_target)  # (batch, 256) — stop gradient for target?

        # Predict anchor from current state
        predicted = f_psi(z_prime_t, train=train)  # (batch, 256)

        return predicted, z_prime_target


def create_train_state(config: dict, rng: jax.Array) -> train_state.TrainState:
    """Initialize model, optimizer, and training state."""
    model = WorldModel(
        projection_head_kwargs=config["model"]["projection_head"],
        dynamics_predictor_kwargs=config["model"]["dynamics_predictor"],
    )

    # Dummy inputs for initialization
    encoder_dim = 768  # Octo-Base
    dummy_z = jnp.zeros((2, encoder_dim))
    dummy_target = jnp.zeros((2, encoder_dim))

    params = model.init(rng, dummy_z, dummy_target, train=False)["params"]

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


@partial(jax.jit, static_argnames=("temperature",))
def train_step(
    state: train_state.TrainState,
    batch: dict,
    temperature: float,
) -> Tuple[train_state.TrainState, dict]:
    """Single training step.

    Args:
        state: Current training state.
        batch: Dict with z_t, z_target.
        temperature: InfoNCE temperature.

    Returns:
        Updated state and metrics dict.
    """

    def loss_fn(params):
        predicted, target_proj = state.apply_fn(
            {"params": params},
            batch["z_t"],
            batch["z_target"],
            train=True,
        )
        loss = infonce_loss(predicted, target_proj, temperature=temperature)
        return loss, {"loss": loss, "predicted": predicted, "target_proj": target_proj}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # Compute retrieval accuracy (is the nearest neighbor the correct target?)
    predicted = aux["predicted"]
    target_proj = aux["target_proj"]
    # Pairwise distances
    diff = predicted[:, None, :] - target_proj[None, :, :]
    distances = jnp.sum(diff ** 2, axis=-1)  # (B, B)
    nearest = jnp.argmin(distances, axis=-1)  # (B,)
    accuracy = jnp.mean(nearest == jnp.arange(predicted.shape[0]))

    metrics = {
        "loss": loss,
        "retrieval_accuracy": accuracy,
        "grad_norm": optax.global_norm(grads),
    }
    return state, metrics


def main(_):
    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

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
    total_steps = config["training"]["total_steps"]
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    # Training loop
    step = 0
    epoch = 0
    t0 = time.time()

    while step < total_steps:
        epoch += 1
        for batch in dataset.get_iterator():
            if step >= total_steps:
                break

            state, metrics = train_step(state, batch, temperature)
            step += 1

            if step % config["training"]["log_every"] == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed
                logging.info(
                    f"Step {step}/{total_steps} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"retrieval_acc={metrics['retrieval_accuracy']:.3f} | "
                    f"grad_norm={metrics['grad_norm']:.3f} | "
                    f"{steps_per_sec:.1f} steps/sec"
                )

            if step % config["training"]["save_every"] == 0:
                checkpoints.save_checkpoint(
                    config["output"]["checkpoint_dir"],
                    state,
                    step,
                    keep=3,
                )
                logging.info(f"Saved checkpoint at step {step}")

    # Final save
    checkpoints.save_checkpoint(
        config["output"]["checkpoint_dir"],
        state,
        step,
        keep=3,
    )
    logging.info(f"Training complete. Final step {step}, final loss {metrics['loss']:.4f}")


if __name__ == "__main__":
    app.run(main)
