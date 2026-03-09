"""Phase 2: Train dynamics transformer on tokenized Bridge V2 trajectories.

Loads raw frames from Bridge V2 trajectories, tokenizes them with the frozen
VQ-VAE encoder, then trains the dynamics transformer to predict future frame
tokens given two consecutive context frames + language instruction.

Co-trains a value head V(z_t, z_{t+1}, lang) that predicts the expected
tracking reward.

Bridge V2 images are 480x640 — resized to 256x256 in the data pipeline.
Language instructions are encoded with a frozen CLIP text encoder.

Usage:
    python -m recap.training.train_dynamics --config configs/train_dynamics.yaml
"""

import os
import time
from typing import Tuple

from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from flax.training import train_state, checkpoints

from recap.models.vqvae_tokenizer import VQVAETokenizer
from recap.models.dynamics_transformer import DynamicsTransformer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/train_dynamics.yaml", "Path to config file.",
                    allow_override=True)


def load_tokenizer(config):
    """Load frozen VQ-VAE tokenizer from checkpoint."""
    tok_cfg = config["tokenizer"]
    model = VQVAETokenizer(
        patch_size=tok_cfg.get("patch_size", 16),
        embed_dim=tok_cfg.get("embed_dim", 512),
        num_codes=tok_cfg.get("num_codes", 512),
        encoder_layers=tok_cfg.get("encoder_layers", 6),
        encoder_heads=tok_cfg.get("encoder_heads", 8),
    )
    dummy_x = jnp.zeros((1, 256, 256, 3))
    variables = model.init(jax.random.PRNGKey(0), dummy_x, train=False)

    restored = checkpoints.restore_checkpoint(tok_cfg["checkpoint_dir"], target=None)
    if "params" in restored:
        tok_params = restored["params"]
    else:
        tok_params = restored

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(tok_params))
    logging.info(f"Loaded frozen VQ-VAE tokenizer ({param_count:,} params)")
    return model, tok_params


def make_tokenize_fn(tokenizer_model, tokenizer_params, dp_sharding=None):
    """Create a sharded function to tokenize a batch of frames across all GPUs."""
    def tokenize(frames):
        """frames: (B, 256, 256, 3) float [0,1] → (B, N) integer indices."""
        return tokenizer_model.apply(
            {"params": tokenizer_params}, frames,
            method=tokenizer_model.encode, train=False,
        )

    if dp_sharding is not None:
        tokenize = jax.jit(
            tokenize,
            in_shardings=(dp_sharding,),
            out_shardings=dp_sharding,
        )
    else:
        tokenize = jax.jit(tokenize)
    return tokenize


def load_lang_encoder(config):
    """Load frozen CLIP text encoder for language conditioning.

    Returns a function: list[str] → (len, lang_dim) numpy array.
    """
    import transformers
    clip_model = config.get("lang_encoder", "openai/clip-vit-base-patch32")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_model)
    model = transformers.FlaxCLIPTextModel.from_pretrained(clip_model)
    lang_dim = model.config.hidden_size  # 512 for clip-vit-base-patch32

    def encode_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True,
                           max_length=77, return_tensors="np")
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        return np.array(outputs.pooler_output)

    logging.info(f"Loaded frozen CLIP text encoder ({clip_model}, dim={lang_dim})")
    return encode_texts, lang_dim


def load_bridge_trajectories(config):
    """Load Bridge V2 as (x_t, x_{t+1}, x_{t+m}) frame triples + language.

    Bridge V2 raw images are 480x640x3. Resized to 256x256 in pipeline.
    Language instruction is extracted per-episode and broadcast to all triples.
    """
    cache_dir = config["data"].get("cache_dir", None)
    m = config["data"]["prediction_horizon"]
    batch_size = config["data"]["batch_size"]

    builder_kwargs = {}
    if cache_dir:
        builder_kwargs["data_dir"] = cache_dir

    ds = tfds.load("bridge_dataset", split="train", **builder_kwargs)

    def extract_triples(episode):
        """RLDS: episode["steps"] is a nested tf.data.Dataset, not a tensor dict.
        Batch all steps into a single tensor, then slice triples."""
        steps_ds = episode["steps"]
        all_steps = steps_ds.batch(10000).get_single_element()
        images = all_steps["observation"]["image_0"]  # (T, 480, 640, 3)
        # Language instruction is the same for all steps — take first
        lang = all_steps["language_instruction"][0]  # scalar string tensor
        T = tf.shape(images)[0]
        max_t = T - m - 1
        max_t = tf.maximum(max_t, 0)
        indices = tf.range(0, max_t)
        x_t = tf.gather(images, indices)
        x_t1 = tf.gather(images, indices + 1)
        x_tm = tf.gather(images, indices + m)
        n = tf.shape(x_t)[0]
        langs = tf.fill([n], lang)
        return tf.data.Dataset.from_tensor_slices({
            "x_t": x_t, "x_t1": x_t1, "x_tm": x_tm, "lang": langs,
        })

    ds = ds.flat_map(extract_triples)

    def preprocess(sample):
        def proc(img):
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, [256, 256], method="bilinear")
            return tf.clip_by_value(img, 0.0, 1.0)
        return {
            "x_t": proc(sample["x_t"]),
            "x_t1": proc(sample["x_t1"]),
            "x_tm": proc(sample["x_tm"]),
            "lang": sample["lang"],
        }

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(config["data"].get("shuffle_buffer", 10000))
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(4)
    return ds


def create_train_state(config, rng, lang_dim=512):
    """Initialize dynamics transformer and optimizer."""
    model_cfg = config["model"]
    tok_cfg = config["tokenizer"]

    model = DynamicsTransformer(
        num_codes=tok_cfg.get("num_codes", 512),
        num_positions=model_cfg.get("num_positions", 256),
        embed_dim=model_cfg.get("embed_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
        num_layers=model_cfg.get("num_layers", 12),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout_rate=model_cfg.get("dropout_rate", 0.1),
        value_hidden_dim=model_cfg.get("value_hidden_dim", 256),
        value_num_layers=model_cfg.get("value_num_layers", 2),
        lang_dim=lang_dim,
        num_lang_tokens=model_cfg.get("num_lang_tokens", 4),
    )

    N = model_cfg.get("num_positions", 256)
    dummy_z = jnp.zeros((2, N), dtype=jnp.int32)
    dummy_lang = jnp.zeros((2, lang_dim))
    params = model.init(rng, dummy_z, dummy_z, lang_embed=dummy_lang, train=False)["params"]

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"Dynamics transformer parameters: {param_count:,}")

    # Log per-component sizes
    for key in params:
        comp_count = sum(x.size for x in jax.tree_util.tree_leaves(params[key]))
        logging.info(f"  {key}: {comp_count:,} params")

    train_cfg = config["training"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg["lr"],
        warmup_steps=train_cfg["warmup_steps"],
        decay_steps=train_cfg["total_steps"],
        end_value=train_cfg["lr"] * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_cfg["grad_clip"]),
        optax.adamw(learning_rate=schedule, weight_decay=train_cfg["weight_decay"]),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def make_train_step(value_loss_weight, replicated_sharding, dp_sharding):
    """Create sharded train step for multi-GPU data parallelism."""

    def train_step(state, z_t, z_t1, z_target, lang_embed):
        def loss_fn(params):
            logits, v_pred = state.apply_fn(
                {"params": params}, z_t, z_t1,
                lang_embed=lang_embed, train=True,
            )

            # Per-position cross-entropy
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, z_target)
            ce_loss = jnp.mean(ce)

            # Tracking reward target for value head
            per_example_ce = jnp.mean(ce, axis=-1)  # (B,)
            tracking_target = -per_example_ce
            tracking_target = jax.lax.stop_gradient(tracking_target)
            value_loss = jnp.mean((v_pred - tracking_target) ** 2)

            total_loss = ce_loss + value_loss_weight * value_loss

            # Top-1 accuracy per position
            predicted_tokens = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(predicted_tokens == z_target)

            # Top-5 accuracy
            top5 = jax.lax.top_k(logits, 5)[1]  # (B, N, 5)
            target_expanded = z_target[:, :, None]  # (B, N, 1)
            top5_correct = jnp.any(top5 == target_expanded, axis=-1)
            top5_accuracy = jnp.mean(top5_correct)

            return total_loss, {
                "ce_loss": ce_loss,
                "value_loss": value_loss,
                "accuracy": accuracy,
                "top5_accuracy": top5_accuracy,
                "v_pred_mean": jnp.mean(v_pred),
                "v_pred_std": jnp.std(v_pred),
                "tracking_target_mean": jnp.mean(tracking_target),
                "tracking_target_std": jnp.std(tracking_target),
                "per_example_ce_min": jnp.min(per_example_ce),
                "per_example_ce_max": jnp.max(per_example_ce),
            }

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = {**aux, "grad_norm": optax.global_norm(grads)}
        return state, metrics

    train_step = jax.jit(
        train_step,
        in_shardings=(replicated_sharding, dp_sharding, dp_sharding, dp_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        donate_argnums=0,
    )
    return train_step


def main(_):
    tf.config.set_visible_devices([], "GPU")

    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

    # --- System info + multi-GPU mesh ---
    devices = jax.devices()
    num_devices = len(devices)
    logging.info(f"JAX devices: {num_devices} × {devices[0].platform}")
    for i, d in enumerate(devices):
        logging.info(f"  Device {i}: {d}")

    batch_size = config["data"]["batch_size"]
    assert batch_size % num_devices == 0, (
        f"batch_size {batch_size} must be divisible by device_count {num_devices}"
    )
    logging.info(f"Config: batch_size={batch_size} ({batch_size // num_devices}/device), "
                 f"prediction_horizon={config['data']['prediction_horizon']}, "
                 f"total_steps={config['training']['total_steps']}")

    mesh = Mesh(jax.devices(), axis_names="batch")
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    # --- Load frozen tokenizer ---
    logging.info("Loading frozen VQ-VAE tokenizer...")
    t_tok = time.time()
    tokenizer_model, tokenizer_params = load_tokenizer(config)
    tokenize = make_tokenize_fn(tokenizer_model, tokenizer_params, dp_sharding=dp_sharding)
    logging.info(f"Tokenizer loaded in {time.time() - t_tok:.1f}s")

    # --- Warm up tokenizer JIT ---
    logging.info("JIT compiling tokenizer...")
    t_jit = time.time()
    dummy_frames = jax.device_put(jnp.zeros((3 * batch_size, 256, 256, 3)), dp_sharding)
    _ = tokenize(dummy_frames)
    jax.block_until_ready(_)
    logging.info(f"Tokenizer JIT: {time.time() - t_jit:.1f}s")

    # --- Load frozen language encoder ---
    logging.info("Loading frozen CLIP text encoder...")
    encode_texts, lang_dim = load_lang_encoder(config)

    # --- Initialize dynamics model ---
    rng = jax.random.PRNGKey(config["data"]["seed"])
    state = create_train_state(config, rng, lang_dim=lang_dim)

    # --- Load data ---
    logging.info("Building data pipeline...")
    t_data = time.time()
    ds = load_bridge_trajectories(config)
    logging.info(f"Data pipeline built in {time.time() - t_data:.1f}s")

    # Skip full dataset scan to avoid idling GPUs for 10-30 min
    steps_per_epoch = None

    value_loss_weight = config["loss"].get("value_loss_weight", 1.0)
    total_steps = config["training"]["total_steps"]
    log_every = config["training"]["log_every"]
    eval_every = config["training"].get("eval_every", 2000)
    save_every = config["training"]["save_every"]
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    train_step = make_train_step(value_loss_weight, replicated_sharding, dp_sharding)

    step = 0
    epoch = 0
    t0 = time.time()
    jit_compiled = False
    step_times = []
    tokenize_times = []

    while step < total_steps:
        epoch += 1
        t_epoch = time.time()

        for batch in ds.as_numpy_iterator():
            if step >= total_steps:
                break

            t_step = time.time()

            # Batch-tokenize all 3 frames together, sharded across all GPUs
            t_tok_start = time.time()
            x_all = np.concatenate([batch["x_t"], batch["x_t1"], batch["x_tm"]], axis=0)  # (3B, 256, 256, 3)
            x_all = jax.device_put(x_all, dp_sharding)  # single transfer, sharded
            z_all = tokenize(x_all)  # (3B, N) — sharded
            z_t, z_t1, z_target = jnp.split(z_all, 3, axis=0)
            if not jit_compiled:
                jax.block_until_ready(z_target)
            tok_dur = time.time() - t_tok_start

            # Encode language instructions (on CPU, frozen CLIP)
            lang_strings = [s.decode("utf-8") if isinstance(s, bytes) else s
                            for s in batch["lang"]]
            lang_embed = encode_texts(lang_strings)  # (B, lang_dim)
            lang_embed = jax.device_put(lang_embed, dp_sharding)

            # Train step
            state, metrics = train_step(state, z_t, z_t1, z_target, lang_embed)

            # First step: measure JIT compilation
            if not jit_compiled:
                jax.block_until_ready(metrics)
                jit_time = time.time() - t_step
                logging.info(f"JIT compilation + first step: {jit_time:.1f}s "
                             f"(tokenize: {tok_dur:.1f}s)")
                jit_compiled = True
                t0 = time.time()

            step += 1
            step_dur = time.time() - t_step
            step_times.append(step_dur)
            tokenize_times.append(tok_dur)
            if len(step_times) > 100:
                step_times.pop(0)
                tokenize_times.pop(0)

            if step % log_every == 0:
                metrics = jax.device_get(metrics)
                elapsed = time.time() - t0
                sps = step / elapsed
                avg_step = sum(step_times) / len(step_times)
                avg_tok = sum(tokenize_times) / len(tokenize_times)
                remaining = total_steps - step
                eta_sec = remaining * avg_step
                eta_hours = eta_sec / 3600
                eta_min = eta_sec / 60
                pct = 100.0 * step / total_steps

                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_min:.1f}min"

                logging.info(
                    f"Step {step:>6d}/{total_steps} ({pct:5.1f}%) | "
                    f"CE={metrics['ce_loss']:.4f} | "
                    f"acc={metrics['accuracy']:.3f} top5={metrics['top5_accuracy']:.3f} | "
                    f"v_loss={metrics['value_loss']:.4f} | "
                    f"grad={metrics['grad_norm']:.3f} | "
                    f"{sps:.1f} steps/s (tok={avg_tok*1000:.0f}ms) | "
                    f"ETA {eta_str}"
                )

            if step % eval_every == 0:
                metrics = jax.device_get(metrics)
                logging.info(
                    f"  [diag] CE range: [{metrics['per_example_ce_min']:.3f}, "
                    f"{metrics['per_example_ce_max']:.3f}] | "
                    f"V: pred={metrics['v_pred_mean']:.4f}±{metrics['v_pred_std']:.4f} "
                    f"target={metrics['tracking_target_mean']:.4f}±{metrics['tracking_target_std']:.4f}"
                )

            if step % save_every == 0:
                checkpoints.save_checkpoint(
                    config["output"]["checkpoint_dir"], state, step, keep=3,
                )
                logging.info(f"Saved checkpoint at step {step}")

        epoch_time = time.time() - t_epoch
        logging.info(f"Epoch {epoch} complete ({epoch_time:.1f}s, step {step}/{total_steps})")

    # Final save
    if step % save_every != 0:
        checkpoints.save_checkpoint(
            config["output"]["checkpoint_dir"], state, step, keep=3,
        )

    total_time = time.time() - t0
    logging.info(f"Training complete in {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    logging.info(f"Final: CE={metrics['ce_loss']:.4f}, acc={metrics['accuracy']:.3f}, "
                 f"top5={metrics['top5_accuracy']:.3f}")


if __name__ == "__main__":
    app.run(main)
