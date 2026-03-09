"""World model + value head diagnostics.

Loads a trained checkpoint and runs representation health checks:
  1. Effective dimension (participation ratio of SVD)
  2. Embedding std (collapsed = near 0)
  3. Cosine similarity distribution (tracking reward range)
  4. Value head collapse check (is V predicting a constant?)
  5. Value head correlation (does V track actual tracking reward?)
  6. Per-dimension activation stats (dead dimensions?)

Usage:
    python tests/test_wm_diagnostics.py --checkpoint checkpoints/world_model_v1
    python tests/test_wm_diagnostics.py --checkpoint checkpoints/world_model_v1 --early_checkpoint checkpoints/world_model_v1/checkpoint_5000
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging
from flax.training import checkpoints

from recap.training.train_world_model import WorldModel

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "checkpoints/world_model_v1", "Checkpoint dir")
flags.DEFINE_string("early_checkpoint", None,
                    "Optional early checkpoint for comparison (e.g. step 5000)")
flags.DEFINE_string("encodings", None,
                    "Path to encodings.h5 for real-data diagnostics (optional)")


def load_checkpoint(ckpt_dir):
    state = checkpoints.restore_checkpoint(ckpt_dir, target=None)
    return state["params"]


def run_diagnostics(params, model, label, data=None):
    """Run full diagnostics on a checkpoint."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Diagnostics: {label}")
    logging.info(f"{'='*60}")

    # Count params
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"Total parameters: {param_count:,}")

    # Check if value head exists
    has_value_head = "ValueHead_0" in str(jax.tree_util.tree_structure(params))
    logging.info(f"Value head present: {has_value_head}")

    # Generate test inputs (random or real data)
    if data is not None:
        N = min(1024, data["z_t"].shape[0])
        idx = np.random.choice(data["z_t"].shape[0], N, replace=False)
        z_t = jnp.array(data["z_t"][idx])
        z_t1 = jnp.array(data["z_t1"][idx])
        z_target = jnp.array(data["z_target"][idx])
        logging.info(f"Using real data: {N} samples")
    else:
        N = 512
        rng = jax.random.PRNGKey(42)
        r1, r2, r3 = jax.random.split(rng, 3)
        z_t = jax.random.normal(r1, (N, 768))
        z_t1 = jax.random.normal(r2, (N, 768))
        z_target = jax.random.normal(r3, (N, 768))
        logging.info(f"Using random data: {N} samples")

    # Forward pass
    result = model.apply({"params": params}, z_t, z_t1, z_target, train=False)
    if len(result) == 3:
        predicted, target_proj, v_pred = result
    else:
        predicted, target_proj = result
        v_pred = None

    # --- 1. Effective dimension ---
    logging.info(f"\n--- 1. Effective Dimension ---")
    for name, emb in [("predicted", predicted), ("target_proj", target_proj)]:
        centered = emb - jnp.mean(emb, axis=0, keepdims=True)
        _, svs, _ = jnp.linalg.svd(centered, full_matrices=False)
        eff_dim = float(jnp.square(jnp.sum(svs)) / (jnp.sum(jnp.square(svs)) + 1e-8))
        top5_svs = np.array(svs[:5])
        sv_ratio = float(svs[0] / (svs[-1] + 1e-8))
        logging.info(f"  {name}: eff_dim={eff_dim:.1f} | top5_svs={top5_svs} | sv_ratio={sv_ratio:.1f}")
        if eff_dim < 5:
            logging.warning(f"  WARNING: {name} eff_dim={eff_dim:.1f} — likely collapsed!")
        elif eff_dim < 20:
            logging.warning(f"  CAUTION: {name} eff_dim={eff_dim:.1f} — low diversity")
        else:
            logging.info(f"  OK: {name} eff_dim={eff_dim:.1f} — healthy")

    # --- 2. Embedding statistics ---
    logging.info(f"\n--- 2. Embedding Statistics ---")
    for name, emb in [("predicted", predicted), ("target_proj", target_proj)]:
        emb_np = np.array(emb)
        logging.info(
            f"  {name}: mean={emb_np.mean():.4f} std={emb_np.std():.4f} "
            f"min={emb_np.min():.4f} max={emb_np.max():.4f} "
            f"norm_mean={float(jnp.mean(jnp.linalg.norm(emb, axis=-1))):.4f}"
        )
        # Per-dimension: how many dims are effectively dead?
        dim_std = np.std(emb_np, axis=0)  # (proj_dim,)
        dead_dims = int(np.sum(dim_std < 1e-4))
        logging.info(
            f"  {name}: per-dim std: min={dim_std.min():.6f} median={np.median(dim_std):.6f} "
            f"max={dim_std.max():.6f} | dead_dims(<1e-4)={dead_dims}/{len(dim_std)}"
        )

    # --- 3. Cosine similarity (tracking reward) distribution ---
    logging.info(f"\n--- 3. Cosine Similarity Distribution ---")
    cosine_sims = np.array(jnp.sum(predicted * target_proj, axis=-1))
    logging.info(
        f"  mean={cosine_sims.mean():.4f} std={cosine_sims.std():.4f} "
        f"min={cosine_sims.min():.4f} max={cosine_sims.max():.4f}"
    )
    # Histogram buckets
    for lo, hi in [(-1, -0.5), (-0.5, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        count = int(np.sum((cosine_sims >= lo) & (cosine_sims < hi)))
        logging.info(f"  [{lo:+.2f}, {hi:+.2f}): {count}/{N} ({100*count/N:.1f}%)")
    if cosine_sims.std() < 0.01:
        logging.warning("  WARNING: cosine sim has near-zero variance — all predictions identical!")

    # --- 4 & 5. Value head diagnostics ---
    if v_pred is not None:
        logging.info(f"\n--- 4. Value Head Collapse Check ---")
        v_np = np.array(v_pred)
        logging.info(
            f"  v_pred: mean={v_np.mean():.4f} std={v_np.std():.4f} "
            f"min={v_np.min():.4f} max={v_np.max():.4f}"
        )
        if v_np.std() < 0.001:
            logging.warning(f"  WARNING: v_pred std={v_np.std():.6f} — VALUE HEAD COLLAPSED to constant!")
        elif v_np.std() < 0.01:
            logging.warning(f"  CAUTION: v_pred std={v_np.std():.4f} — low variance")
        else:
            logging.info(f"  OK: v_pred std={v_np.std():.4f} — not collapsed")

        logging.info(f"\n--- 5. Value Head Correlation ---")
        tracking_target = cosine_sims  # same as what V was trained on
        v_centered = v_np - v_np.mean()
        t_centered = tracking_target - tracking_target.mean()
        corr = np.sum(v_centered * t_centered) / (
            np.sqrt(np.sum(v_centered**2) * np.sum(t_centered**2)) + 1e-8
        )
        logging.info(f"  Pearson correlation(v_pred, tracking_reward): {corr:.4f}")
        # Residual analysis
        residuals = v_np - tracking_target
        logging.info(
            f"  Residuals: mean={residuals.mean():.4f} std={residuals.std():.4f} "
            f"MAE={np.abs(residuals).mean():.4f}"
        )
        if abs(corr) < 0.1:
            logging.warning("  WARNING: V is uncorrelated with tracking reward!")
        elif corr < 0.5:
            logging.info(f"  MODERATE: V partially predicts tracking reward (corr={corr:.3f})")
        else:
            logging.info(f"  GOOD: V strongly predicts tracking reward (corr={corr:.3f})")
    else:
        logging.info("\n--- Value head not present in this checkpoint ---")

    # --- 6. Retrieval accuracy ---
    logging.info(f"\n--- 6. Retrieval Accuracy ---")
    diff = predicted[:, None, :] - target_proj[None, :, :]
    distances = jnp.sum(diff ** 2, axis=-1)
    nearest = jnp.argmin(distances, axis=-1)
    acc = float(jnp.mean(nearest == jnp.arange(N)))
    logging.info(f"  Retrieval accuracy (nearest neighbor): {acc:.3f}")

    pred_c = predicted - jnp.mean(predicted, axis=0, keepdims=True)
    _, sv_final, _ = jnp.linalg.svd(pred_c, full_matrices=False)
    return {
        "eff_dim_predicted": float(jnp.square(jnp.sum(sv_final)) / (jnp.sum(jnp.square(sv_final)) + 1e-8)),
        "embed_std": float(jnp.std(predicted)),
        "cosine_sim_mean": float(cosine_sims.mean()),
        "cosine_sim_std": float(cosine_sims.std()),
        "retrieval_acc": acc,
        "v_pred_std": float(v_np.std()) if v_pred is not None else None,
        "v_correlation": float(corr) if v_pred is not None else None,
    }


def main(_):
    ckpt_dir = FLAGS.checkpoint

    # Build model matching the config
    model = WorldModel(
        projection_head_kwargs={"hidden_dim": 1024, "output_dim": 512},
        dynamics_predictor_kwargs={"hidden_dim": 2048, "num_layers": 4, "output_dim": 512},
        value_head_kwargs={"hidden_dim": 256, "num_layers": 2},
    )

    # Load real data if available
    data = None
    if FLAGS.encodings:
        import h5py
        logging.info(f"Loading encodings from {FLAGS.encodings}...")
        with h5py.File(FLAGS.encodings, "r") as f:
            data = {
                "z_t": np.array(f["z_t"]),
                "z_t1": np.array(f["z_t1"]),
                "z_target": np.array(f["z_target"]),
            }
        logging.info(f"Loaded {data['z_t'].shape[0]:,} transitions")

    # Run on final checkpoint
    params = load_checkpoint(ckpt_dir)
    final_results = run_diagnostics(params, model, f"Final checkpoint ({ckpt_dir})", data)

    # Compare with early checkpoint if provided
    if FLAGS.early_checkpoint:
        early_params = load_checkpoint(FLAGS.early_checkpoint)
        early_results = run_diagnostics(early_params, model, f"Early checkpoint ({FLAGS.early_checkpoint})", data)

        logging.info(f"\n{'='*60}")
        logging.info("Comparison: Early → Final")
        logging.info(f"{'='*60}")
        for key in final_results:
            if final_results[key] is not None and early_results[key] is not None:
                logging.info(f"  {key}: {early_results[key]:.4f} → {final_results[key]:.4f}")

    logging.info("\nDiagnostics complete.")


if __name__ == "__main__":
    app.run(main)
