"""RECAP vs zero-shot baseline on unperturbed carrot_on_plate.

Tracks per-step tracking reward + V baseline during adaptation,
then evaluates success rate before and after.

Usage:
    python tests/test_recap_vs_baseline.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from absl import app, logging

from octo.model.octo_model import OctoModel
from recap.envs.perturbations import make_env, postprocess_octo_action, ActionEnsembler
from recap.training.recap_adaptation import (
    recap_adapt, inject_indicator,
    _run_transformer, sample_actions_from_readouts,
)


def evaluate(octo_model, params, task, num_trials, rng, indicator_embed=None):
    """Run num_trials rollouts, return success rate.

    If indicator_embed is provided, it is added to readout tokens before
    the action head (for adapted policy evaluation).
    """
    action_mean = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    successes = 0
    for trial in range(num_trials):
        env = make_env("PutCarrotOnPlateInScene-v1")
        rng, trial_rng = jax.random.split(rng)
        obs, _ = env.reset()
        done = False
        ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)

        while not done:
            rng, act_rng = jax.random.split(rng)
            pad_mask = obs["pad_mask"]
            trans_out = _run_transformer(
                octo_model.module, params, obs, task, pad_mask
            )
            # Inject indicator embedding if adapted
            if indicator_embed is not None:
                trans_out = inject_indicator(trans_out, indicator_embed)
            norm_actions = sample_actions_from_readouts(
                octo_model, params, trans_out, act_rng
            )
            actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions = np.array(actions[0])
            ensembled = ensembler.ensemble_action(raw_actions)
            action_np = postprocess_octo_action(ensembled)
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        successes += int(success)
        logging.info(
            f"  Trial {trial+1}/{num_trials}: "
            f"success={success}  running={successes/(trial+1):.3f}"
        )
        env.close()

    return successes / num_trials


def main(_):
    logging.info("Loading Octo model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params
    rng = jax.random.PRNGKey(42)

    instruction = "put carrot on plate"
    task = octo_model.create_tasks(texts=[instruction])

    # --- Zero-shot baseline ---
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 1: Zero-shot baseline (10 trials)")
    logging.info("=" * 60)
    rng, eval_rng = jax.random.split(rng)
    baseline_sr = evaluate(octo_model, octo_params, task, 10, eval_rng)
    logging.info(f"Zero-shot success rate: {baseline_sr:.3f}")

    # --- RECAP adaptation ---
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 2: RECAP adaptation (unperturbed)")
    logging.info("=" * 60)

    class Config:
        rank = 8
        backbone_lora = False
        lr = 1e-4
        update_freq = 4
        num_bc_steps = 200
        bc_batch_size = 16
        buffer_size = 256
        min_buffer = 16
        num_episodes = 5
        recap_alpha = 1.0
        wm_checkpoint = "hf://4manifold/ttdr-world-model"
        wm_projection_head_kwargs = {"hidden_dim": 1024, "output_dim": 512}
        wm_dynamics_predictor_kwargs = {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}
        wm_value_head_kwargs = {"hidden_dim": 256, "num_layers": 2}

    adapt_env = make_env("PutCarrotOnPlateInScene-v1")
    rng, adapt_rng = jax.random.split(rng)
    adapted_params, indicator_embed = recap_adapt(
        octo_model, octo_params,
        adapt_env, instruction, Config(), adapt_rng,
    )
    adapt_env.close()

    # --- Post-adaptation evaluation ---
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 3: Post-adaptation evaluation (10 trials)")
    logging.info("=" * 60)
    rng, eval_rng = jax.random.split(rng)
    adapted_sr = evaluate(
        octo_model, adapted_params, task, 10, eval_rng,
        indicator_embed=indicator_embed,
    )
    logging.info(f"Adapted success rate: {adapted_sr:.3f}")

    # --- Summary ---
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Zero-shot:  {baseline_sr:.3f}")
    logging.info(f"  RECAP:      {adapted_sr:.3f}")
    logging.info(f"  Delta:      {adapted_sr - baseline_sr:+.3f}")


if __name__ == "__main__":
    app.run(main)
