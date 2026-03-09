"""Quick diagnostic: does the improvement suffix alone kill performance?

Runs 10 trials each with:
  A) Frozen params + "put carrot on plate" (baseline)
  B) Frozen params + "put carrot on plate World Model Advantage: positive" (suffix only)

If B is 0% but A is ~20%, the suffix is the problem, not the LoRA.
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
from recap.training.recap_adaptation import _run_transformer, sample_actions_from_readouts


def evaluate(octo_model, params, task, num_trials, rng):
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


IMPROVEMENT_SUFFIX = " World Model Advantage: positive"


def main(_):
    logging.info("Loading Octo model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    params = octo_model.params
    rng = jax.random.PRNGKey(42)

    instruction = "put carrot on plate"
    task_plain = octo_model.create_tasks(texts=[instruction])
    task_suffix = octo_model.create_tasks(texts=[instruction + IMPROVEMENT_SUFFIX])

    # --- A: plain task ---
    logging.info("\n" + "=" * 60)
    logging.info("A) Frozen params + plain task (10 trials)")
    logging.info("=" * 60)
    rng, eval_rng = jax.random.split(rng)
    sr_plain = evaluate(octo_model, params, task_plain, 10, eval_rng)
    logging.info(f"Plain success rate: {sr_plain:.3f}")

    # --- B: suffix task ---
    logging.info("\n" + "=" * 60)
    logging.info("B) Frozen params + suffix task (10 trials)")
    logging.info("=" * 60)
    rng, eval_rng = jax.random.split(rng)
    sr_suffix = evaluate(octo_model, params, task_suffix, 10, eval_rng)
    logging.info(f"Suffix success rate: {sr_suffix:.3f}")

    # --- Summary ---
    logging.info("\n" + "=" * 60)
    logging.info(f"  Plain:   {sr_plain:.3f}")
    logging.info(f"  Suffix:  {sr_suffix:.3f}")
    logging.info(f"  Delta:   {sr_suffix - sr_plain:+.3f}")


if __name__ == "__main__":
    app.run(main)
