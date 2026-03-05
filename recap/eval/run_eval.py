"""RECAP evaluation script.

Runs 5 conditions across multiple tasks, seeds, and perturbations:
  1. Octo zero-shot (standard dynamics)
  2. Octo zero-shot (perturbed dynamics)
  3. TTDR RECAP (adaptation with tracking reward)
  4. Task-progress baseline (adaptation with hand-designed reward)
  5. Oracle reward (adaptation with true env reward)

Usage:
    python -m recap.eval.run_eval --config configs/adapt.yaml
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from absl import app, flags, logging

from octo.model.octo_model import OctoModel

from recap.training.recap_adaptation import recap_adapt

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/adapt.yaml", "Path to adaptation config.")
flags.DEFINE_string("output_dir", "results/", "Where to save evaluation results.")
flags.DEFINE_integer("num_trials", 80, "Number of evaluation trials per condition.")
flags.DEFINE_integer("num_seeds", 3, "Number of random seeds.")


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

TASKS = {
    "carrot_on_plate": {
        "env_id": "PutCarrotOnPlateInScene-v1",
        "instruction": "put the carrot on the plate",
    },
    "stack_cube": {
        "env_id": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "instruction": "stack the green cube on the yellow cube",
    },
}

PERTURBATIONS = {
    "object_friction": {"scale": 0.3},
    "object_mass": {"scale": 2.0},
}


def make_env(task_id, perturbation=None, perturbation_scale=None):
    """Create a SimplerEnv environment with optional perturbation.

    Args:
        task_id: SimplerEnv environment ID (e.g. "PutCarrotOnPlateInScene-v1").
        perturbation: Type of perturbation (e.g. "object_friction").
        perturbation_scale: Scale factor for the perturbation.

    Returns:
        env: Environment with .reset() and .step() interface.
    """
    # TODO: import and create SimplerEnv, apply perturbation via SAPIEN API
    # Example:
    #   import simpler_env
    #   env = simpler_env.make(task_id)
    #   if perturbation:
    #       apply_perturbation(env, perturbation, perturbation_scale)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(octo_model, policy_params, env, task, num_trials, rng):
    """Run num_trials rollouts and return success rate.

    Args:
        octo_model: Octo model architecture.
        policy_params: Octo params (possibly with LoRA merged in).
        env: Environment instance.
        task: Task dict from octo_model.create_tasks().
        num_trials: Number of rollouts.
        rng: JAX PRNG key.

    Returns:
        success_rate: float in [0, 1].
        results: list of per-trial dicts.
    """
    successes = 0
    results = []

    for trial in range(num_trials):
        rng, trial_rng = jax.random.split(rng)
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            rng, act_rng = jax.random.split(rng)
            actions = octo_model.sample_actions(policy_params, obs, task, rng=act_rng)
            obs, reward, done, info = env.step(actions)
            total_reward += reward
            steps += 1

        success = info.get("success", total_reward > 0)
        successes += int(success)
        results.append({
            "trial": trial,
            "success": bool(success),
            "total_reward": float(total_reward),
            "steps": steps,
        })

    return successes / num_trials, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Load Octo
    logging.info("Loading Octo model...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    octo_params = octo_model.params

    all_results = {}

    for task_name, task_info in TASKS.items():
        logging.info(f"=== Task: {task_name} ===")
        task = octo_model.create_tasks(texts=[task_info["instruction"]])

        for seed in range(FLAGS.num_seeds):
            rng = jax.random.PRNGKey(seed)
            logging.info(f"  Seed: {seed}")

            # --- Condition 1: Octo zero-shot (standard) ---
            logging.info("    Condition: octo_zero_shot_standard")
            env = make_env(task_info["env_id"])
            success_rate, results = evaluate_policy(
                octo_model, octo_params, env, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {success_rate:.3f}")
            all_results[f"{task_name}/seed{seed}/octo_standard"] = {
                "success_rate": success_rate,
                "results": results,
            }

            # --- Condition 2: Octo zero-shot (perturbed) ---
            perturbation = config["env"]["perturbation"]
            perturbation_scale = config["env"]["perturbation_scale"]
            logging.info(f"    Condition: octo_zero_shot_perturbed ({perturbation} x{perturbation_scale})")
            perturbed_env = make_env(task_info["env_id"], perturbation, perturbation_scale)
            success_rate, results = evaluate_policy(
                octo_model, octo_params, perturbed_env, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {success_rate:.3f}")
            all_results[f"{task_name}/seed{seed}/octo_perturbed"] = {
                "success_rate": success_rate,
                "results": results,
            }

            # --- Condition 3: TTDR RECAP ---
            logging.info("    Condition: ttdr_recap")
            adapt_env = make_env(task_info["env_id"], perturbation, perturbation_scale)
            rng, adapt_rng = jax.random.split(rng)

            # Create a simple config namespace from yaml
            adapt_config = _make_config(config)
            adapted_params = recap_adapt(
                octo_model, octo_params,
                adapt_env, task, adapt_config, adapt_rng,
            )

            # Evaluate the adapted policy
            eval_env = make_env(task_info["env_id"], perturbation, perturbation_scale)
            success_rate, results = evaluate_policy(
                octo_model, adapted_params, eval_env, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {success_rate:.3f}")
            all_results[f"{task_name}/seed{seed}/ttdr_recap"] = {
                "success_rate": success_rate,
                "results": results,
            }

            # --- Condition 4: Task-progress baseline ---
            # TODO: same adaptation loop but with hand-designed task progress reward
            logging.info("    Condition: task_progress_baseline (TODO)")

            # --- Condition 5: Oracle reward ---
            # TODO: same adaptation loop but with true environment reward
            logging.info("    Condition: oracle_reward (TODO)")

    # Save results
    output_path = os.path.join(FLAGS.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {output_path}")

    # Print summary
    logging.info("\n=== Summary ===")
    for key, val in all_results.items():
        logging.info(f"  {key}: {val['success_rate']:.3f}")


def _make_config(yaml_config):
    """Convert yaml dict into a namespace-like object for recap_adapt."""
    class Config:
        pass
    c = Config()
    c.rank = yaml_config["lora"]["rank"]
    c.alpha = yaml_config["lora"]["alpha"]
    c.lr = yaml_config["adaptation"]["lr"]
    c.update_freq = yaml_config["adaptation"]["update_every_M"]
    c.num_bc_steps = yaml_config["adaptation"]["num_bc_steps"]
    c.bc_batch_size = yaml_config["adaptation"]["bc_batch_size"]
    c.advantage_eps = yaml_config["adaptation"]["advantage_eps"]
    c.buffer_size = yaml_config["buffer"]["max_size"]
    c.num_episodes = yaml_config["adaptation"].get("num_episodes", 1)
    c.wm_checkpoint = yaml_config["world_model"]["checkpoint"]
    return c


if __name__ == "__main__":
    app.run(main)
