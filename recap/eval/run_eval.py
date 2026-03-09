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

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't hog all VRAM — SAPIEN needs some for Vulkan

import jax
import jax.numpy as jnp
# Warm up cuDNN before TF import corrupts JAX's cuDNN state.
# On CUDA 12.8 + cuDNN 8.9, if cuDNN is already initialized it survives.
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import json

import numpy as np
import yaml
from absl import app, flags, logging

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel

from recap.envs.perturbations import make_env, postprocess_octo_action, ActionEnsembler
from recap.training.recap_adaptation import recap_adapt, IMPROVEMENT_SUFFIX, sample_actions_from_readouts, _run_transformer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/adapt.yaml", "Path to adaptation config.")
flags.DEFINE_string("output_dir", "results/", "Where to save evaluation results.")
flags.DEFINE_integer("num_trials", 80, "Number of evaluation trials per condition.")
flags.DEFINE_integer("num_seeds", 3, "Number of random seeds.")


# ---------------------------------------------------------------------------
# Task and perturbation definitions
# ---------------------------------------------------------------------------

TASKS = {
    "carrot_on_plate": {
        "env_id": "PutCarrotOnPlateInScene-v1",
        "instruction": "put carrot on plate",
    },
    "stack_cube": {
        "env_id": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
        "instruction": "stack green cube on yellow cube",
    },
}

PERTURBATIONS = {
    "object_friction": {"scale": 0.3},
    "object_mass": {"scale": 2.0},
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(octo_model, policy_params, env, task, num_trials, rng):
    """Run num_trials rollouts and return success rate.

    env should be an OctoEnvWrapper that returns Octo-formatted observations
    ({"image_primary": (B,T,256,256,3), "timestep_pad_mask": (B,T)}).

    Args:
        octo_model: OctoModel instance.
        policy_params: Octo params (possibly with LoRA merged in).
        env: OctoEnvWrapper instance.
        task: Task dict from octo_model.create_tasks().
        num_trials: Number of rollouts.
        rng: JAX PRNG key.

    Returns:
        success_rate: float in [0, 1].
        results: list of per-trial dicts.
    """
    # Action denormalization (Bridge V2 dataset stats)
    action_mean = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    successes = 0
    results = []

    for trial in range(num_trials):
        rng, trial_rng = jax.random.split(rng)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)

        while not done:
            rng, act_rng = jax.random.split(rng)
            # JIT'd transformer + action head (same functions as adaptation loop)
            pad_mask = obs["pad_mask"]
            trans_out = _run_transformer(
                octo_model.module, policy_params, obs, task, pad_mask
            )
            norm_actions = sample_actions_from_readouts(
                octo_model, policy_params, trans_out, act_rng
            )

            # Denormalize: (B, pred_horizon, 7)
            actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions = np.array(actions[0])  # (pred_horizon, 7)

            # Action ensemble (matching SimplerEnv: exec_horizon=1)
            ensembled = ensembler.ensemble_action(raw_actions)

            # Euler→axis-angle + gripper binarization (matches SimplerEnv)
            action_np = postprocess_octo_action(ensembled)

            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        success = info.get("success", total_reward > 0)
        if hasattr(success, "item"):
            success = success.item()
        successes += int(success)
        results.append({
            "trial": trial,
            "success": bool(success),
            "total_reward": float(total_reward),
            "steps": steps,
        })

        if (trial + 1) % 10 == 0:
            logging.info(f"      Trial {trial + 1}/{num_trials}: "
                         f"running success rate = {successes/(trial+1):.3f}")

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
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
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

            # --- Condition 3: TTDR RECAP (standard / unperturbed) ---
            logging.info("    Condition: ttdr_recap_standard")
            adapt_env_std = make_env(task_info["env_id"])  # No perturbation
            rng, adapt_rng = jax.random.split(rng)

            adapt_config = _make_config(config, task_info["instruction"])
            adapted_params_std = recap_adapt(
                octo_model, octo_params,
                adapt_env_std, task_info["instruction"], adapt_config, adapt_rng,
            )

            task_improved = octo_model.create_tasks(
                texts=[task_info["instruction"] + IMPROVEMENT_SUFFIX]
            )
            eval_env_std = make_env(task_info["env_id"])
            success_rate, results = evaluate_policy(
                octo_model, adapted_params_std, eval_env_std, task_improved, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {success_rate:.3f}")
            all_results[f"{task_name}/seed{seed}/ttdr_recap_standard"] = {
                "success_rate": success_rate,
                "results": results,
            }

            # --- Condition 4: TTDR RECAP (perturbed) ---
            logging.info(f"    Condition: ttdr_recap_perturbed ({perturbation} x{perturbation_scale})")
            adapt_env_pert = make_env(task_info["env_id"], perturbation, perturbation_scale)
            rng, adapt_rng = jax.random.split(rng)

            adapted_params_pert = recap_adapt(
                octo_model, octo_params,
                adapt_env_pert, task_info["instruction"], adapt_config, adapt_rng,
            )

            eval_env_pert = make_env(task_info["env_id"], perturbation, perturbation_scale)
            success_rate, results = evaluate_policy(
                octo_model, adapted_params_pert, eval_env_pert, task_improved, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {success_rate:.3f}")
            all_results[f"{task_name}/seed{seed}/ttdr_recap_perturbed"] = {
                "success_rate": success_rate,
                "results": results,
            }

    # Save results
    output_path = os.path.join(FLAGS.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {output_path}")

    # Print summary
    logging.info("\n=== Summary ===")
    for key, val in all_results.items():
        logging.info(f"  {key}: {val['success_rate']:.3f}")


def _make_config(yaml_config, instruction):
    """Convert yaml dict into a namespace-like object for recap_adapt."""
    class Config:
        pass

    c = Config()

    # LoRA
    c.rank = yaml_config["lora"]["rank"]
    c.backbone_lora = yaml_config["lora"].get("backbone_lora", False)

    # Adaptation
    c.lr = yaml_config["adaptation"]["lr"]
    c.update_freq = yaml_config["adaptation"]["update_every_M"]
    c.num_bc_steps = yaml_config["adaptation"]["num_bc_steps"]
    c.bc_batch_size = yaml_config["adaptation"]["bc_batch_size"]
    c.buffer_size = yaml_config["buffer"]["max_size"]
    c.min_buffer = yaml_config["adaptation"].get("min_buffer", 16)
    c.num_episodes = yaml_config["adaptation"].get("num_episodes", 1)

    # RECAP-specific
    c.K = yaml_config["adaptation"].get("K", 3)
    c.recap_alpha = yaml_config["adaptation"].get("recap_alpha", 1.0)

    # World model
    c.wm_checkpoint = yaml_config["world_model"]["checkpoint"]
    c.wm_projection_head_kwargs = yaml_config["world_model"].get(
        "projection_head", {"hidden_dim": 1024, "output_dim": 512}
    )
    c.wm_dynamics_predictor_kwargs = yaml_config["world_model"].get(
        "dynamics_predictor", {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}
    )
    c.wm_value_head_kwargs = yaml_config["world_model"].get("value_head", None)

    return c


if __name__ == "__main__":
    app.run(main)
