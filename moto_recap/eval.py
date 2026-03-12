"""Evaluation script for Moto-RECAP.

Runs 4 conditions across tasks and seeds:
  1. Octo zero-shot (standard dynamics)
  2. Octo zero-shot (perturbed dynamics)
  3. Moto-RECAP adapted (standard dynamics)
  4. Moto-RECAP adapted (perturbed dynamics)

Uses recap.envs.perturbations for environment wrapping (unchanged).

Usage:
    python -m moto_recap.eval --config moto_recap/configs/adapt.yaml
"""

import os
import json

import jax
import jax.numpy as jnp

# Warm up cuDNN before TF import
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
import yaml
from absl import app, flags, logging

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel
from recap.envs.perturbations import make_env, postprocess_octo_action, ActionEnsembler
from moto_recap.adaptation import (
    moto_adapt, _run_transformer, sample_actions_from_readouts,
    inject_anchor_and_indicator,
)
from moto_recap.gpt import load_gpt, encode as gpt_encode
from moto_recap.tokenizer import load_tokenizer
from moto_recap.anchor import AnchorProjection

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "moto_recap/configs/adapt.yaml", "Config path.")
flags.DEFINE_string("output_dir", "results/moto_recap/", "Output directory.")
flags.DEFINE_integer("num_trials", 80, "Trials per condition.")
flags.DEFINE_integer("num_seeds", 3, "Random seeds.")

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


def evaluate_policy(octo_model, policy_params, env, task, num_trials, rng,
                    anchor_proj=None, anchor_proj_params=None,
                    indicator_embed=None, gpt=None, instruction=None, device="cuda"):
    """Run rollouts. If anchor/indicator provided, uses CFG conditioning."""

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
            pad_mask = obs["pad_mask"]

            trans_out = _run_transformer(
                octo_model.module, policy_params, obs, task, pad_mask
            )

            # If anchor-conditioned, inject g + I=1
            if anchor_proj is not None and gpt is not None:
                current_image = np.array(obs["image_primary"][0, -1])
                frame_float = current_image.astype(np.float32) / 255.0
                g_hidden = gpt_encode(gpt, instruction, frame_float[None], device)
                g_jax = jnp.array(g_hidden)
                anchor_embed = anchor_proj.apply(anchor_proj_params, g_jax)
                trans_out = inject_anchor_and_indicator(
                    trans_out, anchor_embed, indicator_embed
                )

            norm_actions = sample_actions_from_readouts(
                octo_model, policy_params, trans_out, act_rng
            )

            actions = norm_actions * action_std[None] + action_mean[None]
            raw_actions = np.array(actions[0])
            ensembled = ensembler.ensemble_action(raw_actions)
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
            "trial": trial, "success": bool(success),
            "total_reward": float(total_reward), "steps": steps,
        })

        if (trial + 1) % 10 == 0:
            logging.info(f"      Trial {trial+1}/{num_trials}: "
                         f"running={successes/(trial+1):.3f}")

    return successes / num_trials, results


def main(_):
    with open(FLAGS.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    device = config.get("device", "cuda")

    # Load Octo (v1.0 — pad_mask, NOT timestep_pad_mask)
    logging.info("Loading Octo (octo-base v1.0)...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params

    all_results = {}

    for task_name, task_info in TASKS.items():
        logging.info(f"=== Task: {task_name} ===")
        task = octo_model.create_tasks(texts=[task_info["instruction"]])

        for seed in range(FLAGS.num_seeds):
            rng = jax.random.PRNGKey(seed)
            logging.info(f"  Seed: {seed}")

            # Condition 1: Zero-shot standard
            logging.info("    Condition: zero_shot_standard")
            env = make_env(task_info["env_id"])
            sr, res = evaluate_policy(
                octo_model, octo_params, env, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {sr:.3f}")
            all_results[f"{task_name}/seed{seed}/zero_shot_standard"] = {
                "success_rate": sr, "results": res,
            }

            # Condition 2: Zero-shot perturbed
            pert = config["env"]["perturbation"]
            pert_scale = config["env"]["perturbation_scale"]
            logging.info(f"    Condition: zero_shot_perturbed ({pert} x{pert_scale})")
            env_p = make_env(task_info["env_id"], pert, pert_scale)
            sr, res = evaluate_policy(
                octo_model, octo_params, env_p, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {sr:.3f}")
            all_results[f"{task_name}/seed{seed}/zero_shot_perturbed"] = {
                "success_rate": sr, "results": res,
            }

            # Condition 3: Moto-RECAP standard
            logging.info("    Condition: moto_recap_standard")
            adapt_env = make_env(task_info["env_id"])
            rng, adapt_rng = jax.random.split(rng)
            adapt_cfg = _make_config(config)
            adapted_params = moto_adapt(
                octo_model, octo_params, adapt_env,
                task_info["instruction"], adapt_cfg, adapt_rng,
                pretrained_checkpoint=config.get("pretrained_checkpoint"),
            )
            eval_env = make_env(task_info["env_id"])
            sr, res = evaluate_policy(
                octo_model, adapted_params, eval_env, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {sr:.3f}")
            all_results[f"{task_name}/seed{seed}/moto_recap_standard"] = {
                "success_rate": sr, "results": res,
            }

            # Condition 4: Moto-RECAP perturbed
            logging.info(f"    Condition: moto_recap_perturbed ({pert} x{pert_scale})")
            adapt_env_p = make_env(task_info["env_id"], pert, pert_scale)
            rng, adapt_rng = jax.random.split(rng)
            adapted_params_p = moto_adapt(
                octo_model, octo_params, adapt_env_p,
                task_info["instruction"], adapt_cfg, adapt_rng,
                pretrained_checkpoint=config.get("pretrained_checkpoint"),
            )
            eval_env_p = make_env(task_info["env_id"], pert, pert_scale)
            sr, res = evaluate_policy(
                octo_model, adapted_params_p, eval_env_p, task, FLAGS.num_trials, rng
            )
            logging.info(f"    Success rate: {sr:.3f}")
            all_results[f"{task_name}/seed{seed}/moto_recap_perturbed"] = {
                "success_rate": sr, "results": res,
            }

    # Save
    out_path = os.path.join(FLAGS.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {out_path}")

    logging.info("\n=== Summary ===")
    for key, val in all_results.items():
        logging.info(f"  {key}: {val['success_rate']:.3f}")


def _make_config(yaml_config):
    """Convert yaml dict to namespace for moto_adapt()."""
    class C:
        pass
    c = C()
    c.rank = yaml_config["lora"]["rank"]
    c.backbone_lora = yaml_config["lora"].get("backbone_lora", False)
    c.lr = yaml_config["adaptation"]["lr"]
    c.update_freq = yaml_config["adaptation"]["update_every_M"]
    c.num_bc_steps = yaml_config["adaptation"]["num_bc_steps"]
    c.bc_batch_size = yaml_config["adaptation"]["bc_batch_size"]
    c.buffer_size = yaml_config["buffer"]["max_size"]
    c.min_buffer = yaml_config["adaptation"].get("min_buffer", 16)
    c.num_episodes = yaml_config["adaptation"].get("num_episodes", 1)
    c.recap_alpha = yaml_config["adaptation"].get("recap_alpha", 1.0)
    c.frame_skip = yaml_config["moto"]["frame_skip"]
    c.adaptation_mode = yaml_config["adaptation"].get("mode", "lora")
    c.moto_tokenizer = yaml_config["moto"]["tokenizer_checkpoint"]
    c.moto_gpt = yaml_config["moto"]["gpt_checkpoint"]
    c.device = yaml_config.get("device", "cuda")
    return c


if __name__ == "__main__":
    app.run(main)
