"""Dry-run the full RECAP adaptation loop with a mock environment.

No SimplerEnv/ManiSkill needed. Uses Octo's example batch as fake observations
and a mock env that returns them. Exercises the full code path:
  - batched_cfg_forward (transformer with batched CFG)
  - sample_actions_from_readouts (action head only, K=3)
  - batched_encode_next (K next-obs in one pass)
  - tracking_reward + grpo_advantage
  - _update_lora (BC loss with classifier-free guidance)

Runs 1 episode, 2 steps, verifies no crashes and LoRA params change.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import jax
import jax.numpy as jnp
# Warm up cuDNN before TF corrupts it
_x = jnp.ones((1, 8, 8, 3))
_k = jnp.ones((3, 3, 3, 16))
_ = jax.lax.conv_general_dilated(
    _x, _k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
)
del _x, _k

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel
from recap.models.lora_adapter import init_lora_params, apply_lora
from recap.losses.tracking_reward import tracking_reward, grpo_advantage
from recap.training.train_world_model import WorldModel
from recap.training.recap_adaptation import (
    IMPROVEMENT_SUFFIX,
    batched_cfg_forward,
    sample_actions_from_readouts,
    batched_encode_next,
    _update_lora,
    load_world_model,
)


class MockEnv:
    """Fake env that returns Octo example observations. No physics."""

    def __init__(self, example_obs, max_steps=3):
        self.example_obs = example_obs
        self.max_steps = max_steps
        self.step_count = 0
        self._state = 0

    def reset(self, **kwargs):
        self.step_count = 0
        self._state = 0
        return self.example_obs, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.example_obs, 0.0, done, False, {"success": False}

    class unwrapped:
        """Mock unwrapped for get_state/set_state."""
        pass

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state


# Patch save/restore to work with mock env
import recap.envs.perturbations as perturb_module
_orig_save = perturb_module.save_env_state
_orig_restore = perturb_module.restore_env_state
perturb_module.save_env_state = lambda env: env.get_state()
perturb_module.restore_env_state = lambda env, s: env.set_state(s)


def make_mock_world_model():
    """Create a randomly initialized world model (no checkpoint needed)."""
    kwargs_proj = {"hidden_dim": 1024, "output_dim": 512}
    kwargs_dyn = {"hidden_dim": 2048, "num_layers": 4, "output_dim": 512}

    wm_model = WorldModel(
        projection_head_kwargs=kwargs_proj,
        dynamics_predictor_kwargs=kwargs_dyn,
    )

    rng = jax.random.PRNGKey(0)
    dummy_z = jnp.zeros((1, 768))
    params = wm_model.init(rng, dummy_z, dummy_z, dummy_z, train=False)["params"]

    return wm_model, params


def main():
    print("Loading Octo...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
    octo_params = octo_model.params

    example_obs = jax.tree.map(lambda x: x[:1], octo_model.example_batch["observation"])
    task = octo_model.create_tasks(texts=["pick up the object"])
    task_improved = octo_model.create_tasks(
        texts=["pick up the object" + IMPROVEMENT_SUFFIX]
    )

    rng = jax.random.PRNGKey(42)
    K = 3

    # ── Test batched_cfg_forward ──────────────────────────────────
    print("\n1. batched_cfg_forward...")
    z_t, z_t1, trans_out_improved = batched_cfg_forward(
        octo_model, octo_params, example_obs, task, task_improved
    )
    print(f"   z_t: {z_t.shape}, z_t1: {z_t1.shape}")
    print(f"   readout_action tokens: {trans_out_improved['readout_action'].tokens.shape}")
    assert z_t.shape == (1, 768)
    print("   OK")

    # ── Test sample_actions_from_readouts ─────────────────────────
    print("\n2. sample_actions_from_readouts (K=3)...")
    rng, lora_rng = jax.random.split(rng)
    lora_params = init_lora_params(lora_rng, octo_params, rank=8)
    merged = apply_lora(octo_params, lora_params)

    all_actions = []
    for k in range(K):
        rng, act_rng = jax.random.split(rng)
        a = sample_actions_from_readouts(
            octo_model, merged, trans_out_improved, act_rng
        )
        all_actions.append(a)
        print(f"   k={k}: actions shape={a.shape}, first={a[0, 0, :3]}")

    # Verify K actions are different (different RNG seeds)
    diff_01 = float(jnp.max(jnp.abs(all_actions[0] - all_actions[1])))
    assert diff_01 > 1e-4, f"K samples identical! diff={diff_01}"
    print(f"   Diff between k=0 and k=1: {diff_01:.4f}")
    print("   OK")

    # ── Test batched_encode_next ──────────────────────────────────
    print("\n3. batched_encode_next...")
    next_obs_list = [example_obs] * K  # mock: same obs for all K
    z_list = batched_encode_next(octo_model, octo_params, next_obs_list, task)
    print(f"   {K} readouts, each shape={z_list[0].shape}")
    assert len(z_list) == K
    assert z_list[0].shape == (1, 768)
    print("   OK")

    # ── Test tracking_reward + grpo_advantage ─────────────────────
    print("\n4. tracking_reward + grpo_advantage...")
    wm_model, wm_params = make_mock_world_model()

    rewards = []
    for k in range(K):
        r = tracking_reward(z_t, z_t1, z_list[k], wm_params, wm_model.apply)
        rewards.append(r)
        print(f"   k={k}: reward={float(r):.6f}")

    rewards_arr = jnp.array(rewards)
    advantages, indicators = grpo_advantage(rewards_arr)
    print(f"   advantages: {advantages}")
    print(f"   indicators: {indicators}")
    print(f"   mean advantage: {float(jnp.mean(advantages)):.6f} (should be ~0)")
    print("   OK")

    # ── Test _update_lora ─────────────────────────────────────────
    print("\n5. _update_lora (2 gradient steps)...")

    # Build a small buffer
    buffer = []
    for k in range(K):
        buffer.append({
            "obs": example_obs,
            "actions": all_actions[k],
            "r_aux": float(rewards[k]),
            "indicator": bool(indicators[k]),
            "z_t": z_t,
            "z_t1": z_t1,
        })

    import optax
    rng, lora_rng2 = jax.random.split(rng)
    lora_params_fresh = init_lora_params(lora_rng2, octo_params, rank=8)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(lora_params_fresh["layers"])

    # Snapshot before
    before = jax.tree.map(lambda x: x.copy(), lora_params_fresh["layers"])

    class MiniConfig:
        num_bc_steps = 2
        bc_batch_size = 2
        recap_alpha = 1.0

    lora_params_trained, _ = _update_lora(
        lora_params_fresh, opt_state, optimizer,
        octo_model, octo_params, buffer,
        task, task_improved, MiniConfig(), rng,
    )

    # Check params changed
    changed = False
    for path in before:
        d = float(jnp.max(jnp.abs(
            before[path]["A"] - lora_params_trained["layers"][path]["A"]
        )))
        if d > 1e-8:
            changed = True
            break
    assert changed, "FAIL: LoRA params didn't change after training!"
    print(f"   LoRA params changed after 2 gradient steps: max delta={d:.6f}")
    print("   OK")

    # ── Test full mock rollout ────────────────────────────────────
    print("\n6. Full mock rollout (1 episode, 2 steps)...")
    env = MockEnv(example_obs, max_steps=2)
    obs, _ = env.reset()
    done = False
    step = 0

    merged = apply_lora(octo_params, lora_params_trained)

    while not done:
        z_t, z_t1, trans_out_imp = batched_cfg_forward(
            octo_model, octo_params, obs, task, task_improved
        )

        sim_state = env.get_state()
        step_actions = []
        step_next_obs = []
        step_done = []

        for k in range(K):
            env.set_state(sim_state)
            rng, act_rng = jax.random.split(rng)
            a = sample_actions_from_readouts(
                octo_model, merged, trans_out_imp, act_rng
            )
            next_obs, _, term, trunc, info = env.step(a)
            step_actions.append(a)
            step_next_obs.append(next_obs)
            step_done.append(term or trunc)

        z_list = batched_encode_next(octo_model, octo_params, step_next_obs, task)

        rs = []
        for k in range(K):
            r = tracking_reward(z_t, z_t1, z_list[k], wm_params, wm_model.apply)
            rs.append(r)

        best_k = int(jnp.argmax(jnp.array(rs)))
        env.set_state(sim_state)
        env.step(step_actions[best_k])
        obs = step_next_obs[best_k]
        done = step_done[best_k]

        step += 1
        print(f"   Step {step}: best_k={best_k}, reward={float(rs[best_k]):.6f}")

    print("   OK")

    print("\n" + "=" * 60)
    print("All dry-run tests passed.")
    print("The full adaptation loop runs end-to-end.")
    print("Next: test with real SimplerEnv on a GPU instance.")


if __name__ == "__main__":
    main()
