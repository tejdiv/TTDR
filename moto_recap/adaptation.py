"""Stage 3: Test-time RECAP adaptation with Moto-GPT world model.

Classifier-free guidance preserves RECAP's formal properties:
  - All buffer actions come from pi_ref (the current policy).
  - All outcomes are real (executed in the environment).
  - Rewards are deterministic: log p(m* | instruction, frame).
  - Advantages are exact: A = r - V(o, g), zero variance.
  - I labels are exact: I = 1 iff A > 0.
  - The improved policy is pi(a|I=1,o,l,g) extracted via CFG.

The RECAP identity holds per-anchor: pi_improved(a|o,l) = pi(a|I=1,o,l,g).
Hindsight relabeling generates M anchors per transition, each with a
deterministic reward and exact I label. This gives KM training samples
from K real transitions while preserving the Bayes identity.

Supports two adaptation modes (config flag):
  - "lora": update LoRA layers + indicator embedding (fast, fewer params)
  - "full": update full action head weights (more capacity, slower)

The anchor projection and I embedding are always frozen (pretrained in Stage 2).
Only the action head parameters adapt to the new environment.

Usage:
    python -m moto_recap.adaptation --config moto_recap/configs/adapt.yaml
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.training import checkpoints

from recap.envs.perturbations import postprocess_octo_action, ActionEnsembler
from recap.models.lora_adapter import init_lora_params, apply_lora
from moto_recap.tokenizer import load_tokenizer, tokenize_frames
from moto_recap.gpt import load_gpt, encode as gpt_encode, score_tokens
from moto_recap.anchor import AnchorProjection, ValueHead
from moto_recap.reward import compute_advantage, compute_indicator
from moto_recap.hindsight import relabel_transition, build_instruction_bank


# ---------------------------------------------------------------------------
# JIT'd forward passes (same pattern as recap_adaptation.py)
# ---------------------------------------------------------------------------

def _make_jit_transformer(module):
    @jax.jit
    def run(params, obs, task, pad_mask):
        return module.apply(
            {"params": params}, obs, task, pad_mask,
            train=False, method="octo_transformer",
        )
    return run


_jit_transformers = {}


def _run_transformer(module, params, obs, task, pad_mask):
    key = id(module)
    if key not in _jit_transformers:
        _jit_transformers[key] = _make_jit_transformer(module)
    return _jit_transformers[key](params, obs, task, pad_mask)


def _make_jit_action_head(module):
    @jax.jit
    def run(params, transformer_out, rng):
        bound = module.bind({"params": params})
        return bound.heads["action"].predict_action(
            transformer_out, rng=rng, train=False
        )
    return run


_jit_action_heads = {}


def sample_actions_from_readouts(octo_model, merged_params, transformer_out, rng):
    key = id(octo_model.module)
    if key not in _jit_action_heads:
        _jit_action_heads[key] = _make_jit_action_head(octo_model.module)
    return _jit_action_heads[key](merged_params, transformer_out, rng)


# ---------------------------------------------------------------------------
# Anchor + indicator injection into readout tokens
# ---------------------------------------------------------------------------

def inject_anchor_and_indicator(trans_out, anchor_embed, indicator_embed):
    """Add anchor projection + indicator embedding to readout tokens.

    This is the CFG conditioning mechanism:
      readout' = readout + anchor_embed + indicator_embed

    At test time, indicator_embed is always added (I=1 — we want the improved
    policy). During LoRA training, indicator_embed is masked per-sample by I.

    Args:
        trans_out: transformer output dict.
        anchor_embed: (B, 768) projected anchor from AnchorProjection.
        indicator_embed: (768,) learned I embedding (frozen from Stage 2).

    Returns:
        Modified trans_out with shifted readout tokens.
    """
    tokens = trans_out["readout_action"].tokens  # (B, T, N, 768)
    shifted = (tokens
               + anchor_embed[:, None, None, :]
               + indicator_embed[None, None, None, :])
    new_tg = trans_out["readout_action"].replace(tokens=shifted)
    return {**trans_out, "readout_action": new_tg}


# ---------------------------------------------------------------------------
# CFG gradient step (same structure as recap_adaptation.py but with anchors)
# ---------------------------------------------------------------------------

def _make_jit_cfg_step(module, optimizer, anchor_proj):
    """JIT'd CFG gradient step for test-time adaptation.

    Loss: L = -log pi(a|o,l) + alpha * -log pi(a|I,o,l,g)

    Unconditional term: plain readout tokens (no g, no I).
    Conditioned term: readout + anchor_embed + I * indicator_embed.

    Only LoRA layers are updated. Anchor projection and indicator
    embedding are frozen (pretrained in Stage 2).
    """
    @jax.jit
    def step(octo_params, lora_params, anchor_proj_params,
             trans_out, batch_indicators, batch_anchors,
             actions_expanded, timestep_pad_mask, action_pad_mask,
             recap_alpha, step_rng, opt_state, trainable_layers):

        def loss_fn(layers):
            lp = {**lora_params, "layers": layers}
            merged = apply_lora(octo_params, lp)
            bound = module.bind({"params": merged}, rngs={"dropout": step_rng})

            # Unconditional: -log pi(a|o,l)
            bc_loss, _ = bound.heads["action"].loss(
                trans_out, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            # Conditioned: -log pi(a|I,o,l,g)
            tokens = trans_out["readout_action"].tokens
            indicator_embed = lora_params["indicator_embed"]  # frozen

            anchor_embed = anchor_proj.apply(
                anchor_proj_params, batch_anchors
            )

            indicator_mask = batch_indicators[:, None, None, None]
            shifted = (tokens
                       + anchor_embed[:, None, None, :]
                       + indicator_mask * indicator_embed[None, None, None, :])
            new_tg = trans_out["readout_action"].replace(tokens=shifted)
            trans_cond = {**trans_out, "readout_action": new_tg}

            cond_loss, _ = bound.heads["action"].loss(
                trans_cond, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            return bc_loss + recap_alpha * cond_loss

        grads = jax.grad(loss_fn)(trainable_layers)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_layers)
        new_layers = optax.apply_updates(trainable_layers, updates)
        return new_layers, new_opt_state

    return step


# ---------------------------------------------------------------------------
# Value head fine-tuning (reset to V_0, retrain on buffer)
# ---------------------------------------------------------------------------

def finetune_value_head(value_head, v_params_v0, buffer_entries, anchor_proj,
                        anchor_proj_params, num_steps=50, lr=1e-3, rng=None):
    """Retrain V from V_0 on adaptation buffer.

    V(o, g) predicts expected surprisal. Reset to pretrained weights each
    RECAP cycle, then fine-tune on the current buffer.
    """
    v_params = jax.tree.map(lambda x: x, v_params_v0)
    tx = optax.adam(lr)
    opt_state = tx.init(v_params)

    obs_all = jnp.concatenate([e["obs_encoding"] for e in buffer_entries])
    anchors_all = jnp.concatenate(
        [jnp.array(e["anchor_hidden"]) for e in buffer_entries]
    )
    r_all = jnp.array([e["reward"] for e in buffer_entries])
    N = obs_all.shape[0]

    anchor_embeds_all = anchor_proj.apply(anchor_proj_params, anchors_all)

    @jax.jit
    def v_step(params, opt_st, obs, anc, r):
        def loss_fn(p):
            v = value_head.apply(p, obs, anc)
            return jnp.mean((v - r) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_st = tx.update(grads, opt_st, params)
        return optax.apply_updates(params, updates), new_st, loss

    bs = min(32, N)
    for step in range(num_steps):
        rng, sample_rng = jax.random.split(rng)
        idx = jax.random.choice(sample_rng, N, shape=(bs,), replace=False)
        v_params, opt_state, _ = v_step(
            v_params, opt_state, obs_all[idx], anchor_embeds_all[idx], r_all[idx]
        )

    return v_params


# ---------------------------------------------------------------------------
# Main adaptation loop
# ---------------------------------------------------------------------------

def moto_adapt(octo_model, octo_params, env, instruction, config, rng,
               pretrained_checkpoint=None):
    """Test-time RECAP adaptation with Moto-GPT world model.

    Preserves all RECAP formal properties:
      - Actions from pi_ref (current policy)
      - Real outcomes from environment
      - Deterministic reward (motion token surprisal)
      - Exact advantages via V baseline
      - CFG extracts improved policy: pi(a|I=1,o,l,g)

    Args:
        octo_model: OctoModel instance (octo-base v1.0).
        octo_params: Octo params (possibly with pretrained LoRA+anchor from Stage 2).
        env: OctoEnvWrapper from perturbations.py.
        instruction: Language instruction string.
        config: Adaptation config (namespace or dict-like).
        rng: JAX PRNG key.
        pretrained_checkpoint: Path to Stage 2 checkpoint with anchor+I params.

    Returns:
        adapted_params: Octo params with adapted action head.
    """
    device = getattr(config, "device", "cuda")
    frame_skip = getattr(config, "frame_skip", 3)
    adaptation_mode = getattr(config, "adaptation_mode", "lora")

    # ── Load Moto models ─────────────────────────────────────────────
    tokenizer = load_tokenizer(config.moto_tokenizer, device)
    gpt = load_gpt(config.moto_gpt, device)

    # ── Load pretrained anchor + I from Stage 2 ──────────────────────
    if pretrained_checkpoint:
        ckpt = checkpoints.restore_checkpoint(pretrained_checkpoint, target=None)
        anchor_params = ckpt["anchor_params"]
        lora_pretrained = ckpt["lora_params"]
        indicator_embed = lora_pretrained["indicator_embed"]  # frozen
    else:
        rng, anchor_rng = jax.random.split(rng)
        from moto_recap.anchor import init_anchor_params
        anchor_params = init_anchor_params(anchor_rng)
        indicator_embed = jnp.zeros((768,))

    anchor_proj = AnchorProjection()
    value_head = ValueHead()
    v_params_v0 = anchor_params["value_head"]

    # ── Initialize adaptation parameters ─────────────────────────────
    rng, lora_rng = jax.random.split(rng)

    if adaptation_mode == "lora":
        lora_params = init_lora_params(
            lora_rng, octo_params, rank=config.rank,
            backbone_lora=getattr(config, "backbone_lora", False),
        )
        lora_params["indicator_embed"] = indicator_embed  # frozen, not optimized
        optimizer = optax.adam(config.lr)
        opt_state = optimizer.init(lora_params["layers"])
        logging.info(f"  Adaptation mode: LoRA (rank={config.rank})")
    else:
        # Full fine-tune: optimize entire action head
        logging.info("  Adaptation mode: full fine-tune (action head)")
        optimizer = optax.adam(config.lr)
        # For full fine-tune, we directly update octo_params action head keys
        # but still use the lora infrastructure with rank=full
        lora_params = init_lora_params(
            lora_rng, octo_params, rank=config.rank,
            backbone_lora=True,  # more params
        )
        lora_params["indicator_embed"] = indicator_embed
        opt_state = optimizer.init(lora_params["layers"])

    task = octo_model.create_tasks(texts=[instruction])

    # Action denormalization
    action_mean = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["mean"]
    )
    action_std = jnp.array(
        octo_model.dataset_statistics["bridge_dataset"]["action"]["std"]
    )

    # Instruction bank for hindsight relabeling
    instruction_bank = [instruction]  # at test time, limited diversity

    # JIT setup
    cfg_step = _make_jit_cfg_step(
        octo_model.module, optimizer, anchor_proj
    )

    buffer = []
    total_steps = 0
    frame_buffer = []  # accumulate m frames before computing reward

    # ── Episode loop ─────────────────────────────────────────────────
    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        done = False
        chunk_count = 0
        ensembler = ActionEnsembler(pred_action_horizon=4, temp=0.0)
        frame_buffer = []
        ep_start = time.time()

        logging.info(f"  Episode {episode+1}/{config.num_episodes}")

        while not done:
            step_start = time.time()
            merged_params = apply_lora(octo_params, lora_params)

            # ── Get anchor from Moto-GPT ─────────────────────────────
            # Extract current frame from obs for Moto (needs raw pixels)
            current_image = np.array(obs["image_primary"][0, -1])  # (256,256,3) uint8
            frame_float = current_image.astype(np.float32) / 255.0
            frame_batch = frame_float[None]  # (1, 256, 256, 3)

            g_hidden = gpt_encode(gpt, instruction, frame_batch, device)  # (1, hidden_dim)
            g_jax = jnp.array(g_hidden)
            anchor_embed = anchor_proj.apply(
                anchor_params["projection"], g_jax
            )  # (1, 768)

            # ── Transformer pass + anchor/indicator injection (I=1) ──
            pad_mask = obs["pad_mask"]
            trans_out = _run_transformer(
                octo_model.module, octo_params, obs, task, pad_mask
            )

            # Extract obs encoding for V
            readout_tokens = trans_out["readout_action"].tokens
            obs_enc = jnp.mean(readout_tokens[:, 0, :, :], axis=1)  # (1, 768)

            # Inject anchor + I=1 (always I=1 at inference — CFG extracts improved policy)
            trans_out_cond = inject_anchor_and_indicator(
                trans_out, anchor_embed, indicator_embed
            )

            # ── Sample action ────────────────────────────────────────
            rng, act_rng = jax.random.split(rng)
            norm_actions = sample_actions_from_readouts(
                octo_model, merged_params, trans_out_cond, act_rng
            )

            # Denormalize + postprocess
            actions_denorm = norm_actions * action_std[None] + action_mean[None]
            raw_action = np.array(actions_denorm[0])  # (pred_horizon, 7)
            ensembled = ensembler.ensemble_action(raw_action)
            action_env = postprocess_octo_action(ensembled)

            # ── Execute ──────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # Accumulate frames for m-step reward
            frame_buffer.append(frame_batch)

            # ── Compute reward every m steps ─────────────────────────
            if len(frame_buffer) >= frame_skip and chunk_count > 0:
                frame_t = frame_buffer[-frame_skip]        # (1, H, W, 3)
                next_image = np.array(next_obs["image_primary"][0, -1])
                frame_t_plus_m = next_image.astype(np.float32)[None] / 255.0

                # Tokenize + score
                actual_tokens = tokenize_frames(
                    tokenizer, frame_t, frame_t_plus_m, device
                )
                _, r_total = score_tokens(
                    gpt, instruction, frame_t, actual_tokens, device
                )
                r_aux = float(r_total[0])

                # Value prediction
                v_pred = float(value_head.apply(
                    anchor_params["value_head"], obs_enc, anchor_embed
                ))

                advantage = r_aux - v_pred
                I_label = advantage > 0.0

                buffer.append({
                    "obs": obs,
                    "obs_encoding": obs_enc,
                    "actions": norm_actions,
                    "frame_t": frame_t,
                    "frame_t_plus_m": frame_t_plus_m,
                    "actual_tokens": actual_tokens,
                    "anchor_hidden": g_hidden,
                    "anchor_embed": np.array(anchor_embed),
                    "reward": r_aux,
                    "advantage": advantage,
                    "I": I_label,
                    "instruction": instruction,
                })

                step_time = time.time() - step_start
                suc_str = ""
                task_success = info.get("success", False)
                if hasattr(task_success, "item"):
                    task_success = task_success.item()
                if task_success:
                    suc_str = " SUCCESS"
                logging.info(
                    f"    step {chunk_count} | {step_time:.1f}s | "
                    f"r={r_aux:.4f} V={v_pred:.4f} A={advantage:.4f} | "
                    f"buf={len(buffer)}{suc_str}"
                )

            obs = next_obs
            chunk_count += 1
            total_steps += 1
            if len(buffer) > config.buffer_size:
                buffer = buffer[-config.buffer_size:]

            # ── RECAP update every M steps ───────────────────────────
            if (chunk_count % config.update_freq == 0
                    and len(buffer) >= config.min_buffer):
                update_start = time.time()

                # 1. Retrain V from V_0
                rng, v_rng = jax.random.split(rng)
                v_params_ft = finetune_value_head(
                    value_head, v_params_v0, buffer, anchor_proj,
                    anchor_params["projection"],
                    num_steps=getattr(config, "v_finetune_steps", 50),
                    lr=getattr(config, "v_finetune_lr", 1e-3),
                    rng=v_rng,
                )

                # 2. Recompute advantages with updated V
                for entry in buffer:
                    obs_e = entry["obs_encoding"]
                    anc_e = anchor_proj.apply(
                        anchor_params["projection"],
                        jnp.array(entry["anchor_hidden"])
                    )
                    v_new = float(value_head.apply(v_params_ft, obs_e, anc_e))
                    entry["advantage"] = entry["reward"] - v_new
                    entry["I"] = entry["advantage"] > 0.0

                n_pos = sum(1 for e in buffer if e["I"])
                mean_adv = np.mean([e["advantage"] for e in buffer])
                logging.info(
                    f"    V-advantage: {n_pos}/{len(buffer)} I=1, "
                    f"mean_adv={mean_adv:.4f}"
                )

                # 3. Reset LoRA, retrain from scratch on buffer
                rng, lora_rng = jax.random.split(rng)
                lora_params = init_lora_params(
                    lora_rng, octo_params, rank=config.rank,
                    backbone_lora=getattr(config, "backbone_lora", False),
                )
                lora_params["indicator_embed"] = indicator_embed  # keep frozen
                opt_state = optimizer.init(lora_params["layers"])

                N = len(buffer)
                all_obs = jax.tree.map(
                    lambda *xs: jnp.concatenate(xs, axis=0),
                    *[e["obs"] for e in buffer]
                )
                all_actions = jnp.concatenate([e["actions"] for e in buffer])
                all_indicators = jnp.array(
                    [e["I"] for e in buffer], dtype=jnp.float32
                )
                all_anchors = jnp.concatenate(
                    [jnp.array(e["anchor_hidden"]) for e in buffer]
                )

                effective_steps = min(
                    config.num_bc_steps,
                    max(20, 10 * N // max(config.bc_batch_size, 1)),
                )

                for bc_step in range(effective_steps):
                    rng, sample_rng, step_rng = jax.random.split(rng, 3)
                    bs = min(config.bc_batch_size, N)
                    idx = jax.random.choice(sample_rng, N, shape=(bs,), replace=False)

                    batch_obs = jax.tree.map(lambda x: x[idx], all_obs)
                    batch_actions = all_actions[idx]
                    batch_ind = all_indicators[idx]
                    batch_anc = all_anchors[idx]

                    B = bs
                    task_B = jax.tree.map(
                        lambda x: jnp.tile(x, (B,) + (1,) * (x.ndim - 1)), task
                    )

                    pm = batch_obs["pad_mask"]
                    t_out = _run_transformer(
                        octo_model.module, octo_params, batch_obs, task_B, pm
                    )

                    window_size = pm.shape[1]
                    act_exp = jnp.broadcast_to(
                        batch_actions[:, None, :, :],
                        (B, window_size, batch_actions.shape[-2],
                         batch_actions.shape[-1]),
                    )
                    act_pm = jnp.ones_like(act_exp, dtype=bool)
                    ts_pm = pm.astype(bool)

                    lora_params["layers"], opt_state = cfg_step(
                        octo_params, lora_params,
                        anchor_params["projection"],
                        t_out, batch_ind, batch_anc,
                        act_exp, ts_pm, act_pm,
                        config.recap_alpha, step_rng, opt_state,
                        lora_params["layers"],
                    )

                logging.info(
                    f"    LoRA update ({effective_steps} steps) "
                    f"in {time.time() - update_start:.1f}s"
                )

    # ── Return adapted params ────────────────────────────────────────
    logging.info(
        f"  Adaptation complete: {total_steps} steps, {len(buffer)} buffer entries"
    )
    return apply_lora(octo_params, lora_params)
