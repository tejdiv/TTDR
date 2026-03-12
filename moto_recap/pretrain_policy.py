"""Stage 2: Pretrain Octo with (g, I) conditioning + hindsight relabeling.

This is where I becomes meaningful. Without this stage, I is a random frozen
vector at test time that carries no information.

With hindsight relabeling, the same action gets I=1 for anchors whose expected
motion matches what happened, and I=0 for mismatched anchors. The model learns
the structural relationship: "I=1 means this action went toward g."

Loss (classifier-free guidance, preserves RECAP identity):
    L = -log pi(a|o,l) + alpha * -log pi(a|I,o,l,g)

The unconditional term ensures the policy stays close to pi_ref.
The conditioned term teaches the policy to use (g, I) for action selection.
At test time, setting I=1 with a novel anchor extracts the improved policy
pi(a|I=1,o,l,g) via CFG — this IS the RECAP mechanism.

Usage:
    python -m moto_recap.pretrain_policy --config moto_recap/configs/pretrain_policy.yaml
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from absl import app, flags, logging
from flax.training import checkpoints

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model import OctoModel
from moto_recap.tokenizer import load_tokenizer, tokenize_frames
from moto_recap.gpt import load_gpt, encode as gpt_encode, score_tokens
from moto_recap.anchor import AnchorProjection, ValueHead, init_anchor_params
from moto_recap.hindsight import relabel_transition, build_instruction_bank
from recap.models.lora_adapter import init_lora_params, apply_lora

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "moto_recap/configs/pretrain_policy.yaml", "Config path.")


def _make_jit_transformer(module):
    """JIT-wrapped transformer (same pattern as recap_adaptation.py)."""
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


def _make_jit_cfg_grad_step(module, optimizer, anchor_proj, value_head):
    """JIT'd gradient step for CFG loss with anchor + I conditioning.

    Loss: L = -log pi(a|o,l) + alpha * -log pi(a|I,o,l,g)

    Gradients flow through: LoRA layers, indicator embedding, anchor projection.
    The unconditional term uses plain readouts (no g, no I).
    The conditioned term adds anchor_embed + I * indicator_embed to readouts.

    This preserves the RECAP identity: pi_improved = pi(a|I=1,o,l,g) via CFG.
    """
    @jax.jit
    def step(octo_params, lora_params, anchor_params,
             trans_out, batch_indicators, batch_anchors,
             actions_expanded, timestep_pad_mask, action_pad_mask,
             recap_alpha, step_rng, opt_state, trainable):

        def loss_fn(trainable):
            # Merge LoRA
            lp = {**lora_params, "layers": trainable["layers"],
                  "indicator_embed": trainable["indicator_embed"]}
            merged = apply_lora(octo_params, lp)
            bound = module.bind({"params": merged}, rngs={"dropout": step_rng})

            # Unconditional BC loss: -log pi(a|o,l) — plain readouts
            bc_loss, _ = bound.heads["action"].loss(
                trans_out, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            # Conditioned loss: -log pi(a|I,o,l,g)
            # Inject anchor + indicator into readout tokens
            tokens = trans_out["readout_action"].tokens  # (B, T, N, 768)
            embed_I = trainable["indicator_embed"]     # (768,)

            # Project anchor: (B, moto_hidden_dim) → (B, 768)
            anchor_embed = anchor_proj.apply(
                trainable["anchor_params"]["projection"],
                batch_anchors
            )  # (B, 768)

            # Per-sample: add anchor always, add I embed only where I=1
            indicator_mask = batch_indicators[:, None, None, None]  # (B,1,1,1)
            anchor_broad = anchor_embed[:, None, None, :]           # (B,1,1,768)
            shifted = (tokens
                       + anchor_broad
                       + indicator_mask * embed_I[None, None, None, :])
            new_tg = trans_out["readout_action"].replace(tokens=shifted)
            trans_cond = {**trans_out, "readout_action": new_tg}

            cond_loss, _ = bound.heads["action"].loss(
                trans_cond, actions_expanded,
                timestep_pad_mask, action_pad_mask, train=True,
            )

            return bc_loss + recap_alpha * cond_loss

        grads = jax.grad(loss_fn)(trainable)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        return new_trainable, new_opt_state

    return step


def main(_):
    with open(FLAGS.config) as f:
        config = yaml.safe_load(f)

    device = "cuda"
    frame_skip = config["moto"]["frame_skip"]

    # ── Load models ──────────────────────────────────────────────────
    logging.info("Loading Octo (octo-base v1.0)...")
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    octo_params = octo_model.params

    logging.info("Loading Moto tokenizer + GPT...")
    tokenizer = load_tokenizer(config["moto"]["tokenizer_checkpoint"], device)
    gpt = load_gpt(config["moto"]["gpt_checkpoint"], device)

    # ── Initialize trainable modules ─────────────────────────────────
    rng = jax.random.PRNGKey(config["training"].get("seed", 0))
    rng, lora_rng, anchor_rng = jax.random.split(rng, 3)

    lora_params = init_lora_params(
        lora_rng, octo_params,
        rank=config["lora"]["rank"],
        backbone_lora=config["lora"].get("backbone_lora", False),
    )
    anchor_params = init_anchor_params(
        anchor_rng,
        moto_hidden_dim=config["moto"].get("hidden_dim", 768),
    )

    # Trainable dict: LoRA layers + indicator + anchor projection
    trainable = {
        "layers": lora_params["layers"],
        "indicator_embed": lora_params["indicator_embed"],
        "anchor_params": anchor_params,
    }

    optimizer = optax.adam(config["training"]["learning_rate"])
    opt_state = optimizer.init(trainable)

    anchor_proj = AnchorProjection(
        moto_hidden_dim=config["moto"].get("hidden_dim", 768)
    )
    value_head = ValueHead()

    # ── Load Bridge V2 data ──────────────────────────────────────────
    logging.info("Loading Bridge V2 trajectories...")
    from moto_recap.data import load_bridge_trajectories

    # Build instruction bank for hindsight relabeling
    logging.info("Building instruction bank...")
    instructions = []
    for traj_data in load_bridge_trajectories(
        config["data"]["data_dir"], train=True, max_trajectories=5000
    ):
        instructions.append(traj_data["instruction"])
    instruction_bank = build_instruction_bank(instructions)
    logging.info(f"Instruction bank: {len(instruction_bank)} unique instructions")

    # ── JIT setup ────────────────────────────────────────────────────
    cfg_step = _make_jit_cfg_grad_step(
        octo_model.module, optimizer, anchor_proj, value_head
    )

    # ── Training loop ────────────────────────────────────────────────
    num_epochs = config["training"]["epochs"]
    recap_alpha = config["training"].get("recap_alpha", 1.0)
    M_relabel = config["training"].get("hindsight_M", 50)
    batch_size = config["training"]["batch_size"]
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    buffer = []  # running buffer for hindsight relabeling

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        num_batches = 0

        for traj_data in load_bridge_trajectories(
            config["data"]["data_dir"], train=True
        ):
            frames = traj_data["frames"]  # (T, H, W, 3) uint8
            instr = traj_data["instruction"]

            T = len(frames)
            if T <= frame_skip:
                continue

            # Process transitions at interval m
            for t in range(0, T - frame_skip, frame_skip):
                f_t = frames[t:t+1].astype(np.float32) / 255.0       # (1,H,W,3)
                f_m = frames[t+frame_skip:t+frame_skip+1].astype(np.float32) / 255.0

                # Tokenize actual transition
                actual_tokens = tokenize_frames(tokenizer, f_t, f_m, device)

                # Get anchor from Moto-GPT
                g_hidden = gpt_encode(gpt, instr, f_t, device)  # (1, hidden_dim)

                # Score: surprisal reward
                _, reward = score_tokens(gpt, instr, f_t, actual_tokens, device)

                # Format obs for Octo (single frame, matching our wrapper format)
                obs_jax = {
                    "image_primary": jnp.array(
                        (frames[t:t+1] * 1.0).astype(np.uint8)
                    )[None],  # (1, 1, H, W, 3)
                    "pad_mask": jnp.ones((1, 1), dtype=jnp.float32),
                }
                task = octo_model.create_tasks(texts=[instr])

                # Encode obs
                trans_out = _run_transformer(
                    octo_model.module, octo_params, obs_jax, task,
                    obs_jax["pad_mask"],
                )
                tokens_readout = trans_out["readout_action"].tokens  # (1,T,N,768)
                obs_enc = jnp.mean(tokens_readout[:, 0, :, :], axis=1)  # (1, 768)

                # Anchor projection + value
                g_jax = jnp.array(g_hidden)
                anchor_embed = anchor_proj.apply(
                    trainable["anchor_params"]["projection"], g_jax
                )
                v_pred = float(value_head.apply(
                    trainable["anchor_params"]["value_head"], obs_enc, anchor_embed
                ))

                advantage = float(reward[0]) - v_pred
                I_label = advantage > 0.0

                transition = {
                    "frame_t": f_t,
                    "frame_t_plus_m": f_m,
                    "action": actions[t:t+1],
                    "actual_tokens": actual_tokens,
                    "obs_encoding": obs_enc,
                    "instruction": instr,
                    "anchor_hidden": g_hidden,
                    "anchor_embed": np.array(anchor_embed),
                    "reward": float(reward[0]),
                    "advantage": advantage,
                    "I": I_label,
                }
                buffer.append(transition)

                # Hindsight relabel
                anchor_proj_fn = partial(
                    anchor_proj.apply,
                    trainable["anchor_params"]["projection"],
                )
                value_fn = partial(
                    value_head.apply,
                    trainable["anchor_params"]["value_head"],
                )
                relabeled = relabel_transition(
                    transition, gpt, tokenizer, anchor_proj_fn, value_fn,
                    instruction_bank, buffer, M=M_relabel, device=device,
                )

                # Accumulate batch from real + relabeled
                if len(buffer) >= batch_size:
                    # Sample a batch from buffer (real + relabeled mixed)
                    batch_indices = np.random.choice(
                        len(buffer), size=min(batch_size, len(buffer)), replace=False
                    )
                    batch_entries = [buffer[i] for i in batch_indices]

                    # Stack for gradient step
                    batch_obs = jax.tree.map(
                        lambda *xs: jnp.concatenate(xs, axis=0),
                        *[{
                            "image_primary": jnp.array(
                                (e["frame_t"] * 255).astype(np.uint8)
                            )[None],
                            "pad_mask": jnp.ones((1, 1), dtype=jnp.float32),
                        } for e in batch_entries]
                    )
                    batch_actions = jnp.array(
                        np.stack([e["action"] for e in batch_entries])
                    )  # (B, 1, 7)
                    batch_anchors_np = np.concatenate(
                        [e["anchor_hidden"] for e in batch_entries]
                    )
                    batch_anchors = jnp.array(batch_anchors_np)
                    batch_indicators = jnp.array(
                        [e["I"] for e in batch_entries], dtype=jnp.float32
                    )

                    B = len(batch_entries)
                    task_B = jax.tree.map(
                        lambda x: jnp.tile(x, (B,) + (1,) * (x.ndim - 1)), task
                    )

                    pad_mask = batch_obs["pad_mask"]
                    trans_out_batch = _run_transformer(
                        octo_model.module, octo_params, batch_obs, task_B, pad_mask
                    )

                    window_size = pad_mask.shape[1]
                    actions_expanded = jnp.broadcast_to(
                        batch_actions[:, None, :, :] if batch_actions.ndim == 3
                        else batch_actions[:, None, None, :],
                        (B, window_size, batch_actions.shape[-2] if batch_actions.ndim >= 3 else 1,
                         batch_actions.shape[-1]),
                    )
                    action_pad_mask = jnp.ones_like(actions_expanded, dtype=bool)
                    timestep_pad_mask = pad_mask.astype(bool)

                    rng, step_rng = jax.random.split(rng)
                    trainable, opt_state = cfg_step(
                        octo_params, lora_params, anchor_params,
                        trans_out_batch, batch_indicators, batch_anchors,
                        actions_expanded, timestep_pad_mask, action_pad_mask,
                        recap_alpha, step_rng, opt_state, trainable,
                    )

                    # Sync back
                    lora_params["layers"] = trainable["layers"]
                    lora_params["indicator_embed"] = trainable["indicator_embed"]
                    anchor_params = trainable["anchor_params"]

                    global_step += 1
                    if global_step % 100 == 0:
                        logging.info(f"  step {global_step}")

                    # Keep buffer bounded
                    max_buf = config["training"].get("max_buffer", 10000)
                    if len(buffer) > max_buf:
                        buffer = buffer[-max_buf:]

        logging.info(
            f"Epoch {epoch+1}/{num_epochs} done in {time.time()-epoch_start:.0f}s, "
            f"global_step={global_step}"
        )

        # Save checkpoint
        ckpt = {
            "lora_params": lora_params,
            "anchor_params": anchor_params,
            "octo_checkpoint": "hf://rail-berkeley/octo-base",
        }
        ckpt_path = os.path.join(save_dir, f"policy_epoch{epoch+1}")
        checkpoints.save_checkpoint(save_dir, ckpt, step=epoch + 1, keep=3)

    logging.info(f"Pretraining complete. {global_step} total steps.")


if __name__ == "__main__":
    app.run(main)
