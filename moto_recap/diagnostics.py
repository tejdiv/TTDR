"""Moto motion token diagnostics.

Validates that the motion tokenizer faithfully captures gripper state,
arm displacement, object movement, and rotation. Run this BEFORE trusting
rewards — if the tokenizer can't distinguish "gripper closed + moved right"
from "gripper open + stayed still", surprisal rewards are meaningless.

Three diagnostic modes:
  1. tokenize_and_decode: round-trip check (tokenize frame pair → decode → compare)
  2. motion_attribution: which of the 8 tokens change when you perturb specific motions
  3. trajectory_tracking: full rollout with per-step token analysis + video

Usage:
    python -m moto_recap.diagnostics --mode all --config moto_recap/configs/adapt.yaml
"""

import os
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "moto_recap/configs/adapt.yaml", "Config path.")
flags.DEFINE_string("mode", "all", "Diagnostic mode: roundtrip, attribution, tracking, all")
flags.DEFINE_string("output_dir", "diagnostics/moto/", "Where to save outputs.")
flags.DEFINE_integer("num_episodes", 3, "Episodes for trajectory tracking.")
flags.DEFINE_integer("max_steps", 100, "Max steps per episode.")


# ---------------------------------------------------------------------------
# 1. Round-trip reconstruction quality
# ---------------------------------------------------------------------------

def run_roundtrip_diagnostic(tokenizer, frame_pairs, output_dir, device="cuda"):
    """Tokenize frame pairs, decode, measure reconstruction error.

    Checks: can the tokenizer faithfully encode and decode visual changes?
    If PSNR is low or SSIM is bad, the tokenizer loses information and
    surprisal rewards will be noisy.

    Args:
        tokenizer: Frozen Moto tokenizer.
        frame_pairs: list of (frame_t, frame_t_plus_m) numpy arrays, each (H,W,3) float [0,1].
        output_dir: where to save comparison images.
        device: torch device.
    """
    from moto_recap.tokenizer import tokenize_frames, decode_tokens

    os.makedirs(os.path.join(output_dir, "roundtrip"), exist_ok=True)
    results = []

    for i, (f_t, f_m) in enumerate(frame_pairs):
        f_t_batch = f_t[None]   # (1,H,W,3)
        f_m_batch = f_m[None]

        tokens = tokenize_frames(tokenizer, f_t_batch, f_m_batch, device)  # (1,8)
        decoded = decode_tokens(tokenizer, tokens, f_t_batch, device)      # (1,H,W,3)

        # Reconstruction metrics
        mse = float(np.mean((decoded[0] - f_m) ** 2))
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        # Per-region error (top=arm, bottom=table, center=gripper)
        H, W = f_m.shape[:2]
        arm_region = np.mean((decoded[0, :H//3] - f_m[:H//3]) ** 2)
        gripper_region = np.mean((decoded[0, H//3:2*H//3, W//4:3*W//4] -
                                   f_m[H//3:2*H//3, W//4:3*W//4]) ** 2)
        table_region = np.mean((decoded[0, 2*H//3:] - f_m[2*H//3:]) ** 2)

        results.append({
            "pair": i, "mse": mse, "psnr": psnr,
            "arm_mse": float(arm_region),
            "gripper_mse": float(gripper_region),
            "table_mse": float(table_region),
            "tokens": tokens[0].tolist(),
        })

        # Save comparison image
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(f_t); axes[0].set_title("Frame t")
        axes[1].imshow(f_m); axes[1].set_title("Frame t+m (actual)")
        axes[2].imshow(np.clip(decoded[0], 0, 1)); axes[2].set_title("Decoded")
        diff = np.abs(decoded[0] - f_m)
        axes[3].imshow(diff / (diff.max() + 1e-8)); axes[3].set_title("Error map")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Pair {i} | PSNR={psnr:.1f}dB | tokens={tokens[0].tolist()}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "roundtrip", f"pair_{i:03d}.png"), dpi=100)
        plt.close(fig)

        logging.info(f"  Pair {i}: PSNR={psnr:.1f}dB | arm={arm_region:.5f} "
                     f"gripper={gripper_region:.5f} table={table_region:.5f}")

    # Summary
    with open(os.path.join(output_dir, "roundtrip", "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    avg_psnr = np.mean([r["psnr"] for r in results])
    logging.info(f"  Roundtrip average PSNR: {avg_psnr:.1f}dB")
    if avg_psnr < 20:
        logging.warning("  LOW PSNR — tokenizer may lose important visual detail")
    return results


# ---------------------------------------------------------------------------
# 2. Motion attribution — which tokens track which DOFs
# ---------------------------------------------------------------------------

def run_attribution_diagnostic(tokenizer, gpt, base_frame, instruction,
                                output_dir, device="cuda"):
    """Perturb individual motion DOFs and see which tokens change.

    Creates synthetic frame pairs where only one thing changes (e.g. gripper
    opens, arm moves right) and checks which of the 8 motion tokens respond.

    This tells you whether token 3 = gripper, token 0-2 = arm position, etc.
    If all tokens change for every perturbation, the representation is
    entangled and harder to interpret (but may still work for rewards).

    Args:
        tokenizer: Frozen Moto tokenizer.
        gpt: Frozen Moto-GPT.
        base_frame: (H, W, 3) float [0,1] — a neutral starting frame.
        instruction: str.
        output_dir: where to save results.
        device: torch device.
    """
    from moto_recap.tokenizer import tokenize_frames
    from moto_recap.gpt import score_tokens

    os.makedirs(os.path.join(output_dir, "attribution"), exist_ok=True)

    # Get baseline tokens (no motion — same frame twice)
    base_batch = base_frame[None]
    baseline_tokens = tokenize_frames(tokenizer, base_batch, base_batch, device)
    logging.info(f"  Baseline (no motion) tokens: {baseline_tokens[0].tolist()}")

    # Synthetic perturbations: shift image regions to simulate motion
    perturbations = {
        "arm_right": lambda f: np.roll(f, 10, axis=1),       # shift right
        "arm_left": lambda f: np.roll(f, -10, axis=1),       # shift left
        "arm_up": lambda f: np.roll(f, -10, axis=0),         # shift up
        "arm_down": lambda f: np.roll(f, 10, axis=0),        # shift down
        "gripper_darken": lambda f: np.clip(f * 0.5, 0, 1),  # proxy for gripper close
        "gripper_brighten": lambda f: np.clip(f * 1.5, 0, 1),# proxy for gripper open
        "no_change": lambda f: f.copy(),                      # control
    }

    results = {}
    for name, perturb_fn in perturbations.items():
        perturbed = perturb_fn(base_frame.copy())
        perturbed_batch = perturbed[None]

        tokens = tokenize_frames(tokenizer, base_batch, perturbed_batch, device)
        token_diff = (tokens[0] != baseline_tokens[0]).astype(int)

        # Also get surprisal
        _, total_lp = score_tokens(gpt, instruction, base_batch, tokens, device)

        results[name] = {
            "tokens": tokens[0].tolist(),
            "baseline_tokens": baseline_tokens[0].tolist(),
            "changed_positions": token_diff.tolist(),
            "num_changed": int(token_diff.sum()),
            "surprisal": float(total_lp[0]),
        }
        logging.info(f"  {name:20s}: changed={token_diff.sum()}/8 "
                     f"positions={np.where(token_diff)[0].tolist()} "
                     f"surprisal={total_lp[0]:.2f}")

    # Save
    with open(os.path.join(output_dir, "attribution", "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: heatmap of which tokens respond to which perturbations
    names = list(results.keys())
    matrix = np.array([results[n]["changed_positions"] for n in names])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"T{i}" for i in range(8)])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Motion Token Position")
    ax.set_ylabel("Perturbation")
    ax.set_title("Motion Token Attribution: which tokens change per DOF")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "attribution", "heatmap.png"), dpi=150)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# 3. Full trajectory tracking — per-step token + reward analysis
# ---------------------------------------------------------------------------

def run_tracking_diagnostic(tokenizer, gpt, env, instruction, output_dir,
                             frame_skip=3, num_episodes=3, max_steps=100,
                             device="cuda"):
    """Run rollouts and log per-step motion tokens, surprisal, and decoded frames.

    This is the key diagnostic: does the surprisal reward track task progress?
    You should see:
      - High surprisal (low reward) when the arm does something unexpected
      - Low surprisal (high reward) when executing the expected manipulation
      - Gripper tokens changing at grasp/release events
      - Token stability during static phases

    Outputs:
      - Per-step CSV with tokens, surprisal, gripper state, arm position
      - Surprisal curve plot
      - Token evolution plot (8 tokens over time)
      - Decoded frame video (what the tokenizer thinks happened)

    Args:
        tokenizer: Frozen Moto tokenizer.
        gpt: Frozen Moto-GPT.
        env: OctoEnvWrapper from perturbations.py.
        instruction: str.
        output_dir: where to save.
        frame_skip: m.
        num_episodes: number of rollouts.
        max_steps: max steps per episode.
        device: torch device.
    """
    from moto_recap.tokenizer import tokenize_frames, decode_tokens
    from moto_recap.gpt import score_tokens

    os.makedirs(os.path.join(output_dir, "tracking"), exist_ok=True)

    all_episode_data = []

    for ep in range(num_episodes):
        logging.info(f"  Episode {ep+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        step = 0
        frame_history = []
        episode_data = []

        while not done and step < max_steps:
            # Extract current frame
            current_image = np.array(obs["image_primary"][0, -1])  # (256,256,3) uint8
            frame_float = current_image.astype(np.float32) / 255.0
            frame_history.append(frame_float)

            # Every m steps, compute tokens + reward
            if len(frame_history) > frame_skip:
                f_t = frame_history[-frame_skip - 1][None]  # (1,H,W,3)
                f_m = frame_float[None]                      # (1,H,W,3)

                tokens = tokenize_frames(tokenizer, f_t, f_m, device)
                per_token_lp, total_lp = score_tokens(
                    gpt, instruction, f_t, tokens, device
                )

                # Decode for visualization
                decoded = decode_tokens(tokenizer, tokens, f_t, device)

                step_data = {
                    "step": step,
                    "tokens": tokens[0].tolist(),
                    "per_token_log_prob": per_token_lp[0].tolist(),
                    "total_log_prob": float(total_lp[0]),
                    "surprisal": float(-total_lp[0]),
                }
                episode_data.append(step_data)

                if step % 10 == 0:
                    logging.info(
                        f"    step {step}: tokens={tokens[0].tolist()} "
                        f"surprisal={-total_lp[0]:.2f}"
                    )

                    # Save frame comparison every 10 steps
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(f_t[0])
                    axes[0].set_title(f"Frame t (step {step-frame_skip})")
                    axes[1].imshow(f_m[0])
                    axes[1].set_title(f"Frame t+m (step {step})")
                    axes[2].imshow(np.clip(decoded[0], 0, 1))
                    axes[2].set_title("Decoded from tokens")
                    for ax in axes:
                        ax.axis("off")
                    fig.suptitle(
                        f"Ep{ep} Step{step} | tokens={tokens[0].tolist()} | "
                        f"surprisal={-total_lp[0]:.2f}"
                    )
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(output_dir, "tracking",
                                     f"ep{ep}_step{step:03d}.png"),
                        dpi=100,
                    )
                    plt.close(fig)

            # Random action (we're just diagnosing the tokenizer, not the policy)
            action = np.zeros(7)
            action[:3] = np.random.randn(3) * 0.01  # small random arm motion
            action[6] = 1.0  # gripper open
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        all_episode_data.append(episode_data)

        # ── Per-episode plots ────────────────────────────────────────
        if len(episode_data) > 1:
            steps_arr = [d["step"] for d in episode_data]
            surprisals = [d["surprisal"] for d in episode_data]
            token_matrix = np.array([d["tokens"] for d in episode_data])
            per_token_lps = np.array([d["per_token_log_prob"] for d in episode_data])

            # Surprisal over time
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps_arr, surprisals, "b-o", markersize=3)
            ax.set_xlabel("Step")
            ax.set_ylabel("Surprisal (negative log prob)")
            ax.set_title(f"Episode {ep}: Surprisal over time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, "tracking", f"ep{ep}_surprisal.png"),
                dpi=150,
            )
            plt.close(fig)

            # Token evolution (8 tokens over time)
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            for t_idx in range(8):
                axes[0].plot(steps_arr, token_matrix[:, t_idx],
                            label=f"T{t_idx}", alpha=0.7)
            axes[0].set_ylabel("Token ID")
            axes[0].set_title(f"Episode {ep}: Motion token IDs over time")
            axes[0].legend(loc="upper right", ncol=4, fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Per-token log probs over time
            for t_idx in range(8):
                axes[1].plot(steps_arr, per_token_lps[:, t_idx],
                            label=f"T{t_idx}", alpha=0.7)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Log probability")
            axes[1].set_title(f"Episode {ep}: Per-token log probs")
            axes[1].legend(loc="upper right", ncol=4, fontsize=8)
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, "tracking", f"ep{ep}_tokens.png"),
                dpi=150,
            )
            plt.close(fig)

    # ── Save all data ────────────────────────────────────────────────
    with open(os.path.join(output_dir, "tracking", "all_episodes.json"), "w") as f:
        json.dump(all_episode_data, f, indent=2)

    logging.info(f"  Tracking diagnostic saved to {output_dir}/tracking/")
    return all_episode_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    import yaml
    with open(FLAGS.config) as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cuda")
    frame_skip = config["moto"]["frame_skip"]
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    from moto_recap.tokenizer import load_tokenizer
    from moto_recap.gpt import load_gpt

    logging.info("Loading Moto tokenizer + GPT...")
    tokenizer = load_tokenizer(config["moto"]["tokenizer_checkpoint"], device)
    gpt = load_gpt(config["moto"]["gpt_checkpoint"], device)

    modes = FLAGS.mode.split(",") if FLAGS.mode != "all" else [
        "roundtrip", "attribution", "tracking"
    ]

    if "roundtrip" in modes or "attribution" in modes:
        # Need some frames — grab from a quick env rollout
        logging.info("Collecting sample frames from env...")
        from recap.envs.perturbations import make_env
        env = make_env("PutCarrotOnPlateInScene-v1")
        obs, _ = env.reset()
        sample_frames = []
        for _ in range(20):
            current = np.array(obs["image_primary"][0, -1]).astype(np.float32) / 255.0
            sample_frames.append(current)
            action = np.zeros(7)
            action[:3] = np.random.randn(3) * 0.01
            action[6] = 1.0
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    if "roundtrip" in modes:
        logging.info("=== Round-trip reconstruction diagnostic ===")
        pairs = [(sample_frames[i], sample_frames[i + frame_skip])
                 for i in range(0, len(sample_frames) - frame_skip, frame_skip)]
        run_roundtrip_diagnostic(tokenizer, pairs, FLAGS.output_dir, device)

    if "attribution" in modes:
        logging.info("=== Motion attribution diagnostic ===")
        run_attribution_diagnostic(
            tokenizer, gpt, sample_frames[0], "put carrot on plate",
            FLAGS.output_dir, device,
        )

    if "tracking" in modes:
        logging.info("=== Trajectory tracking diagnostic ===")
        from recap.envs.perturbations import make_env
        env = make_env("PutCarrotOnPlateInScene-v1")
        run_tracking_diagnostic(
            tokenizer, gpt, env, "put carrot on plate", FLAGS.output_dir,
            frame_skip=frame_skip, num_episodes=FLAGS.num_episodes,
            max_steps=FLAGS.max_steps, device=device,
        )

    logging.info("=== All diagnostics complete ===")


if __name__ == "__main__":
    app.run(main)
