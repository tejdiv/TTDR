"""Fine-tune FlexTok on Bridge V2 robotics images with rectified flow loss.

Streams Bridge V2 frames via Octo's OXE dataloader, feeds them through
FlexTok's forward pass (VAE encode → encoder → regularizer → noise →
decoder), and trains with the rectified flow matching objective:

    L = || v_θ(x_t, t) - (ε - x_0) ||²

where x_0 is the clean VAE latent, ε is noise, x_t = (1-t)·x_0 + t·ε,
and v_θ is the decoder's velocity prediction.

The VAE is frozen. Only the encoder and decoder are trained.

Usage:
    python -m recap.training.train_flextok \
        --data_dir /path/to/bridge_v2_rlds \
        --num_steps 50000 --batch_size 8 --lr 1e-4
"""

import os
from functools import partial

import numpy as np
import tensorflow as tf
# Prevent TF from grabbing GPU
tf.config.set_visible_devices([], "GPU")

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from absl import app, flags, logging

from flextok.flextok_wrapper import FlexTokFromHub

from octo.data.dataset import make_dataset_from_rlds
from octo.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
from octo.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from octo.utils.spec import ModuleSpec

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/home/ubuntu/data/rlds", "Path to Bridge V2 RLDS data directory.")
flags.DEFINE_boolean("auto_download", True, "Auto-download Bridge V2 if not found.")
flags.DEFINE_integer("batch_size", 8, "Training batch size.")
flags.DEFINE_integer("image_size", 256, "Image resize dimension.")
flags.DEFINE_integer("max_frames_per_traj", 0, "Max frames per trajectory (0 = all).")
flags.DEFINE_float("lr", 1e-4, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_integer("num_steps", 50000, "Total training steps.")
flags.DEFINE_integer("log_every", 50, "Log every N steps.")
flags.DEFINE_integer("save_every", 5000, "Save checkpoint every N steps.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/flextok", "Checkpoint save directory.")
flags.DEFINE_float("grad_clip", 1.0, "Gradient clipping norm.")
flags.DEFINE_string("hf_repo", None, "HuggingFace repo to push final model (e.g. 'username/flextok-bridge-v2').")


def _decode_image(raw_bytes):
    """Decode a single JPEG byte string to numpy (H, W, 3) uint8."""
    return tf.io.decode_image(
        raw_bytes, channels=3, expand_animations=False, dtype=tf.uint8
    ).numpy()


class BridgeV2ImageDataset(IterableDataset):
    """Streams individual frames from Bridge V2 as PyTorch tensors.

    Each item is a (C, H, W) float32 tensor normalized to [0, 1].
    Trajectories are loaded lazily via tf.data — only one is in memory at a time.
    """

    def __init__(self, data_dir, image_size=256, max_frames_per_traj=0, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_frames_per_traj = max_frames_per_traj
        self.shuffle = shuffle

    def _make_dataset(self):
        config = OXE_DATASET_CONFIGS["bridge_dataset"]
        standardize_fn = ModuleSpec.create(
            OXE_STANDARDIZATION_TRANSFORMS["bridge_dataset"]
        )

        dataset, _ = make_dataset_from_rlds(
            name="bridge_dataset",
            data_dir=self.data_dir,
            train=True,
            standardize_fn=standardize_fn,
            image_obs_keys=config["image_obs_keys"],
            language_key="language_instruction",
            shuffle=self.shuffle,
        )

        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )
        return dataset

    def __iter__(self):
        dataset = self._make_dataset()

        for traj in dataset.as_numpy_iterator():
            image_bytes = traj["observation"]["image_primary"]  # (T,) JPEG bytes
            T = len(image_bytes)

            if self.max_frames_per_traj > 0 and T > self.max_frames_per_traj:
                indices = np.random.choice(T, self.max_frames_per_traj, replace=False)
                indices.sort()
            else:
                indices = range(T)

            for i in indices:
                raw = image_bytes[i]
                if not isinstance(raw, (bytes, np.bytes_)) or len(raw) == 0:
                    continue

                img = _decode_image(raw)  # (H, W, 3) uint8

                if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                    img = tf.image.resize(
                        img, (self.image_size, self.image_size),
                        method="lanczos3", antialias=True,
                    ).numpy().astype(np.uint8)

                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                yield img_tensor


def compute_flow_loss(model, data_dict):
    """Compute rectified flow matching loss from FlexTok forward output.

    The noise module creates x_t = (1-t)·clean + t·noise and stores the noise.
    The decoder predicts a velocity v_θ(x_t, t).
    Target velocity = noise - clean (direction from clean to noise).

    Loss = mean || v_θ - (noise - clean) ||²

    Key names are read from the model's modules to avoid hardcoding.
    """
    noise_module = model.flow_matching_noise_module
    clean_key = noise_module.clean_images_read_key
    noise_key = noise_module.noise_write_key

    # The decoder's final head writes the prediction
    dec_head = model.decoder.module_dict["dec_head"]
    pred_key = dec_head.write_key

    noise_list = data_dict[noise_key]
    clean_list = data_dict[clean_key]
    pred_list = data_dict[pred_key]

    total_loss = 0.0
    count = 0
    for noise, clean, pred in zip(noise_list, clean_list, pred_list):
        target = noise - clean
        total_loss = total_loss + F.mse_loss(pred, target)
        count += 1

    return total_loss / count


BRIDGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"


def download_bridge_v2(data_dir):
    """Download Bridge V2 RLDS dataset if not already present."""
    import subprocess

    target = os.path.join(data_dir, "bridge_dataset", "1.0.0")
    marker = os.path.join(target, "dataset_info.json")

    if os.path.exists(marker):
        logging.info(f"Bridge V2 already exists at {target}")
        return

    logging.info(f"Downloading Bridge V2 to {target}...")
    os.makedirs(target, exist_ok=True)
    subprocess.run([
        "wget", "-c", "-r", "-np", "-nH", "--cut-dirs=4",
        "--reject", "index.html*",
        "-P", target,
        BRIDGE_URL,
    ], check=True)
    logging.info("Download complete.")


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if FLAGS.auto_download:
        download_bridge_v2(FLAGS.data_dir)
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    # --- Load FlexTok ---
    logging.info("Loading FlexTok model...")
    model = FlexTokFromHub.from_pretrained("EPFL-VILAB/flextok_d18_d28_dfn")
    model = model.to(device)

    # Freeze VAE — only train encoder + decoder
    for param in model.vae.parameters():
        param.requires_grad = False

    model.train()
    model.vae.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    logging.info(f"Trainable parameters: {total_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        trainable_params, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay,
    )

    # --- Data ---
    dataset = BridgeV2ImageDataset(
        data_dir=FLAGS.data_dir,
        image_size=FLAGS.image_size,
        max_frames_per_traj=FLAGS.max_frames_per_traj,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=0,
        drop_last=True,
    )

    # --- Training loop ---
    logging.info(f"Training for {FLAGS.num_steps} steps, batch_size={FLAGS.batch_size}")
    step = 0
    running_loss = 0.0

    while step < FLAGS.num_steps:
        for images in dataloader:
            if step >= FLAGS.num_steps:
                break

            # images: (B, 3, 256, 256) float32 [0, 1]
            images = images.to(device)

            # FlexTok expects data_dict with "rgb" as list of (1, C, H, W)
            data_dict = {model.vae.images_read_key: images.split(1)}

            # Forward: VAE encode (frozen) → encoder → regularizer → noise → decoder
            data_dict = model(data_dict)

            # Rectified flow loss: || v_θ - (noise - clean) ||²
            loss = compute_flow_loss(data_dict)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, FLAGS.grad_clip)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % FLAGS.log_every == 0:
                avg_loss = running_loss / FLAGS.log_every
                logging.info(f"Step {step}/{FLAGS.num_steps} | loss={avg_loss:.6f}")
                running_loss = 0.0

            if step % FLAGS.save_every == 0:
                if FLAGS.hf_repo:
                    logging.info(f"Pushing checkpoint at step {step} to {FLAGS.hf_repo}...")
                    model.push_to_hub(FLAGS.hf_repo, commit_message=f"checkpoint step {step}")
                    logging.info("Upload complete.")
                else:
                    ckpt_path = os.path.join(FLAGS.checkpoint_dir, f"flextok_step{step}.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": {
                            k: v for k, v in model.state_dict().items()
                            if "vae" not in k
                        },
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    logging.info(f"Saved checkpoint to {ckpt_path}")

    logging.info(f"Training complete at step {step}")

    # Push final model to HuggingFace
    if FLAGS.hf_repo:
        logging.info(f"Pushing model to HuggingFace: {FLAGS.hf_repo}")
        model.push_to_hub(FLAGS.hf_repo)
        logging.info("Upload complete.")


if __name__ == "__main__":
    app.run(main)
