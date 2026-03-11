"""Fine-tune FlexTok on Bridge V2 robotics images with rectified flow loss.

Streams Bridge V2 frames, feeds them through FlexTok's forward pass
(VAE encode → encoder → regularizer → noise → decoder), and trains
with the rectified flow matching objective:

    L = || v_θ(x_t, t) - (ε - x_0) ||²

where x_0 is the clean VAE latent, ε is noise, x_t = (1-t)·x_0 + t·ε,
and v_θ is the decoder's velocity prediction.

The VAE is frozen. Only the encoder and decoder are trained.

Usage:
    python -m recap.training.train_flextok \
        --data_dir /path/to/bridge_v2_rlds \
        --num_steps 50000 --batch_size 8 --lr 1e-4 \
        --hf_repo your-username/flextok-bridge-v2
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from absl import app, flags, logging

from flextok.flextok_wrapper import FlexTokFromHub

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/home/ubuntu/data/rlds/bridge_dataset/1.0.0", "Path to Bridge V2 RLDS data directory.")
flags.DEFINE_boolean("auto_download", True, "Auto-download Bridge V2 if not found.")
flags.DEFINE_integer("batch_size", 8, "Training batch size.")
flags.DEFINE_integer("image_size", 256, "Image resize dimension.")
flags.DEFINE_integer("max_frames_per_traj", 0, "Max frames per trajectory (0 = all).")
flags.DEFINE_float("lr", 1e-4, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_integer("num_steps", 5000, "Total training steps.")
flags.DEFINE_integer("log_every", 50, "Log every N steps.")
flags.DEFINE_integer("save_every", 500, "Save checkpoint every N steps.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/flextok", "Checkpoint save directory.")
flags.DEFINE_float("grad_clip", 1.0, "Gradient clipping norm.")
flags.DEFINE_string("hf_repo", None, "HuggingFace repo to push final model (e.g. 'username/flextok-bridge-v2').")


class BridgeV2ImageDataset(IterableDataset):
    """Streams individual frames from Bridge V2 as PyTorch tensors.

    Each item is a (C, H, W) float32 tensor normalized to [-1, 1].
    Reads RLDS tfrecords directly via tf.data (no tfds dependency).
    """

    def __init__(self, data_dir, image_size=256, max_frames_per_traj=0, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_frames_per_traj = max_frames_per_traj
        self.shuffle = shuffle

    def __iter__(self):
        import glob
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except AttributeError:
            pass

        # Find all tfrecord shards
        pattern = os.path.join(self.data_dir, "bridge_dataset-train.tfrecord-*")
        files = sorted(glob.glob(pattern))
        assert len(files) > 0, f"No tfrecord files found at {pattern}"

        if self.shuffle:
            np.random.shuffle(files)

        # Each tfrecord is a tf.train.Example with flat features.
        # Images are stored as variable-length byte lists in steps/observation/image_0.
        feature_desc = {
            "steps/observation/image_0": tf.io.VarLenFeature(tf.string),
        }

        dataset = tf.data.TFRecordDataset(files)

        for raw_record in dataset:
            parsed = tf.io.parse_single_example(raw_record, feature_desc)
            image_bytes = tf.sparse.to_dense(parsed["steps/observation/image_0"], default_value=b"")
            T = len(image_bytes)

            if self.max_frames_per_traj > 0 and T > self.max_frames_per_traj:
                indices = np.random.choice(T, self.max_frames_per_traj, replace=False)
                indices.sort()
            else:
                indices = range(T)

            for i in indices:
                raw = image_bytes[i].numpy()
                if len(raw) == 0:
                    continue
                img = tf.io.decode_image(raw, channels=3, expand_animations=False).numpy()  # (H, W, 3) uint8

                if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                    img = tf.image.resize(
                        img, (self.image_size, self.image_size),
                        method="lanczos3", antialias=True,
                    ).numpy().astype(np.uint8)

                # Normalize to [-1, 1] (FlexTok convention)
                img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
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

    # Find the decoder's final output head (last module with a write_key)
    pred_key = None
    for name, module in reversed(list(model.decoder.module_dict.items())):
        if hasattr(module, "write_key"):
            pred_key = module.write_key
            break
    assert pred_key is not None, (
        f"Could not find decoder output key. Decoder modules: {list(model.decoder.module_dict.keys())}"
    )

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

    marker = os.path.join(data_dir, "dataset_info.json")

    if os.path.exists(marker):
        logging.info(f"Bridge V2 already exists at {data_dir}")
        return

    logging.info(f"Downloading Bridge V2 to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    subprocess.run([
        "wget", "-c", "-r", "-np", "-nH", "--cut-dirs=6",
        "--reject", "index.html*",
        "-P", data_dir,
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

            images = images.to(device)

            # FlexTok expects data_dict with images as list of (1, C, H, W)
            data_dict = {model.vae.images_read_key: list(images.split(1))}

            # Forward: VAE encode (frozen) → encoder → regularizer → noise → decoder
            data_dict = model(data_dict)

            # Rectified flow loss: || v_θ - (noise - clean) ||²
            loss = compute_flow_loss(model, data_dict)

            # Diagnostics on first step
            if step == 0:
                noise_module = model.flow_matching_noise_module
                clean_key = noise_module.clean_images_read_key
                noise_key = noise_module.noise_write_key
                pred_key = None
                for name, module in reversed(list(model.decoder.module_dict.items())):
                    if hasattr(module, "write_key"):
                        pred_key = module.write_key
                        break
                logging.info(f"[DIAG] clean_key={clean_key}, noise_key={noise_key}, pred_key={pred_key}")
                logging.info(f"[DIAG] data_dict keys: {list(data_dict.keys())}")
                logging.info(f"[DIAG] decoder modules: {list(model.decoder.module_dict.keys())}")
                c = data_dict[clean_key][0]
                n = data_dict[noise_key][0]
                p = data_dict[pred_key][0]
                t = n - c
                logging.info(f"[DIAG] clean: mean={c.mean():.4f} std={c.std():.4f} shape={c.shape}")
                logging.info(f"[DIAG] noise: mean={n.mean():.4f} std={n.std():.4f} shape={n.shape}")
                logging.info(f"[DIAG] pred:  mean={p.mean():.4f} std={p.std():.4f} shape={p.shape}")
                logging.info(f"[DIAG] target (noise-clean): mean={t.mean():.4f} std={t.std():.4f}")
                logging.info(f"[DIAG] input images: min={images.min():.4f} max={images.max():.4f}")
                logging.info(f"[DIAG] initial loss={loss.item():.6f}")

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, FLAGS.grad_clip)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % FLAGS.log_every == 0:
                avg_loss = running_loss / FLAGS.log_every
                logging.info(f"Step {step}/{FLAGS.num_steps} | loss={avg_loss:.6f} | grad_norm={grad_norm:.4f}")
                running_loss = 0.0

            if step % FLAGS.save_every == 0 and FLAGS.hf_repo:
                logging.info(f"Pushing checkpoint at step {step} to {FLAGS.hf_repo}...")
                model.push_to_hub(FLAGS.hf_repo, commit_message=f"checkpoint step {step}")
                logging.info("Upload complete.")

    logging.info(f"Training complete at step {step}")

    # Push final model to HuggingFace
    if FLAGS.hf_repo:
        logging.info(f"Pushing model to HuggingFace: {FLAGS.hf_repo}")
        model.push_to_hub(FLAGS.hf_repo)
        logging.info("Upload complete.")


if __name__ == "__main__":
    app.run(main)
