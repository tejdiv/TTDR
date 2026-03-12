"""Fine-tune a VQGAN (taming transformers) on Bridge V2 robotics images.

Streams Bridge V2 frames and trains a VQGAN with the standard three-part
loss from taming transformers:

    L_gen  = L1 + perceptual (LPIPS) + adaptive_weight * GAN_loss + codebook_loss
    L_disc = hinge_loss(real, fake)

Two optimizers alternate: one for the autoencoder (encoder + decoder +
quantizer), one for the PatchGAN discriminator.

Usage:
    python -m recap.training.train_taming \
        --data_dir /path/to/bridge_v2_rlds \
        --num_steps 50000 --batch_size 8
"""

import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import IterableDataset, DataLoader
from absl import app, flags, logging

# Add taming_transformers to path so its internal imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "taming_transformers"))

from taming.models.vqgan import VQModel

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/home/ubuntu/data/rlds/bridge_dataset/1.0.0/",
                    "Path to Bridge V2 RLDS data directory.")
flags.DEFINE_boolean("auto_download", True, "Auto-download Bridge V2 if not found.")
flags.DEFINE_integer("batch_size", 8, "Training batch size.")
flags.DEFINE_integer("image_size", 256, "Image resize dimension.")
flags.DEFINE_integer("max_frames_per_traj", 0, "Max frames per trajectory (0 = all).")
flags.DEFINE_float("lr", 4.5e-6, "Learning rate.")
flags.DEFINE_integer("num_steps", 50000, "Total training steps.")
flags.DEFINE_integer("log_every", 50, "Log every N steps.")
flags.DEFINE_integer("save_every", 5000, "Save checkpoint every N steps.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/taming_finetuned", "Checkpoint save directory.")
flags.DEFINE_float("grad_clip", 1.0, "Gradient clipping norm.")
flags.DEFINE_string("config_path", "checkpoints/taming/vqgan_imagenet_f16_16384.yaml",
                    "Path to VQGAN config YAML.")
flags.DEFINE_string("ckpt_path", "checkpoints/taming/vqgan_imagenet_f16_16384.ckpt",
                    "Path to pretrained VQGAN checkpoint.")


class BridgeV2ImageDataset(IterableDataset):
    """Streams individual frames from Bridge V2 as PyTorch tensors.

    Each item is a (C, H, W) float32 tensor normalized to [-1, 1].
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

        pattern = os.path.join(self.data_dir, "bridge_dataset-train.tfrecord-*")
        files = sorted(glob.glob(pattern))
        assert len(files) > 0, f"No tfrecord files found at {pattern}"

        if self.shuffle:
            np.random.shuffle(files)

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
                img = tf.io.decode_image(raw, channels=3, expand_animations=False).numpy()

                if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                    img = tf.image.resize(
                        img, (self.image_size, self.image_size),
                        method="lanczos3", antialias=True,
                    ).numpy().astype(np.uint8)

                img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
                yield img_tensor


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

    # --- Load config from YAML ---
    with open(FLAGS.config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    ddconfig = model_config["params"]["ddconfig"]
    lossconfig = model_config["params"]["lossconfig"]
    n_embed = model_config["params"]["n_embed"]
    embed_dim = model_config["params"]["embed_dim"]

    # Use learning rate from config if not overridden
    lr = FLAGS.lr or model_config.get("base_learning_rate", 4.5e-6)

    # --- Build VQGAN from config + load pretrained weights ---
    model = VQModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=FLAGS.ckpt_path,
    )
    model = model.to(device)
    model.train()
    # LPIPS must stay in eval mode
    model.loss.perceptual_loss.eval()

    # Set learning_rate attribute (used by VQModel.configure_optimizers)
    model.learning_rate = lr

    # Use VQModel's own optimizer setup: Adam(betas=(0.5, 0.9))
    optimizers, _ = model.configure_optimizers()
    opt_ae, opt_disc = optimizers

    ae_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
                list(model.quantize.parameters()) + list(model.quant_conv.parameters()) + \
                list(model.post_quant_conv.parameters())
    disc_params = list(model.loss.discriminator.parameters())

    total_ae = sum(p.numel() for p in ae_params)
    total_disc = sum(p.numel() for p in disc_params)
    logging.info(f"Autoencoder parameters: {total_ae:,}")
    logging.info(f"Discriminator parameters: {total_disc:,}")

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
    running_ae_loss = 0.0
    running_disc_loss = 0.0

    while step < FLAGS.num_steps:
        for images in dataloader:
            if step >= FLAGS.num_steps:
                break

            images = images.to(device)

            # Forward pass through VQModel
            xrec, qloss = model(images)

            # ---- Step 1: Update autoencoder ----
            ae_loss, log_ae = model.loss(
                qloss, images, xrec, optimizer_idx=0,
                global_step=step, last_layer=model.get_last_layer(), split="train",
            )

            opt_ae.zero_grad()
            ae_loss.backward()
            ae_grad_norm = torch.nn.utils.clip_grad_norm_(ae_params, FLAGS.grad_clip)
            opt_ae.step()

            # ---- Step 2: Update discriminator ----
            # Re-forward to get fresh reconstructions (detached inside loss)
            xrec, qloss = model(images)

            disc_loss, log_disc = model.loss(
                qloss, images, xrec, optimizer_idx=1,
                global_step=step, last_layer=model.get_last_layer(), split="train",
            )

            opt_disc.zero_grad()
            disc_loss.backward()
            disc_grad_norm = torch.nn.utils.clip_grad_norm_(disc_params, FLAGS.grad_clip)
            opt_disc.step()

            step += 1
            running_ae_loss += ae_loss.item()
            running_disc_loss += disc_loss.item()

            if step % FLAGS.log_every == 0:
                avg_ae = running_ae_loss / FLAGS.log_every
                avg_disc = running_disc_loss / FLAGS.log_every
                rec = log_ae["train/rec_loss"].item()
                quant = log_ae["train/quant_loss"].item()
                g = log_ae["train/g_loss"].item()
                d_w = log_ae["train/d_weight"].item()
                logging.info(
                    f"Step {step}/{FLAGS.num_steps} | "
                    f"ae={avg_ae:.4f} disc={avg_disc:.4f} | "
                    f"rec={rec:.4f} quant={quant:.4f} g={g:.4f} d_w={d_w:.4f} | "
                    f"ae_gn={ae_grad_norm:.4f} disc_gn={disc_grad_norm:.4f}"
                )
                running_ae_loss = 0.0
                running_disc_loss = 0.0

            if step % FLAGS.save_every == 0:
                ckpt_path = os.path.join(FLAGS.checkpoint_dir, f"vqgan_step{step}.pt")
                torch.save({"step": step, "state_dict": model.state_dict()}, ckpt_path)
                logging.info(f"Saved checkpoint to {ckpt_path}")

    logging.info(f"Training complete at step {step}")

    # Save final checkpoint
    final_path = os.path.join(FLAGS.checkpoint_dir, "vqgan_final.pt")
    torch.save({"step": step, "state_dict": model.state_dict()}, final_path)
    logging.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    app.run(main)
