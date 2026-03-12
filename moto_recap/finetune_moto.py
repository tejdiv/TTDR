"""Stage 1: Fine-tune Moto-GPT on Bridge V2 with random context truncation.

Expects pre-tokenized data from pretokenize.py. If not found, runs pretokenize first.

Usage:
    python -m moto_recap.finetune_moto --config moto_recap/configs/finetune_moto.yaml
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml
from absl import app, flags, logging

from moto_recap.gpt import load_gpt, _tokenize_instruction, _preprocess_frames

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "moto_recap/configs/finetune_moto.yaml", "Config path.")


class PreTokenizedDataset(Dataset):
    """Loads pre-tokenized motion tokens from .npz file."""

    def __init__(self, npz_path):
        logging.info(f"Loading pre-tokenized data from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        self.tokens = data["tokens"]          # (total_pairs, 8)
        self.pair_counts = data["pair_counts"]  # (num_traj,)
        self.instructions = data["instructions"]  # (num_traj,) str
        self.first_frames = data["first_frames"]  # (num_traj, 224, 224, 3) uint8

        # Build index: for each trajectory, (start_idx, num_pairs)
        self.traj_offsets = np.zeros(len(self.pair_counts), dtype=np.int64)
        self.traj_offsets[1:] = np.cumsum(self.pair_counts[:-1])

        logging.info(f"  {len(self.pair_counts)} trajectories, "
                     f"{self.tokens.shape[0]} total pairs")

    def __len__(self):
        return len(self.pair_counts)

    def __getitem__(self, idx):
        offset = self.traj_offsets[idx]
        n = self.pair_counts[idx]
        traj_tokens = self.tokens[offset:offset+n]  # (n, 8)

        # Moto expects sequence_length=2
        if n >= 2:
            start = np.random.randint(0, n - 1)
            tokens = traj_tokens[start:start + 2]
            mask = np.array([0, 1] if np.random.random() < 0.5 else [1, 1],
                            dtype=np.int64)
        else:
            tokens = np.stack([traj_tokens[0], np.zeros(8, dtype=np.int64)])
            mask = np.array([1, 0], dtype=np.int64)

        first_frame = self.first_frames[idx].astype(np.float32) / 255.0

        return {
            "instruction": str(self.instructions[idx]),
            "motion_tokens": tokens,
            "latent_mask": mask,
            "first_frame": first_frame,
        }


def collate_fn(batch):
    return {
        "motion_tokens": torch.from_numpy(
            np.stack([b["motion_tokens"] for b in batch])).long(),
        "latent_mask": torch.from_numpy(
            np.stack([b["latent_mask"] for b in batch])),
        "instructions": [b["instruction"] for b in batch],
        "first_frames": np.stack([b["first_frame"] for b in batch]),
    }


def main(_):
    with open(FLAGS.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for pre-tokenized data
    npz_path = config["data"].get("pretokenized",
        os.path.join(config["data"]["data_dir"], "bridge_motion_tokens.npz"))

    if not os.path.exists(npz_path):
        logging.info(f"Pre-tokenized data not found at {npz_path}")
        logging.info("Run: python -m moto_recap.pretokenize first")
        return

    # Load GPT for fine-tuning
    logging.info("Loading Moto-GPT...")
    gpt = load_gpt(config["moto"]["gpt_checkpoint"], device)
    for p in gpt.parameters():
        p.requires_grad_(True)

    dataset = PreTokenizedDataset(npz_path)
    dataloader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=4,
    )

    optimizer = torch.optim.AdamW(
        gpt.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    num_epochs = config["training"]["epochs"]
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            tokens = batch["motion_tokens"].to(device)
            latent_mask = batch["latent_mask"].to(device)
            lang_ids, lang_attn = _tokenize_instruction(batch["instructions"], device)
            rgb = _preprocess_frames(batch["first_frames"], device)

            outputs = gpt.forward(
                rgb=rgb, language=lang_ids,
                attention_mask=latent_mask.clone(),
                latent_motion_ids=tokens, latent_mask=latent_mask,
                train=True, lang_attention_mask=lang_attn,
            )

            logits = outputs["latent_motion_preds"]
            B, T, n_tok = tokens.shape
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1), reduction="none"
            ).reshape(B, T, n_tok)
            loss = (ce * latent_mask.unsqueeze(-1).float()).sum() / \
                   (latent_mask.sum() * n_tok + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                logging.info(f"  Epoch {epoch+1}, batch {num_batches}: "
                             f"loss={total_loss/num_batches:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        logging.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        torch.save(gpt.state_dict(),
                    os.path.join(save_dir, f"moto_gpt_epoch{epoch+1}.pt"))

    final_path = os.path.join(save_dir, "moto_gpt_finetuned.pt")
    torch.save(gpt.state_dict(), final_path)
    logging.info(f"Saved fine-tuned GPT to {final_path}")


if __name__ == "__main__":
    app.run(main)
