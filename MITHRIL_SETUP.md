# Mithril 8xA100 Setup

## 1. Reserve an instance

Go to [mithril.ai](https://mithril.ai/) → Instances → Create.

- **Instance type:** 8x A100 80GB SXM (204 CPU cores, 1520GB RAM, 14TB ephemeral)
- **Region:** any US region with availability
- **Duration:** minimum 3 hours, 1-hour increments, up to 2 weeks
- **SSH keys:** upload your public key during setup
- **Startup script (optional):** can add setup commands that run on allocation

Reservations are non-cancellable. Instance becomes accessible ~15-20 min after start time.
Spot bids are also available if you want cheaper preemptible access.

## 2. Connect

Once the instance shows "Allocated" status on the Instances page:

```bash
chmod 600 ~/.ssh/id_ed25519_mithril
ssh -i ~/.ssh/id_ed25519_mithril ubuntu@35.87.44.192
```

Replace `35.87.44.192` with the IP shown on the instance detail page.

## 3. Sync code (from local Mac)

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_mithril" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@35.87.44.192:~/TTDR/
```

## 4. Create venv and install

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
cd ~/TTDR
pip install "jax[cuda12]==0.4.20" flax==0.7.5 optax==0.1.5 \
  tensorflow==2.15.0 tensorflow_probability==0.23.0 \
  tensorflow_datasets==4.9.2 \
  chex==0.1.85 distrax==0.1.5 ml_dtypes==0.2.0 \
  orbax-checkpoint==0.5.3 tensorstore==0.1.45 h5py absl-py pyyaml \
  numpy==1.24.3 scipy wandb tqdm transformers==4.36.2 einops \
  huggingface_hub
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"
pip install -e .
python -c "from huggingface_hub import login; login()"
```

## 5. Verify GPUs

```bash
python -c "import jax; print(jax.devices())"
```

Should print 8 CudaDevices.

## 6. Download Bridge V2 data

```bash
mkdir -p /home/ubuntu/data/rlds/bridge_dataset/1.0.0

wget -r -np -nH --cut-dirs=4 \
  -P /home/ubuntu/data/rlds/bridge_dataset/1.0.0 \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
```

## 7. Precompute encoder outputs

Test with 3 trajectories first (single GPU):

```bash
mkdir -p data/bridge_v2_encodings
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 \
  --batch_size 64 \
  --window_size 2 \
  --max_trajectories 3
```

Then run the full precompute across all 8 GPUs (~4-5hrs), merge, and upload to HF:

```bash
bash scripts/launch_precompute.sh --hf tejasrao/ttdr-bridge-encodings
```

This runs Octo's frozen encoder (window_size=2) over all Bridge V2
trajectories in parallel across 8 GPUs, merges shards into a single
`encodings.h5` with (z_t, z_t1, z_target, traj_id), and uploads to HF Hub.

To download the encodings on another machine:

```bash
huggingface-cli download tejasrao/ttdr-bridge-encodings encodings.h5 \
  --local-dir data/bridge_v2_encodings --repo-type dataset
```

## 8. Train world model

```bash
python -m recap.training.train_world_model \
  --config configs/train_wm.yaml
```

Should log:
- `JAX devices: 8`
- `batch_size 1024, 128 per device`
- `World model parameters: ~15M`
- Loss decreasing, retrieval accuracy increasing

---

## Quick re-sync and run (from local Mac)

After making local code changes:

```bash
rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519_mithril" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@35.87.44.192:~/TTDR/
```

Then SSH in and run:

```bash
ssh -i ~/.ssh/id_ed25519_mithril ubuntu@35.87.44.192
source ~/venv/bin/activate
cd ~/TTDR

# Quick test (single GPU, 3 trajectories):
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 --batch_size 64 --window_size 2 \
  --max_trajectories 3

# Full run (8 GPUs, ~4-5hrs, uploads to HF):
bash scripts/launch_precompute.sh --hf tejasrao/ttdr-bridge-encodings

# Train (~2-5 min):
python -m recap.training.train_world_model --config configs/train_wm.yaml
```

## Re-activate venv (if you disconnect)

```bash
source ~/venv/bin/activate
cd ~/TTDR
```
