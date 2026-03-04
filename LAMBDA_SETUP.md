# Lambda 8xA100 Setup

## 1. Connect

```
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@35.87.44.192
```

## 2. Sync code (from local Mac)

```
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_lambda" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@35.87.44.192:~/TTDR/
```

## 3. Create venv and install

```
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

## 4. Verify GPUs

```
python -c "import jax; print(jax.devices())"
```

Should print 8 CudaDevices.

## 5. Download Bridge V2 data

Source: RAIL Berkeley (the up-to-date copy Octo expects).

```
mkdir -p /home/ubuntu/data/rlds/bridge_dataset/1.0.0

wget -r -np -nH --cut-dirs=4 \
  -P /home/ubuntu/data/rlds/bridge_dataset/1.0.0 \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
```

## 6. Precompute encoder outputs

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

## 7. Train world model

```
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
# SCP/rsync code to Lambda
rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519_lambda" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@35.87.44.192:~/TTDR/

# SSH in
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@35.87.44.192
```

```bash
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
