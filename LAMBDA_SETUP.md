# Lambda 8xA100 Setup

## 1. Connect

```
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@129.213.145.219
```

## 2. Sync code (from local Mac)

```
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_lambda" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@129.213.145.219:~/TTDR/
```

## 3. Create venv and install

```
python3 -m venv ~/venv
source ~/venv/bin/activate
cd ~/TTDR
pip install -U jax[cuda12] flax optax tensorflow numpy \
  orbax-checkpoint tensorstore h5py absl-py pyyaml \
  scipy wandb tqdm transformers einops
pip install -e .
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

Test with 3 trajectories first:

```bash
mkdir -p data/bridge_v2_encodings
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 \
  --batch_size 32 \
  --max_trajectories 3
```

Then run the full precompute (all ~60k trajectories):

```bash
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 \
  --batch_size 64
```

This runs Octo's frozen encoder over all Bridge V2
trajectories and caches (z_t, z_{t+m}) to HDF5.
One-time cost.

## 7. Train world model

```
python -m recap.training.train_world_model \
  --config configs/train_wm.yaml
```

Should log:
- `JAX devices: 8`
- `batch_size 1024, 128 per device`
- `World model parameters: ~3.8M`
- Loss decreasing, retrieval accuracy increasing

---

## Quick re-sync and run (from local Mac)

After making local code changes:

```bash
rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519_lambda" \
  /Users/tejasrao/Desktop/TTDR/ \
  ubuntu@129.213.145.219:~/TTDR/
```

Then SSH in and run precompute:

```bash
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@129.213.145.219
```

```bash
source ~/venv/bin/activate
cd ~/TTDR
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 \
  --batch_size 32 \
  --max_trajectories 3
```

## Re-activate venv (if you disconnect)

```bash
source ~/venv/bin/activate
cd ~/TTDR
```
