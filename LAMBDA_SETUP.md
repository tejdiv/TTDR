# Lambda Setup

Two tiers:
- **1×A10 ($0.75/hr)**: RECAP tests, bind roundtrip, dry-run adaptation, SimplerEnv
- **8×A100 ($10/hr)**: precompute encodings, train world model

## 0. Quick RECAP test (1×A10, ~10 min)

From your Mac:

```bash
LAMBDA_IP=<your-ip>
SSH_KEY=~/.ssh/id_ed25519_lambda

rsync -avz -e "ssh -i $SSH_KEY" /Users/tejasrao/Desktop/TTDR/ ubuntu@$LAMBDA_IP:~/TTDR/
ssh -i $SSH_KEY ubuntu@$LAMBDA_IP
```

On the instance:

```bash
cd ~/TTDR
export HF_TOKEN=hf_...
bash tests/run_on_lambda.sh
```

This runs 3 tests in sequence:
1. **bind/unbind roundtrip** — verifies LoRA works through split transformer+action head
2. **WM checkpoint load** — downloads from `hf://4manifold/ttdr-world-model`, checks format
3. **dry-run adaptation** — full RECAP loop with mock env (no SimplerEnv), 1 episode, 2 steps

If test 1 fails, we fall back to `octo_model.replace()` (correct but slower).
If test 2 fails, the HF checkpoint format doesn't match what `flax.training.checkpoints` expects.

---

## 0.5. SimplerEnv + full RECAP adaptation (1×A10)

After `run_on_lambda.sh` passes, install SimplerEnv and run full adaptation:

```bash
# Install SimplerEnv (ManiSkill3 branch with visual matching)
pip install mani_skill gymnasium

# CRITICAL: pin numpy back — mani_skill pulls in numpy 2.x which breaks jaxlib
pip install numpy==1.24.3

# Verify numpy is 1.x
python -c "import numpy; print(numpy.__version__)"  # should be 1.24.3
```

```bash
# Run full RECAP adaptation
python -m recap.eval.run_eval --config configs/adapt.yaml
```

**Known issues:**
- `mani_skill`/`gymnasium` install pulls in numpy 2.x → must re-pin `numpy==1.24.3` after
- System cuDNN is required (the script installs it via apt). Pip cuDNN doesn't work for convolutions on CUDA 12.8
- All transformer calls must be JIT-wrapped (non-JIT cuDNN algorithm selection is broken on CUDA 12.8 + cuDNN 8.9)

---

## 1. Connect (8×A100 for training)

```bash
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@129.213.26.7
```

## 2. Sync code (from local Mac)

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_lambda" /Users/tejasrao/Desktop/TTDR/ ubuntu@129.213.26.7:~/TTDR/
```

## 3. Create venv and install

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
cd ~/TTDR
pip install --upgrade pip
```

```bash
# System cuDNN (required — pip cuDNN doesn't work for convolutions)
sudo apt-get update -qq && sudo apt-get install -y -qq libcudnn8 libcudnn8-dev
```

```bash
pip install numpy==1.24.3
pip install jax==0.4.20 jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```bash
pip install ml_dtypes==0.2.0
```

```bash
pip install tensorflow==2.15.0 tensorflow_probability==0.23.0 tensorflow_datasets==4.9.2
pip install --no-deps ml_dtypes==0.2.0
```

```bash
pip install flax==0.7.5 optax==0.1.5 chex==0.1.85 distrax==0.1.5
pip install --no-deps orbax-checkpoint==0.5.3 tensorstore==0.1.45
```

```bash
pip install h5py absl-py pyyaml scipy==1.11.4 wandb tqdm transformers==4.36.2 einops huggingface_hub
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"
pip install -e .
```

```bash
# Pin numpy back (some deps pull in numpy 2.x)
pip install numpy==1.24.3
```

```bash
python -c "from huggingface_hub import login; login()"
```

## 4. Verify GPUs

```bash
python -c "import jax; print(jax.devices())"
```

Should print 8 CudaDevices.

## 5. Download Bridge V2 data

Source: RAIL Berkeley (the up-to-date copy Octo expects).

```bash
mkdir -p /home/ubuntu/data/rlds/bridge_dataset/1.0.0

wget -r -np -nH --cut-dirs=4 -P /home/ubuntu/data/rlds/bridge_dataset/1.0.0 https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
```

## 6. Precompute encoder outputs

Test with 3 trajectories first (single GPU):

```bash
mkdir -p data/bridge_v2_encodings
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 --batch_size 64 --window_size 2 --max_trajectories 3
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
huggingface-cli download tejasrao/ttdr-bridge-encodings encodings.h5 --local-dir data/bridge_v2_encodings --repo-type dataset
```

## 7. Train world model

```bash
python -m recap.training.train_world_model --config configs/train_wm.yaml
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
rsync -avz -e "ssh -i ~/.ssh/id_ed25519_lambda" /Users/tejasrao/Desktop/TTDR/ ubuntu@129.213.26.7:~/TTDR/
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@129.213.26.7
```

```bash
source ~/venv/bin/activate
cd ~/TTDR

# Quick test (single GPU, 3 trajectories):
python scripts/precompute_encodings.py \
  --data_dir /home/ubuntu/data/rlds \
  --output_dir data/bridge_v2_encodings \
  --chunk_size 4 --batch_size 64 --window_size 2 --max_trajectories 3

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
