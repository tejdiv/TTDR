#!/bin/bash
# Train VQ-VAE tokenizer + dynamics transformer on Bridge V2 (4×H100).
# Uses the same ttdr-train Baseten project — Bridge V2 tfrecords already cached.
#
# Phase 1: VQ-VAE tokenizer (100K steps, ~1-2h)
# Phase 2: Dynamics transformer (200K steps, ~2-3h)
# Total: ~3-5h
set -e

echo "============================================================"
echo "VQ-VAE + Dynamics Training Pipeline (8×H200)"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# ---- CPU thread tuning: prevent tf.data from bottlenecking 8 GPUs ----
export TF_NUM_INTEROP_THREADS=16
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16
echo "CPU threads: TF_NUM_INTEROP=$TF_NUM_INTEROP_THREADS, TF_NUM_INTRAOP=$TF_NUM_INTRAOP_THREADS"

# ---- System packages ----
apt-get update && apt-get install -y python3 python3-pip python3-venv git wget
ln -sf /usr/bin/python3 /usr/bin/python

# ---- Install deps ----
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  flax==0.7.5 optax==0.1.5 \
  tensorflow==2.15.0 tensorflow_probability==0.23.0 \
  tensorflow_datasets==4.9.2 \
  chex==0.1.85 distrax==0.1.5 ml_dtypes==0.2.0 \
  h5py absl-py pyyaml \
  numpy==1.24.3 "scipy==1.11.4" tqdm transformers==4.36.2 einops \
  huggingface_hub
pip install --no-deps orbax-checkpoint==0.5.3 tensorstore==0.1.51
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"

# ---- Install TTDR ----
pip install -e .

# ---- Verify GPUs ----
python -c "
import jax
devs = jax.devices()
print(f'JAX devices: {len(devs)}')
for i, d in enumerate(devs):
    print(f'  Device {i}: {d}')
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), 'No GPU!'
"

# ---- Locate or download Bridge V2 data ----
echo "=== DISK SPACE ==="
df -h
echo ""
# Try project cache first, then checkpoint dir, then download
if [ -n "$BT_PROJECT_CACHE_DIR" ]; then
  echo "BT_PROJECT_CACHE_DIR=$BT_PROJECT_CACHE_DIR"
  SEARCH_DIR="$BT_PROJECT_CACHE_DIR"
elif [ -n "$BT_CHECKPOINT_DIR" ]; then
  echo "No project cache, searching BT_CHECKPOINT_DIR=$BT_CHECKPOINT_DIR"
  SEARCH_DIR="$BT_CHECKPOINT_DIR"
else
  echo "No cache dirs set, will download to /tmp"
  SEARCH_DIR="/tmp"
fi

TFRECORD_FILE=$(find "$SEARCH_DIR" -name "bridge_dataset-train.tfrecord-00000-of-01024" -type f 2>/dev/null | head -1)

if [ -z "$TFRECORD_FILE" ]; then
  DOWNLOAD_DIR="${BT_PROJECT_CACHE_DIR:-${BT_CHECKPOINT_DIR:-/tmp}}/rlds"
  DOWNLOAD_TARGET="${BT_PROJECT_CACHE_DIR:-${BT_CHECKPOINT_DIR:-/tmp}}"
  if [ -f "ttdr-precompute_3yd4743_checkpoints.json" ]; then
    echo "=== DOWNLOADING FROM S3 (previous job's cached data) ==="
    echo "Download target: $DOWNLOAD_TARGET"
    python download_from_s3.py \
      --json ttdr-precompute_3yd4743_checkpoints.json \
      --output_dir "$DOWNLOAD_TARGET"
  else
    echo "No cached Bridge V2 data found. Downloading from source (~60GB)..."
    mkdir -p "$DOWNLOAD_DIR/bridge_dataset/1.0.0"
    wget -c -r -np -nH --cut-dirs=4 \
      --reject "index.html*" \
      -P "$DOWNLOAD_DIR" \
      https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
  fi
  TFRECORD_FILE=$(find "$DOWNLOAD_TARGET" -name "bridge_dataset-train.tfrecord-00000-of-01024" -type f | head -1)
fi

if [ -z "$TFRECORD_FILE" ]; then
  echo "ERROR: Failed to find or download Bridge V2 data"
  exit 1
fi

TFRECORD_DIR=$(dirname "$TFRECORD_FILE")
DATA_DIR=$(dirname "$(dirname "$TFRECORD_DIR")")
SHARD_COUNT=$(find "$TFRECORD_DIR" -name "bridge_dataset-train.tfrecord-*" -type f | wc -l)
echo "Bridge V2 data: $DATA_DIR ($SHARD_COUNT shards)"

if [ "$SHARD_COUNT" -lt 1024 ]; then
  echo "WARNING: Expected 1024 shards, found $SHARD_COUNT. Data may be incomplete."
fi

# ---- Set checkpoint dirs ----
CKPT_BASE="${BT_CHECKPOINT_DIR:-/tmp/checkpoints}"
TOK_CKPT_DIR="$CKPT_BASE/vqvae_tokenizer"
DYN_CKPT_DIR="$CKPT_BASE/dynamics_transformer"
mkdir -p "$TOK_CKPT_DIR" "$DYN_CKPT_DIR"
echo "Tokenizer checkpoints: $TOK_CKPT_DIR"
echo "Dynamics checkpoints:  $DYN_CKPT_DIR"

# ---- Phase 1: VQ-VAE Tokenizer ----
echo ""
echo "============================================================"
echo "PHASE 1: VQ-VAE Tokenizer Training"
echo "============================================================"
echo "Phase 1 start: $(date)"

python -c "
import yaml
with open('configs/train_tokenizer.yaml') as f:
    config = yaml.safe_load(f)
config['data']['cache_dir'] = '$DATA_DIR'
config['output']['checkpoint_dir'] = '$TOK_CKPT_DIR'
with open('/tmp/train_tokenizer_runtime.yaml', 'w') as f:
    yaml.dump(config, f)
print('Runtime config written to /tmp/train_tokenizer_runtime.yaml')
print(f'  cache_dir: {config[\"data\"][\"cache_dir\"]}')
print(f'  checkpoint_dir: {config[\"output\"][\"checkpoint_dir\"]}')
print(f'  batch_size: {config[\"data\"][\"batch_size\"]}')
print(f'  total_steps: {config[\"training\"][\"total_steps\"]}')
"

python -m recap.training.train_tokenizer --config /tmp/train_tokenizer_runtime.yaml

echo "Phase 1 complete: $(date)"

# ---- Phase 2: Dynamics Transformer ----
echo ""
echo "============================================================"
echo "PHASE 2: Dynamics Transformer Training"
echo "============================================================"
echo "Phase 2 start: $(date)"

python -c "
import yaml
with open('configs/train_dynamics.yaml') as f:
    config = yaml.safe_load(f)
config['data']['cache_dir'] = '$DATA_DIR'
config['tokenizer']['checkpoint_dir'] = '$TOK_CKPT_DIR'
config['output']['checkpoint_dir'] = '$DYN_CKPT_DIR'
with open('/tmp/train_dynamics_runtime.yaml', 'w') as f:
    yaml.dump(config, f)
print('Runtime config written to /tmp/train_dynamics_runtime.yaml')
print(f'  data cache_dir: {config[\"data\"][\"cache_dir\"]}')
print(f'  tokenizer checkpoint: {config[\"tokenizer\"][\"checkpoint_dir\"]}')
print(f'  dynamics checkpoint: {config[\"output\"][\"checkpoint_dir\"]}')
print(f'  batch_size: {config[\"data\"][\"batch_size\"]}')
print(f'  prediction_horizon: {config[\"data\"][\"prediction_horizon\"]}')
print(f'  total_steps: {config[\"training\"][\"total_steps\"]}')
"

python -m recap.training.train_dynamics --config /tmp/train_dynamics_runtime.yaml

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Tokenizer checkpoint: $TOK_CKPT_DIR"
echo "Dynamics checkpoint:  $DYN_CKPT_DIR"
echo "Checkpoints synced to Baseten cloud via BT_CHECKPOINT_DIR."
