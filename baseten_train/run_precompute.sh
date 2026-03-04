#!/bin/bash
# Step 1: Download Bridge V2, precompute encodings, upload to HF.
# Run this first. If it succeeds, encodings are safe on HF.
# Then run config_train_only.py for training.
#
# CRASH SAFETY:
#   - Everything goes to BT_CHECKPOINT_DIR (500GB, synced to cloud)
#   - Download + encoding shards both survive crashes
#   - Merge + HF upload is fast and can be re-run cheaply
set -e

# ---- Storage setup ----
echo "=== STORAGE ==="
echo ""
echo "BT_CHECKPOINT_DIR=$BT_CHECKPOINT_DIR"
echo "BT_PROJECT_CACHE_DIR=$BT_PROJECT_CACHE_DIR"
echo "BT_LOAD_CHECKPOINT_DIR=$BT_LOAD_CHECKPOINT_DIR"
df -h
echo ""

# Download goes to project cache (persists across ALL jobs in this project)
if [ -z "$BT_PROJECT_CACHE_DIR" ]; then
  echo "WARNING: BT_PROJECT_CACHE_DIR not set. Using /tmp for download (NOT persistent!)."
  CACHE_DIR=/tmp
else
  echo "OK: Project cache available at $BT_PROJECT_CACHE_DIR"
  CACHE_DIR="$BT_PROJECT_CACHE_DIR"
fi
DATA_DIR="$CACHE_DIR/rlds"

# Encoding shards go to checkpoint dir (per-job, synced to cloud)
if [ -z "$BT_CHECKPOINT_DIR" ]; then
  echo "WARNING: BT_CHECKPOINT_DIR not set. Using /tmp for encodings."
  ENCODING_DIR=/tmp/encoding_shards
else
  echo "OK: Checkpoint dir available at $BT_CHECKPOINT_DIR"
  ENCODING_DIR="$BT_CHECKPOINT_DIR/encoding_shards"
fi

echo "Download dir (cached):   $DATA_DIR"
echo "Encoding shards dir:     $ENCODING_DIR"
echo ""

# ---- Smoke test: verify imports before downloading anything ----
echo "=== SMOKE TEST ==="

# ---- System packages (bare CUDA image has no Python/git/wget) ----
apt-get update && apt-get install -y python3 python3-pip python3-venv git wget
ln -sf /usr/bin/python3 /usr/bin/python

# ---- Install deps (pinned to Octo-compatible versions) ----
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  flax==0.7.5 optax==0.1.5 \
  tensorflow==2.15.0 tensorflow_probability==0.23.0 \
  tensorflow_datasets==4.9.2 \
  chex==0.1.85 distrax==0.1.5 ml_dtypes==0.2.0 \
  h5py absl-py pyyaml \
  numpy==1.24.3 "scipy==1.11.4" wandb tqdm transformers==4.36.2 einops \
  huggingface_hub
pip install --no-deps orbax-checkpoint==0.5.3 tensorstore==0.1.51
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"

# ---- Install TTDR (code is bundled by push.sh) ----
pip install -e .

# ---- Verify imports BEFORE spending 2hrs downloading ----
python -c "from octo.model.octo_model import OctoModel; print('octo import OK')"
python -c "from recap.data.oxe_contrastive import make_bridge_trajectory_dataset; print('recap import OK')"
python -c "import jax; devs = jax.devices(); print(f'JAX devices: {devs}'); assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), 'ERROR: No GPU found! Check JAX CUDA install.'"

# ---- HF login ----
python -c "import huggingface_hub; huggingface_hub.login(token='$hf_token')"

echo "=== SMOKE TEST PASSED — starting download ==="

# ---- Download Bridge V2 data (cached across jobs) ----
EXPECTED_DIR="$DATA_DIR/bridge_dataset/1.0.0"

# Check anywhere under CACHE_DIR for existing shards
EXISTING_SHARDS=$(find "$CACHE_DIR" -name "bridge_dataset-train.tfrecord-*" -type f 2>/dev/null | wc -l)
echo "Existing shards in project cache: $EXISTING_SHARDS"

if [ "$EXISTING_SHARDS" -ge 1024 ]; then
  echo "=== FOUND $EXISTING_SHARDS shards in cache — skipping download ==="
elif [ -f "ttdr-precompute_3yd4743_checkpoints.json" ]; then
  echo "=== DOWNLOADING FROM S3 (previous job's cached data) ==="
  python download_from_s3.py \
    --json ttdr-precompute_3yd4743_checkpoints.json \
    --output_dir "$CACHE_DIR"
else
  echo "=== DOWNLOADING BRIDGE V2 FROM SOURCE ==="
  mkdir -p "$CACHE_DIR"
  wget -c -r -np -nH --cut-dirs=4 \
    --reject "index.html*" \
    -P "$CACHE_DIR" \
    https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
  echo "Download complete."
fi

# ---- Find where tfrecords actually are and set DATA_DIR accordingly ----
# TFDS expects: DATA_DIR/bridge_dataset/1.0.0/*.tfrecord*
TFRECORD_FILE=$(find "$CACHE_DIR" -name "bridge_dataset-train.tfrecord-00000-of-01024" -type f | head -1)
if [ -z "$TFRECORD_FILE" ]; then
  echo "ERROR: No tfrecord files found anywhere under $CACHE_DIR"
  find "$CACHE_DIR" -type f | head -10
  exit 1
fi
TFRECORD_DIR=$(dirname "$TFRECORD_FILE")
# DATA_DIR needs to be two levels above: DATA_DIR/bridge_dataset/1.0.0/
DATA_DIR=$(dirname "$(dirname "$TFRECORD_DIR")")
echo "TFRecords at: $TFRECORD_DIR"
echo "DATA_DIR set to: $DATA_DIR"

# ---- Verify ----
SHARD_COUNT=$(find "$TFRECORD_DIR" -name "bridge_dataset-train.tfrecord-*" -type f | wc -l)
echo "Found $SHARD_COUNT train shards"
if [ "$SHARD_COUNT" -lt 1024 ]; then
  echo "ERROR: Expected 1024 train shards, found $SHARD_COUNT."
  exit 1
fi
echo "All 1024 train shards present."

# ---- Precompute (4 GPUs, shards saved to BT_CHECKPOINT_DIR for crash safety) ----
export NUM_SHARDS=4
export DATA_DIR
export OUTPUT_DIR="$ENCODING_DIR"
bash scripts/launch_precompute.sh --hf tejasrao/ttdr-bridge-encodings

echo ""
echo "Done. Encodings uploaded to HF: tejasrao/ttdr-bridge-encodings"
echo "Everything persisted in BT_CHECKPOINT_DIR."
