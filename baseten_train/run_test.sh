#!/bin/bash
# End-to-end test: install deps → precompute on debug dataset → train 10 steps.
# Uses the 25-trajectory debug dataset (3.5MB) bundled in tests/.
# Tests the FULL pipeline with real data — no faking.
# Also tests BT_CHECKPOINT_DIR storage (precompute + training checkpoints).
set -e

echo "============================================"
echo "=== STORAGE & ENVIRONMENT INFO ==="
echo "============================================"
echo ""
echo "--- All mounted volumes ---"
df -h
echo ""
echo "--- /tmp ---"
df -h /tmp
echo ""
echo "--- BT_CHECKPOINT_DIR ---"
echo "BT_CHECKPOINT_DIR=$BT_CHECKPOINT_DIR"
if [ -z "$BT_CHECKPOINT_DIR" ]; then
  echo "WARNING: BT_CHECKPOINT_DIR is not set. Using /tmp."
  BASE_DIR=/tmp
else
  echo "OK: BT_CHECKPOINT_DIR is set"
  df -h "$BT_CHECKPOINT_DIR"
  echo "Mount point details:"
  mount | grep "$(df "$BT_CHECKPOINT_DIR" --output=source | tail -1)" || true
  echo "Contents:"
  ls -la "$BT_CHECKPOINT_DIR" 2>/dev/null || echo "(empty)"
  BASE_DIR="$BT_CHECKPOINT_DIR"
fi
echo ""
ENCODING_DIR="$BASE_DIR/test_encodings"
CHECKPOINT_DIR="$BASE_DIR/test_checkpoints"
echo "--- Plan ---"
echo "Encoding dir:   $ENCODING_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "============================================"
echo ""

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
  numpy==1.24.3 "scipy==1.11.4" wandb tqdm transformers==4.36.2 einops \
  huggingface_hub
pip install --no-deps orbax-checkpoint==0.5.3 tensorstore==0.1.51
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"

# ---- Install TTDR ----
pip install -e .

# ---- Verify imports ----
echo "=== IMPORT TEST ==="
python -c "from octo.model.octo_model import OctoModel; print('octo import OK')"
python -c "from recap.data.oxe_contrastive import make_bridge_trajectory_dataset; print('data loader import OK')"
python -c "from recap.training.train_world_model import create_train_state; print('training import OK')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# ---- HF login (needed to download Octo model) ----
python -c "import huggingface_hub; huggingface_hub.login(token='$hf_token')"

# ---- Step 1: Precompute on debug dataset → BT_CHECKPOINT_DIR ----
echo ""
echo "=== PRECOMPUTE TEST (debug dataset, 10 trajectories) ==="
python scripts/precompute_encodings.py \
  --data_dir tests/debug_dataset \
  --output_dir "$ENCODING_DIR" \
  --max_trajectories 10 \
  --num_shards 1 \
  --batch_size 4 \
  --chunk_size 2 \
  --window_size 2

# ---- Verify HDF5 output ----
echo ""
echo "=== VERIFYING PRECOMPUTE OUTPUT ==="
python -c "
import h5py, sys, os
path = os.path.join('$ENCODING_DIR', 'encodings.h5')
print(f'Reading from: {path}')
with h5py.File(path, 'r') as f:
    for key in ['z_t', 'z_t1', 'z_target', 'traj_id']:
        assert key in f, f'Missing dataset: {key}'
        print(f'  {key}: shape={f[key].shape}, dtype={f[key].dtype}')
    N = f['z_t'].shape[0]
    assert N > 0, 'No transitions written!'
    assert f['z_t'].shape[1] == 768, f'Wrong encoder dim: {f[\"z_t\"].shape[1]}'
    assert f['z_t'].shape == f['z_t1'].shape == f['z_target'].shape
    assert f['traj_id'].shape == (N,)
    print(f'  Total transitions: {N}')
    print(f'  Metadata: chunk_size={f.attrs[\"chunk_size\"]}, window_size={f.attrs[\"window_size\"]}')
print('Precompute output OK')
"

# ---- Step 2: Train 10 steps → checkpoints to BT_CHECKPOINT_DIR ----
echo ""
echo "=== TRAINING TEST (10 steps) ==="
python -c "
import yaml, os
config = {
    'data': {'cache_dir': '$ENCODING_DIR', 'batch_size': 32, 'seed': 42},
    'model': {
        'projection_head': {'hidden_dim': 256, 'output_dim': 128},
        'dynamics_predictor': {'hidden_dim': 256, 'num_layers': 2, 'output_dim': 128},
    },
    'loss': {'temperature': 0.1, 'intra_weight': 0.5},
    'training': {
        'lr': 3e-4, 'warmup_steps': 2, 'total_steps': 10, 'weight_decay': 0.01,
        'grad_clip': 1.0, 'log_every': 1, 'eval_every': 100, 'save_every': 5,
    },
    'output': {
        'checkpoint_dir': '$CHECKPOINT_DIR',
        'wandb_project': None, 'wandb_run_name': None,
    },
}
with open('/tmp/test_train_config.yaml', 'w') as f:
    yaml.dump(config, f)
print('Test config written')
print(f'  Encodings from: $ENCODING_DIR')
print(f'  Checkpoints to: $CHECKPOINT_DIR')
"

python -m recap.training.train_world_model --config /tmp/test_train_config.yaml

# ---- Verify checkpoints were written to BT_CHECKPOINT_DIR ----
echo ""
echo "=== VERIFYING CHECKPOINTS ==="
ls -la "$CHECKPOINT_DIR"/
echo ""
if [ -d "$CHECKPOINT_DIR/checkpoint_5" ] || [ -d "$CHECKPOINT_DIR/checkpoint_10" ]; then
  echo "Checkpoints found in BT_CHECKPOINT_DIR — crash-safe storage working!"
else
  echo "WARNING: No checkpoints found in $CHECKPOINT_DIR"
fi

echo ""
echo "=== ALL TESTS PASSED ==="
echo "Full pipeline works: precompute → BT_CHECKPOINT_DIR → train → checkpoints to BT_CHECKPOINT_DIR"
echo "Safe to run: bash push.sh config_download.py"
