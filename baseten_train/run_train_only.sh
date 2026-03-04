#!/bin/bash
# Train only: download pre-computed encodings from HF → train
set -e

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

# ---- HF login ----
python -c "import huggingface_hub; huggingface_hub.login(token='$hf_token')"

# ---- Download pre-computed encodings from HF ----
mkdir -p data/bridge_v2_encodings
huggingface-cli download tejasrao/ttdr-bridge-encodings encodings.h5 \
  --local-dir data/bridge_v2_encodings --repo-type dataset

# ---- Patch config to save checkpoints directly to BT_CHECKPOINT_DIR ----
# This way Baseten's background sync uploads them continuously,
# instead of only copying after training finishes.
python -c "
import yaml, os
with open('configs/train_wm.yaml') as f:
    config = yaml.safe_load(f)
ckpt_dir = os.environ.get('BT_CHECKPOINT_DIR', 'checkpoints/world_model')
config['output']['checkpoint_dir'] = ckpt_dir
with open('/tmp/train_wm_runtime.yaml', 'w') as f:
    yaml.dump(config, f)
print(f'Checkpoint dir set to: {ckpt_dir}')
"

# ---- Train world model ----
python -m recap.training.train_world_model --config /tmp/train_wm_runtime.yaml

echo "Done. Checkpoints saved to BT_CHECKPOINT_DIR and synced to Baseten cloud."
