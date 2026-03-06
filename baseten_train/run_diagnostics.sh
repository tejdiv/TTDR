#!/bin/bash
# Run WM + value head diagnostics on real data (encodings + checkpoint from S3).
set -e

apt-get update && apt-get install -y python3 python3-pip python3-venv git wget
ln -sf /usr/bin/python3 /usr/bin/python

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
pip install -e .

# Download encodings from S3
mkdir -p data/bridge_v2_encodings
python -c "
import json, urllib.request, os
with open('ttdr-precompute_wgl7go3_checkpoints.json') as f:
    data = json.load(f)
for a in data['checkpoint_artifacts']:
    if a['relative_file_name'] == 'encoding_shards/encodings.h5':
        dest = 'data/bridge_v2_encodings/encodings.h5'
        print(f'Downloading encodings.h5 ({a[\"size_bytes\"]/1e9:.1f} GB)...')
        urllib.request.urlretrieve(a['url'], dest)
        print(f'Done: {os.path.getsize(dest)/1e9:.1f} GB')
        break
"

# Download checkpoint from S3 (final + early)
mkdir -p checkpoints/world_model_v1
python -c "
import json, urllib.request, os
with open('ttdr-train_w5d09j3_checkpoints.json') as f:
    data = json.load(f)
for a in data['checkpoint_artifacts']:
    name = a['relative_file_name']
    if 'checkpoint_50000' in name or 'checkpoint_5000/' in name:
        dest = os.path.join('checkpoints/world_model_v1', name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(a['url'], dest)
print('Checkpoints downloaded.')
"

# Run diagnostics
python tests/test_wm_diagnostics.py \
  --checkpoint checkpoints/world_model_v1 \
  --early_checkpoint checkpoints/world_model_v1/checkpoint_5000 \
  --encodings data/bridge_v2_encodings/encodings.h5

echo "Diagnostics complete."
