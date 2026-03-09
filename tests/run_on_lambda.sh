#!/bin/bash
# Run all RECAP tests on a Lambda A10 instance.
#
# From your Mac:
#   1. Spin up a 1×A10 on Lambda Cloud
#   2. rsync -avz -e "ssh -i ~/.ssh/id_ed25519_lambda" \
#        /Users/tejasrao/Desktop/TTDR/ ubuntu@<IP>:~/TTDR/
#   3. ssh -i ~/.ssh/id_ed25519_lambda ubuntu@<IP>
#   4. cd ~/TTDR && bash tests/run_on_lambda.sh
#
# Needs: HF_TOKEN env var (for private world model checkpoint)
#   export HF_TOKEN=hf_...
#   bash tests/run_on_lambda.sh

set -e

# ── Check HF token ───────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN first:"
    echo "  export HF_TOKEN=hf_..."
    echo "  bash tests/run_on_lambda.sh"
    exit 1
fi

# ── System cuDNN (required — pip cuDNN doesn't work for convolutions) ──
echo "=== Installing system cuDNN ==="
sudo apt-get update -qq
sudo apt-get install -y -qq libcudnn8 libcudnn8-dev 2>/dev/null || {
    echo "WARNING: Could not install system cuDNN via apt."
    echo "If convolutions fail, install manually:"
    echo "  sudo apt-get install libcudnn8"
}

# ── Install Python deps ────────────────────────────────────────────
echo ""
echo "=== Installing Python dependencies ==="
pip install --upgrade pip

# Stage 1: numpy first (1.x required for jaxlib 0.4.20), then JAX with CUDA
pip install numpy==1.24.3
pip install jax==0.4.20 jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Stage 2: ml_dtypes
pip install -q ml_dtypes==0.2.0

# Stage 3: tensorflow + tensorflow_probability (pin versions for TF 2.15)
pip install -q tensorflow==2.15.0 tensorflow_probability==0.23.0 tensorflow_datasets==4.9.2
pip install -q --no-deps ml_dtypes==0.2.0

# Stage 4: flax ecosystem (--no-deps for orbax/tensorstore to avoid conflicts)
pip install -q flax==0.7.5 optax==0.1.5 chex==0.1.85 distrax==0.1.5
pip install -q --no-deps orbax-checkpoint==0.5.3 tensorstore==0.1.45

# Stage 5: everything else (scipy pinned — 1.12+ removed scipy.linalg.tril)
pip install -q h5py absl-py pyyaml scipy==1.11.4 wandb tqdm \
    transformers==4.36.2 einops huggingface_hub nest_asyncio

pip install -q --no-deps "dlimp @ git+https://github.com/kvablack/dlimp.git"
pip install -e .

# Stage 6: pin numpy back (some deps pull in numpy 2.x)
pip install numpy==1.24.3

# ── HF login ─────────────────────────────────────────────────────
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# ── Verify GPU ───────────────────────────────────────────────────
echo ""
echo "=== GPU check ==="
python -c "import jax; print(f'Devices: {jax.devices()}'); assert jax.default_backend() == 'gpu', 'No GPU!'"

# ── Test 1: bind/unbind roundtrip ────────────────────────────────
echo ""
echo "=== Test 1: bind/unbind roundtrip ==="
python -m tests.test_bind_roundtrip

# ── Test 2: world model loads from HuggingFace ───────────────────
echo ""
echo "=== Test 2: world model checkpoint load ==="
python -c "
from flax.training import checkpoints
from huggingface_hub import snapshot_download

print('Downloading 4manifold/ttdr-world-model...')
local_path = snapshot_download('4manifold/ttdr-world-model')
print(f'Downloaded to: {local_path}')

import os
print(f'Contents: {os.listdir(local_path)}')

state = checkpoints.restore_checkpoint(local_path, target=None)
print(f'Checkpoint keys: {list(state.keys()) if isinstance(state, dict) else type(state)}')
if isinstance(state, dict) and 'params' in state:
    import jax
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state['params']))
    print(f'World model params: {param_count:,}')
    print('OK: checkpoint loads correctly')
else:
    print(f'WARNING: unexpected checkpoint structure: {type(state)}')
    print('May need to adjust load_world_model() in recap_adaptation.py')
"

# ── Test 3: dry-run adaptation loop (mock env) ───────────────────
echo ""
echo "=== Test 3: dry-run adaptation (mock env, 1 episode, 2 steps) ==="
python tests/test_adaptation_dry_run.py

echo ""
echo "=== All tests passed ==="
