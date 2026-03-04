# Baseten 4×H100 Setup

Config files live in `baseten_train/`. Edit `HF_TOKEN` before pushing.

## 0. Check available GPUs

```bash
export BASETEN_API_KEY="your-key-from-app.baseten.co/settings/account"

# List all instance types
curl -s https://api.baseten.co/v1/instance_types \
  -H "Authorization: Api-Key $BASETEN_API_KEY" | python3 -m json.tool

# Filter to multi-GPU only
curl -s https://api.baseten.co/v1/instance_types \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for it in data.get('instance_types', []):
    if it.get('gpu_count', 0) >= 2:
        gpu = it.get('gpu_type', 'cpu')
        cnt = it.get('gpu_count', 0)
        mem = it.get('gpu_memory_limit_mib', 0) // 1024
        name = it.get('name', it.get('id', ''))
        print(f'{cnt}x {gpu} ({mem} GiB)  —  {name}')
"
```

## 1. Install & login

```bash
pip install --upgrade truss
truss login
```

## 2. Set your HF token

Go to [app.baseten.co](https://app.baseten.co) → **Settings → Secrets** → add `HF_TOKEN` with your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## 3. Run precompute (step 1)

```bash
cd baseten_train
bash push.sh config_download.py
```

`push.sh` bundles your TTDR code into the upload directory, pushes to Baseten, then cleans up.

Downloads Bridge V2, precomputes encodings on 4 GPUs, uploads `encodings.h5` to HF.
Once this succeeds, encodings are safe on HF — you never need to re-download Bridge V2.

**Crash safe:** Downloaded data is saved to `$BT_CHECKPOINT_DIR` (Baseten's persistent
storage). If precompute crashes, just re-run the same command — it detects the cached
data and skips the download.

## 4. Run training (step 2)

```bash
cd baseten_train
bash push.sh config_train_only.py
```

Downloads `encodings.h5` from HF and trains. If this crashes, just re-run it —
no data is lost.

## 5. Run everything in one shot (if you're feeling lucky)

```bash
cd baseten_train
bash push.sh config.py
```

Does steps 3+4 in a single job. If it crashes mid-train, you lose the download time.

## 6. Monitor

```bash
truss train logs --job-id <job_id> --tail
truss train metrics --job-id <job_id>
```

Or: [app.baseten.co/training](https://app.baseten.co/training/)

## 7. Debug (rSSH)

Connect to a running/failed container via VS Code or Cursor Remote Tunnels.

## 8. Interactive session (if you want a shell)

```bash
cd baseten_train
truss train push config_interactive.py
```

This starts 4×H100s and keeps the container alive for 24h. Connect via rSSH
in VS Code/Cursor and run commands manually — just like SSH on Lambda/Mithril.

---

## Files

| File | Purpose |
|------|---------|
| `baseten_train/config_download.py` | Precompute-only config (4×H100) |
| `baseten_train/run_precompute.sh` | Download data → precompute → upload encodings to HF |
| `baseten_train/config_train_only.py` | Train-only config (4×H100) |
| `baseten_train/run_train_only.sh` | Download encodings from HF → train |
| `baseten_train/config.py` | Full pipeline config (both steps in one job) |
| `baseten_train/run.sh` | Full pipeline script (data → precompute → train) |
| `baseten_train/config_interactive.py` | Interactive session (sleep 24h, connect via rSSH) |

## Pricing

| GPU | VRAM | $/min (×1) | $/hr (×4) |
|-----|------|-----------|-----------|
| A100 | 80 GiB | $0.067 | ~$16.00 |
| H100 | 80 GiB | $0.108 | ~$26.00 |
| B200 | 180 GiB | $0.166 | ~$39.90 |
