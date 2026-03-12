# moto_recap — Goal-conditioned RECAP with Moto-GPT world model.
#
# Replaces contrastive/VQ-VAE world model with Moto motion token surprisal.
# Adds anchor conditioning (g) and hindsight relabeling for KM data efficiency.
#
# Stages:
#   1. finetune_moto.py   — fine-tune Moto-GPT on Bridge V2 (context truncation)
#   2. pretrain_policy.py — pretrain Octo with (g, I) + hindsight relabeling
#   3. adaptation.py      — test-time RECAP (LoRA or full fine-tune)
