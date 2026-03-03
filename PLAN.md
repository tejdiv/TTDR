# TTDR Implementation Plan

## Context

**Problem:** Generalist robot policies (VLAs like Octo) work on their training distribution but fail under dynamics shift (changed friction, mass, payload). Existing test-time adaptation uses coarse task-level rewards. TTDR proposes dense, per-chunk anchor-tracking rewards from a contrastive world model for faster, more targeted adaptation.

**Approach:** Train a contrastive world model on OXE Bridge V2 (WidowX) data. Deploy Octo zero-shot (no finetuning) on SimplerEnv's ManiSkill3 GPU-parallelized WidowX Bridge tasks with visual matching. Apply dynamics perturbations to degrade performance. Use RECAP adaptation with anchor-tracking rewards to recover.

**Why WidowX Bridge:** Octo's strongest embodiment — `bridge_dataset` has sampling weight 1.0 in OXE Magic Soup (highest of any dataset), with 60k language-annotated trajectories. SimplerEnv's ManiSkill3 branch provides 4 GPU-parallelized WidowX tasks with visual matching, enabling fast rollout collection for RECAP.

---

## Part 1: Environment Setup

### SimplerEnv WidowX Bridge (ManiSkill3/SAPIEN, GPU-parallelized)

**Install:**
```bash
git clone -b maniskill3 https://github.com/simpler-env/SimplerEnv.git
pip install -e SimplerEnv
pip install mani_skill
```

**Primary evaluation tasks (GPU-parallelized, visual matching):**
| Task | Env ID | Language Instruction |
|------|--------|---------------------|
| Carrot on Plate | `PutCarrotOnPlateInScene-v1` | "put the carrot on the plate" |
| Stack Cube | `StackGreenCubeOnYellowCubeBakedTexInScene-v1` | "stack the green cube on the yellow cube" |
| Spoon on Towel | `PutSpoonOnTableClothInScene-v1` | "put the spoon on the towel" |
| Eggplant in Basket | `PutEggplantInBasketScene-v1` | "put the eggplant in the basket" |

**Action space:** 7D (δxyz + δrotation + gripper) at 5 Hz. SimplerEnv's `OctoInference` handles denormalization using `bridge_dataset` statistics, binarized gripper, Euler→axis-angle.

### Dynamics Perturbations (SAPIEN API)

| Perturbation | Values | What Breaks |
|---|---|---|
| Object friction | ×0.3, ×0.5, ×2.0, ×3.0 | Grasp slip / over-grip |
| Object mass | ×0.5, ×2.0, ×3.0 | Lift/place dynamics |
| Joint damping | ×0.5, ×2.0 | Arm responsiveness |
| Gripper friction | ×0.3, ×0.5 | Grasp reliability |

### Files
```
recap/envs/__init__.py
recap/envs/perturbations.py
```

---

## Part 2: World Model Training on OXE Bridge V2 Data

### Data source
`bridge_dataset` from OXE:
- ~60,096 trajectories, all language-annotated
- WidowX 250 6-DOF arm, teleoperated at 5 Hz
- 13 skills across 24 environments
- Two RGB cameras: `image_0` (over-shoulder), `image_1` (side)
- RLDS format, Octo dataloader config at `octo/data/oxe/oxe_dataset_configs.py:56`
- Transform at `octo/data/oxe/oxe_standardization_transforms.py:27`

### Step 2a: Pre-compute encoder outputs
Run Octo's frozen encoder over all 60k trajectories. Cache to HDF5:
- `z_t`: (N, 768) — encoder readout tokens
- `text_embed`: (N, D) — T5 language embedding
- `z_{t+m}`: (N, 768) — readout tokens m steps later (m=4)

### Step 2b: Train h + f_ψ
```
z_t (768) → h → z'_t (256, L2-norm) → f_ψ → ẑ'_{t+m} (256, L2-norm)
```
- `h`: MLP 768→512→256, LayerNorm+GELU, L2-norm. ~1.2M params. Flax.
- `f_ψ`: MLP 256→1024→1024→1024→256, LayerNorm+GELU, L2-norm. ~2.6M params. Flax.
  Language conditioning is already encoded in z'_t via Octo's cross-attention.
- Loss: InfoNCE with L2 distance, τ=0.1, batch 256-512
- Training: AdamW lr=3e-4, warmup 1k, cosine, ~50k steps

### Files
```
recap/models/__init__.py
recap/models/projection_head.py
recap/models/dynamics_predictor.py
recap/losses/__init__.py
recap/losses/contrastive.py
recap/data/__init__.py
recap/data/oxe_contrastive.py
scripts/precompute_encodings.py
recap/training/__init__.py
recap/training/train_world_model.py
configs/train_wm.yaml
```

---

## Part 3: RECAP Test-Time Adaptation

```
FROZEN: encoder φ, world model (h + f_ψ)
ADAPTED: LoRA on DiffusionActionHead (rank=8)

for each step t:
    anchor g = f_ψ(h(φ(o_t)))
    execute action, observe o_{t+m}
    r_aux = -||h(φ(o_{t+m})) - g||² / (||g - h(φ(o_t))||² + ε)
    buffer.append(...)

    every M chunks:
        rewards = [r_1, r_2, ..., r_M] from buffer
        A_i = (r_i - mean(rewards)) / (std(rewards) + ε)
        update LoRA via advantage-conditioned BC (weighted by A_i)
```

No learned value function. Advantages are computed by normalizing tracking rewards
within each buffer batch — the batch mean replaces V_ϕ as the baseline.

LoRA targets: `nn.Dense` in DiffusionActionHead's MLPResNet (`action_heads.py:386` → `diffusion.py`).

### Files
```
recap/models/lora_adapter.py
recap/losses/tracking_reward.py
recap/data/replay_buffer.py
recap/training/recap_adaptation.py
configs/adapt.yaml
```

---

## Part 4: Evaluation

| Condition | Method |
|---|---|
| Standard dynamics | Octo zero-shot (baseline) |
| Perturbed dynamics | Octo zero-shot (perf drop) |
| Perturbed dynamics | **TTDR anchor-tracking RECAP** |
| Perturbed dynamics | Task-level progress (TT-VLA baseline) |
| Perturbed dynamics | Oracle reward (upper bound) |

80 trials × 3 seeds. Tasks: Carrot on Plate, Stack Cube.

```
recap/eval/__init__.py
recap/eval/run_eval.py
recap/eval/perturbation_eval.py
```

---

## Verification Checkpoints

0. SimplerEnv ManiSkill3 WidowX runs, GPU parallelization works, perturbations degrade perf
1. Encoder z_t from sim clusters near z_t from real Bridge V2 (domain gap check)
2. World model retrieval accuracy >60% on held-out Bridge V2
3. Tracking reward correlates with chunk-level success
4. Buffer-normalized advantages show meaningful variance across chunks within a batch
5. RECAP recovers 30-60% of lost performance within 5-10 cycles

## Key Risk

Encoder domain gap (Octo never seen synthetic images). Visual matching mitigates. Fallbacks: light encoder finetune, learned adapter between φ and h.
