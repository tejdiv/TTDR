# Part 3: RECAP Adaptation — Detailed Implementation

## File 1: `recap/losses/tracking_reward.py`

**What it computes:** The dense per-chunk reward signal. At time `t`, the world model predicts an "anchor" — where the observation embedding *should* be `m` steps later. After executing `m` steps, you compare reality to the prediction.

**Inputs at each chunk boundary:**
- `z_t` — Octo encoder output at current time (768-dim), from a live forward pass through the frozen encoder
- `z_t1` — Octo encoder output with 2-frame context (768-dim), same frozen encoder
- `z_t_plus_m` — Octo encoder output `m` steps later (768-dim), same frozen encoder
- Frozen world model params (projection head `h` + dynamics predictor `f_ψ`)

**The computation:**
```python
def tracking_reward(z_t, z_t1, z_t_plus_m, world_model_params, world_model_apply_fn):
    # Project current state through frozen world model
    z_prime_t = h(z_t)              # (256,) on unit sphere
    z_prime_t1 = h(z_t1)           # (256,) on unit sphere

    # Anchor: where the world model predicts we should be
    anchor = f_psi(concat(z_prime_t, z_prime_t1))  # (256,) on unit sphere

    # Where we actually ended up
    actual = h(z_t_plus_m)          # (256,) on unit sphere

    # Tracking error (squared L2 distance on unit sphere)
    tracking_error = ||actual - anchor||²

    # Normalize by predicted displacement (how far anchor is from start)
    displacement = ||anchor - z_prime_t||² + ε

    r_aux = -tracking_error / displacement
    return r_aux
```

**Why normalize:** Without normalization, chunks where the robot is supposed to move a lot (big displacement) would always have larger absolute error than chunks where it stays still. Dividing by displacement makes the reward scale-invariant — it measures *relative* tracking quality.

**Implementation detail:** The existing `WorldModel.__call__` in `train_world_model.py` already returns `(predicted, target_proj)` which is `(anchor, h(z_target))`. You also need `h(z_t)` for the displacement denominator, which means you either:
- Add a method to `WorldModel` that also returns the intermediate `z_prime_t`, or
- Call `ProjectionHead` separately on `z_t`

Cleanest approach: the reward function takes the raw encoder outputs and the frozen world model, calls `ProjectionHead` directly for `z_prime_t`, and uses the full `WorldModel` forward pass for the anchor + actual projection.

---

## File 2: `recap/models/lora_adapter.py`

**What it does:** Injects low-rank adapters into Octo's diffusion action head so you can adapt the policy with very few trainable parameters.

**LoRA mechanics:** For a frozen `nn.Dense` with weight `W` (shape `[in, out]`), LoRA adds a low-rank bypass:
```
output = x @ W + x @ A @ B     (A: [in, rank], B: [rank, out])
```
Only `A` and `B` are trainable. With rank=8, a 256×1024 Dense goes from 262k params to (256×8 + 8×1024) = 10k trainable params.

### Target layers in Octo's DiffusionActionHead

The `ScoreActor` inside `DiffusionActionHead` contains these Dense layers:

| Path | Shape | Role |
|---|---|---|
| `diffusion_model/reverse_network/Dense_0` | `input → 256` | MLPResNet input projection |
| `diffusion_model/reverse_network/MLPResNetBlock_0/Dense_0` | `256 → 1024` | Block 0 expand |
| `diffusion_model/reverse_network/MLPResNetBlock_0/Dense_1` | `1024 → 256` | Block 0 contract |
| `diffusion_model/reverse_network/MLPResNetBlock_1/Dense_0` | `256 → 1024` | Block 1 expand |
| `diffusion_model/reverse_network/MLPResNetBlock_1/Dense_1` | `1024 → 256` | Block 1 contract |
| `diffusion_model/reverse_network/MLPResNetBlock_2/Dense_0` | `256 → 1024` | Block 2 expand |
| `diffusion_model/reverse_network/MLPResNetBlock_2/Dense_1` | `1024 → 256` | Block 2 contract |
| `diffusion_model/reverse_network/Dense_1` | `256 → out_dim` | Output projection |

The 2 Dense layers in `diffusion_model/cond_encoder/` (time embedding MLP) should be skipped — they're tiny and encode diffusion timestep, not action-relevant information.

### Implementation approach in Flax

Work at the param-tree level — no monkey-patching of modules:

1. Load Octo's full params as a frozen pytree
2. Create a separate LoRA param pytree: for each target Dense kernel `W` at path `p`, create `lora_A[p]` (shape `[in, rank]`, init random small) and `lora_B[p]` (shape `[rank, out]`, init zeros so LoRA starts as identity)
3. Write a custom `apply` that intercepts Dense calls: `output = x @ (W_frozen + A @ B)` — or equivalently, do a forward pass with modified params where each target kernel is replaced by `W + A @ B`
4. Only `lora_A` and `lora_B` are in the optimizer's param set

**Concretely:** Write a function `inject_lora(params, target_paths, rank)` that:
- Walks the param tree
- For each Dense kernel at a target path, creates `(A, B)` pair
- Returns `lora_params` (the trainable A/B pairs) and a function `merge_params(frozen_params, lora_params)` that produces the full param tree with LoRA applied

This way the existing `DiffusionActionHead` code runs unchanged — you just pass it modified params.

**Total trainable params:** ~8 Dense layers × rank 8 × (in+out) ≈ roughly 80k params. Tiny compared to the full action head.

---

## File 3: `recap/data/replay_buffer.py`

**What it stores:** Transitions collected during adaptation rollouts in the perturbed environment.

**Each entry:**
```python
{
    "z_t":        (768,),    # encoder output at chunk start
    "z_t1":       (768,),    # encoder output (2-frame context) at chunk start
    "z_t_plus_m": (768,),    # encoder output m steps later
    "action":     (m, 7),    # the m actions executed during this chunk
    "r_aux":      scalar,    # tracking reward for this chunk
}
```

**Note:** Store encoder outputs, not raw images. The frozen encoder runs once per observation during rollout — no need to store and re-encode images.

**Behavior:**
- Fixed max capacity (e.g., 2000 chunks)
- FIFO eviction when full (oldest chunks dropped)
- `sample(batch_size)` returns a random batch for BC updates
- `get_last_M()` returns the most recent M chunks for advantage computation
- Straightforward NumPy arrays, nothing fancy needed

---

## File 4: `recap/training/recap_adaptation.py`

This is the main loop that ties everything together.

### Full algorithm

```
Setup:
    Load frozen Octo model (encoder + action head)
    Load frozen world model checkpoint (h + f_ψ)
    Initialize LoRA params on DiffusionActionHead (A=random small, B=zeros)
    Create optimizer for LoRA params only (AdamW, small lr e.g. 1e-4)
    Initialize replay buffer
    Create SimplerEnv with perturbation applied

Rollout + Adaptation loop:
    obs = env.reset()

    for chunk c = 0, 1, 2, ...:
        # --- Collect experience ---
        z_t = frozen_encoder(obs_t)           # encode current observation
        z_t1 = frozen_encoder(obs_t, obs_t-1) # 2-frame context encoding

        # Execute m actions using current policy (frozen weights + LoRA)
        merged_params = merge_lora(frozen_action_head_params, lora_params)
        for step in range(m):
            action = diffusion_sample(merged_params, z_t, ...)  # DDPM reverse
            obs = env.step(action)

        z_t_plus_m = frozen_encoder(obs)      # encode observation after m steps

        # Compute tracking reward
        r_aux = tracking_reward(z_t, z_t1, z_t_plus_m, frozen_world_model)

        # Store in buffer
        buffer.append(z_t, z_t1, z_t_plus_m, actions, r_aux)

        # --- Update every M chunks ---
        if c % M == 0 and len(buffer) >= M:
            recent = buffer.get_last_M()
            rewards = recent["r_aux"]                          # (M,)
            advantages = (rewards - mean(rewards)) / (std(rewards) + ε)

            # Advantage-conditioned BC: update LoRA to upweight good actions
            for update_step in range(num_bc_steps):
                batch = buffer.sample(batch_size)

                # Recompute advantages for sampled batch
                r_batch = batch["r_aux"]
                A_batch = (r_batch - mean(rewards)) / (std(rewards) + ε)

                # Weighted BC loss
                loss = advantage_weighted_bc_loss(
                    lora_params, frozen_params,
                    batch["z_t"], batch["action"], A_batch
                )
                lora_params = optimizer.update(lora_params, grads)
```

### Key design decisions in the update step

**1. Advantage normalization uses the recent M chunk stats.** `mean(rewards)` and `std(rewards)` come from the last M chunks, providing a rolling baseline. This is the GRPO-style trick — no value function needed.

**2. Advantage-conditioned BC loss.** Two common approaches:
- **Filtered BC:** Only clone actions where `A_i > 0` (above average), ignore the rest
- **Weighted BC:** Weight all actions by `max(A_i, 0)` or `exp(A_i / β)` — smoother gradient signal

The weighted version is more standard. The loss for a single sample is:
```
w_i = max(A_i, 0)   # or softmax(A / β)
L_i = w_i * ||π(o_i) - a_i||²   # diffusion action head already uses MSE
```

**3. How BC interacts with diffusion.** Octo's `DiffusionActionHead` already has a `loss` method that computes the denoising MSE. You just call it with the stored actions and weight each sample's loss by its advantage. No need to reinvent the diffusion training — you're fine-tuning the same denoising objective, just with non-uniform sample weights.

**4. The encoder forward pass during rollout.** You need to run Octo's full encoder (ViT + transformer) on each live observation to get `z_t`. This is the same encoder that was used during precompute, just now on sim images instead of real Bridge V2 images. This is where the domain gap risk lives.

---

## File 5: `configs/adapt.yaml`

```yaml
lora:
  rank: 8
  target_modules: "reverse_network"  # MLPResNet Dense layers only
  alpha: 16                          # LoRA scaling factor

adaptation:
  chunk_size: 4                      # m = action chunk length
  update_every_M: 16                 # update LoRA every M chunks
  num_bc_steps: 10                   # gradient steps per update
  bc_batch_size: 64                  # samples per BC step
  lr: 1e-4
  advantage_eps: 1e-8                # epsilon for std normalization
  advantage_clip: null               # optional clip range for advantages

buffer:
  max_size: 2000

env:
  task: "PutCarrotOnPlateInScene-v1"
  perturbation: "object_friction"
  perturbation_scale: 0.3

world_model:
  checkpoint: "checkpoints/world_model/step_50000"
```

---

## Implementation order

1. **`tracking_reward.py`** — standalone, just needs the frozen world model forward pass
2. **`lora_adapter.py`** — standalone, just needs Octo's param tree structure
3. **`replay_buffer.py`** — standalone, simple data structure
4. **`recap_adaptation.py`** — ties everything together, depends on all three above + Part 1 (perturbations)
5. **`adapt.yaml`** — just config, write alongside step 4

Steps 1–3 are independent and can be built in parallel.
