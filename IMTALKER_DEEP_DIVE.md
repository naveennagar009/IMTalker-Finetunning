# IMTalker: Complete Technical Deep-Dive & Finetuning Guide

> **Paper**: *IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer*
> **Authors**: Bo Chen, Tao Liu, Qi Chen, Xie Chen, Zilong Zheng (SJTU & BIGAI)
> **Project**: https://cbsjtu01.github.io/IMTalker/

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Two-Stage Architecture](#2-two-stage-architecture)
3. [Stage 1: IMT Renderer (Deep Dive)](#3-stage-1-imt-renderer)
4. [Stage 2: Flow-Matching Motion Generator (Deep Dive)](#4-stage-2-flow-matching-motion-generator)
5. [The Motion Latent Space — The Bridge](#5-the-motion-latent-space)
6. [Training Losses Explained](#6-training-losses-explained)
7. [Data Pipeline](#7-data-pipeline)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Finetuning Guide](#9-finetuning-guide)
10. [Hyperparameter Reference](#10-hyperparameter-reference)
11. [Troubleshooting & Common Issues](#11-troubleshooting--common-issues)

---

## 1. High-Level Overview

IMTalker generates realistic talking face videos from a single portrait image + audio. Unlike prior methods that use explicit optical flow warping (which breaks under large head movements), IMTalker uses **implicit motion transfer** via cross-attention — it never computes pixel-level flow fields.

### What Makes IMTalker Different

| Approach | How Motion Works | Weakness |
|---|---|---|
| FOMM, LIA | Explicit optical flow → warp source pixels | Tearing, stretching under large poses |
| Diffusion (EMO, Hallo) | End-to-end video diffusion | Extremely slow, identity leakage |
| **IMTalker** | Implicit motion latent → cross-attention alignment | None of the above ✅ |

### Performance
- **40 FPS** video-driven, **42 FPS** audio-driven (RTX 4090, 512×512)
- State-of-the-art on HDTF and CelebV benchmarks

---

## 2. Two-Stage Architecture

IMTalker splits the problem into two independently trained models:

```
┌─────────────────────────────────────────────────────────┐
│                    INFERENCE FLOW                        │
│                                                         │
│  Audio ──→ [Generator] ──→ Motion Latent (32-dim)       │
│                               │                         │
│  Portrait ──→ [Renderer] ←────┘                         │
│                   │                                     │
│                   ▼                                     │
│            Output Video Frame                           │
└─────────────────────────────────────────────────────────┘
```

**Stage 1 — Renderer (124M params)**: Given a reference portrait and a motion latent vector, renders the output face. Trained on video frames with self-supervised reconstruction.

**Stage 2 — Generator (FMGenerator)**: Given audio + pose + gaze signals, produces motion latent vectors that the renderer consumes. Trained on pre-extracted motion latents from the renderer.

**Critical dependency**: The Generator's output must match the Renderer's motion latent space. If you retrain the Renderer, you MUST re-extract generator training data and retrain the Generator.

---

## 3. Stage 1: IMT Renderer

### 3.1 Architecture Components

The renderer (`IMTRenderer`, 124M params) has 5 sub-modules:

#### A. Identity Encoder (`IdentityEncoder`, aka `dense_feature_encoder`)
```
Input:  Reference face image (B, 3, 512, 512)
Output: Multi-scale features (list of 6 tensors) + Identity vector (B, 512)
```
- **Purpose**: Extracts appearance/identity information from the reference portrait
- **Architecture**: Initial 7×7 conv → 6 DownConvResBlocks → EqualConv2d → 4 EqualLinear layers → final 512-dim identity vector
- **Multi-scale features**: Produced at resolutions [256, 128, 64, 32, 16, 8] with channels [32, 64, 128, 256, 512, 512]
- **Key insight**: Returns features in REVERSE order (coarsest first) for the cross-attention hierarchy

#### B. Motion Encoder (`MotionEncoder`, aka `latent_token_encoder`)
```
Input:  Face image (B, 3, 512, 512)
Output: Motion latent vector (B, 32)
```
- **Purpose**: Compresses a face image into a compact 32-dimensional motion representation
- **Architecture**: Conv → 5 StyleGAN-style ResBlocks [128, 256, 512, 512, 512] → EqualConv2d → 4 EqualLinear → 32-dim output
- **Key insight**: This is the SAME encoder used during generator data prep to extract motion latents from training videos
- **Latent dimension**: 32 (very compact — forces the model to learn only motion, not identity)

#### C. Identity-Adaptive Module (`IdentidyAdaptive`, aka `adapt`)
```
Input:  Motion latent (B, 32) + Identity vector (B, 512)
Output: Personalized motion latent (B, 32)
```
- **Purpose**: Projects a generic motion latent into a person-specific motion space
- **Architecture**: Concat(mot, id) → EqualLinear(544→512) → 4 EqualLinear(512→512) → EqualLinear(512→32)
- **Why it matters**: The same head turn looks different on different people (face shape, proportions). This module adapts the motion to each identity.
- **Training constraint**: A distance loss ensures that the same motion projected through different identities maintains consistent relationships (prevents identity leakage into the motion space)

#### D. Motion Decoder (`MotionDecoder`, aka `latent_token_decoder`)
```
Input:  Personalized motion latent (B, 32)
Output: Multi-scale motion maps (4 tensors at resolutions 8, 16, 32, 64)
```
- **Purpose**: Expands the compact 32-dim motion latent into spatial feature maps
- **Architecture**: Learned constant (1, 32, 4, 4) → 13 StyleGAN StyledConv layers with progressive upsampling
- **Output resolutions**: 8×8 (512ch), 16×16 (512ch), 32×32 (256ch), 64×64 (128ch)
- **StyleGAN influence**: Uses style modulation from the motion latent at every layer — the latent "styles" the spatial features

#### E. Implicit Motion Transfer Module (`imt` — CrossAttention blocks)
```
Input:  Motion maps (driving + reference) + Identity features
Output: Aligned features for rendering
```
- **Purpose**: THE core innovation. Instead of warping pixels with optical flow, it uses cross-attention to implicitly align motion discrepancies with identity features
- **Architecture**: 6 CrossAttention blocks, one per spatial resolution
- **Coarse-to-fine**: At low resolutions (8×8, 16×16), uses standard cross-attention (global receptive field). At high resolutions (32×32+), uses Swin-style windowed attention for efficiency.
- **How it works**:
  1. Computes attention between driving motion map and reference motion map → gets attention map
  2. Uses this attention map to rearrange the reference identity features
  3. The rearranged features represent "what the reference person would look like in the driving pose"

#### F. Synthesis Network (`frame_decoder`)
```
Input:  Aligned features (6 levels)
Output: Final rendered face (B, 3, 512, 512)
```
- **Purpose**: Decodes the aligned multi-scale features into the final RGB image
- **Architecture**: Progressive upsampling with UpConvResBlocks + skip connections + SelfAttention at each level
- **Final output**: Conv → PixelShuffle(2×) → Sigmoid → RGB image in [0, 1]

### 3.2 Renderer Forward Pass (Training)

During training, the renderer sees two frames from the same video + one negative frame from a different video:

```python
# ref = reference frame (image_0), real = target frame (image_1), neg = negative identity

# 1. Extract identity features and vectors for all three
f_0, id_0 = app_encode(ref)      # Reference identity
f_1, id_1 = app_encode(real)     # Target identity (same person)
_,   id_2 = app_encode(neg)      # Negative identity (different person)

# 2. Extract motion latents
t_0 = mot_encode(ref)            # Reference motion
t_1 = mot_encode(real)           # Target motion (what we want to render)

# 3. Project motions through identity-adaptive module
ta_10 = id_adapt(t_1, id_0)     # Target motion in reference's identity space
ta_11 = id_adapt(t_1, id_1)     # Target motion in its own identity space
ta_12 = id_adapt(t_1, id_2)     # Target motion in negative's identity space
ta_00 = id_adapt(t_0, id_0)     # Reference motion in reference's identity space

# 4. Decode to motion maps
ma_10 = mot_decode(ta_10)        # Driving motion maps
ma_00 = mot_decode(ta_00)        # Reference motion maps

# 5. Cross-attention implicit motion transfer + rendering
pred = decode(ma_10, ma_00, f_0) # Output: rendered face with target motion, reference identity
```

### 3.3 Discriminator

- **Type**: Multi-scale PatchDiscriminator
- **Scale 1**: Full resolution (512×512) — StyleGAN2 Discriminator architecture
- **Scale 2**: Half resolution (256×256) — Same architecture, downsampled input
- **Loss**: Non-saturating GAN loss with R1 regularization (softplus formulation)
- **Output**: Two scalar predictions (one per scale), combined as a list

---

## 4. Stage 2: Flow-Matching Motion Generator

### 4.1 What is Flow Matching?

Flow Matching is a generative modeling technique (alternative to diffusion) that learns to transform noise into data by modeling the velocity field of an ODE:

```
x(0) = noise  ──velocity field──→  x(1) = motion latent
```

At training time:
1. Sample a real motion latent `m` and random noise `ε`
2. Create a noised version: `m_noised = t * m + (1-t) * ε` where `t ~ Uniform(0,1)`
3. The model predicts the flow: `predicted_flow = m - ε`
4. Loss = L1(predicted_flow, ground_truth_flow)

At inference time:
- Start from noise
- Use an ODE solver (Euler, 10 steps) to iteratively denoise
- Each step asks the model "what direction should I move?"

### 4.2 Architecture Components

#### A. Audio Encoder (`AudioEncoder`)
```
Input:  Raw audio waveform (B, num_samples)
Output: Audio features (B, T, 768)
```
- **Backbone**: Wav2Vec2-Base (frozen, no gradient)
- **Feature extraction**: Uses last hidden state (768-dim per frame)
- **Frame alignment**: Automatically resamples to match video FPS (25 fps)
- **Window**: 2 seconds = 50 frames per chunk

#### B. Condition Projections
All conditions are projected to the same dimension (`dim_c = 32`):
```
Audio:  (B, T, 768)  → Linear + LayerNorm + SiLU → (B, T, 32)
Gaze:   (B, T, 2)    → Linear + LayerNorm + SiLU → (B, T, 32)
Pose:   (B, T, 3)    → Linear + LayerNorm + SiLU → (B, T, 32)
Camera: (B, T, 3)    → Linear + LayerNorm + SiLU → (B, T, 32)
```
These are then **summed element-wise** (not concatenated) before being fed to the transformer.

#### C. Flow-Matching Transformer (`FlowMatchingTransformer`)
```
Input:  Noised motion (B, T, 32), conditions, timestep
Output: Predicted flow (B, T, 32)
```

**Architecture details**:
- **Input embedding**: `x_embedder` takes concat of [ref_motion, current_motion] = (B, T, 64) → Linear → (B, T, 512)
- **Timestep embedding**: Sinusoidal embedding (256-dim) → MLP → (B, 1, 512)
- **Condition embedding**: `c_embedder` maps (B, T, 32) → (B, T, 512)
- **Combined conditioning**: `c = timestep_embed + condition_embed` (additive)
- **Transformer blocks**: 8 FMTBlocks with:
  - AdaLN (Adaptive Layer Normalization) — modulates via 6 learned affine parameters from the condition
  - Multi-head self-attention (8 heads, 64 dim/head) with **RoPE** (Rotary Position Embeddings)
  - MLP (hidden = 4× = 2048) with GELU activation
- **Decoder**: AdaLN → Linear(512→32) — outputs predicted flow

**Temporal context**: The model uses `num_prev_frames = 10` frames of lookback context to maintain temporal coherence between chunks.

**Classifier-Free Guidance (CFG)**: During inference, audio can be dropped with `a_cfg_scale` to control how strongly audio influences the output. Scale of 3.0 provides good audio-lip sync.

### 4.3 Generator Training

```python
# 1. Sample real motion latent from dataset
m_now = batch["m_now"]           # (B, 50, 32) — 50 frames of motion

# 2. Create noised version (flow matching)
noise = randn_like(m_now)
t = rand(B)                      # Random timestep per sample
m_noised = t * m_now + (1-t) * noise
gt_flow = m_now - noise          # Ground truth flow

# 3. Model predicts the flow
pred_flow = model(m_noised, audio, pose, gaze, cam, t)

# 4. Losses
fm_loss = L1(pred_flow, gt_flow)
velocity_loss = L1(pred_flow[:, 1:] - pred_flow[:, :-1],
                   gt_flow[:, 1:] - gt_flow[:, :-1])  # Temporal smoothness
total_loss = fm_loss + velocity_loss
```

### 4.4 EMA (Exponential Moving Average)

The generator uses EMA with decay=0.9999 to maintain a smoothed copy of the weights:
- Updated after every training step
- Used during validation and inference
- Saved in checkpoints as `ema_state_dict`
- Provides more stable and higher quality outputs than raw training weights

---

## 5. The Motion Latent Space

The 32-dimensional motion latent vector is the **critical bridge** between the two stages:

```
Renderer's MotionEncoder:  Image → 32-dim vector (encodes motion)
Generator's output:        Audio → 32-dim vector (predicted motion)
```

### What the 32 dimensions encode
The latent space is learned, not hand-designed. But it captures:
- **Lip shape** (open/closed, width, shape)
- **Jaw position** (open amount)
- **Eye state** (open/closed, gaze direction)
- **Eyebrow position** (raised, furrowed)
- **Head rotation** (yaw, pitch, roll — though pose is also given as a separate condition)

### Why alignment matters
If you retrain the Renderer, its MotionEncoder changes → the latent space changes → old Generator outputs become meaningless. This is why:
1. Train Renderer first
2. Extract motion latents with the NEW Renderer
3. Train Generator on those new latents

---

## 6. Training Losses Explained

### 6.1 Renderer Losses

```python
total_g_loss = 1.0 * l1_loss        # Pixel reconstruction
             + 10.0 * vgg_loss_all   # Global perceptual similarity
             + 100.0 * vgg_loss_face # Face-region perceptual (HIGHEST weight)
             + 1.0 * dist_loss       # Identity disentanglement
             + 1.0 * g_gan_loss      # Adversarial (fool discriminator)
```

| Loss | Weight | What It Does | Healthy Range |
|---|---|---|---|
| **L1** | ×1 | Pixel-level reconstruction accuracy | 0.02-0.05 (finetune), 0.15-0.25 (scratch) |
| **VGG All** | ×10 | Multi-scale perceptual similarity (global) | 3-5 (finetune), 13-17 (scratch) |
| **VGG Face** | ×100 | Perceptual similarity in eye+mouth regions | 1,000-2,000 (finetune), 4,000-7,000 (scratch) |
| **Dist** | ×1 | \|‖t₁-ta₁₁‖ - ‖t₁-ta₁₂‖\| — identity distance constraint | 0.01-0.3 |
| **G GAN** | ×1 | Generator adversarial loss (fool D) | Rises over training (normal) |
| **D Loss** | - | Discriminator loss | Should oscillate, not collapse to 0 |

#### VGG Loss Details
- Uses VGG19 pretrained on ImageNet (frozen)
- Computes L1 distance between feature maps at 5 layers
- **Global**: Full image comparison at 4 scales (1.0, 0.5, 0.25, 0.125)
- **Face-masked**: Same but weighted by eye+mouth binary mask → forces high fidelity in critical face regions

#### Distance Loss (Identity Disentanglement)
```
t_1 = motion of target frame
ta_11 = id_adapt(t_1, id_same)     → motion in same person's space
ta_12 = id_adapt(t_1, id_different) → motion in different person's space

dist_loss = |‖t_1 - ta_11‖ - ‖t_1 - ta_12‖|
```
This ensures the identity-adaptive module creates DIFFERENT projections for different people — preventing identity information from leaking into the motion space.

### 6.2 Generator Losses

```python
total_loss = fm_loss + velocity_loss
```

| Loss | What It Does | Healthy Range |
|---|---|---|
| **FM Loss** | L1 between predicted and ground-truth flow | 0.3-0.7 |
| **Velocity Loss** | Temporal smoothness of flow predictions | Included in fm_loss range |

The generator loss is much simpler because it operates in latent space (32-dim) rather than pixel space (512×512×3).

---

## 7. Data Pipeline

### 7.1 Renderer Dataset

**Input**: Raw MP4 videos
**Preprocessing** (`prepare_renderer_data.py`):
1. **Face detection**: Sample 3 frames per video, detect face bounding box
2. **Cropping**: Square crop around face (1.6× face size), resize to 512×512
3. **Frame extraction**: FFmpeg at 25 fps
4. **Landmark extraction**: face_alignment library, 68 landmarks per frame

**Output structure**:
```
renderer_dataset/
├── video_frame/
│   ├── video_001/
│   │   ├── image_0001.jpg    # 512×512 cropped face
│   │   ├── image_0002.jpg
│   │   └── ...
│   └── video_002/
│       └── ...
└── lmd/
    ├── video_001.txt          # 68 landmarks per frame
    └── video_002.txt
```

**Dataset class** (`TFDataset`):
- Randomly samples 2 frames from same video (ref + target)
- Randomly samples 1 frame from different video (negative)
- Creates eye+mouth binary masks from landmarks
- 95% train / 5% validation split

### 7.2 Generator Dataset

**Input**: Renderer dataset + trained renderer checkpoint + raw MP4s
**Preprocessing** (`prepare_generator_data.py`):
1. **Motion latents**: Run each frame through renderer's `latent_token_encoder` → (N, 32) per video
2. **Audio features**: Extract wav2vec2 last hidden state → (N, 768) per video
3. **Pose**: Estimated from landmarks (pitch, yaw, roll) → (N, 3)
4. **Camera**: Estimated from landmarks (scale, x_offset, y_offset) → (N, 3)
5. **Gaze**: Estimated from eye landmarks → (N, 2)

**Output structure**:
```
generator_dataset/
├── motion/video_001.pt        # (N, 32) motion latents
├── audio/video_001.npy        # (N, 768) wav2vec2 features
├── smirk/video_001.pt         # {pose_params: (N,3), cam: (N,3)}
└── gaze/video_001.npy         # (N, 2) gaze direction
```

**Dataset class** (`AudioMotionSmirkGazeDataset`):
- Clips 2-second windows (50 frames) + 10 prev frames for context
- Random start position within each video
- Returns: `m_now`, `a_now`, `gaze`, `pose`, `cam`, `m_prev`, `a_prev`, `gaze_prev`, `pose_prev`, `cam_prev`, `m_ref`

---

## 8. Inference Pipeline

### Audio-Driven Generation
```
Input: Portrait image + Audio file
                │                    │
                ▼                    ▼
        IdentityEncoder        Wav2Vec2 (frozen)
         ↓ features              ↓ audio features
         ↓ id_vector             ↓
         │              ┌─── FlowMatchingTransformer ◄── pose, gaze (optional)
         │              │    (ODE solver, 10 steps)
         │              ↓
         │         Motion latents (T, 32)
         │              │
         │              ↓
         │        IdentityAdaptive
         │              │
         │              ↓
         │        MotionDecoder
         │              │
         │              ↓
         └──────► CrossAttention IMT
                        │
                        ▼
                  SynthesisNetwork
                        │
                        ▼
                  Output frames (T, 3, 512, 512)
```

### Chunked Generation
For long audio, the generator processes in 2-second chunks (50 frames) with 10-frame overlap for smooth transitions:
- Chunk 0: noise → ODE → frames 1-50 (no previous context)
- Chunk 1: noise → ODE → frames 51-100 (uses last 10 frames from chunk 0 as context)
- ...

### Classifier-Free Guidance
At inference, audio CFG scale (default 3.0) controls lip-sync strength:
- `scale=1.0`: No guidance (natural but may have weak lip sync)
- `scale=3.0`: Default (good balance)
- `scale=5.0+`: Strong lip sync but may look over-articulated

---

## 9. Finetuning Guide

### 9.1 Why Finetune vs Train from Scratch

| Aspect | From Scratch | Finetuning |
|---|---|---|
| Starting L1 loss | ~0.27 | ~0.03 |
| Starting VGG face | ~7,000 | ~1,700 |
| Convergence time | 50k+ steps | 10-20k steps |
| Quality risk | May not converge well | Retains pretrained quality |
| Data requirement | Large diverse dataset | Can work with domain-specific data |

### 9.2 Finetuning Configuration

#### Renderer Finetuning
```bash
# Key settings
LR=1e-5              # 10× lower than original 1e-4
RESUME_CKPT=renderer.ckpt   # Pretrained on VFHQ
ITERATIONS=50000
BATCH_SIZE=4          # Per GPU (4 GPUs = effective 16)
```

**What gets loaded**:
- Generator (IMTRenderer): All 827 parameters ✅
- Discriminator (PatchDiscriminator): All 107 parameters ✅
- VGG (perceptual loss): From torchvision pretrained (frozen) ✅
- Optimizer: Created FRESH with new LR (no stale momentum) ✅

**Learning rate schedule**:
- Cosine annealing: 1e-5 → 1e-7 over 50k steps
- Both G and D use the same LR schedule

#### Generator Finetuning
```bash
# Key settings
LR=2e-5              # 5× lower than original 1e-4
RESUME_CKPT=generator.ckpt  # Pretrained
ITERATIONS=50000
BATCH_SIZE=16         # Per GPU (4 GPUs = effective 64)
```

**What gets loaded**:
- FMGenerator: EMA weights preferred (smoother) — 108 parameters ✅
- Wav2Vec2: Stays frozen (no training) ✅
- EMA: Re-registered from loaded weights ✅
- Optimizer: Created FRESH with new LR ✅

**Learning rate schedule**:
- Cosine annealing: 2e-5 → 2e-7 over 50k steps

### 9.3 Correct Finetuning Order

```
1. Renderer Data Prep     ← Only once (already done, skips)
2. Renderer Finetuning    ← Adapts renderer to your faces
3. Generator Data Prep    ← Extracts NEW motion latents from YOUR renderer
4. Generator Finetuning   ← Adapts generator to YOUR motion space
```

⚠️ **NEVER skip step 3** — if you finetune the renderer and reuse old generator data, the generator will produce motion latents that don't match the finetuned renderer's latent space.

### 9.4 Expected Loss Behavior During Finetuning

**Renderer** (starting from pretrained):
- L1: Starts ~0.03, may initially rise to ~0.04-0.05 as model adapts to new data distribution, then settles at ~0.02-0.03
- VGG face: Starts ~1,700, should stabilize at ~1,000-1,500 after adaptation
- D loss: Starts ~4.0 (healthy), should oscillate between 0.5-5.0
- G GAN: Should stabilize around 5-15

**Generator** (starting from pretrained):
- FM loss: Starts ~0.3-0.5, should decrease to ~0.2-0.3
- Val loss: Should track train loss closely (if diverging = overfitting)

### 9.5 Signs of Trouble

| Symptom | Cause | Fix |
|---|---|---|
| L1 suddenly spikes to >0.5 | LR too high | Lower LR by 2-5× |
| D loss → 0 and stays there | D collapsed | Lower D LR or increase G LR |
| D loss oscillates wildly (>10) | GAN instability | Lower both LRs |
| VGG face increases over time | Model forgetting | Lower LR, add warmup |
| Val loss >> train loss | Overfitting | More data, augmentation, or fewer steps |

---

## 10. Hyperparameter Reference

### 10.1 Renderer Architecture

| Parameter | Value | Description |
|---|---|---|
| `depth` | 2 | Depth of attention blocks |
| `latent_dim` | 32 | Motion latent dimension |
| `swin_res_threshold` | 128 | Resolution below which to use Swin attention |
| `num_heads` | 8 | Attention heads |
| `window_size` | 8 | Swin attention window size |
| `drop_path` | 0.1 | Stochastic depth rate |
| `low_res_depth` | 2 | Depth for low-res attention blocks |
| Feature dims | [32, 64, 128, 256, 512, 512] | Channel dims per spatial level |
| Spatial dims | [256, 128, 64, 32, 16, 8] | Spatial resolution per level |

### 10.2 Generator Architecture

| Parameter | Value | Description |
|---|---|---|
| `dim_motion` | 32 | Motion latent dimension (must match renderer) |
| `dim_c` | 32 | Condition projection dimension |
| `dim_h` | 512 | Transformer hidden dimension |
| `dim_w` | 32 | Output dimension |
| `fmt_depth` | 8 | Number of transformer blocks |
| `num_heads` | 8 | Attention heads |
| `mlp_ratio` | 4.0 | MLP expansion ratio |
| `num_prev_frames` | 10 | Temporal context frames |
| `wav2vec_sec` | 2.0 | Audio window (seconds) = 50 frames |
| `fps` | 25.0 | Video frame rate |
| `sampling_rate` | 16000 | Audio sample rate |

### 10.3 Training Hyperparameters

| Parameter | Renderer (finetune) | Generator (finetune) |
|---|---|---|
| Learning rate | 1e-5 | 2e-5 |
| LR schedule | Cosine → 1e-7 | Cosine → 2e-7 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) | Adam (β₁=0.5, β₂=0.999) |
| Batch size | 4 per GPU | 16 per GPU |
| Max steps | 50,000 | 50,000 |
| Grad clip | 1.0 | 1.0 (max_grad_norm) |
| DDP strategy | ddp_find_unused_parameters_true | ddp_find_unused_parameters_true |
| EMA | No | Yes (decay=0.9999) |

### 10.4 Loss Weights

| Loss | Weight | Purpose |
|---|---|---|
| `loss_l1` | 1.0 | Pixel reconstruction |
| `loss_vgg_all` | 10.0 | Global perceptual |
| `loss_vgg_face` | 100.0 | Face-region perceptual |
| `loss_dist` | 1.0 | Identity disentanglement |
| `gan_weight` | 1.0 | Adversarial loss scale |
| `audio_dropout_prob` | 0.1 | Audio/condition dropout (enables CFG) |

### 10.5 Inference Parameters

| Parameter | Value | Description |
|---|---|---|
| `nfe` | 10 | ODE solver steps (more = better quality, slower) |
| `ode_method` | euler | ODE solver type |
| `a_cfg_scale` | 3.0 | Audio classifier-free guidance strength |
| `ode_atol` | 1e-5 | ODE absolute tolerance |
| `ode_rtol` | 1e-5 | ODE relative tolerance |

---

## 11. Troubleshooting & Common Issues

### 11.1 Data Preparation Issues

**Problem**: `ValueError: inhomogeneous shape` in landmark files
**Cause**: Some landmark files have lines with != 68 coordinate pairs
**Fix**: The dataset's `read_landmark_info` pads/truncates to exactly 68 landmarks per frame

**Problem**: `no_face` returned for many videos
**Cause**: Face detector can't find faces (side profiles, occlusions, very small faces)
**Fix**: Increase `max_samples` in face detection, or filter out problematic videos

### 11.2 Training Issues

**Problem**: `checkpoint_callback` undefined
**Cause**: Missing ModelCheckpoint definition in renderer training script
**Fix**: Add `pl.callbacks.ModelCheckpoint(...)` before trainer initialization

**Problem**: Cosine LR scheduler not decaying
**Cause**: `eta_min` set equal to `lr` (e.g., both 1e-4)
**Fix**: Set `eta_min = lr * 0.01` for actual decay

**Problem**: Training stops silently with `| tee`
**Cause**: `set -e` doesn't catch errors through pipes; `set -eo pipefail` required
**Fix**: Use `set -eo pipefail` at top of shell scripts

**Problem**: Generator trained on wrong motion latents
**Cause**: Generator data was extracted with old/pretrained renderer, not the finetuned one
**Fix**: Always re-extract generator data after renderer finetuning

### 11.3 Inference Issues

**Problem**: Generated face doesn't match input identity
**Cause**: Identity leakage in motion latent space, or insufficient finetuning
**Fix**: Check dist_loss is low (<0.3), finetune longer

**Problem**: Lip sync is weak
**Cause**: Audio CFG scale too low, or audio features not aligned
**Fix**: Increase `a_cfg_scale` (try 3.0-5.0)

**Problem**: Temporal jitter between frames
**Cause**: Insufficient `num_prev_frames` or ODE steps
**Fix**: Increase `nfe` to 15-20 for smoother results

---

## Appendix: File Locations

```
/workspace/IMTalker/
├── renderer/
│   ├── train.py              # Renderer training script
│   ├── models.py             # IMTRenderer architecture
│   ├── discriminator.py      # PatchDiscriminator
│   ├── dataset.py            # TFDataset
│   ├── vgg19_mask.py         # VGGLoss_mask
│   ├── attention_modules.py  # CrossAttention, SelfAttention
│   └── lia_resblocks.py      # StyledConv, EqualConv2d, EqualLinear
├── generator/
│   ├── train.py              # Generator training script
│   ├── FM.py                 # FMGenerator + AudioEncoder
│   ├── FMT.py                # FlowMatchingTransformer
│   ├── dataset.py            # AudioMotionSmirkGazeDataset
│   ├── wav2vec2.py           # Wav2VecModel wrapper
│   └── options/base_options.py
├── checkpoints/
│   ├── renderer.ckpt         # Pretrained renderer (VFHQ)
│   ├── generator.ckpt        # Pretrained generator
│   └── wav2vec2-base-960h/   # Wav2Vec2 model
├── app.py                    # Inference / demo script
└── assets/                   # Sample inputs

/workspace/
├── prepare_renderer_data.py   # Multi-GPU renderer data prep
├── prepare_generator_data.py  # Multi-GPU generator data prep
├── train_renderer.sh          # Renderer training launcher
├── train_generator.sh         # Generator training launcher
├── launch_full_training.sh    # Master pipeline orchestrator
├── liveavatar_data/
│   ├── mp4/                   # Raw training videos (15,743)
│   ├── renderer_dataset/      # Preprocessed frames + landmarks
│   └── generator_dataset/     # Pre-extracted features
└── exps/
    ├── custom_renderer/       # Finetuned renderer outputs
    │   ├── checkpoints/
    │   └── loss_plots/
    └── custom_generator/      # Finetuned generator outputs
        ├── checkpoints/
        └── loss_plots/
```

---

*Document generated: February 10, 2026*
*Hardware: 4× NVIDIA A100-SXM4-80GB, 128 CPU cores, 1.7TB RAM*
*Dataset: 15,743 custom videos*
