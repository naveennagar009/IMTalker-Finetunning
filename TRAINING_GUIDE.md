# IMTalker Half-Body Fine-tuning — Training Guide

## Server Requirements

- **GPU**: 4x A100 (80GB) recommended. 2x A100 minimum.
- **Disk**: ~2TB for 298K videos + processed datasets
- **CUDA**: 12.1
- **Python**: 3.10

## Environment Setup

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate IMTalker
```

### Option B: Pip

```bash
conda create -n IMTalker python=3.10 -y
conda activate IMTalker

# Install ffmpeg via conda (required for video processing)
conda install -c conda-forge ffmpeg -y

# PyTorch (CUDA 12.1)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import face_alignment; print('face_alignment OK')"
ffmpeg -version | head -1
```

## Data Layout

```
/workspace/Final_Data/
├── mp4/          # 298K source videos (input)
├── mp3/          # audio files (unused by training)
├── renderer_dataset/    # created by Phase 1
│   ├── video_frame/     #   512x512 cropped frames per video
│   ├── lmd/             #   68-point landmarks per video
│   └── face_bbox/       #   face bbox JSON per video (halfbody only)
└── generator_dataset/   # created by Phase 3
    ├── motion/          #   32-d latents per video
    ├── audio/           #   wav2vec2 features per video
    ├── smirk/           #   pose params per video
    └── gaze/            #   gaze direction per video
```

## Pretrained Checkpoints

Place pretrained IMTalker checkpoints before training:

```
IMTalker/checkpoints/
├── renderer.ckpt              # pretrained face renderer
├── generator.ckpt             # pretrained face generator
├── config.yaml
└── wav2vec2-base-960h/        # wav2vec2 weights
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```

## Running the Full Pipeline

### Half-body training (recommended)

```bash
HALFBODY=1 bash launch_full_training.sh
```

### Face-only training (baseline)

```bash
bash launch_full_training.sh
```

The pipeline runs 4 phases sequentially:

| Phase | Script | Duration (298K videos, 4x A100) | Output |
|-------|--------|----------------------------------|--------|
| 1. Data prep | `prepare_renderer_data.py` | ~12-24 hours | `renderer_dataset/` |
| 2. Renderer training | `train_renderer.sh` | ~2-4 weeks (50K steps) | `exps/halfbody_renderer/` |
| 3. Generator data prep | `prepare_generator_data.py` | ~12-24 hours | `generator_dataset/` |
| 4. Generator training | `train_generator.sh` | ~6-12 hours (50K steps) | `exps/halfbody_generator/` |

## Running Phases Individually

### Phase 1: Data preparation

```bash
python prepare_renderer_data.py \
    --input_dir /workspace/Final_Data/mp4 \
    --output_dir /workspace/Final_Data/renderer_dataset \
    --num_gpus 4 --threads_per_gpu 24 \
    --halfbody
```

### Phase 2: Renderer training

```bash
HALFBODY=1 bash train_renderer.sh
```

Training params (edit `train_renderer.sh`):
- `LR=1e-5` (10x lower than original — protects pretrained face quality)
- `ITERATIONS=50000`
- `BATCH_SIZE=4`
- Half-body losses: `--face_weight 3.0 --body_weight 1.0 --face_disc_weight 2.0`

### Phase 3: Generator data extraction

```bash
python prepare_generator_data.py \
    --renderer_data_dir /workspace/Final_Data/renderer_dataset \
    --mp4_dir /workspace/Final_Data/mp4 \
    --output_dir /workspace/Final_Data/generator_dataset \
    --renderer_ckpt ./exps/halfbody_renderer/checkpoints/last.ckpt \
    --wav2vec_path ./IMTalker/checkpoints/wav2vec2-base-960h \
    --num_gpus 4 --threads_per_gpu 8
```

### Phase 4: Generator training

```bash
HALFBODY=1 bash train_generator.sh
```

## Monitoring Training

TensorBoard logs are saved under `exps/<exp_name>/`:

```bash
tensorboard --logdir exps/ --port 6006 --bind_all
```

Loss plots are auto-saved to `exps/<exp_name>/loss_plots/`.

Checkpoints saved every 1000 steps (renderer) / 2000 steps (generator) under `exps/<exp_name>/checkpoints/`.

## Resuming Training

If training is interrupted, it resumes from the last Lightning checkpoint automatically. Just re-run the same command — Lightning picks up `last.ckpt`.

To resume from a specific checkpoint, edit the `--resume_ckpt` path in the training script.

## Inference

```bash
cd IMTalker
python generator/generate.py \
    --renderer_ckpt ../exps/halfbody_renderer/checkpoints/last.ckpt \
    --generator_ckpt ../exps/halfbody_generator/checkpoints/last.ckpt \
    --reference_image halfbody_photo.jpg \
    --audio input.wav \
    --output output.mp4
```

## Half-body vs Face-only: Key Differences

| Aspect | Face-only | Half-body |
|--------|-----------|-----------|
| Crop multiplier | 1.6x face bbox | 3.2x face bbox |
| L1 loss | Uniform | Face 3x, body 1x |
| Discriminator | Global only | Global + face-crop via `disc.scale2` |
| Generator LR | Uniform | Per-module (encoders 0.1x, decoders 0.5x) |
| Extra VGG loss | No | Face-crop perceptual loss |
| Landmark hint | Full frame | Tight face bbox |
| Reference image | Face crop | Half-body photo |

## Troubleshooting

**OOM during renderer training**: Reduce `BATCH_SIZE` from 4 to 2 in `train_renderer.sh`.

**Slow data prep**: Increase `--threads_per_gpu` (max ~32 on A100 nodes). Data prep is CPU/IO bound.

**Bad landmarks in half-body mode**: The `face_bbox` JSON should contain valid face bounding boxes. If a video has no detectable face, it's skipped during data prep.

**Generator training diverges**: Lower LR from `5e-5` to `2e-5`. The generator is lightweight and converges quickly.
