#!/bin/bash
set -eo pipefail

echo "[$(date)] Starting Multi-Phase Training Pipeline for 15k videos..."
echo "Order: Renderer Prep → Renderer Train → Generator Prep (with NEW renderer) → Generator Train"

# Ensure IMTalker environment is used
export PATH=/opt/conda/envs/IMTalker/bin:$PATH
export PYTHONPATH="/workspace/IMTalker:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Phase 1: Renderer Data Prep (already done, will skip) ──
echo "[$(date)] Phase 1: Preparing Renderer Dataset..."
python /workspace/prepare_renderer_data.py \
    --input_dir "/workspace/liveavatar_data/mp4" \
    --output_dir "/workspace/liveavatar_data/renderer_dataset" \
    --num_gpus 4 --threads_per_gpu 24 2>&1 | tee /workspace/full_prep_renderer.log

# ── Phase 2: Renderer Training (50k steps) ──
echo "[$(date)] Phase 2: Training Renderer (50k steps)..."
bash /workspace/train_renderer.sh 2>&1 | tee /workspace/renderer_train.log

# ── Phase 3: Generator Data Prep (uses NEW custom renderer checkpoint) ──
# Find the best renderer checkpoint from training
RENDERER_CKPT=$(ls -t /workspace/exps/custom_renderer/checkpoints/last.ckpt 2>/dev/null || echo "")
if [ -z "$RENDERER_CKPT" ]; then
    echo "[ERROR] No renderer checkpoint found! Cannot proceed with generator data prep."
    exit 1
fi
echo "[$(date)] Phase 3: Preparing Generator Dataset using custom renderer: $RENDERER_CKPT"
python /workspace/prepare_generator_data.py \
    --renderer_data_dir "/workspace/liveavatar_data/renderer_dataset" \
    --mp4_dir "/workspace/liveavatar_data/mp4" \
    --output_dir "/workspace/liveavatar_data/generator_dataset" \
    --renderer_ckpt "$RENDERER_CKPT" \
    --wav2vec_path "/workspace/IMTalker/checkpoints/wav2vec2-base-960h" \
    --num_gpus 4 --threads_per_gpu 8 2>&1 | tee /workspace/full_prep_generator.log

# ── Phase 4: Generator Training (50k steps) ──
echo "[$(date)] Phase 4: Training Generator (50k steps)..."
bash /workspace/train_generator.sh 2>&1 | tee /workspace/generator_train.log

echo "[$(date)] Training Pipeline Completed Successfully!"
