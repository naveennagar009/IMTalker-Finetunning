#!/bin/bash

# Training arguments — FINETUNING from pretrained checkpoint
DATASET_PATH="/workspace/liveavatar_data/renderer_dataset"
EXP_NAME="custom_renderer"
BATCH_SIZE=4
ITERATIONS=50000
SAVE_FREQ=1000
LR=1e-5  # 10x lower than default (1e-4) for smooth finetuning

echo "Starting Renderer Finetuning on 4x A100 GPUs (LR=$LR)..."
export PATH=/opt/conda/envs/IMTalker/bin:$PATH
export PYTHONPATH="/workspace/IMTalker:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/opt/conda/envs/IMTalker/bin/python /workspace/IMTalker/renderer/train.py \
    --dataset_path "$DATASET_PATH" \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --iter "$ITERATIONS" \
    --save_freq "$SAVE_FREQ" \
    --display_freq 1000 \
    --lr "$LR" \
    --resume_ckpt "/workspace/IMTalker/checkpoints/renderer.ckpt"
