#!/bin/bash

# Training arguments — FINETUNING from pretrained checkpoint
DATASET_PATH="/workspace/liveavatar_data/generator_dataset"
EXP_NAME="custom_generator"
BATCH_SIZE=16
ITERATIONS=50000
SAVE_FREQ=2000
LR=2e-5  # 5x lower than default (1e-4) for smooth finetuning

echo "Starting Generator Finetuning on 4x A100 GPUs (LR=$LR)..."
export PYTHONPATH="/workspace/IMTalker:$PYTHONPATH"
python /workspace/IMTalker/generator/train.py \
    --wav2vec_model_path "/workspace/IMTalker/checkpoints/wav2vec2-base-960h" \
    --dataset_path "$DATASET_PATH" \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --iter "$ITERATIONS" \
    --save_freq "$SAVE_FREQ" \
    --display_freq 1000 \
    --lr "$LR" \
    --resume_ckpt "/workspace/IMTalker/checkpoints/generator.ckpt"
