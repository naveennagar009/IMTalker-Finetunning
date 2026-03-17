#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/workspace/Final_Data"

HALFBODY=${HALFBODY:-0}
DATASET_PATH="${DATA_ROOT}/generator_dataset"
BATCH_SIZE=16
ITERATIONS=50000
SAVE_FREQ=2000

if [ "$HALFBODY" = "1" ]; then
    EXP_NAME="halfbody_generator"
    LR=5e-5
    echo "Starting Half-Body Generator Finetuning (LR=$LR)..."
else
    EXP_NAME="custom_generator"
    LR=2e-5
    echo "Starting Face-Only Generator Finetuning (LR=$LR)..."
fi

export PYTHONPATH="${REPO_ROOT}/IMTalker:$PYTHONPATH"
python "${REPO_ROOT}/IMTalker/generator/train.py" \
    --wav2vec_model_path "${REPO_ROOT}/IMTalker/checkpoints/wav2vec2-base-960h" \
    --dataset_path "$DATASET_PATH" \
    --exp_name "$EXP_NAME" \
    --exp_path "${REPO_ROOT}/exps" \
    --batch_size "$BATCH_SIZE" \
    --iter "$ITERATIONS" \
    --save_freq "$SAVE_FREQ" \
    --display_freq 1000 \
    --lr "$LR" \
    --resume_ckpt "${REPO_ROOT}/IMTalker/checkpoints/generator.ckpt"
