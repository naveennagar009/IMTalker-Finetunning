#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/workspace/Final_Data"

HALFBODY=${HALFBODY:-0}
DATASET_PATH="${DATA_ROOT}/renderer_dataset"
BATCH_SIZE=4
ITERATIONS=50000
SAVE_FREQ=1000
LR=1e-5

if [ "$HALFBODY" = "1" ]; then
    EXP_NAME="halfbody_renderer"
    HALFBODY_FLAGS="--halfbody --face_weight 3.0 --body_weight 1.0 --face_disc_weight 2.0"
    echo "Starting Half-Body Renderer Finetuning (LR=$LR)..."
else
    EXP_NAME="custom_renderer"
    HALFBODY_FLAGS=""
    echo "Starting Face-Only Renderer Finetuning (LR=$LR)..."
fi

export PYTHONPATH="${REPO_ROOT}/IMTalker:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python "${REPO_ROOT}/IMTalker/renderer/train.py" \
    --dataset_path "$DATASET_PATH" \
    --exp_name "$EXP_NAME" \
    --exp_path "${REPO_ROOT}/exps" \
    --batch_size "$BATCH_SIZE" \
    --iter "$ITERATIONS" \
    --save_freq "$SAVE_FREQ" \
    --display_freq 1000 \
    --lr "$LR" \
    --resume_ckpt "${REPO_ROOT}/IMTalker/checkpoints/renderer.ckpt" \
    $HALFBODY_FLAGS
