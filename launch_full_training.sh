#!/bin/bash
set -eo pipefail

# ── Paths (edit these) ──
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/workspace/Final_Data"
MP4_DIR="${DATA_ROOT}/mp4"
RENDERER_DATASET="${DATA_ROOT}/renderer_dataset"
GENERATOR_DATASET="${DATA_ROOT}/generator_dataset"
CHECKPOINTS="${REPO_ROOT}/IMTalker/checkpoints"
NUM_GPUS=4

# Toggle: set HALFBODY=1 for half-body pipeline, 0 for face-only
export HALFBODY=${HALFBODY:-0}

if [ "$HALFBODY" = "1" ]; then
    echo "[$(date)] Starting Half-Body Training Pipeline..."
    RENDERER_EXP="halfbody_renderer"
    PREP_FLAGS="--halfbody"
else
    echo "[$(date)] Starting Face-Only Training Pipeline..."
    RENDERER_EXP="custom_renderer"
    PREP_FLAGS=""
fi
echo "Order: Renderer Prep → Renderer Train → Generator Prep (with NEW renderer) → Generator Train"
echo "Data: ${MP4_DIR} ($(find ${MP4_DIR} -name '*.mp4' | head -1 | xargs dirname))"

export PYTHONPATH="${REPO_ROOT}/IMTalker:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Phase 1: Renderer Data Prep ──
echo "[$(date)] Phase 1: Preparing Renderer Dataset..."
python "${REPO_ROOT}/prepare_renderer_data.py" \
    --input_dir "${MP4_DIR}" \
    --output_dir "${RENDERER_DATASET}" \
    --num_gpus ${NUM_GPUS} --threads_per_gpu 24 $PREP_FLAGS 2>&1 | tee "${REPO_ROOT}/full_prep_renderer.log"

# ── Phase 2: Renderer Training (50k steps) ──
echo "[$(date)] Phase 2: Training Renderer (50k steps)..."
bash "${REPO_ROOT}/train_renderer.sh" 2>&1 | tee "${REPO_ROOT}/renderer_train.log"

# ── Phase 3: Generator Data Prep (uses NEW renderer checkpoint) ──
RENDERER_CKPT=$(ls -t "${REPO_ROOT}/exps/${RENDERER_EXP}/checkpoints/last.ckpt" 2>/dev/null || echo "")
if [ -z "$RENDERER_CKPT" ]; then
    echo "[ERROR] No renderer checkpoint found at ${REPO_ROOT}/exps/${RENDERER_EXP}/checkpoints/"
    exit 1
fi
echo "[$(date)] Phase 3: Preparing Generator Dataset using renderer: $RENDERER_CKPT"
python "${REPO_ROOT}/prepare_generator_data.py" \
    --renderer_data_dir "${RENDERER_DATASET}" \
    --mp4_dir "${MP4_DIR}" \
    --output_dir "${GENERATOR_DATASET}" \
    --renderer_ckpt "$RENDERER_CKPT" \
    --wav2vec_path "${CHECKPOINTS}/wav2vec2-base-960h" \
    --num_gpus ${NUM_GPUS} --threads_per_gpu 8 2>&1 | tee "${REPO_ROOT}/full_prep_generator.log"

# ── Phase 4: Generator Training (50k steps) ──
echo "[$(date)] Phase 4: Training Generator (50k steps)..."
bash "${REPO_ROOT}/train_generator.sh" 2>&1 | tee "${REPO_ROOT}/generator_train.log"

echo "[$(date)] Training Pipeline Completed Successfully!"
