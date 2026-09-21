#!/usr/bin/env bash
set -euo pipefail

TRAIN_DATA="/video_ssd/lpm/ImageNet/train"
VAL_DATA="/video_ssd/lpm/ImageNet/val"

CONFIG_DIR="./configs"
SCRIPT="eval_linear_vae.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/losses/linear_probe_gap"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${OUT_ROOT}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "SDVAE.yaml"
  "SD3VAE.yaml"
  "FLUXVAE.yaml"
  "DCAE.yaml"
  "SOFTVQ.yaml"
  "VAVAE.yaml"
  "VAVAE64.yaml"
  "EQVAE.yaml"
  "MAETOK.yaml"
  "INVAE.yaml"
  "REPAEVAE.yaml"
  "DETOK.yaml"
  "QWVAE.yaml"
  "RAE.yaml"
  "SVG.yaml"
  "FLUX2VAE.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
  "DMVAE.yaml"
  "UAE.yaml"
  "VTPS.yaml"
  "VTPB.yaml"
  "VTPL.yaml"
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE_WEBSSL.yaml"
  "PAE_DINOv2L_d32.yaml"
)

NUM_GPUS=4
BASE_PORT=29800

EPOCHS=100
BATCH_SIZE=128
LR=0.001
RESOLUTION=256
NUM_WORKERS=4
MIXED_PRECISION="no"

REPORT_TO="wandb"
# REPORT_TO="none"

TRAIN_SIZE=-1
VAL_SIZE=-1

FEATURE_TYPE="gap"

for ((i=0; i<${#CONFIGS[@]}; i++)); do
  CONFIG="${CONFIGS[$i]}"
  GPU_ID=$((i % NUM_GPUS))
  PORT=$((BASE_PORT + GPU_ID))

  NAME="${CONFIG%.yaml}"
  NAME_LOWER="$(echo "${NAME}" | tr '[:upper:]' '[:lower:]')"

  EXP_NAME="linear_${NAME_LOWER}_${FEATURE_TYPE}"
  OUT_DIR="${OUT_ROOT}/${EXP_NAME}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  mkdir -p "${OUT_DIR}"

  echo "Launching GAP linear probe:"
  echo "  config:       ${CONFIG}"
  echo "  gpu:          ${GPU_ID}"
  echo "  port:         ${PORT}"
  echo "  feature_type: ${FEATURE_TYPE}"
  echo "  batch_size:   ${BATCH_SIZE}"
  echo "  out_dir:      ${OUT_DIR}"
  echo "  log:          ${OUT_FILE}"

  nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch \
    --num_processes=1 \
    --main_process_port "${PORT}" \
    "${SCRIPT}" \
    --vae-config "${CONFIG_DIR}/${CONFIG}" \
    --train-data "${TRAIN_DATA}" \
    --val-data "${VAL_DATA}" \
    --output-dir "${OUT_DIR}" \
    --exp-name "${EXP_NAME}" \
    --feature-type "${FEATURE_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --resolution "${RESOLUTION}" \
    --num-workers "${NUM_WORKERS}" \
    --train-size "${TRAIN_SIZE}" \
    --val-size "${VAL_SIZE}" \
    --mixed-precision "${MIXED_PRECISION}" \
    --report-to "${REPORT_TO}" \
    > "${OUT_FILE}" 2>&1 &

  if (( (i + 1) % NUM_GPUS == 0 )); then
    echo "Waiting for current batch of ${NUM_GPUS} jobs..."
    wait
    echo "Current batch finished."
  fi
done

wait

echo "All GAP linear probe jobs finished."