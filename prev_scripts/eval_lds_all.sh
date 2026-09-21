#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/video_ssd/kongzishang/imagenet_val"
CONFIG_DIR="./configs"
LOG_DIR="/video_ssd/kongzishang/xutongda/git/IFID/losses/lds_logs"
OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/losses/lds"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

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

NUM_GPUS=8
BASE_PORT=29800

for ((i=0; i<${#CONFIGS[@]}; i++)); do
  CONFIG="${CONFIGS[$i]}"
  GPU_ID=$((i % NUM_GPUS))
  PORT=$((BASE_PORT + GPU_ID))

  NAME="${CONFIG%.yaml}"
  NAME_LOWER="$(echo "${NAME}" | tr '[:upper:]' '[:lower:]')"

  OUT_DIR="${OUT_ROOT}/${NAME_LOWER}"
  OUT_FILE="${LOG_DIR}/eval_lds_${NAME_LOWER}.out"

  mkdir -p "${OUT_DIR}"

  echo "Launching eval_lds.py with ${CONFIG} on GPU ${GPU_ID}, port ${PORT}"
  echo "Out dir: ${OUT_DIR}"
  echo "Log: ${OUT_FILE}"

  nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch \
    --num_processes=1 \
    --main_process_port ${PORT} \
    eval_lds.py \
    --bs 8 \
    --num_workers 8 \
    --base_path "${BASE_PATH}" \
    --vae-config "${CONFIG_DIR}/${CONFIG}" \
    --out_dir "${OUT_DIR}" \
    --num_images 50000 \
    > "${OUT_FILE}" 2>&1 &

  if (( (i + 1) % NUM_GPUS == 0 )); then
    echo "Waiting for current batch of ${NUM_GPUS} jobs..."
    wait
    echo "Current batch finished."
  fi
done

wait

echo "All eval_lds jobs finished."