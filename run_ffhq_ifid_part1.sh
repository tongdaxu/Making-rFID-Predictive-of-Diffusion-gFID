#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

SAMPLE_DIR="./samples_ffhq"
DATASET="/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/imgs"
DATASET_REF="/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/imgs"
CONFIG_DIR="./configs"

SMALL="${SMALL:-70000}"
SMALL_VAL="${SMALL_VAL:-70000}"

SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_MAX="${SAVE_MAX:-100}"

LOG_DIR="./nohup_logs_ffhq"
mkdir -p "${LOG_DIR}"

JOBS=(
  # "SD-VAE|SDVAE.yaml|ffhq-ifid-sdvae"
  # "FLUX-VAE|FLUXVAE.yaml|ffhq-ifid-fluxvae"
  # "QwenImg-VAE|QWVAE.yaml|ffhq-ifid-qwenimgvae"
  # "SD3-VAE|SD3VAE.yaml|ffhq-ifid-sd3vae"
  # "SOFT-VQ|SOFTVQ.yaml|ffhq-ifid-softvq"
  # "MAE-TOK|MAETOK.yaml|ffhq-ifid-maetok"
  # "DE-TOK|DETOK.yaml|ffhq-ifid-detok"
  # "DM-VAE|DMVAE.yaml|ffhq-ifid-dmvae"
)

echo "============================================================"
echo "[FFHQ IFID PART 1]"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GPU_IDS=${GPU_IDS}"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

for job in "${JOBS[@]}"; do
  IFS="|" read -r NAME CONFIG_FILE EXP_NAME <<< "${job}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}"
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}" > "${OUT_FILE}"
    continue
  fi

  if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    echo "[SKIP DONE] ${NAME}: ${OUT_FILE}"
    continue
  fi

  echo "============================================================"
  echo "[START] ${NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[EXP] ${EXP_NAME}"
  echo "[LOG] ${OUT_FILE}"
  echo "============================================================"

  accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPU_IDS}" \
    evalvae_ffhq.py \
    --seed=0 \
    --sample-dir="${SAMPLE_DIR}" \
    --exp-name="${EXP_NAME}" \
    --dataset="${DATASET}" \
    --dataset-ref="${DATASET_REF}" \
    --vae-config="${CONFIG_PATH}" \
    --small "${SMALL}" \
    --small_val "${SMALL_VAL}" \
    -save \
    --save-every "${SAVE_EVERY}" \
    --save-max "${SAVE_MAX}" \
    > "${OUT_FILE}" 2>&1

  echo "[DONE] ${NAME}"
  tail -n 20 "${OUT_FILE}" || true
  echo
done

echo "All FFHQ IFID PART 1 jobs finished."
