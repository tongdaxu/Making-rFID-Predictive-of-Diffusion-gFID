#!/usr/bin/env bash
set -uo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

SCRIPT="${SCRIPT:-evalvae_fix_recon.py}"

SAMPLE_DIR="${SAMPLE_DIR:-./samples_imagenet_ifid_alpha20k}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

SMALL="${SMALL:-20000}"
SMALL_VAL="${SMALL_VAL:-20000}"
BS="${BS:-25}"

SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_MAX="${SAVE_MAX:-50}"

LOG_DIR="${LOG_DIR:-./nohup_logs_imagenet_ifid_alpha20k}"
mkdir -p "${LOG_DIR}"

ALPHAS=(
  "0.1"
  "0.2"
  "0.3"
  "0.4"
  "0.5"
)

INTPS=(
  "linear"
  "slerp"
)

JOBS=(
  "DE-TOK|DETOK.yaml|detok"
  "DM-VAE|DMVAE.yaml|dmvae"
  "REPAE-SDVAE|REPAEVAE.yaml|repae-sdvae"
  "VTPB|VTPB.yaml|vtpb"
  "VTPL|VTPL.yaml|vtpl"
  "VTPS|VTPS.yaml|vtps"
)

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

echo "============================================================"
echo "[ImageNet IFID linear vs slerp alpha sweep]"
echo "SCRIPT=${SCRIPT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GPU_IDS=${GPU_IDS}"
echo "SAVE_IMAGES=${SAVE_IMAGES}"
echo "LOG_DIR=${LOG_DIR}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "============================================================"

for job in "${JOBS[@]}"; do
  IFS="|" read -r NAME CONFIG_FILE MODEL_TAG <<< "${job}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}"
    continue
  fi

  for INTP in "${INTPS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
      ATAG="$(alpha_tag "${ALPHA}")"
      EXP_NAME="imagenet20k-ifid-${MODEL_TAG}-${INTP}-${ATAG}"
      OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

      if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
        echo "[SKIP DONE] ${NAME} ${INTP} alpha=${ALPHA}: ${OUT_FILE}"
        continue
      fi

      echo "============================================================"
      echo "[START] ${NAME}"
      echo "[CONFIG] ${CONFIG_PATH}"
      echo "[INTP] ${INTP}"
      echo "[ALPHA] ${ALPHA}"
      echo "[EXP] ${EXP_NAME}"
      echo "[LOG] ${OUT_FILE}"
      echo "============================================================"

      SAVE_ARGS=()
      if [[ "${SAVE_IMAGES}" == "1" ]]; then
        SAVE_ARGS=(-save --save-every "${SAVE_EVERY}" --save-max "${SAVE_MAX}")
      fi

      accelerate launch \
        --num_processes="${NUM_PROCESSES}" \
        --gpu_ids="${GPU_IDS}" \
        "${SCRIPT}" \
        --seed=0 \
        --sample-dir="${SAMPLE_DIR}" \
        --exp-name="${EXP_NAME}" \
        --dataset="${DATASET}" \
        --dataset-ref="${DATASET_REF}" \
        --vae-config="${CONFIG_PATH}" \
        --small "${SMALL}" \
        --small_val "${SMALL_VAL}" \
        --bs "${BS}" \
        --intp "${INTP}" \
        --alpha "${ALPHA}" \
        "${SAVE_ARGS[@]}" \
        > "${OUT_FILE}" 2>&1

      STATUS=$?

      if [[ "${STATUS}" -ne 0 ]]; then
        echo "[ERROR] ${NAME} ${INTP} alpha=${ALPHA} failed, status=${STATUS}"
        echo "[ERROR] log: ${OUT_FILE}"
        echo "[ERROR] Last 80 lines:"
        tail -n 80 "${OUT_FILE}" || true
        echo "${STATUS}" > "${OUT_FILE}.failed"
        echo
        continue
      fi

      if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
        echo "[DONE] ${NAME} ${INTP} alpha=${ALPHA}"
        rm -f "${OUT_FILE}.failed"
        touch "${OUT_FILE}.done"
        tail -n 20 "${OUT_FILE}" || true
      else
        echo "[WARN] ${NAME} ${INTP} alpha=${ALPHA} exited 0 but final metrics not found."
        echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
        tail -n 80 "${OUT_FILE}" || true
      fi

      echo
    done
  done
done

echo "All ImageNet IFID alpha sweep jobs finished."
