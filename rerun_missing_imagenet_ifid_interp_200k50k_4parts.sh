#!/usr/bin/env bash
set -uo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

SCRIPT="${SCRIPT:-evalvae_fix_recon.py}"

SAMPLE_DIR="${SAMPLE_DIR:-./samples_imagenet_ifid_interp_200k50k}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
BS="${BS:-8}"
TOP="${TOP:-10}"
SEED="${SEED:-0}"

CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_MAX="${SAVE_MAX:-50}"

LOG_DIR="${LOG_DIR:-./nohup_logs_imagenet_ifid_interp_200k50k}"
mkdir -p "${LOG_DIR}"

PART_ID="${PART_ID:-1}"
BACKUP_OLD="${BACKUP_OLD:-1}"

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

TASKS=()

if [[ "${PART_ID}" == "1" ]]; then
  TASKS=(
    "RAE.yaml|rae|linear|0.1"
    "RAE.yaml|rae|linear|0.25"
    "RAE.yaml|rae|linear|0.5"
    "RAE.yaml|rae|slerp|0.1"
    "RAE.yaml|rae|slerp|0.25"
  )
elif [[ "${PART_ID}" == "2" ]]; then
  TASKS=(
    "SVG.yaml|svg|linear|0.1"
    "SVG.yaml|svg|linear|0.25"
    "SVG.yaml|svg|linear|0.5"
    "SVG.yaml|svg|slerp|0.1"
    "SVG.yaml|svg|slerp|0.25"
  )
elif [[ "${PART_ID}" == "3" ]]; then
  TASKS=(
    "SVGT2I_P_STAGE1_256.yaml|svg-t2i-p256|linear|0.1"
    "SVGT2I_P_STAGE1_256.yaml|svg-t2i-p256|linear|0.25"
    "SVGT2I_P_STAGE1_256.yaml|svg-t2i-p256|linear|0.5"
  )
elif [[ "${PART_ID}" == "4" ]]; then
  TASKS=(
    "VTPB.yaml|vtpb|linear|0.25"
    "VTPB.yaml|vtpb|linear|0.5"
    "VTPB.yaml|vtpb|slerp|0.1"
    "VTPB.yaml|vtpb|slerp|0.25"
  )
else
  echo "[FATAL] PART_ID must be 1, 2, 3, or 4. Got: ${PART_ID}"
  exit 1
fi

echo "================================================================================"
echo "[RERUN MISSING ImageNet iFID interp 200k50k]"
echo "PART_ID=${PART_ID}"
echo "SCRIPT=${SCRIPT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "CPU_OFFLOAD=${CPU_OFFLOAD}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GPU_IDS=${GPU_IDS}"
echo "LOG_DIR=${LOG_DIR}"
echo "TASKS=${#TASKS[@]}"
echo "================================================================================"

CPU_ARGS=()
if [[ "${CPU_OFFLOAD}" == "1" ]]; then
  CPU_ARGS=(-cpu)
fi

SAVE_ARGS=()
if [[ "${SAVE_IMAGES}" == "1" ]]; then
  SAVE_ARGS=(-save --save-every "${SAVE_EVERY}" --save-max "${SAVE_MAX}")
fi

for task in "${TASKS[@]}"; do
  IFS="|" read -r CONFIG_FILE MODEL_TAG INTP ALPHA <<< "${task}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  ATAG="$(alpha_tag "${ALPHA}")"
  EXP_NAME="imagenet-ifid-200k50k-${MODEL_TAG}-${INTP}-${ATAG}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP CONFIG MISSING] ${CONFIG_PATH}"
    continue
  fi

  if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    touch "${OUT_FILE}.done"
    echo "[SKIP DONE] ${EXP_NAME}"
    continue
  fi

  if [[ "${BACKUP_OLD}" == "1" && -f "${OUT_FILE}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    cp "${OUT_FILE}" "${OUT_FILE}.bak.${TS}"
    echo "[BACKUP] ${OUT_FILE}.bak.${TS}"
  fi

  echo "================================================================================"
  echo "[START] ${EXP_NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[INTP] ${INTP}"
  echo "[ALPHA] ${ALPHA}"
  echo "[LOG] ${OUT_FILE}"
  echo "================================================================================"

  accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPU_IDS}" \
    "${SCRIPT}" \
    --seed="${SEED}" \
    --sample-dir="${SAMPLE_DIR}" \
    --exp-name="${EXP_NAME}" \
    --dataset="${DATASET}" \
    --dataset-ref="${DATASET_REF}" \
    --vae-config="${CONFIG_PATH}" \
    --small "${SMALL}" \
    --small_val "${SMALL_VAL}" \
    --bs "${BS}" \
    --top "${TOP}" \
    --intp "${INTP}" \
    --alpha "${ALPHA}" \
    "${CPU_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    > "${OUT_FILE}" 2>&1

  STATUS=$?

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[ERROR] ${EXP_NAME} failed, status=${STATUS}"
    echo "${STATUS}" > "${OUT_FILE}.failed"
    tail -n 100 "${OUT_FILE}" || true
    echo
    continue
  fi

  if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    echo "[DONE] ${EXP_NAME}"
    rm -f "${OUT_FILE}.failed"
    touch "${OUT_FILE}.done"
    tail -n 20 "${OUT_FILE}" || true
  else
    echo "[WARN] ${EXP_NAME} exited 0 but final metrics not found."
    echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
    tail -n 100 "${OUT_FILE}" || true
  fi

  echo
done

echo "================================================================================"
echo "[PART FINISHED] PART_ID=${PART_ID}"
echo "================================================================================"
