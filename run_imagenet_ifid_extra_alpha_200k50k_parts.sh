#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# ImageNet iFID extra alpha sweep
#
# Newly added experiments:
#   linear alpha=0.2
#   linear alpha=0.3
#   linear alpha=0.4
#   slerp  alpha=0.2
#   slerp  alpha=0.3
#   slerp  alpha=0.4
#
# Dataset:
#   ref/train = 200k
#   val       = 50k
#
# Usage:
#   MODE_GROUP=regular NUM_PARTS=5 PART_ID=1..5   for 4090 groups
#   MODE_GROUP=special NUM_PARTS=3 PART_ID=1..3   for A800 groups
# ============================================================

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

SCRIPT="${SCRIPT:-evalvae_fix_recon.py}"

SAMPLE_DIR="${SAMPLE_DIR:-./samples_imagenet_ifid_interp_200k50k}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
BS="${BS:-25}"
TOP="${TOP:-10}"
SEED="${SEED:-0}"

SAVE_IMAGES="${SAVE_IMAGES:-0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_MAX="${SAVE_MAX:-50}"

CPU_OFFLOAD="${CPU_OFFLOAD:-0}"

LOG_DIR="${LOG_DIR:-./nohup_logs_imagenet_ifid_interp_200k50k}"
mkdir -p "${LOG_DIR}"

MODE_GROUP="${MODE_GROUP:-regular}"   # regular or special
NUM_PARTS="${NUM_PARTS:-1}"
PART_ID="${PART_ID:-1}"

# 22 models total.
# regular = exclude RAE / SVG / SVGT2I_P_STAGE1_256
# special = only RAE / SVG / SVGT2I_P_STAGE1_256
ALL_MODELS=(
  "SDVAE.yaml|SDVAE|sdvae|regular"
  "SD3VAE.yaml|SD3VAE|sd3vae|regular"
  "FLUXVAE.yaml|FLUXVAE|fluxvae|regular"
  "DCAE.yaml|DCAE|dcae|regular"
  "SOFTVQ.yaml|SOFTVQ|softvq|regular"
  "VAVAE.yaml|VAVAE|vavae|regular"
  "VAVAE64.yaml|VAVAE64|vavae64|regular"
  "EQVAE.yaml|EQVAE|eqvae|regular"
  "MAETOK.yaml|MAETOK|maetok|regular"
  "INVAE.yaml|INVAE|invae|regular"
  "REPAEVAE.yaml|REPAEVAE|repae-sdvae|regular"
  "DETOK.yaml|DETOK|detok|regular"
  "QWVAE.yaml|QWVAE|qwenimgvae|regular"
  "RAE.yaml|RAE|rae|special"
  "SVG.yaml|SVG|svg|special"
  "FLUX2VAE.yaml|FLUX2VAE|flux2|regular"
  "SVGT2I_P_STAGE1_256.yaml|SVGT2I_P_STAGE1_256|svg-t2i-p256|special"
  "DMVAE.yaml|DMVAE|dmvae|regular"
  "VTPS.yaml|VTPS|vtps|regular"
  "VTPB.yaml|VTPB|vtpb|regular"
  "VTPL.yaml|VTPL|vtpl|regular"
  "PAE_DINOv2L_d32.yaml|PAE_DINOv2L_d32|pae-dino|regular"
)

EXPERIMENTS=(
  "linear|0.2"
  "linear|0.3"
  "linear|0.4"
  "slerp|0.2"
  "slerp|0.3"
  "slerp|0.4"
)

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

model_in_this_part() {
  local idx="$1"
  local assigned=$(( ((idx - 1) % NUM_PARTS) + 1 ))
  [[ "${assigned}" == "${PART_ID}" ]]
}

echo "================================================================================"
echo "[ImageNet iFID extra alpha sweep: 200k train / 50k val]"
echo "MODE_GROUP=${MODE_GROUP}"
echo "NUM_PARTS=${NUM_PARTS}"
echo "PART_ID=${PART_ID}"
echo "SCRIPT=${SCRIPT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "SEED=${SEED}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GPU_IDS=${GPU_IDS}"
echo "CPU_OFFLOAD=${CPU_OFFLOAD}"
echo "SAVE_IMAGES=${SAVE_IMAGES}"
echo "LOG_DIR=${LOG_DIR}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "================================================================================"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[FATAL] script not found: ${SCRIPT}"
  exit 1
fi

if [[ ! -d "${DATASET}" ]]; then
  echo "[FATAL] DATASET not found: ${DATASET}"
  exit 1
fi

if [[ ! -d "${DATASET_REF}" ]]; then
  echo "[FATAL] DATASET_REF not found: ${DATASET_REF}"
  exit 1
fi

SELECTED_MODELS=()
for model in "${ALL_MODELS[@]}"; do
  IFS="|" read -r CONFIG_FILE MODEL_NAME MODEL_TAG GROUP <<< "${model}"
  if [[ "${GROUP}" == "${MODE_GROUP}" ]]; then
    SELECTED_MODELS+=("${model}")
  fi
done

echo
echo "[SELECTED MODELS: ${#SELECTED_MODELS[@]}]"
idx=0
for model in "${SELECTED_MODELS[@]}"; do
  idx=$((idx + 1))
  IFS="|" read -r CONFIG_FILE MODEL_NAME MODEL_TAG GROUP <<< "${model}"
  if model_in_this_part "${idx}"; then
    echo "  local#${idx}: ${MODEL_NAME} (${CONFIG_FILE})"
  fi
done
echo

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

idx=0
for model in "${SELECTED_MODELS[@]}"; do
  idx=$((idx + 1))

  if ! model_in_this_part "${idx}"; then
    continue
  fi

  IFS="|" read -r CONFIG_FILE MODEL_NAME MODEL_TAG GROUP <<< "${model}"
  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP CONFIG MISSING] ${MODEL_NAME}: ${CONFIG_PATH}"
    TOTAL_SKIP=$((TOTAL_SKIP + 6))
    continue
  fi

  for exp in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r INTP ALPHA <<< "${exp}"

    ATAG="$(alpha_tag "${ALPHA}")"
    EXP_NAME="imagenet-ifid-200k50k-${MODEL_TAG}-${INTP}-${ATAG}"
    OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

    if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
      touch "${OUT_FILE}.done"
      echo "[SKIP DONE] ${MODEL_NAME} ${INTP} alpha=${ALPHA}: ${OUT_FILE}"
      TOTAL_SKIP=$((TOTAL_SKIP + 1))
      continue
    fi

    echo "================================================================================"
    echo "[START] ${MODEL_NAME}"
    echo "[CONFIG] ${CONFIG_PATH}"
    echo "[INTP] ${INTP}"
    echo "[ALPHA] ${ALPHA}"
    echo "[EXP] ${EXP_NAME}"
    echo "[LOG] ${OUT_FILE}"
    echo "================================================================================"

    SAVE_ARGS=()
    if [[ "${SAVE_IMAGES}" == "1" ]]; then
      SAVE_ARGS=(-save --save-every "${SAVE_EVERY}" --save-max "${SAVE_MAX}")
    fi

    CPU_ARGS=()
    if [[ "${CPU_OFFLOAD}" == "1" ]]; then
      CPU_ARGS=(-cpu)
    fi

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
      echo "[ERROR] ${MODEL_NAME} ${INTP} alpha=${ALPHA} failed, status=${STATUS}"
      echo "[ERROR] log: ${OUT_FILE}"
      echo "[ERROR] Last 100 lines:"
      tail -n 100 "${OUT_FILE}" || true
      echo "${STATUS}" > "${OUT_FILE}.failed"
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      echo
      continue
    fi

    if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
      echo "[DONE] ${MODEL_NAME} ${INTP} alpha=${ALPHA}"
      rm -f "${OUT_FILE}.failed"
      touch "${OUT_FILE}.done"
      TOTAL_RUN=$((TOTAL_RUN + 1))
      tail -n 20 "${OUT_FILE}" || true
    else
      echo "[WARN] ${MODEL_NAME} ${INTP} alpha=${ALPHA} exited 0 but final metrics not found."
      echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      tail -n 100 "${OUT_FILE}" || true
    fi

    echo
  done
done

echo "================================================================================"
echo "[PART FINISHED]"
echo "MODE_GROUP=${MODE_GROUP}"
echo "PART_ID=${PART_ID}/${NUM_PARTS}"
echo "TOTAL_RUN=${TOTAL_RUN}"
echo "TOTAL_SKIP=${TOTAL_SKIP}"
echo "TOTAL_FAIL=${TOTAL_FAIL}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"
