#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"
SAMPLE_DIR="${SAMPLE_DIR:-${ROOT}/ifid_saved_images}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_ifid_extra3}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"
SCRIPT="${SCRIPT:-evalvae_fix_recon.py}"

RUN_MODE="${RUN_MODE:-full}"       # smoke / full
ONLY_TAG="${ONLY_TAG:-all}"        # all / uae / scalerae-siglip2 / scalerae-webssl

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29961}"

TOP="${TOP:-10}"
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

mkdir -p "${SAMPLE_DIR}" "${LOG_DIR}"

case "${RUN_MODE}" in
  smoke)
    SMALL=2000
    SMALL_VAL=400
    SAVE_EVERY=1
    SAVE_MAX=40
    EXP_PREFIX="smoke-ifid-iis"
    ;;
  full)
    SMALL=200000
    SMALL_VAL=50000
    SAVE_EVERY=1
    SAVE_MAX=50000
    EXP_PREFIX="ifid-iis"
    ;;
  *)
    echo "[FATAL] RUN_MODE must be smoke or full, got: ${RUN_MODE}"
    exit 1
    ;;
esac

# config | display name | tag | per-rank batch size
JOBS=(
  "UAE.yaml|UAE|uae|10"
  "ScaleRAE_SIGLIP2_NONORM.yaml|ScaleRAE-SigLIP2|scalerae-siglip2|1"
  "ScaleRAE_WEBSSL.yaml|ScaleRAE-WEBSSL|scalerae-webssl|1"
)

echo "================================================================================"
echo "[EXTRA-3 iFID INTERPOLATION IMAGES]"
echo "RUN_MODE=${RUN_MODE}"
echo "ONLY_TAG=${ONLY_TAG}"
echo "SCRIPT=${SCRIPT}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "TOP=${TOP}"
echo "INTP=${INTP}"
echo "ALPHA=${ALPHA}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "SAVE_MAX=${SAVE_MAX}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[FATAL] Missing evaluator: ${SCRIPT}"
  exit 1
fi

FAILED=0
DONE=0
SKIPPED=0

for item in "${JOBS[@]}"; do
  IFS="|" read -r CONFIG_FILE MODEL TAG BS <<< "${item}"

  if [[ "${ONLY_TAG}" != "all" && "${ONLY_TAG}" != "${TAG}" ]]; then
    continue
  fi

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

  if [[ "${RUN_MODE}" == "full" ]]; then
    EXP_NAME="${EXP_PREFIX}-${TAG}-${INTP}-a05-top${TOP}-${SMALL}x${SMALL_VAL}"
  else
    EXP_NAME="${EXP_PREFIX}-${TAG}-${INTP}-a05-top${TOP}-${SMALL}x${SMALL_VAL}"
  fi

  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[FAILED] Missing config: ${CONFIG_PATH}"
    echo "MISSING_CONFIG" > "${OUT_FILE}.failed"
    FAILED=$((FAILED + 1))
    continue
  fi

  if [[ -f "${OUT_FILE}.done" ]]; then
    echo "[SKIP DONE] ${MODEL}: ${OUT_FILE}.done"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo
  echo "================================================================================"
  echo "[START] ${MODEL}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[TAG] ${TAG}"
  echo "[BS] ${BS}"
  echo "[EXP] ${EXP_NAME}"
  echo "[LOG] ${OUT_FILE}"
  echo "================================================================================"

  rm -f "${OUT_FILE}.failed"

  set +e

  accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPU_IDS}" \
    --main_process_port="${MASTER_PORT}" \
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
    --top "${TOP}" \
    --intp "${INTP}" \
    --alpha "${ALPHA}" \
    -save \
    --save-every "${SAVE_EVERY}" \
    --save-max "${SAVE_MAX}" \
    > "${OUT_FILE}" 2>&1

  STATUS=$?

  set -e

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "${STATUS}" > "${OUT_FILE}.failed"
    echo "[FAILED] ${MODEL}, status=${STATUS}"
    tail -n 120 "${OUT_FILE}" || true
    FAILED=$((FAILED + 1))
    continue
  fi

  if grep -q '^ifid=' "${OUT_FILE}" &&
     grep -q '^rfid=' "${OUT_FILE}" &&
     grep -q '^lpips=' "${OUT_FILE}"; then

    rm -f "${OUT_FILE}.failed"
    touch "${OUT_FILE}.done"

    echo "[DONE] ${MODEL}"
    tail -n 20 "${OUT_FILE}" || true
    DONE=$((DONE + 1))
  else
    echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
    echo "[FAILED] ${MODEL}: process exited 0 but final metrics are missing"
    tail -n 120 "${OUT_FILE}" || true
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "================================================================================"
echo "[FINISHED]"
echo "DONE=${DONE}"
echo "SKIPPED=${SKIPPED}"
echo "FAILED=${FAILED}"
echo "================================================================================"

if (( FAILED > 0 )); then
  exit 1
fi
