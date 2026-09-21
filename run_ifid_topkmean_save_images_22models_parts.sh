#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"
SAMPLE_DIR="${SAMPLE_DIR:-${ROOT}/ifid_topkmean_saved_images}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_ifid_topkmean}"
mkdir -p "${ROOT}" "${SAMPLE_DIR}" "${LOG_DIR}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

SCRIPT="${SCRIPT:-evalvae_fix_recon_topkmean.py}"

MODEL_GROUP="${MODEL_GROUP:-all}"   # all / regular / special

PART_ID="${1:-${PART_ID:-1}}"
SHARD_TOTAL="${SHARD_TOTAL:-1}"

if ! [[ "${PART_ID}" =~ ^[0-9]+$ ]]; then
  echo "[FATAL] PART_ID must be integer, got: ${PART_ID}"
  exit 1
fi

if ! [[ "${SHARD_TOTAL}" =~ ^[0-9]+$ ]]; then
  echo "[FATAL] SHARD_TOTAL must be integer, got: ${SHARD_TOTAL}"
  exit 1
fi

if (( PART_ID < 1 || PART_ID > SHARD_TOTAL )); then
  echo "[FATAL] PART_ID must be in [1, ${SHARD_TOTAL}], got: ${PART_ID}"
  exit 1
fi

SHARD_INDEX=$((PART_ID - 1))

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-$((29820 + PART_ID))}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
BS="${BS:-25}"
TOP="${TOP:-10}"
TOP_AGG="${TOP_AGG:-mean}"          # mean / nearest / sample
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

SAVE_EVERY="${SAVE_EVERY:-1}"
SAVE_MAX="${SAVE_MAX:-50000}"

OVERWRITE="${OVERWRITE:-0}"

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

ATAG="$(alpha_tag "${ALPHA}")"

# config | display name | tag | group
JOBS=(
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

SELECTED=()
for item in "${JOBS[@]}"; do
  IFS="|" read -r CONFIG_FILE NAME TAG GROUP <<< "${item}"

  if [[ "${MODEL_GROUP}" == "all" || "${MODEL_GROUP}" == "${GROUP}" ]]; then
    SELECTED+=("${item}")
  fi
done

echo "================================================================================"
echo "[iFID top-k mean rerun with saved interpolate images]"
echo "ROOT=${ROOT}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "SCRIPT=${SCRIPT}"
echo "MODEL_GROUP=${MODEL_GROUP}"
echo "SELECTED=${#SELECTED[@]}"
echo "PART_ID=${PART_ID}"
echo "SHARD_TOTAL=${SHARD_TOTAL}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "TOP_AGG=${TOP_AGG}"
echo "INTP=${INTP}"
echo "ALPHA=${ALPHA}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "SAVE_MAX=${SAVE_MAX}"
echo "OVERWRITE=${OVERWRITE}"
echo "================================================================================"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[FATAL] SCRIPT not found: ${SCRIPT}"
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

echo
echo "[MODELS IN THIS PART]"
for i in "${!SELECTED[@]}"; do
  if (( i % SHARD_TOTAL != SHARD_INDEX )); then
    continue
  fi

  IFS="|" read -r CONFIG_FILE NAME TAG GROUP <<< "${SELECTED[$i]}"
  echo "  selected_index=$((i + 1)) model=${NAME} config=${CONFIG_FILE} group=${GROUP}"
done
echo

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

for i in "${!SELECTED[@]}"; do
  if (( i % SHARD_TOTAL != SHARD_INDEX )); then
    continue
  fi

  IFS="|" read -r CONFIG_FILE NAME TAG GROUP <<< "${SELECTED[$i]}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  EXP_NAME="ifid-iis-topkmean-${TAG}-${INTP}-${ATAG}-top${TOP}-${SMALL}x${SMALL_VAL}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}"
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}" > "${OUT_FILE}"
    TOTAL_SKIP=$((TOTAL_SKIP + 1))
    continue
  fi

  if [[ "${OVERWRITE}" != "1" && -f "${OUT_FILE}.done" ]]; then
    echo "[SKIP DONE] ${NAME}: ${OUT_FILE}.done"
    TOTAL_SKIP=$((TOTAL_SKIP + 1))
    continue
  fi

  echo "================================================================================"
  echo "[START] ${NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[EXP] ${EXP_NAME}"
  echo "[LOG] ${OUT_FILE}"
  echo "================================================================================"

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
    --top-agg "${TOP_AGG}" \
    --intp "${INTP}" \
    --alpha "${ALPHA}" \
    -save \
    --save-every "${SAVE_EVERY}" \
    --save-max "${SAVE_MAX}" \
    > "${OUT_FILE}" 2>&1

  STATUS=$?

  set -e

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[ERROR] ${NAME} failed, status=${STATUS}"
    echo "${STATUS}" > "${OUT_FILE}.failed"
    tail -n 100 "${OUT_FILE}" || true
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    echo
    continue
  fi

  if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    echo "[DONE] ${NAME}"
    rm -f "${OUT_FILE}.failed"
    touch "${OUT_FILE}.done"
    TOTAL_RUN=$((TOTAL_RUN + 1))
    tail -n 30 "${OUT_FILE}" || true
  else
    echo "[WARN] ${NAME} exited 0 but final metrics not found."
    echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
    tail -n 100 "${OUT_FILE}" || true
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi

  echo
done

echo "================================================================================"
echo "[PART FINISHED]"
echo "MODEL_GROUP=${MODEL_GROUP}"
echo "PART_ID=${PART_ID}/${SHARD_TOTAL}"
echo "TOTAL_RUN=${TOTAL_RUN}"
echo "TOTAL_SKIP=${TOTAL_SKIP}"
echo "TOTAL_FAIL=${TOTAL_FAIL}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"
