#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"
CONFIG_DIR="${CONFIG_DIR:-/video_ssd/kongzishang/xutongda/git/IFID/configs}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

SCRIPT="${SCRIPT:-generate_interpolate.py}"

MODEL_GROUP="${MODEL_GROUP:-all}"   # all / regular / special

PART_ID="${1:-${PART_ID:-1}}"
SHARD_TOTAL="${SHARD_TOTAL:-1}"
SHARD_INDEX=$((PART_ID - 1))

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MASTER_PORT="${MASTER_PORT:-$((29620 + PART_ID))}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
BS="${BS:-25}"
TOP="${TOP:-10}"
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"
NN_SAMPLE="${NN_SAMPLE:-categorical}"

NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"

CREATE_NPZ="${CREATE_NPZ:-1}"
KEEP_SAMPLES="${KEEP_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-0}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${ROOT}" "${LOG_DIR}"

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

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

ATAG="$(alpha_tag "${ALPHA}")"

# config | model_name | tag | group
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

SELECTED_MODELS=()
for item in "${ALL_MODELS[@]}"; do
  IFS="|" read -r CONFIG_FILE MODEL_NAME TAG GROUP <<< "${item}"

  if [[ "${MODEL_GROUP}" == "all" ]]; then
    SELECTED_MODELS+=("${item}")
  elif [[ "${MODEL_GROUP}" == "${GROUP}" ]]; then
    SELECTED_MODELS+=("${item}")
  fi
done

echo "================================================================================"
echo "[run_generate_interpolate_22models_parts.sh]"
echo "ROOT=${ROOT}"
echo "MODEL_GROUP=${MODEL_GROUP}"
echo "SELECTED_MODELS=${#SELECTED_MODELS[@]}"
echo "PART_ID=${PART_ID}"
echo "SHARD_TOTAL=${SHARD_TOTAL}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "GPUS=${GPUS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "INTP=${INTP}"
echo "ALPHA=${ALPHA}"
echo "NN_SAMPLE=${NN_SAMPLE}"
echo "LOG_DIR=${LOG_DIR}"
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
for i in "${!SELECTED_MODELS[@]}"; do
  if (( i % SHARD_TOTAL != SHARD_INDEX )); then
    continue
  fi

  IFS="|" read -r CONFIG_FILE MODEL_NAME TAG GROUP <<< "${SELECTED_MODELS[$i]}"
  echo "  global_selected_index=$((i + 1)) model=${MODEL_NAME} tag=${TAG} group=${GROUP}"
done
echo

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

for i in "${!SELECTED_MODELS[@]}"; do
  if (( i % SHARD_TOTAL != SHARD_INDEX )); then
    continue
  fi

  IFS="|" read -r CONFIG_FILE MODEL_NAME TAG GROUP <<< "${SELECTED_MODELS[$i]}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  EXP_NAME="imagenet-interpolate-${TAG}-${INTP}-${ATAG}-top${TOP}-${SMALL}x${SMALL_VAL}"

  NPZ_PATH="${ROOT}/${EXP_NAME}.npz"
  OUT_LOG="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP CONFIG MISSING] ${MODEL_NAME}: ${CONFIG_PATH}"
    TOTAL_SKIP=$((TOTAL_SKIP + 1))
    continue
  fi

  if [[ -s "${NPZ_PATH}" && "${OVERWRITE}" != "1" ]]; then
    echo "[SKIP DONE] ${MODEL_NAME}: ${NPZ_PATH}"
    TOTAL_SKIP=$((TOTAL_SKIP + 1))
    continue
  fi

  echo "================================================================================"
  echo "[START] ${MODEL_NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[EXP_NAME] ${EXP_NAME}"
  echo "[NPZ] ${NPZ_PATH}"
  echo "[LOG] ${OUT_LOG}"
  echo "================================================================================"

  CREATE_ARGS=()
  if [[ "${CREATE_NPZ}" == "1" ]]; then
    CREATE_ARGS=(--create-npz)
  else
    CREATE_ARGS=(--no-create-npz)
  fi

  KEEP_ARGS=()
  if [[ "${KEEP_SAMPLES}" == "1" ]]; then
    KEEP_ARGS=(--keep-samples)
  else
    KEEP_ARGS=(--no-keep-samples)
  fi

  OVERWRITE_ARGS=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    OVERWRITE_ARGS=(--overwrite)
  fi

  set +e

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPUS}" \
    --main_process_port="${MASTER_PORT}" \
    "${SCRIPT}" \
    --vae-config "${CONFIG_PATH}" \
    --dataset "${DATASET}" \
    --dataset-ref "${DATASET_REF}" \
    --small "${SMALL}" \
    --small-val "${SMALL_VAL}" \
    --sample-dir "${ROOT}" \
    --exp-name "${EXP_NAME}" \
    --bs "${BS}" \
    --num-workers "${NUM_WORKERS}" \
    --top "${TOP}" \
    --intp "${INTP}" \
    --alpha "${ALPHA}" \
    --nn-sample "${NN_SAMPLE}" \
    --seed "${SEED}" \
    "${CREATE_ARGS[@]}" \
    "${KEEP_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    > "${OUT_LOG}" 2>&1

  STATUS=$?

  set -e

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[ERROR] ${MODEL_NAME} failed, status=${STATUS}"
    echo "${STATUS}" > "${OUT_LOG}.failed"
    tail -n 100 "${OUT_LOG}" || true
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    echo
    continue
  fi

  if [[ -s "${NPZ_PATH}" ]]; then
    echo "[DONE] ${MODEL_NAME}: ${NPZ_PATH}"
    rm -f "${OUT_LOG}.failed"
    touch "${OUT_LOG}.done"
    TOTAL_RUN=$((TOTAL_RUN + 1))
    tail -n 30 "${OUT_LOG}" || true
  else
    echo "[WARN] ${MODEL_NAME} exited 0 but npz not found: ${NPZ_PATH}"
    echo "NO_NPZ" > "${OUT_LOG}.failed"
    tail -n 100 "${OUT_LOG}" || true
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
echo "ROOT=${ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"
