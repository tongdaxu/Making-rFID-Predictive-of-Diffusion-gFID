#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# ImageNet iFID interpolation ablation
#
# Dataset:
#   train/ref = 200k
#   val       = 50k full ImageNet val
#
# Experiments per model:
#   linear alpha=0.1
#   linear alpha=0.25
#   linear alpha=0.5
#   slerp  alpha=0.1
#   slerp  alpha=0.25
#
# We intentionally do NOT run slerp alpha=0.5 here,
# because it is the main experiment.
#
# Part mode:
#   NUM_PARTS=11 PART_ID=1  -> run models assigned to part 1
#   NUM_PARTS=11 PART_ID=2  -> run models assigned to part 2
#   ...
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

# Automatic part split over models.
# Example: NUM_PARTS=11 and PART_ID=1..11.
NUM_PARTS="${NUM_PARTS:-1}"
PART_ID="${PART_ID:-1}"

# Optional explicit range override, 1-indexed inclusive.
# If JOB_START/JOB_END are set, they override modulo part assignment.
JOB_START="${JOB_START:-}"
JOB_END="${JOB_END:-}"

# 22 models, in the order you requested before.
# Format:
#   config_file|model_name|model_tag
MODELS=(
  "SDVAE.yaml|SDVAE|sdvae"
  "SD3VAE.yaml|SD3VAE|sd3vae"
  "FLUXVAE.yaml|FLUXVAE|fluxvae"
  "DCAE.yaml|DCAE|dcae"
  "SOFTVQ.yaml|SOFTVQ|softvq"
  "VAVAE.yaml|VAVAE|vavae"
  "VAVAE64.yaml|VAVAE64|vavae64"
  "EQVAE.yaml|EQVAE|eqvae"
  "MAETOK.yaml|MAETOK|maetok"
  "INVAE.yaml|INVAE|invae"
  "REPAEVAE.yaml|REPAEVAE|repae-sdvae"
  "DETOK.yaml|DETOK|detok"
  "QWVAE.yaml|QWVAE|qwenimgvae"
  "RAE.yaml|RAE|rae"
  "SVG.yaml|SVG|svg"
  "FLUX2VAE.yaml|FLUX2VAE|flux2"
  "SVGT2I_P_STAGE1_256.yaml|SVGT2I_P_STAGE1_256|svg-t2i-p256"
  "DMVAE.yaml|DMVAE|dmvae"
  "VTPS.yaml|VTPS|vtps"
  "VTPB.yaml|VTPB|vtpb"
  "VTPL.yaml|VTPL|vtpl"
  "PAE_DINOv2L_d32.yaml|PAE_DINOv2L_d32|pae-dino"
)

# 5 experiments per model.
# Format:
#   intp|alpha
EXPERIMENTS=(
  "linear|0.1"
  "linear|0.25"
  "linear|0.5"
  "slerp|0.1"
  "slerp|0.25"
)

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

model_in_this_part() {
  local idx="$1"

  if [[ -n "${JOB_START}" && -n "${JOB_END}" ]]; then
    if (( idx >= JOB_START && idx <= JOB_END )); then
      return 0
    else
      return 1
    fi
  fi

  # modulo split: model idx 1 goes to part 1, idx 2 to part 2, ...
  # when idx > NUM_PARTS, wrap around.
  local assigned=$(( ((idx - 1) % NUM_PARTS) + 1 ))
  if (( assigned == PART_ID )); then
    return 0
  else
    return 1
  fi
}

echo "================================================================================"
echo "[ImageNet iFID linear/slerp ablation: train200k val50k]"
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
echo "SAVE_IMAGES=${SAVE_IMAGES}"
echo "CPU_OFFLOAD=${CPU_OFFLOAD}"
echo "LOG_DIR=${LOG_DIR}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "NUM_PARTS=${NUM_PARTS}"
echo "PART_ID=${PART_ID}"
echo "JOB_START=${JOB_START:-<unset>}"
echo "JOB_END=${JOB_END:-<unset>}"
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

echo
echo "[MODELS IN THIS PART]"
idx=0
for model in "${MODELS[@]}"; do
  idx=$((idx + 1))
  IFS="|" read -r CONFIG_FILE MODEL_NAME MODEL_TAG <<< "${model}"
  if model_in_this_part "${idx}"; then
    echo "  #${idx}: ${MODEL_NAME} (${CONFIG_FILE})"
  fi
done
echo

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

idx=0
for model in "${MODELS[@]}"; do
  idx=$((idx + 1))

  if ! model_in_this_part "${idx}"; then
    continue
  fi

  IFS="|" read -r CONFIG_FILE MODEL_NAME MODEL_TAG <<< "${model}"
  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP CONFIG MISSING] #${idx} ${MODEL_NAME}: ${CONFIG_PATH}"
    TOTAL_SKIP=$((TOTAL_SKIP + 5))
    continue
  fi

  for exp in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r INTP ALPHA <<< "${exp}"

    ATAG="$(alpha_tag "${ALPHA}")"
    EXP_NAME="imagenet-ifid-200k50k-${MODEL_TAG}-${INTP}-${ATAG}"
    OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

    if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
      echo "[SKIP DONE] #${idx} ${MODEL_NAME} ${INTP} alpha=${ALPHA}: ${OUT_FILE}"
      TOTAL_SKIP=$((TOTAL_SKIP + 1))
      continue
    fi

    echo "================================================================================"
    echo "[START] #${idx} ${MODEL_NAME}"
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
      echo "[ERROR] #${idx} ${MODEL_NAME} ${INTP} alpha=${ALPHA} failed, status=${STATUS}"
      echo "[ERROR] log: ${OUT_FILE}"
      echo "[ERROR] Last 100 lines:"
      tail -n 100 "${OUT_FILE}" || true
      echo "${STATUS}" > "${OUT_FILE}.failed"
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      echo
      continue
    fi

    if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
      echo "[DONE] #${idx} ${MODEL_NAME} ${INTP} alpha=${ALPHA}"
      rm -f "${OUT_FILE}.failed"
      touch "${OUT_FILE}.done"
      TOTAL_RUN=$((TOTAL_RUN + 1))
      tail -n 20 "${OUT_FILE}" || true
    else
      echo "[WARN] #${idx} ${MODEL_NAME} ${INTP} alpha=${ALPHA} exited 0 but final metrics not found."
      echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
      tail -n 100 "${OUT_FILE}" || true
    fi

    echo
  done
done

echo "================================================================================"
echo "[PART FINISHED]"
echo "PART_ID=${PART_ID}/${NUM_PARTS}"
echo "TOTAL_RUN=${TOTAL_RUN}"
echo "TOTAL_SKIP=${TOTAL_SKIP}"
echo "TOTAL_FAIL=${TOTAL_FAIL}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"
