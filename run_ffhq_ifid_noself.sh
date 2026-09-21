#!/usr/bin/env bash
set -uo pipefail

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

SCRIPT="${SCRIPT:-evalvae_ffhq_noself.py}"

SAMPLE_DIR="${SAMPLE_DIR:-./samples_ffhq_noself}"
DATASET="${DATASET:-/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/imgs}"
DATASET_REF="${DATASET_REF:-/video_ssd/kongzishang/xutongda/git/IFID/datasets/ffhq/imgs}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

SMALL="${SMALL:-70000}"
SMALL_VAL="${SMALL_VAL:-70000}"
BS="${BS:-25}"
TOP="${TOP:-10}"
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

SAVE_IMAGES="${SAVE_IMAGES:-1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
SAVE_MAX="${SAVE_MAX:-100}"

LOG_DIR="${LOG_DIR:-./nohup_logs_ffhq_noself}"
mkdir -p "${LOG_DIR}"

# 1-indexed inclusive. Example:
# JOB_START=1 JOB_END=11  -> first machine
# JOB_START=12 JOB_END=22 -> second machine
JOB_START="${JOB_START:-1}"
JOB_END="${JOB_END:-9999}"

JOBS=(
  "SDVAE.yaml|SDVAE|ffhq-ifid-sdvae"
  "SD3VAE.yaml|SD3VAE|ffhq-ifid-sd3vae"
  "FLUXVAE.yaml|FLUXVAE|ffhq-ifid-fluxvae"
  "DCAE.yaml|DCAE|ffhq-ifid-dcae"
  "SOFTVQ.yaml|SOFTVQ|ffhq-ifid-softvq"
  "VAVAE.yaml|VAVAE|ffhq-ifid-vavae"
  "VAVAE64.yaml|VAVAE64|ffhq-ifid-vavae64"
  "EQVAE.yaml|EQVAE|ffhq-ifid-eqvae"
  "MAETOK.yaml|MAETOK|ffhq-ifid-maetok"
  "INVAE.yaml|INVAE|ffhq-ifid-invae"
  "REPAEVAE.yaml|REPAEVAE|ffhq-ifid-repae-sdvae"
  "DETOK.yaml|DETOK|ffhq-ifid-detok"
  "QWVAE.yaml|QWVAE|ffhq-ifid-qwenimgvae"
  "RAE.yaml|RAE|ffhq-ifid-rae"
  "SVG.yaml|SVG|ffhq-ifid-svg"
  "FLUX2VAE.yaml|FLUX2VAE|ffhq-ifid-flux2"
  "SVGT2I_P_STAGE1_256.yaml|SVGT2I_P_STAGE1_256|ffhq-ifid-svg-t2i-p256"
  "DMVAE.yaml|DMVAE|ffhq-ifid-dmvae"
  "VTPS.yaml|VTPS|ffhq-ifid-vtps"
  "VTPB.yaml|VTPB|ffhq-ifid-vtpb"
  "VTPL.yaml|VTPL|ffhq-ifid-vtpl"
  "PAE_DINOv2L_d32.yaml|PAE_DINOv2L_d32|ffhq-ifid-pae-dino"
)

echo "============================================================"
echo "[FFHQ iFID no-self rerun]"
echo "SCRIPT=${SCRIPT}"
echo "DATASET=${DATASET}"
echo "DATASET_REF=${DATASET_REF}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "INTP=${INTP}"
echo "ALPHA=${ALPHA}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GPU_IDS=${GPU_IDS}"
echo "JOB_START=${JOB_START}"
echo "JOB_END=${JOB_END}"
echo "LOG_DIR=${LOG_DIR}"
echo "SAMPLE_DIR=${SAMPLE_DIR}"
echo "============================================================"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[FATAL] script not found: ${SCRIPT}"
  exit 1
fi

if [[ ! -d "${DATASET}" ]]; then
  echo "[FATAL] dataset not found: ${DATASET}"
  exit 1
fi

NUM_IMAGES=$(find "${DATASET}" -type f -name "*.png" | wc -l | tr -d ' ')
echo "[DATASET PNG COUNT] ${NUM_IMAGES}"

idx=0
for job in "${JOBS[@]}"; do
  idx=$((idx + 1))

  if (( idx < JOB_START || idx > JOB_END )); then
    continue
  fi

  IFS="|" read -r CONFIG_FILE MODEL_NAME EXP_NAME <<< "${job}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP] ${MODEL_NAME}: config not found: ${CONFIG_PATH}"
    echo "[SKIP] ${MODEL_NAME}: config not found: ${CONFIG_PATH}" > "${OUT_FILE}"
    continue
  fi

  if [[ -f "${OUT_FILE}" ]] && grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    echo "[SKIP DONE] #${idx} ${MODEL_NAME}: ${OUT_FILE}"
    continue
  fi

  echo "============================================================"
  echo "[START] #${idx} ${MODEL_NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
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
    --top "${TOP}" \
    --intp "${INTP}" \
    --alpha "${ALPHA}" \
    --exclude-self \
    "${SAVE_ARGS[@]}" \
    > "${OUT_FILE}" 2>&1

  STATUS=$?

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[ERROR] #${idx} ${MODEL_NAME} failed, status=${STATUS}"
    echo "[ERROR] log: ${OUT_FILE}"
    echo "[ERROR] Last 100 lines:"
    tail -n 100 "${OUT_FILE}" || true
    echo "${STATUS}" > "${OUT_FILE}.failed"
    echo
    continue
  fi

  if grep -q '^ifid=' "${OUT_FILE}" && grep -q '^lpips=' "${OUT_FILE}"; then
    echo "[DONE] #${idx} ${MODEL_NAME}"
    rm -f "${OUT_FILE}.failed"
    touch "${OUT_FILE}.done"
    tail -n 20 "${OUT_FILE}" || true
  else
    echo "[WARN] #${idx} ${MODEL_NAME} exited 0 but final metrics not found."
    echo "NO_FINAL_METRICS" > "${OUT_FILE}.failed"
    tail -n 100 "${OUT_FILE}" || true
  fi

  echo
done

echo "All selected FFHQ no-self jobs finished."
