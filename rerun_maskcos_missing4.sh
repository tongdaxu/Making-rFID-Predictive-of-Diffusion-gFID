#!/usr/bin/env bash
set -uo pipefail

CONFIG_DIR="${CONFIG_DIR:-./configs}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
OUT_ROOT="${OUT_ROOT:-/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_maskcos}"

GPU_LIST_STR="${GPU_LIST:-0,1,2,3,4,5,6,7}"

NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

MASK_RATIO="${MASK_RATIO:-0.75}"
T_VALUES_STR="${T_VALUES:-0.1 0.4 0.6 0.9}"
NORMALIZE_LATENT="${NORMALIZE_LATENT:-none}"

FORCE="${FORCE:-1}"

SCRIPT="${SCRIPT:-./eval_latent_maskcos_metrics.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-./summarize_latent_maskcos_metrics.py}"

LOG_DIR="${OUT_ROOT}/logs"
WORK_DIR="${OUT_ROOT}/work_missing4"
TASK_FILE="${WORK_DIR}/tasks.tsv"
IDX_FILE="${WORK_DIR}/next_idx.txt"
LOCK_FILE="${WORK_DIR}/queue.lock"
WORKER_LOG_DIR="${WORK_DIR}/worker_logs"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${WORK_DIR}" "${WORKER_LOG_DIR}"

CONFIGS=(
  "EQVAE.yaml"
  "DETOK.yaml"
  "SVG.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
)

: > "${TASK_FILE}"

for CFG in "${CONFIGS[@]}"; do
  CFG_PATH="${CONFIG_DIR}/${CFG}"
  if [[ ! -f "${CFG_PATH}" ]]; then
    echo "[WARN] missing config, skip: ${CFG_PATH}"
    continue
  fi

  NAME="${CFG%.yaml}"
  OUT_DIR="${OUT_ROOT}/${NAME}"
  LOG_FILE="${LOG_DIR}/${NAME}.out"

  # 这里不要给 DETOK 传 --token-grid 8 16。
  # DETOK 实际 token length 是 N=256，脚本会自动推成 16x16。
  EXTRA_ARGS=""

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${NAME}" "${CFG_PATH}" "${OUT_DIR}" "${LOG_FILE}" "${EXTRA_ARGS}" \
    >> "${TASK_FILE}"
done

echo 0 > "${IDX_FILE}"
TOTAL_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')

echo "================================================================================"
echo "[START] rerun missing 4 maskcos metrics"
echo "[CONFIG_DIR]       ${CONFIG_DIR}"
echo "[DATASET]          ${DATASET}"
echo "[OUT_ROOT]         ${OUT_ROOT}"
echo "[GPU_LIST]         ${GPU_LIST_STR}"
echo "[NUM_IMAGES]       ${NUM_IMAGES}"
echo "[BATCH_SIZE]       ${BATCH_SIZE}"
echo "[NUM_WORKERS]      ${NUM_WORKERS}"
echo "[MASK_RATIO]       ${MASK_RATIO}"
echo "[T_VALUES]         ${T_VALUES_STR}"
echo "[NORMALIZE_LATENT] ${NORMALIZE_LATENT}"
echo "[FORCE]            ${FORCE}"
echo "[TOTAL_TASKS]      ${TOTAL_TASKS}"
echo "================================================================================"

column -t -s $'\t' "${TASK_FILE}" || cat "${TASK_FILE}"

run_one_task() {
  local GPU="$1"
  local LINE="$2"

  IFS=$'\t' read -r NAME CFG_PATH OUT_DIR LOG_FILE EXTRA_ARGS <<< "${LINE}"

  local METRICS_JSON="${OUT_DIR}/metrics.json"

  echo "--------------------------------------------------------------------------------"
  echo "[GPU ${GPU}] START ${NAME}"
  echo "[GPU ${GPU}] CFG=${CFG_PATH}"
  echo "[GPU ${GPU}] OUT=${OUT_DIR}"
  echo "[GPU ${GPU}] LOG=${LOG_FILE}"
  echo "--------------------------------------------------------------------------------"

  if [[ "${FORCE}" != "1" && -f "${METRICS_JSON}" ]]; then
    if grep -q '"align_0.1"' "${METRICS_JSON}" && grep -q '"entropy_0.9"' "${METRICS_JSON}"; then
      echo "[GPU ${GPU}] SKIP existing metrics: ${METRICS_JSON}"
      return 0
    fi
  fi

  mkdir -p "${OUT_DIR}"

  # shellcheck disable=SC2206
  local T_ARGS=( ${T_VALUES_STR} )

  CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT}" \
    --vae-config "${CFG_PATH}" \
    --dataset "${DATASET}" \
    --output-dir "${OUT_DIR}" \
    --num-images "${NUM_IMAGES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --t-values "${T_ARGS[@]}" \
    --mask-ratio "${MASK_RATIO}" \
    --normalize-latent "${NORMALIZE_LATENT}" \
    > "${LOG_FILE}" 2>&1

  local STATUS=$?

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[GPU ${GPU}] ERROR ${NAME}, status=${STATUS}"
    echo "[GPU ${GPU}] Last 100 lines:"
    tail -n 100 "${LOG_FILE}" || true
    return 0
  fi

  echo "[GPU ${GPU}] DONE ${NAME}"
  tail -n 20 "${LOG_FILE}" || true
}

worker_loop() {
  local GPU="$1"
  local WLOG="${WORKER_LOG_DIR}/gpu${GPU}.out"

  {
    echo "[GPU ${GPU}] worker started"

    while true; do
      local IDX
      local LINE

      {
        flock 200
        IDX=$(cat "${IDX_FILE}")
        IDX=$((IDX + 1))
        echo "${IDX}" > "${IDX_FILE}"
      } 200>"${LOCK_FILE}"

      LINE=$(sed -n "${IDX}p" "${TASK_FILE}" || true)

      if [[ -z "${LINE}" ]]; then
        echo "[GPU ${GPU}] no more tasks, exit"
        break
      fi

      echo
      echo "[GPU ${GPU}] picked task #${IDX}/${TOTAL_TASKS}"
      run_one_task "${GPU}" "${LINE}"
      echo
    done
  } > "${WLOG}" 2>&1
}

IFS=',' read -r -a GPU_LIST_ARR <<< "${GPU_LIST_STR}"

PIDS=()
for GPU in "${GPU_LIST_ARR[@]}"; do
  worker_loop "${GPU}" &
  PIDS+=("$!")
  echo "[LAUNCH] worker GPU=${GPU}, pid=${PIDS[-1]}"
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

echo "================================================================================"
echo "[DONE] missing 4 workers finished"
echo "[SUMMARY] start summarizing..."
echo "================================================================================"

python "${SUMMARY_SCRIPT}" --root "${OUT_ROOT}"

echo "================================================================================"
echo "[ALL DONE]"
echo "[SUMMARY CSV] ${OUT_ROOT}/maskcos_summary.csv"
echo "[SUMMARY MD]  ${OUT_ROOT}/maskcos_summary.md"
echo "[WORKER_LOG]  ${WORKER_LOG_DIR}"
echo "================================================================================"

exit "${FAIL}"
