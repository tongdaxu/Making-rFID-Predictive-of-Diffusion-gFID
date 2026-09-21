#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"

IMAGE_ROOT="${IMAGE_ROOT:-${ROOT}/ifid_saved_images}"
NPZ_DIR="${NPZ_DIR:-${ROOT}/iis_npz_from_intp_dist}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/guided_activation_cache}"

RESULT_DIR="${RESULT_DIR:-${ROOT}/results_iis_iprec_irec_from_intp_dist}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_iis_iprec_irec_from_intp_dist}"

PACK_SCRIPT="${PACK_SCRIPT:-pack_intp_images_to_npz.py}"
CACHE_SCRIPT="${CACHE_SCRIPT:-cache_guided_reference_pool.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_interpolate_iis_iprec_irec.py}"

GPUS_STR="${GPUS:-0 1 2 3}"

TF_BATCH_SIZE="${TF_BATCH_SIZE:-64}"
SOFTMAX_BATCH_SIZE="${SOFTMAX_BATCH_SIZE:-512}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-4}"

OVERWRITE_NPZ="${OVERWRITE_NPZ:-0}"
OVERWRITE_METRICS="${OVERWRITE_METRICS:-0}"

mkdir -p \
  "${NPZ_DIR}" \
  "${CACHE_DIR}" \
  "${RESULT_DIR}" \
  "${LOG_DIR}"

# ============================================================
# Locate guided-diffusion
# ============================================================

if [[ -z "${GUIDED_DIFFUSION_DIR:-}" ]]; then
  for candidate in \
    "/video_ssd/kongzishang/xutongda/git/IFID/third_party/guided-diffusion" \
    "/video_ssd/kongzishang/xutongda/git/guided-diffusion" \
    "/video_ssd/kongzishang/xutongda/git/IFID/guided-diffusion"
  do
    if [[ -f "${candidate}/evaluations/evaluator.py" ]]; then
      GUIDED_DIFFUSION_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${GUIDED_DIFFUSION_DIR:-}" ]]; then
  echo "[FATAL] Cannot locate guided-diffusion."
  exit 1
fi

if [[ ! -f "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py" ]]; then
  echo "[FATAL] Missing:"
  echo "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py"
  exit 1
fi

for script in \
  "${PACK_SCRIPT}" \
  "${CACHE_SCRIPT}" \
  "${EVAL_SCRIPT}"
do
  if [[ ! -f "${script}" ]]; then
    echo "[FATAL] Missing script: ${script}"
    exit 1
  fi

  python -m py_compile "${script}" || exit 1
done

# ============================================================
# 25 models in fixed order
# display name | tag
# ============================================================

MODELS=(
  "SDVAE|sdvae"
  "SD3VAE|sd3vae"
  "FLUXVAE|fluxvae"
  "DCAE|dcae"
  "SOFTVQ|softvq"
  "VAVAE|vavae"
  "VAVAE64|vavae64"
  "EQVAE|eqvae"
  "MAETOK|maetok"
  "INVAE|invae"
  "REPAEVAE|repae-sdvae"
  "DETOK|detok"
  "QWVAE|qwenimgvae"
  "RAE|rae"
  "SVG|svg"
  "FLUX2VAE|flux2"
  "SVGT2I_P_STAGE1_256|svg-t2i-p256"
  "DMVAE|dmvae"
  "UAE|uae"
  "VTPS|vtps"
  "VTPB|vtpb"
  "VTPL|vtpl"
  "ScaleRAE-SigLIP2|scalerae-siglip2"
  "ScaleRAE-WEBSSL|scalerae-webssl"
  "PAE_DINOv2L_d32|pae-dino"
)

is_complete_json() {
  local json_path="$1"

  python - "${json_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.exists():
    raise SystemExit(1)

try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)

iis = data.get("iIS", data.get("inception_score"))
iprec = data.get("iPrec", data.get("precision"))
irec = data.get("iRec", data.get("recall"))

if iis is None or iprec is None or irec is None:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

valid_npz() {
  local npz_path="$1"

  if [[ -f "${npz_path}" ]]; then
    local size
    size=$(stat -c '%s' "${npz_path}")

    if [[ "${size}" -gt 1000000000 ]]; then
      return 0
    fi
  fi

  return 1
}

IFS=' ' read -r -a GPU_ARRAY <<< "${GPUS_STR}"
NUM_WORKERS="${#GPU_ARRAY[@]}"

if (( NUM_WORKERS == 0 )); then
  echo "[FATAL] No GPUs supplied."
  exit 1
fi

echo "================================================================================"
echo "[25-MODEL INTERPOLATE METRICS — 4 GPU]"
echo "GPUS=${GPUS_STR}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "NPZ_DIR=${NPZ_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"

echo
echo "[GPU INFORMATION]"

nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,name,memory.total \
  --format=csv,noheader || true

# ============================================================
# Stage 1: prepare common reference NPZ
# ============================================================

REF_NPZ="${NPZ_DIR}/imagenet-val-src_dist-50000.npz"

if ! valid_npz "${REF_NPZ}"; then
  REF_IMAGE_DIR="${IMAGE_ROOT}/ifid-iis-dcae-slerp-a05-top10-200000x50000_extra/src_dist"

  echo
  echo "================================================================================"
  echo "[STAGE 1] Packing common reference NPZ"
  echo "IMAGE_DIR=${REF_IMAGE_DIR}"
  echo "OUT_NPZ=${REF_NPZ}"
  echo "================================================================================"

  python "${PACK_SCRIPT}" \
    --image-dir="${REF_IMAGE_DIR}" \
    --out-npz="${REF_NPZ}" \
    --expected=50000 \
    --image-size=256

  status=$?

  if [[ "${status}" -ne 0 ]]; then
    echo "[FATAL] Failed to pack reference NPZ."
    exit 1
  fi
else
  echo "[REFERENCE NPZ EXISTS] ${REF_NPZ}"
fi

# ============================================================
# Stage 2: prepare all missing sample NPZs sequentially
#
# Do this before GPU parallel evaluation to avoid four workers
# simultaneously writing approximately 10 GiB NPZ files.
# ============================================================

echo
echo "================================================================================"
echo "[STAGE 2] Checking and packing missing sample NPZs"
echo "================================================================================"

PACK_FAILED=0
PACK_CREATED=0
PACK_SKIPPED=0

for item in "${MODELS[@]}"; do
  IFS="|" read -r MODEL TAG <<< "${item}"

  EXP_NAME="ifid-iis-${TAG}-slerp-a05-top10-200000x50000"

  IMAGE_DIR="${IMAGE_ROOT}/${EXP_NAME}_extra/intp_dist"
  SAMPLE_NPZ="${NPZ_DIR}/${EXP_NAME}.npz"
  OUT_JSON="${RESULT_DIR}/${EXP_NAME}.json"
  PACK_LOG="${LOG_DIR}/${EXP_NAME}.pack.out"

  # Already has all three metrics: no need to require NPZ.
  if is_complete_json "${OUT_JSON}" &&
     [[ "${OVERWRITE_METRICS}" -eq 0 ]]; then
    echo "[PACK SKIP COMPLETE RESULT] ${MODEL}"
    PACK_SKIPPED=$((PACK_SKIPPED + 1))
    continue
  fi

  if valid_npz "${SAMPLE_NPZ}" &&
     [[ "${OVERWRITE_NPZ}" -eq 0 ]]; then
    echo "[PACK REUSE] ${MODEL}: ${SAMPLE_NPZ}"
    PACK_SKIPPED=$((PACK_SKIPPED + 1))
    continue
  fi

  if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "[PACK FAILED] Missing intp_dist: ${IMAGE_DIR}"
    echo "MISSING_IMAGE_DIR" > "${PACK_LOG}.failed"
    PACK_FAILED=$((PACK_FAILED + 1))
    continue
  fi

  IMAGE_COUNT="$(
    find "${IMAGE_DIR}" \
      -type f \
      \( \
        -iname '*.png' \
        -o -iname '*.jpg' \
        -o -iname '*.jpeg' \
        -o -iname '*.webp' \
        -o -iname '*.bmp' \
      \) \
      | wc -l
  )"

  if [[ "${IMAGE_COUNT}" -ne 50000 ]]; then
    echo "[PACK FAILED] ${MODEL}: found ${IMAGE_COUNT}, expected 50000"
    echo "IMAGE_COUNT_${IMAGE_COUNT}" > "${PACK_LOG}.failed"
    PACK_FAILED=$((PACK_FAILED + 1))
    continue
  fi

  echo "[PACK START] ${MODEL}"

  PACK_ARGS=(
    --image-dir="${IMAGE_DIR}"
    --out-npz="${SAMPLE_NPZ}"
    --expected=50000
    --image-size=256
  )

  if [[ "${OVERWRITE_NPZ}" -eq 1 ]]; then
    PACK_ARGS+=(--overwrite)
  fi

  python "${PACK_SCRIPT}" "${PACK_ARGS[@]}" \
    > "${PACK_LOG}" 2>&1

  status=$?

  if [[ "${status}" -eq 0 ]] && valid_npz "${SAMPLE_NPZ}"; then
    rm -f "${PACK_LOG}.failed"
    touch "${PACK_LOG}.done"

    echo "[PACK DONE] ${MODEL}"
    PACK_CREATED=$((PACK_CREATED + 1))
  else
    echo "${status}" > "${PACK_LOG}.failed"

    echo "[PACK FAILED] ${MODEL}, status=${status}"
    tail -n 100 "${PACK_LOG}" || true

    PACK_FAILED=$((PACK_FAILED + 1))
  fi
done

echo
echo "[PACK SUMMARY]"
echo "created=${PACK_CREATED}"
echo "skipped=${PACK_SKIPPED}"
echo "failed=${PACK_FAILED}"

if (( PACK_FAILED > 0 )); then
  echo "[FATAL] Some sample NPZs could not be prepared."
  exit 1
fi

# ============================================================
# Stage 3: compute common reference pool once
# ============================================================

REF_POOL="${CACHE_DIR}/imagenet-val-src_dist-pool-50000.npy"
REF_LOG="${LOG_DIR}/cache_reference_pool.out"

if [[ ! -s "${REF_POOL}" ]]; then
  echo
  echo "================================================================================"
  echo "[STAGE 3] Computing shared reference activations on GPU ${GPU_ARRAY[0]}"
  echo "================================================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  python "${CACHE_SCRIPT}" \
    --guided-diffusion-dir="${GUIDED_DIFFUSION_DIR}" \
    --ref-npz="${REF_NPZ}" \
    --out-npy="${REF_POOL}" \
    --tf-batch-size="${TF_BATCH_SIZE}" \
    --softmax-batch-size="${SOFTMAX_BATCH_SIZE}" \
    > "${REF_LOG}" 2>&1

  status=$?

  if [[ "${status}" -ne 0 || ! -s "${REF_POOL}" ]]; then
    echo "[FATAL] Reference activation cache failed."
    tail -n 120 "${REF_LOG}" || true
    exit 1
  fi
else
  echo "[REFERENCE POOL EXISTS] ${REF_POOL}"
fi

# ============================================================
# Stage 4: four GPU workers
# ============================================================

run_worker() {
  local worker_id="$1"
  local physical_gpu="$2"

  local index=0
  local worker_done=0
  local worker_skip=0
  local worker_failed=0

  echo "================================================================================"
  echo "[WORKER ${worker_id}] START"
  echo "physical_gpu=${physical_gpu}"
  echo "================================================================================"

  for item in "${MODELS[@]}"; do
    if (( index % NUM_WORKERS != worker_id )); then
      index=$((index + 1))
      continue
    fi

    IFS="|" read -r MODEL TAG <<< "${item}"

    EXP_NAME="ifid-iis-${TAG}-slerp-a05-top10-200000x50000"

    SAMPLE_NPZ="${NPZ_DIR}/${EXP_NAME}.npz"
    OUT_JSON="${RESULT_DIR}/${EXP_NAME}.json"
    METRIC_LOG="${LOG_DIR}/${EXP_NAME}.metrics.out"

    echo
    echo "--------------------------------------------------------------------------------"
    echo "[WORKER ${worker_id}] MODEL=${MODEL}"
    echo "[WORKER ${worker_id}] GPU=${physical_gpu}"
    echo "[WORKER ${worker_id}] JSON=${OUT_JSON}"
    echo "--------------------------------------------------------------------------------"

    if is_complete_json "${OUT_JSON}" &&
       [[ "${OVERWRITE_METRICS}" -eq 0 ]]; then
      echo "[SKIP COMPLETE] ${MODEL}"
      worker_skip=$((worker_skip + 1))
      index=$((index + 1))
      continue
    fi

    if ! valid_npz "${SAMPLE_NPZ}"; then
      echo "[FAILED] Missing or invalid NPZ: ${SAMPLE_NPZ}"
      echo "MISSING_OR_INVALID_NPZ" > "${METRIC_LOG}.failed"

      worker_failed=$((worker_failed + 1))
      index=$((index + 1))
      continue
    fi

    rm -f \
      "${METRIC_LOG}.done" \
      "${METRIC_LOG}.failed"

    echo "[START] ${MODEL} on GPU ${physical_gpu}"

    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
    MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
    python "${EVAL_SCRIPT}" \
      --guided-diffusion-dir="${GUIDED_DIFFUSION_DIR}" \
      --reference-pool="${REF_POOL}" \
      --sample-npz="${SAMPLE_NPZ}" \
      --out-json="${OUT_JSON}" \
      --model="${MODEL}" \
      --tag="${TAG}" \
      --tf-batch-size="${TF_BATCH_SIZE}" \
      --softmax-batch-size="${SOFTMAX_BATCH_SIZE}" \
      > "${METRIC_LOG}" 2>&1

    status=$?

    if [[ "${status}" -eq 0 ]] &&
       is_complete_json "${OUT_JSON}"; then

      touch "${METRIC_LOG}.done"
      rm -f "${METRIC_LOG}.failed"

      echo "[DONE] ${MODEL}"
      tail -n 25 "${METRIC_LOG}" || true

      worker_done=$((worker_done + 1))
    else
      echo "${status}" > "${METRIC_LOG}.failed"

      echo "[FAILED] ${MODEL}, status=${status}"
      tail -n 120 "${METRIC_LOG}" || true

      worker_failed=$((worker_failed + 1))
    fi

    index=$((index + 1))
  done

  echo
  echo "================================================================================"
  echo "[WORKER ${worker_id}] FINISHED"
  echo "GPU=${physical_gpu}"
  echo "done=${worker_done}"
  echo "skipped=${worker_skip}"
  echo "failed=${worker_failed}"
  echo "================================================================================"

  if (( worker_failed > 0 )); then
    return 1
  fi
}

echo
echo "================================================================================"
echo "[STAGE 4] Starting ${NUM_WORKERS} parallel GPU workers"
echo "================================================================================"

PIDS=()

for worker_id in "${!GPU_ARRAY[@]}"; do
  physical_gpu="${GPU_ARRAY[$worker_id]}"
  worker_log="${LOG_DIR}/worker_gpu${physical_gpu}.out"

  run_worker "${worker_id}" "${physical_gpu}" \
    > "${worker_log}" 2>&1 &

  PIDS+=("$!")

  echo \
    "[LAUNCHED] worker=${worker_id}, " \
    "gpu=${physical_gpu}, pid=${PIDS[-1]}, log=${worker_log}"
done

WORKER_FAILURES=0

for pid in "${PIDS[@]}"; do
  if wait "${pid}"; then
    :
  else
    WORKER_FAILURES=$((WORKER_FAILURES + 1))
  fi
done

echo
echo "================================================================================"
echo "[ALL GPU WORKERS FINISHED]"
echo "worker_failures=${WORKER_FAILURES}"
echo "================================================================================"

# ============================================================
# Final validation
# ============================================================

COMPLETE=0
INCOMPLETE=0

for item in "${MODELS[@]}"; do
  IFS="|" read -r MODEL TAG <<< "${item}"

  EXP_NAME="ifid-iis-${TAG}-slerp-a05-top10-200000x50000"
  OUT_JSON="${RESULT_DIR}/${EXP_NAME}.json"

  if is_complete_json "${OUT_JSON}"; then
    COMPLETE=$((COMPLETE + 1))
  else
    echo "[INCOMPLETE] ${MODEL}: ${OUT_JSON}"
    INCOMPLETE=$((INCOMPLETE + 1))
  fi
done

echo
echo "================================================================================"
echo "[FINAL SUMMARY]"
echo "complete=${COMPLETE}/25"
echo "incomplete=${INCOMPLETE}"
echo
echo "[FAILED FILES]"
find "${LOG_DIR}" \
  -maxdepth 1 \
  -type f \
  -name "*.failed" \
  -print
echo "================================================================================"

if (( INCOMPLETE > 0 || WORKER_FAILURES > 0 )); then
  exit 1
fi
