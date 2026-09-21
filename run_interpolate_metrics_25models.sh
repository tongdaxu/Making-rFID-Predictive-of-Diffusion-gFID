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

GPU="${GPU:-0}"
SCOPE="${SCOPE:-all25}"  # all25 / extra3

TF_BATCH_SIZE="${TF_BATCH_SIZE:-64}"
SOFTMAX_BATCH_SIZE="${SOFTMAX_BATCH_SIZE:-512}"

OVERWRITE_NPZ="${OVERWRITE_NPZ:-0}"
OVERWRITE_METRICS="${OVERWRITE_METRICS:-0}"

mkdir -p \
  "${NPZ_DIR}" \
  "${CACHE_DIR}" \
  "${RESULT_DIR}" \
  "${LOG_DIR}"

# ------------------------------------------------------------------
# Locate guided-diffusion
# ------------------------------------------------------------------
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
  echo "[FATAL] Missing evaluator.py under ${GUIDED_DIFFUSION_DIR}"
  exit 1
fi

for script in "${PACK_SCRIPT}" "${CACHE_SCRIPT}" "${EVAL_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "[FATAL] Missing script: ${script}"
    exit 1
  fi
done

# display | tag
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

is_extra3() {
  case "$1" in
    uae|scalerae-siglip2|scalerae-webssl)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_complete_json() {
  python - "$1" <<'PY'
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

# ------------------------------------------------------------------
# Prepare common ImageNet validation reference NPZ
# ------------------------------------------------------------------
REF_NPZ="${NPZ_DIR}/imagenet-val-src_dist-50000.npz"

if [[ ! -s "${REF_NPZ}" ]]; then
  REF_IMAGE_DIR="${IMAGE_ROOT}/ifid-iis-dcae-slerp-a05-top10-200000x50000_extra/src_dist"

  echo "[REFERENCE NPZ] Packing DCAE src_dist..."

  python "${PACK_SCRIPT}" \
    --image-dir="${REF_IMAGE_DIR}" \
    --out-npz="${REF_NPZ}" \
    --expected=50000 \
    --image-size=256
else
  echo "[REFERENCE NPZ] Reusing ${REF_NPZ}"
fi

# ------------------------------------------------------------------
# Prepare shared reference activations
# ------------------------------------------------------------------
REF_POOL="${CACHE_DIR}/imagenet-val-src_dist-pool-50000.npy"
REF_LOG="${LOG_DIR}/cache_reference_pool.out"

if [[ ! -s "${REF_POOL}" ]]; then
  echo "[REFERENCE POOL] Computing common reference activations..."

  CUDA_VISIBLE_DEVICES="${GPU}" \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  python "${CACHE_SCRIPT}" \
    --guided-diffusion-dir="${GUIDED_DIFFUSION_DIR}" \
    --ref-npz="${REF_NPZ}" \
    --out-npy="${REF_POOL}" \
    --tf-batch-size="${TF_BATCH_SIZE}" \
    --softmax-batch-size="${SOFTMAX_BATCH_SIZE}" \
    > "${REF_LOG}" 2>&1

  status=$?

  if [[ "${status}" -ne 0 ]]; then
    echo "[FATAL] Reference activation cache failed."
    tail -n 120 "${REF_LOG}" || true
    exit 1
  fi
else
  echo "[REFERENCE POOL] Reusing ${REF_POOL}"
fi

echo "================================================================================"
echo "[INTERPOLATE iIS / iPrec / iRec]"
echo "SCOPE=${SCOPE}"
echo "GPU=${GPU}"
echo "GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "NPZ_DIR=${NPZ_DIR}"
echo "REF_POOL=${REF_POOL}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"

DONE=0
SKIPPED=0
FAILED=0

for item in "${MODELS[@]}"; do
  IFS="|" read -r MODEL TAG <<< "${item}"

  if [[ "${SCOPE}" == "extra3" ]] && ! is_extra3 "${TAG}"; then
    continue
  fi

  EXP_NAME="ifid-iis-${TAG}-slerp-a05-top10-200000x50000"

  IMAGE_DIR="${IMAGE_ROOT}/${EXP_NAME}_extra/intp_dist"
  SAMPLE_NPZ="${NPZ_DIR}/${EXP_NAME}.npz"
  OUT_JSON="${RESULT_DIR}/${EXP_NAME}.json"

  PACK_LOG="${LOG_DIR}/${EXP_NAME}.pack.out"
  METRIC_LOG="${LOG_DIR}/${EXP_NAME}.metrics.out"

  echo
  echo "================================================================================"
  echo "[MODEL] ${MODEL}"
  echo "[TAG] ${TAG}"
  echo "[IMAGE_DIR] ${IMAGE_DIR}"
  echo "[SAMPLE_NPZ] ${SAMPLE_NPZ}"
  echo "[OUT_JSON] ${OUT_JSON}"
  echo "================================================================================"

  if is_complete_json "${OUT_JSON}" &&
     [[ "${OVERWRITE_METRICS}" -eq 0 ]]; then
    echo "[SKIP COMPLETE] ${MODEL}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if [[ ! -d "${IMAGE_DIR}" ]]; then
    echo "[FAILED] Missing intp_dist: ${IMAGE_DIR}"
    echo "MISSING_IMAGE_DIR" > "${METRIC_LOG}.failed"
    FAILED=$((FAILED + 1))
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
      \) \
      | wc -l
  )"

  if [[ "${IMAGE_COUNT}" -ne 50000 ]]; then
    echo "[FAILED] ${MODEL} has ${IMAGE_COUNT} images, expected 50000."
    echo "IMAGE_COUNT_${IMAGE_COUNT}" > "${METRIC_LOG}.failed"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Pack only when missing, unless overwrite requested.
  if [[ ! -s "${SAMPLE_NPZ}" || "${OVERWRITE_NPZ}" -eq 1 ]]; then
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

    if [[ "${status}" -ne 0 ]]; then
      echo "[FAILED] NPZ packing failed for ${MODEL}"
      echo "${status}" > "${PACK_LOG}.failed"
      tail -n 120 "${PACK_LOG}" || true
      FAILED=$((FAILED + 1))
      continue
    fi
  else
    echo "[REUSE NPZ] ${SAMPLE_NPZ}"
  fi

  rm -f \
    "${METRIC_LOG}.failed" \
    "${METRIC_LOG}.done"

  set +e

  CUDA_VISIBLE_DEVICES="${GPU}" \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
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

  set -e

  if [[ "${status}" -eq 0 ]] && is_complete_json "${OUT_JSON}"; then
    touch "${METRIC_LOG}.done"
    rm -f "${METRIC_LOG}.failed"

    echo "[DONE] ${MODEL}"
    tail -n 30 "${METRIC_LOG}" || true

    DONE=$((DONE + 1))
  else
    echo "${status}" > "${METRIC_LOG}.failed"

    echo "[FAILED] ${MODEL}, status=${status}"
    tail -n 120 "${METRIC_LOG}" || true

    FAILED=$((FAILED + 1))
  fi
done

echo
echo "================================================================================"
echo "[FINISHED]"
echo "DONE=${DONE}"
echo "SKIPPED=${SKIPPED}"
echo "FAILED=${FAILED}"
echo
echo "[FAILED FILES]"
find "${LOG_DIR}" \
  -maxdepth 1 \
  -type f \
  -name "*.failed" \
  -print
echo "================================================================================"

if (( FAILED > 0 )); then
  exit 1
fi
