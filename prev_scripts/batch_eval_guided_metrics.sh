#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation for already-generated VAE diffusion image folders.
# This follows guided-diffusion evaluator input style:
#   generated images -> uint8 NHWC .npz, no resize/crop by default.

GUIDED_DIFFUSION_DIR="./third_party/guided-diffusion"
REF_NPZ="/video_ssd/kongzishang/xutongda/git/IFID/guided_eval_direct/VIRTUAL_imagenet256_labeled.npz"
NPZ_DIR="./guided_eval_npz"
RESULT_DIR="./guided_eval_results"
LOG_DIR="./guided_eval_logs"

mkdir -p "${NPZ_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

SAMPLE_DIRS=(
  # "samples/sit-xl-sdvae_400000_cfg1.0"
  # "samples/sit-xl-sd3vae_400000_cfg1.0"
  # "samples/sit-xl-fluxvae_400000_cfg1.0"
)

LIMIT=50000

# Strict default: no resize. If your sample folders are not already 256x256,
# fix generation/export instead of post-resizing for the main table.
RESIZE=-1

for SAMPLE_DIR in "${SAMPLE_DIRS[@]}"; do
  NAME="$(basename "${SAMPLE_DIR}")"
  SAMPLE_NPZ="${NPZ_DIR}/${NAME}.npz"
  OUT_JSON="${RESULT_DIR}/${NAME}.json"
  OUT_LOG="${LOG_DIR}/${NAME}.out"

  echo "========== ${NAME} =========="
  echo "sample dir: ${SAMPLE_DIR}"
  echo "sample npz: ${SAMPLE_NPZ}"
  echo "out json:   ${OUT_JSON}"
  echo "log:        ${OUT_LOG}"

  if [[ ! -f "${SAMPLE_NPZ}" ]]; then
    python make_guided_npz_from_images.py \
      --image-dir "${SAMPLE_DIR}" \
      --out-npz "${SAMPLE_NPZ}" \
      --limit "${LIMIT}" \
      --resize "${RESIZE}" \
      > "${OUT_LOG}" 2>&1
  else
    echo "NPZ exists, skip conversion: ${SAMPLE_NPZ}" | tee "${OUT_LOG}"
  fi

  python eval_guided_diffusion_metrics.py \
    --guided-diffusion-dir "${GUIDED_DIFFUSION_DIR}" \
    --ref-npz "${REF_NPZ}" \
    --sample-npz "${SAMPLE_NPZ}" \
    --out-json "${OUT_JSON}" \
    >> "${OUT_LOG}" 2>&1

  cat "${OUT_JSON}"
done

echo "All guided-diffusion metrics finished."
