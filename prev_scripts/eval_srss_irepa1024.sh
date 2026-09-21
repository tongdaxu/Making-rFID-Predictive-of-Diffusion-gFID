#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run SRSS only for all VAE configs on the iREPA masked subset.
#
# Required prepared data:
#   /video_ssd/kongzishang/xutongda/git/IFID/iREPA/metrics/spatial-metrics-data/images
#   /video_ssd/kongzishang/xutongda/git/IFID/iREPA/metrics/spatial-metrics-data/masks
#
# This script creates an ImageFolder-style temporary dataset:
#   spatial-metrics-data/images_imagefolder_1class/irepa/*.jpg
# so that ImageNet-style dataset loaders can read the 1024 iREPA images.
# ============================================================

DATA_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/iREPA/metrics/spatial-metrics-data"
IMAGE_SRC="${DATA_ROOT}/images"
MASK_DIR="${DATA_ROOT}/masks"
DATASET_ROOT="${DATA_ROOT}/images_imagefolder_1class"

CONFIG_DIR="./configs"
SCRIPT="eval_latent_viv_cds_srss.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_viv_cds_srss/srss"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

CONFIGS=(
  # "SDVAE.yaml"
  # "SD3VAE.yaml"
  # "FLUXVAE.yaml"
  # "DCAE.yaml"
  # "SOFTVQ.yaml"
  # "VAVAE.yaml"
  # "VAVAE64.yaml"
  # "EQVAE.yaml"
  # "MAETOK.yaml"
  # "INVAE.yaml"
  # "REPAEVAE.yaml"
  "DETOK.yaml"
  # "QWVAE.yaml"
  # "RAE.yaml"
  # "SVG.yaml"
  # "FLUX2VAE.yaml"
  # "SVGT2I_P_STAGE1_256.yaml"
  # "DMVAE.yaml"
  # "UAE.yaml"
  # "VTPS.yaml"
  # "VTPB.yaml"
  # "VTPL.yaml"
  # "ScaleRAE_SIGLIP2_NONORM.yaml"
  # "ScaleRAE_WEBSSL.yaml"
  # "PAE_DINOv2L_d32.yaml"
)

GPU_LIST=(0 1 2 3 4 5 6 7)

BATCH_SIZE=16
NUM_WORKERS=4
STORE_DTYPE="fp32"

# For a cross-VAE table with 8x8 / 16x16 / 32x32 grids, adaptive is recommended.
# It keeps the iREPA ratio: 64/256 = 25% foreground and 25% background.
# If you want strict original 16x16 iREPA filtering, set SRSS_FILTER_MODE="strict",
# but 8x8 latents will be filtered out.
SRSS_FILTER_MODE="adaptive"  # adaptive | strict | none
SRSS_MIN_POS_FRAC=0.25
SRSS_MIN_NEG_FRAC=0.25
SRSS_STRICT_MIN_POS_PATCHES=64
SRSS_STRICT_MIN_NEG_PATCHES=64

POLL_INTERVAL=10
SKIP_DONE=1

prepare_dataset() {
  if [[ ! -d "${IMAGE_SRC}" ]]; then
    echo "ERROR: IMAGE_SRC not found: ${IMAGE_SRC}" >&2
    exit 1
  fi
  if [[ ! -d "${MASK_DIR}" ]]; then
    echo "ERROR: MASK_DIR not found: ${MASK_DIR}" >&2
    exit 1
  fi

  echo "[Prepare] creating ImageFolder-style dataset at ${DATASET_ROOT}"
  rm -rf "${DATASET_ROOT}"
  mkdir -p "${DATASET_ROOT}/irepa"

  while IFS= read -r -d '' f; do
    ln -sf "${f}" "${DATASET_ROOT}/irepa/$(basename "${f}")"
  done < <(find "${IMAGE_SRC}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) -print0 | sort -z)

  IMAGE_COUNT=$(find "${DATASET_ROOT}/irepa" -maxdepth 1 \( -type f -o -type l \) | wc -l)
  MASK_COUNT=$(find "${MASK_DIR}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l)

  echo "[Check] image_count=${IMAGE_COUNT}"
  echo "[Check] mask_count=${MASK_COUNT}"

  if (( IMAGE_COUNT == 0 )); then
    echo "ERROR: no images found under ${IMAGE_SRC}" >&2
    exit 1
  fi
  if (( MASK_COUNT == 0 )); then
    echo "ERROR: no masks found under ${MASK_DIR}" >&2
    exit 1
  fi
  if (( IMAGE_COUNT != MASK_COUNT )); then
    echo "WARNING: image_count != mask_count. Will use NUM_IMAGES=min(image_count, mask_count)." >&2
  fi

  if (( MASK_COUNT < IMAGE_COUNT )); then
    NUM_IMAGES="${MASK_COUNT}"
  else
    NUM_IMAGES="${IMAGE_COUNT}"
  fi
  export NUM_IMAGES

  echo "[Check] NUM_IMAGES=${NUM_IMAGES}"

  # Fail early if image stems do not have matching masks.
  python - <<PY
from pathlib import Path
import sys

img_dir = Path("${DATASET_ROOT}") / "irepa"
mask_dir = Path("${MASK_DIR}")
imgs = sorted([p for p in img_dir.iterdir() if p.is_file() or p.is_symlink()])
exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

missing = []
for img in imgs[:min(len(imgs), int("${NUM_IMAGES}"))]:
    stem = img.stem
    ok = any((mask_dir / f"{stem}{ext}").exists() for ext in exts)
    if not ok:
        missing.append(img.name)
        if len(missing) >= 10:
            break

if missing:
    print("ERROR: Some image files do not have same-stem masks under mask dir.", file=sys.stderr)
    print("Examples:", missing, file=sys.stderr)
    print("This script expects image stem xxx and mask xxx.png/jpg/...", file=sys.stderr)
    sys.exit(2)

print("[Check] image-mask stem matching looks OK.")
PY
}

extra_args_for_config() {
  local config="$1"
  case "${config}" in
    # These have 128 tokens x 32 dim in your table. If the actual layout is 16x8,
    # change 8 16 to 16 8.
    "MAETOK.yaml")
      echo "--token-grid 8 16"
      ;;
    *)
      echo ""
      ;;
  esac
}

is_done() {
  local out_dir="$1"
  if (( SKIP_DONE == 0 )); then
    return 1
  fi
  python - "$out_dir/metrics.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(1)
try:
    m = json.loads(p.read_text())
except Exception:
    sys.exit(1)
if "SRSS" in m and m["SRSS"] is not None:
    sys.exit(0)
sys.exit(1)
PY
}

declare -a GPU_PID GPU_CONFIG GPU_LOG GPU_STATUS GPU_PHYSICAL_ID
NEXT_IDX=0
FAILED=0

launch_job() {
  local slot="$1"
  local physical_gpu="$2"
  local config="$3"

  local name="${config%.yaml}"
  local name_lower
  name_lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"

  local out_dir="${OUT_ROOT}/${name_lower}"
  local out_file="${LOG_DIR}/${name_lower}.out"
  local status_file="${LOG_DIR}/${name_lower}.status"
  local extra_args
  extra_args="$(extra_args_for_config "${config}")"

  mkdir -p "${out_dir}"
  rm -f "${status_file}"

  echo "Launching SRSS ${config} on GPU ${physical_gpu}; NUM_IMAGES=${NUM_IMAGES}"
  (
    set +e
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" python "${SCRIPT}" \
      --vae-config "${CONFIG_DIR}/${config}" \
      --dataset "${DATASET_ROOT}" \
      --output-dir "${out_dir}" \
      --num-images "${NUM_IMAGES}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --store-dtype "${STORE_DTYPE}" \
      --metrics srss \
      --mask-dir "${MASK_DIR}" \
      --srss-filter-mode "${SRSS_FILTER_MODE}" \
      --srss-min-pos-frac "${SRSS_MIN_POS_FRAC}" \
      --srss-min-neg-frac "${SRSS_MIN_NEG_FRAC}" \
      --srss-min-pos-patches "${SRSS_STRICT_MIN_POS_PATCHES}" \
      --srss-min-neg-patches "${SRSS_STRICT_MIN_NEG_PATCHES}" \
      ${extra_args} \
      > "${out_file}" 2>&1
    code=$?
    echo "${code}" > "${status_file}"
    exit "${code}"
  ) &

  GPU_PID["${slot}"]="$!"
  GPU_CONFIG["${slot}"]="${config}"
  GPU_LOG["${slot}"]="${out_file}"
  GPU_STATUS["${slot}"]="${status_file}"
  GPU_PHYSICAL_ID["${slot}"]="${physical_gpu}"
}

launch_next_if_any() {
  local slot="$1"
  local physical_gpu="$2"

  while (( NEXT_IDX < ${#CONFIGS[@]} )); do
    local config="${CONFIGS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))

    local name="${config%.yaml}"
    local name_lower
    name_lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"
    local out_dir="${OUT_ROOT}/${name_lower}"

    if is_done "${out_dir}"; then
      echo "Skip done: ${config}"
      continue
    fi

    launch_job "${slot}" "${physical_gpu}" "${config}"
    return
  done

  GPU_PID["${slot}"]=""
}

cleanup() {
  echo "Caught signal. Killing child jobs..."
  for pid in "${GPU_PID[@]:-}"; do
    [[ -n "${pid:-}" ]] && kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

prepare_dataset

for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
  launch_next_if_any "${slot}" "${GPU_LIST[$slot]}"
done

while true; do
  ACTIVE=0
  for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
    pid="${GPU_PID[$slot]:-}"
    [[ -z "${pid}" ]] && continue
    ACTIVE=$((ACTIVE + 1))

    if ! kill -0 "${pid}" 2>/dev/null; then
      config="${GPU_CONFIG[$slot]}"
      log_file="${GPU_LOG[$slot]}"
      status_file="${GPU_STATUS[$slot]}"
      physical_gpu="${GPU_PHYSICAL_ID[$slot]}"
      code="unknown"
      [[ -f "${status_file}" ]] && code="$(cat "${status_file}")"

      if [[ "${code}" == "0" ]]; then
        echo "Finished OK: ${config} on GPU ${physical_gpu}"
      else
        echo "FAILED: ${config} on GPU ${physical_gpu}, code=${code}, log=${log_file}"
        FAILED=1
      fi

      launch_next_if_any "${slot}" "${physical_gpu}"
    fi
  done

  if (( ACTIVE == 0 && NEXT_IDX >= ${#CONFIGS[@]} )); then
    break
  fi

  sleep "${POLL_INTERVAL}"
done

if (( FAILED != 0 )); then
  echo "Some SRSS jobs failed. Check ${LOG_DIR}"
  exit 1
fi

echo "All SRSS jobs finished. Results under ${OUT_ROOT}"
