#!/usr/bin/env bash
set -euo pipefail

DATASET="/video_ssd/lpm/ImageNet/val"
CONFIG_DIR="./configs"
SCRIPT="eval_latent_viv_cds_srss.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_viv_cds_srss"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

CONFIGS=(
  "SDVAE.yaml"
  "SD3VAE.yaml"
  "FLUXVAE.yaml"
  "DCAE.yaml"
  "SOFTVQ.yaml"
  "VAVAE.yaml"
  "VAVAE64.yaml"
  "EQVAE.yaml"
  "MAETOK.yaml"
  "INVAE.yaml"
  "REPAEVAE.yaml"
  "DETOK.yaml"
  "QWVAE.yaml"
  "RAE.yaml"
  "SVG.yaml"
  "FLUX2VAE.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
  "DMVAE.yaml"
  "UAE.yaml"
  "VTPS.yaml"
  "VTPB.yaml"
  "VTPL.yaml"
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE_WEBSSL.yaml"
  "PAE_DINOv2L_d32.yaml"
)

GPU_LIST=(0 1 2 3 4 5 6 7)

NUM_IMAGES=50000
BATCH_SIZE=16
NUM_WORKERS=4
STORE_DTYPE="fp32"

# SRSS is an official mask-aware metric. To compute all three metrics, set one of:
#   MASK_ARGS=(--mask-npz /path/to/masks_val_50k.npz)
#   MASK_ARGS=(--mask-dir /path/to/mask_images)
#
# If MASK_ARGS is empty, this script intentionally computes only VIV/CDS.
# Set REQUIRE_SRSS=1 to force the script to fail when masks are missing.
MASK_ARGS=()
REQUIRE_SRSS=0

# SRSS mask filtering.
# adaptive is recommended for this VAE table because latent grids include 8x8, 16x16, 32x32,
# and token latents. It preserves iREPA's ratio: 64/256 = 25% foreground and 25% background.
#   8x8   -> 16/16
#   16x16 -> 64/64
#   32x32 -> 256/256
# strict reproduces iREPA's fixed 64/64 comparison threshold.
SRSS_FILTER_MODE="adaptive"  # adaptive | strict | none
SRSS_MIN_POS_FRAC=0.25
SRSS_MIN_NEG_FRAC=0.25
SRSS_STRICT_MIN_POS_PATCHES=64
SRSS_STRICT_MIN_NEG_PATCHES=64

if (( ${#MASK_ARGS[@]} == 0 )); then
  if (( REQUIRE_SRSS != 0 )); then
    echo "ERROR: REQUIRE_SRSS=1 but MASK_ARGS is empty. Set --mask-npz or --mask-dir." >&2
    exit 1
  fi
  echo "WARNING: MASK_ARGS is empty; computing only VIV/CDS, not SRSS." >&2
  METRICS=(viv cds)
else
  METRICS=(viv cds srss)
fi

POLL_INTERVAL=10
declare -a GPU_PID GPU_CONFIG GPU_LOG GPU_STATUS GPU_PHYSICAL_ID
NEXT_IDX=0
FAILED=0

extra_args_for_config() {
  local config="$1"
  case "${config}" in
    "MAETOK.yaml")
      echo "--token-grid 8 16"
      ;;
    *)
      echo ""
      ;;
  esac
}

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

  echo "Launching ${config} on GPU ${physical_gpu}; metrics=${METRICS[*]}"
  (
    set +e
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" python "${SCRIPT}" \
      --vae-config "${CONFIG_DIR}/${config}" \
      --dataset "${DATASET}" \
      --output-dir "${out_dir}" \
      --num-images "${NUM_IMAGES}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --store-dtype "${STORE_DTYPE}" \
      --metrics "${METRICS[@]}" \
      --srss-filter-mode "${SRSS_FILTER_MODE}" \
      --srss-min-pos-frac "${SRSS_MIN_POS_FRAC}" \
      --srss-min-neg-frac "${SRSS_MIN_NEG_FRAC}" \
      --srss-min-pos-patches "${SRSS_STRICT_MIN_POS_PATCHES}" \
      --srss-min-neg-patches "${SRSS_STRICT_MIN_NEG_PATCHES}" \
      ${extra_args} \
      "${MASK_ARGS[@]}" \
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
  if (( NEXT_IDX < ${#CONFIGS[@]} )); then
    local config="${CONFIGS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))
    launch_job "${slot}" "${physical_gpu}" "${config}"
  else
    GPU_PID["${slot}"]=""
  fi
}

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
  echo "Some jobs failed. Check ${LOG_DIR}"
  exit 1
fi

echo "All latent metric jobs finished."
