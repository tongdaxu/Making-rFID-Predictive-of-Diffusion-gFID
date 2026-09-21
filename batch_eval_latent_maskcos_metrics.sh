#!/usr/bin/env bash
set -euo pipefail

# Dynamic multi-GPU runner for eval_latent_maskcos_metrics.py.
# Each GPU runs one VAE config; when it finishes, the next config starts.

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"
SCRIPT="${SCRIPT:-eval_latent_maskcos_metrics.py}"
OUT_ROOT="${OUT_ROOT:-/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_maskcos}"
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

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})

NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESOLUTION="${RESOLUTION:-256}"
MASK_RATIO="${MASK_RATIO:-0.75}"
SEED="${SEED:-1234}"
NORMALIZE_LATENT="${NORMALIZE_LATENT:-none}"  # none | per_sample_channel | per_sample_all
T_VALUES=(${T_VALUES:-0.1 0.4 0.6 0.9})
DROP_LAST="${DROP_LAST:-0}"
SAVE_PER_BATCH="${SAVE_PER_BATCH:-0}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

extra_args_for_config() {
  local config="$1"
  case "${config}" in
    # If a wrapper returns [B,N,C] with non-square N, fill exact layout here.
    # Examples, only enable if you have verified the wrapper output layout:
    # "MAETOK.yaml") echo "--token-grid 8 16" ;;
    # "DETOK.yaml") echo "--token-grid 8 16" ;;
    # "DMVAE.yaml") echo "--token-grid 16 16" ;;
    *) echo "" ;;
  esac
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

  echo "Launching ${config} on GPU ${physical_gpu}; output=${out_dir}"
  (
    set +e
    args=(
      --vae-config "${CONFIG_DIR}/${config}"
      --dataset "${DATASET}"
      --output-dir "${out_dir}"
      --num-images "${NUM_IMAGES}"
      --batch-size "${BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --resolution "${RESOLUTION}"
      --mask-ratio "${MASK_RATIO}"
      --seed "${SEED}"
      --normalize-latent "${NORMALIZE_LATENT}"
      --t-values "${T_VALUES[@]}"
    )
    if [[ "${DROP_LAST}" == "1" ]]; then
      args+=(--drop-last)
    fi
    if [[ "${SAVE_PER_BATCH}" == "1" ]]; then
      args+=(--save-per-batch)
    fi

    # shellcheck disable=SC2206
    extra_arr=(${extra_args})

    env CUDA_VISIBLE_DEVICES="${physical_gpu}" python "${SCRIPT}" \
      "${args[@]}" \
      "${extra_arr[@]}" \
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

echo "All mask-cos latent metric jobs finished."
