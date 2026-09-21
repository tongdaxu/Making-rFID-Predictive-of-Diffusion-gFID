#!/usr/bin/env bash
set -euo pipefail

# Dynamic scheduler for VAE linear probing.
# When one GPU finishes, the next VAE config starts immediately on that GPU.

TRAIN_DATA="/video_ssd/lpm/ImageNet/train"
VAL_DATA="/video_ssd/lpm/ImageNet/val"

CONFIG_DIR="./configs"
SCRIPT="eval_linear_vae_pool.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/linear_probemaxpool"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/status"

mkdir -p "${OUT_ROOT}"
mkdir -p "${LOG_DIR}"
mkdir -p "${STATUS_DIR}"

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
  # "DETOK.yaml"
  "QWVAE.yaml"
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

# Default for one 8-GPU group visible as 0..7.
# If avgpool/maxpool are launched on the SAME 16-GPU machine, set one script to:
#   GPU_LIST=(0 1 2 3 4 5 6 7)
# and the other to:
#   GPU_LIST=(8 9 10 11 12 13 14 15)
GPU_LIST=(0)

BASE_PORT=29900

EPOCHS=5
BATCH_SIZE=64
LR=0.001
RESOLUTION=256
NUM_WORKERS=4
MIXED_PRECISION="no"

REPORT_TO="wandb"
# REPORT_TO="none"

TRAIN_SIZE=-1
VAL_SIZE=-1

FEATURE_TYPE="maxpool"
POLL_INTERVAL=10

declare -a GPU_PID
declare -a GPU_CONFIG
declare -a GPU_LOG
declare -a GPU_STATUS
declare -a GPU_PHYSICAL_ID

NEXT_IDX=0
FAILED=0

launch_job() {
  local slot="$1"
  local physical_gpu="$2"
  local config="$3"

  local port=$((BASE_PORT + slot))
  local name="${config%.yaml}"
  local name_lower
  name_lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"

  local exp_name="linear_${name_lower}_${FEATURE_TYPE}"
  local out_dir="${OUT_ROOT}/${exp_name}"
  local out_file="${LOG_DIR}/${exp_name}.out"
  local status_file="${STATUS_DIR}/${exp_name}.status"

  mkdir -p "${out_dir}"
  rm -f "${status_file}"

  echo "Launching:"
  echo "  slot:         ${slot}"
  echo "  gpu:          ${physical_gpu}"
  echo "  port:         ${port}"
  echo "  config:       ${config}"
  echo "  feature_type: ${FEATURE_TYPE}"
  echo "  out_dir:      ${out_dir}"
  echo "  log:          ${out_file}"

  (
    set +e

    env CUDA_VISIBLE_DEVICES="${physical_gpu}" accelerate launch \
      --num_processes=1 \
      --main_process_port "${port}" \
      "${SCRIPT}" \
      --vae-config "${CONFIG_DIR}/${config}" \
      --train-data "${TRAIN_DATA}" \
      --val-data "${VAL_DATA}" \
      --output-dir "${out_dir}" \
      --exp-name "${exp_name}" \
      --feature-type "${FEATURE_TYPE}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --resolution "${RESOLUTION}" \
      --num-workers "${NUM_WORKERS}" \
      --train-size "${TRAIN_SIZE}" \
      --val-size "${VAL_SIZE}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --report-to "${REPORT_TO}" \
      > "${out_file}" 2>&1

    code=$?
    echo "${code}" > "${status_file}"
    exit "${code}"
  ) &

  local pid=$!

  GPU_PID["${slot}"]="${pid}"
  GPU_CONFIG["${slot}"]="${config}"
  GPU_LOG["${slot}"]="${out_file}"
  GPU_STATUS["${slot}"]="${status_file}"
  GPU_PHYSICAL_ID["${slot}"]="${physical_gpu}"

  echo "  pid:          ${pid}"
  echo
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
    GPU_CONFIG["${slot}"]=""
    GPU_LOG["${slot}"]=""
    GPU_STATUS["${slot}"]=""
  fi
}

# Fill GPUs first.
for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
  physical_gpu="${GPU_LIST[$slot]}"

  GPU_PID["${slot}"]=""
  GPU_CONFIG["${slot}"]=""
  GPU_LOG["${slot}"]=""
  GPU_STATUS["${slot}"]=""
  GPU_PHYSICAL_ID["${slot}"]="${physical_gpu}"

  launch_next_if_any "${slot}" "${physical_gpu}"
done

# Dynamic scheduling.
while true; do
  ACTIVE=0

  for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
    pid="${GPU_PID[$slot]:-}"
    physical_gpu="${GPU_PHYSICAL_ID[$slot]:-${GPU_LIST[$slot]}}"

    if [[ -z "${pid}" ]]; then
      continue
    fi

    ACTIVE=$((ACTIVE + 1))

    if ! kill -0 "${pid}" 2>/dev/null; then
      config="${GPU_CONFIG[$slot]}"
      log_file="${GPU_LOG[$slot]}"
      status_file="${GPU_STATUS[$slot]}"

      code="unknown"
      if [[ -f "${status_file}" ]]; then
        code="$(cat "${status_file}")"
      fi

      if [[ "${code}" == "0" ]]; then
        echo "Finished OK: ${config} on GPU ${physical_gpu}"
      else
        echo "FAILED: ${config} on GPU ${physical_gpu}, exit_code=${code}"
        echo "Log: ${log_file}"
        FAILED=1
      fi

      echo "GPU ${physical_gpu} is free. Launching next job if available..."
      launch_next_if_any "${slot}" "${physical_gpu}"
    fi
  done

  if (( ACTIVE == 0 && NEXT_IDX >= ${#CONFIGS[@]} )); then
    break
  fi

  sleep "${POLL_INTERVAL}"
done

if (( FAILED != 0 )); then
  echo "Some ${FEATURE_TYPE} linear probe jobs failed. Check logs in ${LOG_DIR}"
  exit 1
fi

echo "All ${FEATURE_TYPE} linear probe jobs finished."
