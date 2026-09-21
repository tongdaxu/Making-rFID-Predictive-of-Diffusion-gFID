#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/video_ssd/kongzishang/imagenet_val"
CONFIG_DIR="./configs"
SCRIPT="eval_gmm.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/losses/gmm"
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
  # "DETOK.yaml"
  # "QWVAE.yaml"
  "RAE.yaml"
  "SVG.yaml"
  # "FLUX2VAE.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
  # "DMVAE.yaml"
  "UAE.yaml"
  # "VTPS.yaml"
  # "VTPB.yaml"
  # "VTPL.yaml"
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE_WEBSSL.yaml"
  # "PAE_DINOv2L_d32.yaml"
)

GPU_LIST=(0 1 2 3 4 5)
BASE_PORT=29800
POLL_INTERVAL=10

N_ITER=500
BS=64
BATCH_SIZE=1024
NUM_WORKERS=8

# UMAP is CPU-heavy and can leave the GPU idle for a long time.
# For a quick sanity/full-table run under a strict GPU-reclaim policy, consider PCA first.
DIM_REDUCTION="UMAP"   # UMAP | PCA | None
TARGET_DIM=100
FACTOR=1.0
USE_WANDB=1

declare -a GPU_PID GPU_CONFIG GPU_LOG GPU_STATUS GPU_PHYSICAL_ID
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

  local exp_name="${name_lower}_vae_gmm"
  local out_file="${LOG_DIR}/eval_gmm_${name_lower}.out"
  local status_file="${LOG_DIR}/eval_gmm_${name_lower}.status"

  rm -f "${status_file}"

  echo "Launching ${config} on GPU ${physical_gpu}, port ${port}"
  echo "Exp: ${exp_name}"
  echo "Log: ${out_file}"

  (
    set +e
    env CUDA_VISIBLE_DEVICES="${physical_gpu}" accelerate launch \
      --num_processes=1 \
      --main_process_port "${port}" \
      "${SCRIPT}" \
      --base_path "${BASE_PATH}" \
      --out_dir "${OUT_ROOT}" \
      --exp "${exp_name}" \
      --vae-config "${CONFIG_DIR}/${config}" \
      --n_iter "${N_ITER}" \
      --bs "${BS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --dim_reduction "${DIM_REDUCTION}" \
      --target_dim "${TARGET_DIM}" \
      --factor "${FACTOR}" \
      --use_wandb "${USE_WANDB}" \
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
  echo "Some eval_gmm jobs failed. Check ${LOG_DIR}"
  exit 1
fi

echo "All eval_gmm jobs finished."
