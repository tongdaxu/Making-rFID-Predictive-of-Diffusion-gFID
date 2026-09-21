#!/usr/bin/env bash
set -euo pipefail

DATASET="/video_ssd/lpm/ImageNet/val"
CONFIG_DIR="./configs"
SCRIPT="eval_latent_sec.py"

OUT_ROOT="/video_ssd/kongzishang/xutongda/git/IFID/latent_metrics_viv_cds_srss/sec"
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
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  # "ScaleRAE_WEBSSL.yaml"
  # "PAE_DINOv2L_d32.yaml"
)

GPU_LIST=(4 5 6 7)

NUM_IMAGES=50000
BATCH_SIZE=16
NUM_WORKERS=4
STORE_DTYPE="fp32"
THRESHOLDS=(0.25 0.5)
DIST_TYPE="Manhattan"

POLL_INTERVAL=10
SKIP_DONE=1

extra_args_for_config() {
  local config="$1"
  case "${config}" in
    # 128-token latents. Based on your MAETOK test, 16x8 is the better layout.
    "MAETOK.yaml")
      echo "--token-grid 16 8"
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
if "SEC@0.5" in m and m["SEC@0.5"] is not None:
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

  echo "Launching SEC ${config} on GPU ${physical_gpu}"
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
      --thresholds "${THRESHOLDS[@]}" \
      --dist-type "${DIST_TYPE}" \
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
  echo "Some SEC jobs failed. Check ${LOG_DIR}"
  exit 1
fi

echo "All SEC jobs finished. Results under ${OUT_ROOT}"
