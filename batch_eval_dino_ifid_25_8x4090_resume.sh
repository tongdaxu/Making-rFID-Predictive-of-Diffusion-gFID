#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_dino_ifid_25_4090.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
NUM_WORKERS="${NUM_WORKERS:-4}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"

NN_CACHE="${NN_CACHE:-/share/xutongda/REPA-E/exps_interpolate/dino_ifid_25/dinov2_large_cls_ref50000_val50000}"

# RTX 4090 defaults.
DEFAULT_BS="${DEFAULT_BS:-16}"
HIGH_DIM_BS="${HIGH_DIM_BS:-8}"

INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

OUT_TAG="${OUT_TAG:-dinov2_large_cls_cos_top1_slerp_a05_50k50k}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/dino_ifid_25/${OUT_TAG}}"
LOG_DIR="${OUT_ROOT}/logs"

SKIP_DONE="${SKIP_DONE:-1}"
ONLY_CONFIG="${ONLY_CONFIG:-}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

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

batch_size_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_BS}"
      ;;
    *)
      echo "${DEFAULT_BS}"
      ;;
  esac
}

FILTERED=()
for cfg in "${CONFIGS[@]}"; do
  if [[ -z "${ONLY_CONFIG}" || "${cfg}" == "${ONLY_CONFIG}" ]]; then
    FILTERED+=("${cfg}")
  fi
done
CONFIGS=("${FILTERED[@]}")

if (( ${#CONFIGS[@]} == 0 )); then
  echo "No configs selected. ONLY_CONFIG=${ONLY_CONFIG}" >&2
  exit 2
fi

declare -a GPU_PID GPU_CONFIG GPU_LOG GPU_STATUS GPU_PHYSICAL
NEXT_IDX=0
FAILED=0

launch_job() {
  local slot="$1"
  local gpu="$2"
  local cfg="$3"

  local name="${cfg%.yaml}"
  local lower
  lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"

  local bs
  bs="$(batch_size_for_config "${cfg}")"

  local log="${LOG_DIR}/${lower}.out"
  local status="${LOG_DIR}/${lower}.status"

  echo "Launching ${cfg} on RTX 4090 GPU ${gpu}: bs=${bs}, shared DINO pairs, ref=${REF_SIZE}, val=${VAL_SIZE}"

  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
      --vae-config "${CONFIG_DIR}/${cfg}" \
      --exp-name "${name}" \
      --dataset "${DATASET}" \
      --dataset-ref "${DATASET_REF}" \
      --ref-size "${REF_SIZE}" \
      --val-size "${VAL_SIZE}" \
      --nn-cache "${NN_CACHE}" \
      --bs "${bs}" \
      --num-workers "${NUM_WORKERS}" \
      --intp "${INTP}" \
      --alpha "${ALPHA}" \
      --verify-names \
      --out-dir "${OUT_ROOT}" \
      > "${log}" 2>&1

    code=$?
    echo "${code}" > "${status}"
    exit "${code}"
  ) &

  GPU_PID["${slot}"]="$!"
  GPU_CONFIG["${slot}"]="${cfg}"
  GPU_LOG["${slot}"]="${log}"
  GPU_STATUS["${slot}"]="${status}"
  GPU_PHYSICAL["${slot}"]="${gpu}"
}

launch_next() {
  local slot="$1"
  local gpu="$2"

  while (( NEXT_IDX < ${#CONFIGS[@]} )); do
    local cfg="${CONFIGS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))

    local name="${cfg%.yaml}"
    local metrics="${OUT_ROOT}/${name}/metrics.json"

    if [[ "${SKIP_DONE}" == "1" && -s "${metrics}" ]]; then
      echo "[SKIP DONE] ${cfg}: ${metrics}"
      continue
    fi

    launch_job "${slot}" "${gpu}" "${cfg}"
    return
  done

  GPU_PID["${slot}"]=""
}

for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
  launch_next "${slot}" "${GPU_LIST[$slot]}"
done

while true; do
  ACTIVE=0

  for ((slot=0; slot<${#GPU_LIST[@]}; slot++)); do
    pid="${GPU_PID[$slot]:-}"
    [[ -z "${pid}" ]] && continue

    ACTIVE=$((ACTIVE + 1))

    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true

      cfg="${GPU_CONFIG[$slot]}"
      status_file="${GPU_STATUS[$slot]}"
      code=999
      [[ -f "${status_file}" ]] && code="$(cat "${status_file}")"

      if [[ "${code}" != "0" ]]; then
        echo "[FAIL] ${cfg} exit=${code}; log=${GPU_LOG[$slot]}" >&2
        FAILED=$((FAILED + 1))
      else
        echo "[DONE] ${cfg}"
      fi

      launch_next "${slot}" "${GPU_PHYSICAL[$slot]}"
    fi
  done

  (( ACTIVE == 0 )) && break
  sleep "${POLL_INTERVAL}"
done

echo "All jobs finished. failures=${FAILED}"
exit "${FAILED}"
