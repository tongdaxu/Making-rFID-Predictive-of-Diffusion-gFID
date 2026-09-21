#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_local_ifid_p1_25_fast.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

OUT_TAG="${OUT_TAG:-local_ifid_p1_${INTP}_a05_${REF_SIZE}x${VAL_SIZE}}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_ifid_p1_25/${OUT_TAG}}"

# Reuse the exact raw 50k reference caches from the recent LSM benchmark.
CACHE_ROOT="${CACHE_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache}"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

ONLY_CONFIG="${ONLY_CONFIG:-}"
SKIP_DONE="${SKIP_DONE:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

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

extra_args_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "SOFTVQ.yaml") echo "--token-grid 8 8" ;;
    "MAETOK.yaml") echo "--token-grid 8 16" ;;
    "DETOK.yaml") echo "--token-grid 16 16" ;;
    "DMVAE.yaml") echo "--token-grid 16 16" ;;
    "ScaleRAE_SIGLIP2_NONORM.yaml") echo "--token-grid 16 16" ;;
    "ScaleRAE_WEBSSL.yaml") echo "--token-grid 16 16" ;;
    *) echo "" ;;
  esac
}

query_bs_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_BS:-32}" ;;
    *)
      echo "${DEFAULT_BS:-64}" ;;
  esac
}

ref_bs_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_REF_BS:-64}" ;;
    *)
      echo "${DEFAULT_REF_BS:-128}" ;;
  esac
}

ref_chunk_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_REF_CHUNK:-1024}" ;;
    "SVG.yaml"|"SVGT2I_P_STAGE1_256.yaml"|"FLUX2VAE.yaml"|"PAE_DINOv2L_d32.yaml")
      echo "${MID_DIM_REF_CHUNK:-2048}" ;;
    *)
      echo "${DEFAULT_REF_CHUNK:-4096}" ;;
  esac
}

position_chunk_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "SDVAE.yaml"|"SD3VAE.yaml"|"FLUXVAE.yaml"|"REPAEVAE.yaml"|"QWVAE.yaml")
      echo "${LARGE_GRID_POS_CHUNK:-64}" ;;
    *)
      echo "${DEFAULT_POS_CHUNK:-64}" ;;
  esac
}

query_chunk_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_QUERY_CHUNK:-16}" ;;
    *)
      echo "${DEFAULT_QUERY_CHUNK:-64}" ;;
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

  local out_file="${LOG_DIR}/${lower}.out"
  local status_file="${LOG_DIR}/${lower}.status"

  local bs refbs rchunk pchunk qchunk extra
  bs="$(query_bs_for_config "${cfg}")"
  refbs="$(ref_bs_for_config "${cfg}")"
  rchunk="$(ref_chunk_for_config "${cfg}")"
  pchunk="$(position_chunk_for_config "${cfg}")"
  qchunk="$(query_chunk_for_config "${cfg}")"
  extra="$(extra_args_for_config "${cfg}")"

  echo "Launching ${cfg} on GPU ${gpu}: bs=${bs}, ref_bs=${refbs}, ref_chunk=${rchunk}, pos_chunk=${pchunk}, query_chunk=${qchunk}, exact same-position P=1 NN over ${REF_SIZE} refs"

  (
    set +e

    args=(
      --vae-config "${CONFIG_DIR}/${cfg}"
      --exp-name "${name}"
      --dataset "${DATASET}"
      --dataset-ref "${DATASET_REF}"
      --ref-size "${REF_SIZE}"
      --val-size "${VAL_SIZE}"
      --bs "${bs}"
      --ref-bs "${refbs}"
      --num-workers "${NUM_WORKERS}"
      --intp "${INTP}"
      --alpha "${ALPHA}"
      --ref-chunk-size "${rchunk}"
      --position-chunk-size "${pchunk}"
      --query-chunk-size "${qchunk}"
      --metric-dtype "${METRIC_DTYPE:-fp32}"
      --gpu-bank-dtype "${GPU_BANK_DTYPE:-fp16}"
      --latent-cache-dir "${CACHE_ROOT}"
      --cache-group-batches "${CACHE_GROUP_BATCHES:-20}"
      --cache-dtype "${CACHE_DTYPE:-fp16}"
      --reuse-cache
      --out-dir "${OUT_ROOT}"
      --seed "${SEED}"
    )

    # shellcheck disable=SC2206
    extra_arr=(${extra})

    env CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
      "${args[@]}" "${extra_arr[@]}" > "${out_file}" 2>&1

    code=$?
    echo "${code}" > "${status_file}"
    exit "${code}"
  ) &

  GPU_PID["${slot}"]="$!"
  GPU_CONFIG["${slot}"]="${cfg}"
  GPU_LOG["${slot}"]="${out_file}"
  GPU_STATUS["${slot}"]="${status_file}"
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
