#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_patch_ifid_sweep_25_fast.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

PATCH_SIZES="${PATCH_SIZES:-1 3 5 7}"
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

OUT_TAG="${OUT_TAG:-patch_ifid_p1357_${INTP}_a05_${REF_SIZE}x${VAL_SIZE}}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/patch_ifid_sweep_25/${OUT_TAG}}"

# Reuse the raw 50k reference caches already built for LSM.
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
    # Large-channel models: smaller chunk for matmul/workspace stability.
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_REF_CHUNK:-512}" ;;
    *)
      echo "${DEFAULT_REF_CHUNK:-1024}" ;;
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

  local bs refbs rchunk extra
  bs="$(query_bs_for_config "${cfg}")"
  refbs="$(ref_bs_for_config "${cfg}")"
  rchunk="$(ref_chunk_for_config "${cfg}")"
  extra="$(extra_args_for_config "${cfg}")"

  echo "Launching ${cfg} on GPU ${gpu}: bs=${bs}, ref_bs=${refbs}, ref_chunk=${rchunk}, P=[${PATCH_SIZES}], exact same-position all-${REF_SIZE} hard NN"

  (
    set +e

    # shellcheck disable=SC2206
    patch_arr=(${PATCH_SIZES})
    # shellcheck disable=SC2206
    extra_arr=(${extra})

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
      --patch-sizes "${patch_arr[@]}"
      --intp "${INTP}"
      --alpha "${ALPHA}"
      --ref-chunk-size "${rchunk}"
      --metric-dtype "${METRIC_DTYPE:-fp32}"
      --gpu-bank-dtype "${GPU_BANK_DTYPE:-fp16}"
      --latent-cache-dir "${CACHE_ROOT}"
      --cache-group-batches "${CACHE_GROUP_BATCHES:-20}"
      --cache-dtype "${CACHE_DTYPE:-fp16}"
      --reuse-cache
      --out-dir "${OUT_ROOT}"
      --seed "${SEED}"
    )

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
