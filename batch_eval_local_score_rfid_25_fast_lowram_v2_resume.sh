#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_local_score_rfid_25_fast_lowram_v2.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-512}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-0}"
NOISE_SEED="${NOISE_SEED:-12345}"
TIME_MODE="${TIME_MODE:-shifted_raw}"
OUT_TAG="${OUT_TAG:-lsm_fast_p1_t02_t04_50k50k_k${CANDIDATE_TOPK}}"

OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/${OUT_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

ONLY_CONFIG="${ONLY_CONFIG:-}"
SKIP_DONE="${SKIP_DONE:-1}"

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
      echo "${HIGH_DIM_BS:-1}" ;;
    "SVG.yaml"|"SVGT2I_P_STAGE1_256.yaml"|"FLUX2VAE.yaml"|"PAE_DINOv2L_d32.yaml")
      echo "${MID_DIM_BS:-2}" ;;
    *)
      echo "${DEFAULT_BS:-4}" ;;
  esac
}

ref_bs_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_REF_BS:-4}" ;;
    "SVG.yaml"|"SVGT2I_P_STAGE1_256.yaml"|"FLUX2VAE.yaml"|"PAE_DINOv2L_d32.yaml")
      echo "${MID_DIM_REF_BS:-16}" ;;
    *)
      echo "${DEFAULT_REF_BS:-32}" ;;
  esac
}

# Aim for roughly the same 500-query block used by the optimized original iFID.
query_block_for_config() {
  local cfg="$1"
  local bs
  bs="$(query_bs_for_config "$cfg")"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_QUERY_BLOCK:-256}" ;;   # bs=1 -> 256 queries
    *)
      # regular: bs=4 -> 512 queries; mid: bs=2 -> 512 queries
      if [[ "$bs" -eq 4 ]]; then
        echo "${DEFAULT_QUERY_BLOCK:-128}"
      elif [[ "$bs" -eq 2 ]]; then
        echo "${MID_DIM_QUERY_BLOCK:-256}"
      else
        echo "${DEFAULT_QUERY_BLOCK:-128}"
      fi
      ;;
  esac
}

candidate_chunk_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_CAND_CHUNK:-8}" ;;
    *)
      echo "${DEFAULT_CAND_CHUNK:-32}" ;;
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
POLL_INTERVAL="${POLL_INTERVAL:-10}"

launch_job() {
  local slot="$1"
  local gpu="$2"
  local cfg="$3"
  local name="${cfg%.yaml}"
  local lower
  lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"
  local out_file="${LOG_DIR}/${lower}.out"
  local status_file="${LOG_DIR}/${lower}.status"

  local bs refbs qblock cchunk extra
  bs="$(query_bs_for_config "${cfg}")"
  refbs="$(ref_bs_for_config "${cfg}")"
  qblock="$(query_block_for_config "${cfg}")"
  cchunk="$(candidate_chunk_for_config "${cfg}")"
  extra="$(extra_args_for_config "${cfg}")"

  echo "Launching ${cfg} on GPU ${gpu}: query_bs=${bs}, ref_bs=${refbs}, qblock=${qblock}, query_imgs=$((bs*qblock)), cand_chunk=${cchunk}"

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
      --query-block-batches "${qblock}"
      --settings 0.2:1 0.4:1
      --time-mode "${TIME_MODE}"
      --candidate-topk "${CANDIDATE_TOPK}"
      --candidate-chunk-size "${cchunk}"
      --metric-chunk-size "${METRIC_CHUNK_SIZE:-8192}"
      --metric-dtype "${METRIC_DTYPE:-fp32}"
      --gpu-bank-dtype "${GPU_BANK_DTYPE:-fp32}"
      --bank-load-lock "${BANK_LOAD_LOCK:-/tmp/lsm_reference_bank_load.lock}"
      --cache-dtype "${CACHE_DTYPE:-fp16}"
      --cache-group-batches "${CACHE_GROUP_BATCHES:-2}"
      --latent-cache-dir "${CACHE_ROOT}"
      --out-dir "${OUT_ROOT}"
      --seed "${SEED}"
      --noise-seed "${NOISE_SEED}"
      --reuse-cache
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
exit $(( FAILED > 0 ? 1 : 0 ))
