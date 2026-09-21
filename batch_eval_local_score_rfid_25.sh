#!/usr/bin/env bash
set -euo pipefail

# 25-VAE Local-Score rFID runner.
#
# Defaults are the requested full protocol:
#   50k ImageNet train cross-class refs
#   50k ImageNet val queries
#   t=.2,P=1 and t=.4,P=3
#   candidate_topK=512
#
# Recommended smoke test first:
#   ONLY_CONFIG=SDVAE.yaml VAL_SIZE=1000 bash batch_eval_local_score_rfid_25.sh
#
# Then full:
#   bash batch_eval_local_score_rfid_25.sh
#
# Candidate-K sensitivity later:
#   CANDIDATE_TOPK=256 OUT_TAG=k256 bash batch_eval_local_score_rfid_25.sh
#   CANDIDATE_TOPK=1024 OUT_TAG=k1024 bash batch_eval_local_score_rfid_25.sh

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_local_score_rfid_25.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-512}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
NOISE_SEED="${NOISE_SEED:-12345}"
TIME_MODE="${TIME_MODE:-shifted_raw}"
OUT_TAG="${OUT_TAG:-k${CANDIDATE_TOPK}}"

OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/${OUT_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"

ONLY_CONFIG="${ONLY_CONFIG:-}"

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

# Native token-grid mappings established in the current 25-VAE benchmark.
# BCHW models do not need this.
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

# Keep the metric identical across models; only memory scheduling changes.
batch_size_for_config() {
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

query_block_for_config() {
  local cfg="$1"
  case "${cfg}" in
    "RAE.yaml"|"UAE.yaml"|"ScaleRAE_SIGLIP2_NONORM.yaml"|"ScaleRAE_WEBSSL.yaml")
      echo "${HIGH_DIM_QUERY_BLOCK:-1}" ;;
    *)
      echo "${DEFAULT_QUERY_BLOCK:-4}" ;;
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
  local name_lower
  name_lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"

  local out_file="${LOG_DIR}/${name_lower}.out"
  local status_file="${LOG_DIR}/${name_lower}.status"
  local bs qblock cchunk extra
  bs="$(batch_size_for_config "${cfg}")"
  qblock="$(query_block_for_config "${cfg}")"
  cchunk="$(candidate_chunk_for_config "${cfg}")"
  extra="$(extra_args_for_config "${cfg}")"

  echo "Launching ${cfg} on GPU ${gpu}: bs=${bs}, qblock=${qblock}, cand_chunk=${cchunk}"

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
      --num-workers "${NUM_WORKERS}"
      --query-block-batches "${qblock}"
      --settings 0.2:1 0.4:1
      --time-mode "${TIME_MODE}"
      --candidate-topk "${CANDIDATE_TOPK}"
      --candidate-chunk-size "${cchunk}"
      --metric-chunk-size "${METRIC_CHUNK_SIZE:-256}"
      --metric-dtype "${METRIC_DTYPE:-fp32}"
      --cache-dtype "${CACHE_DTYPE:-fp16}"
      --cache-group-batches "${CACHE_GROUP_BATCHES:-20}"
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
  if (( NEXT_IDX < ${#CONFIGS[@]} )); then
    local cfg="${CONFIGS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))
    launch_job "${slot}" "${gpu}" "${cfg}"
  else
    GPU_PID["${slot}"]=""
  fi
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
      cfg="${GPU_CONFIG[$slot]}"
      status="${GPU_STATUS[$slot]}"
      log="${GPU_LOG[$slot]}"
      gpu="${GPU_PHYSICAL[$slot]}"
      code="unknown"
      [[ -f "${status}" ]] && code="$(cat "${status}")"

      if [[ "${code}" == "0" ]]; then
        echo "Finished OK: ${cfg} on GPU ${gpu}"
      else
        echo "FAILED: ${cfg} on GPU ${gpu}, code=${code}, log=${log}"
        FAILED=1
      fi
      launch_next "${slot}" "${gpu}"
    fi
  done

  if (( ACTIVE == 0 && NEXT_IDX >= ${#CONFIGS[@]} )); then
    break
  fi
  sleep "${POLL_INTERVAL}"
done

echo "Results: ${OUT_ROOT}"
if (( FAILED != 0 )); then
  echo "Some jobs failed; inspect ${LOG_DIR}" >&2
  exit 1
fi
