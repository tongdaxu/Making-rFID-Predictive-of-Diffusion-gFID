#!/usr/bin/env bash
set -uo pipefail

cd /video_ssd/kongzishang/xutongda/git/IFID

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_PORT="${BASE_PORT:-31000}"
QUERY_BLOCK_BATCHES="${QUERY_BLOCK_BATCHES:-100}"
CANDIDATE_BLOCK_BATCHES="${CANDIDATE_BLOCK_BATCHES:-10}"
VERIFY_FIRST_BATCH="${VERIFY_FIRST_BATCH:-1}"
ONLY_TAG="${ONLY_TAG:-all}"

REF_SIZE=1000000
VAL_SIZE=50000
BS=25
TOP=10
ALPHA=0.5

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"
EVAL_SCRIPT="${EVAL_SCRIPT:-evalvae_fix_diskcache_blocked_4to8_v2.py}"
REUSE_CACHE_WORLD_SIZE="${REUSE_CACHE_WORLD_SIZE:-0}"

LOG_DIR="${ROOT}/logs_ifid_refsize_1000000"
SAMPLE_DIR="${ROOT}/tmp_ifid_refsize_1000000"
CACHE_BASE="${CACHE_BASE:-/video_ssd/kongzishang/xutongda/ifid_latent_cache_refsize_1000000}"

mkdir -p "${LOG_DIR}" "${SAMPLE_DIR}" "${CACHE_BASE}"

MODELS=(
  "RAE.yaml|RAE|rae"
  "SVG.yaml|SVG|svg"
  "SVGT2I_P_STAGE1_256.yaml|SVG-T2I|svg-t2i-p256"
)

for idx in "${!MODELS[@]}"; do
  IFS='|' read -r CFG MODEL TAG <<< "${MODELS[$idx]}"

  if [[ "${ONLY_TAG}" != "all" && "${ONLY_TAG}" != "${TAG}" ]]; then
    continue
  fi

  EXP="ifid-ref1000000-${TAG}-slerp-a05-top10"
  OUT="${LOG_DIR}/${EXP}.out"
  CACHE_ROOT="${CACHE_BASE}/${EXP}_small1000000_seed0"

  if [[ -f "${OUT}.done" ]] || { [[ -f "${OUT}" ]] && grep -q '^ifid=' "${OUT}"; }; then
    echo "[SKIP DONE] ${MODEL}"
    touch "${OUT}.done"
    continue
  fi

  rm -f "${OUT}.failed"

  EXTRA_ARGS=(
    --query-block-batches "${QUERY_BLOCK_BATCHES}"
    --candidate-block-batches "${CANDIDATE_BLOCK_BATCHES}"
  )

  # Reuse an existing cache. If it was built with fewer GPUs (e.g. 4) than
  # the current run (e.g. 8), pass REUSE_CACHE_WORLD_SIZE=4. The evaluator
  # zero-copy reshards the existing files and reconstructs the old logical-rank
  # top-K ordering before global retrieval.
  if [[ -d "${CACHE_ROOT}/rank000" ]] && compgen -G "${CACHE_ROOT}/rank000/rank000_chunk*.pt" > /dev/null; then
    echo "[CACHE FOUND] ${MODEL}: attempting exact cache reuse"
    EXTRA_ARGS+=(--reuse-cache)
    if [[ "${REUSE_CACHE_WORLD_SIZE}" -gt 0 ]]; then
      EXTRA_ARGS+=(--reuse-cache-world-size "${REUSE_CACHE_WORLD_SIZE}")
      echo "[CACHE RESHARD] existing cache world size=${REUSE_CACHE_WORLD_SIZE}, current GPUs=${NUM_GPUS}"
    fi
  else
    echo "[CACHE BUILD] ${MODEL}: no reusable cache found"
  fi

  if [[ "${VERIFY_FIRST_BATCH}" == "1" ]]; then
    EXTRA_ARGS+=(--verify-first-batch)
  fi

  echo "======================================================================"
  echo "[START] ${MODEL}"
  echo "script=${EVAL_SCRIPT}"
  echo "query_block_batches=${QUERY_BLOCK_BATCHES} (queries/block=$((QUERY_BLOCK_BATCHES * BS)))"
  echo "candidate_block_batches=${CANDIDATE_BLOCK_BATCHES}"
  echo "cache=${CACHE_ROOT}"
  echo "extra_args=${EXTRA_ARGS[*]}"
  echo "======================================================================"

  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  OMP_NUM_THREADS=4 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" \
    --master-port="$((BASE_PORT + idx))" \
    "${EVAL_SCRIPT}" \
    --seed=0 \
    --sample-dir="${SAMPLE_DIR}" \
    --exp-name="${EXP}" \
    --dataset="${DATASET}" \
    --dataset-ref="${DATASET_REF}" \
    --vae-config="${CONFIG_DIR}/${CFG}" \
    --small "${REF_SIZE}" \
    --small_val "${VAL_SIZE}" \
    --bs "${BS}" \
    --top "${TOP}" \
    --intp slerp \
    --alpha "${ALPHA}" \
    --latent-cache-dir="${CACHE_BASE}" \
    --cache-dtype=fp32 \
    --metric-dtype=fp32 \
    --metric-norm=none \
    --metric-chunk-size=64 \
    --cache-group-size=20 \
    "${EXTRA_ARGS[@]}" \
    > "${OUT}" 2>&1

  STATUS=$?

  if [[ "${STATUS}" -eq 0 ]] && grep -q '^ifid=' "${OUT}"; then
    touch "${OUT}.done"
    rm -f "${OUT}.failed"
    echo "[DONE] ${MODEL}"
    grep -E '^ifid=|^rfid=|^lpips=' "${OUT}" || true

    echo "[CLEAN CACHE] ${CACHE_ROOT}"
    rm -rf "${CACHE_ROOT}"
  else
    echo "${STATUS}" > "${OUT}.failed"
    echo "[FAILED] ${MODEL}, status=${STATUS}"
    tail -n 120 "${OUT}" || true
    exit 1
  fi

done
