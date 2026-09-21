#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"

REF_SIZE="${REF_SIZE:?REF_SIZE must be 50000 or 1000000}"
GROUP="${GROUP:?GROUP is required}"

ONLY_TAG="${ONLY_TAG:-all}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-29600}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

CONFIG_DIR="${CONFIG_DIR:-./configs}"

PREALLOC_SCRIPT="${PREALLOC_SCRIPT:-evalvae_fix_recon_prealloc.py}"
DISKCACHE_SCRIPT="${DISKCACHE_SCRIPT:-evalvae_fix_diskcache.py}"

LOG_DIR="${ROOT}/logs_ifid_refsize_${REF_SIZE}"
SAMPLE_DIR="${ROOT}/tmp_ifid_refsize_${REF_SIZE}"

# 对特殊的 1M latent，最好放本地 SSD。
CACHE_BASE="${CACHE_BASE:-/video_ssd/kongzishang/xutongda/ifid_latent_cache_refsize_${REF_SIZE}}"

VAL_SIZE=50000
BS=25
TOP=10
ALPHA=0.5

CACHE_DTYPE="${CACHE_DTYPE:-fp32}"
METRIC_DTYPE="${METRIC_DTYPE:-fp32}"
METRIC_CHUNK_SIZE="${METRIC_CHUNK_SIZE:-64}"
CACHE_GROUP_SIZE="${CACHE_GROUP_SIZE:-20}"

mkdir -p \
  "${LOG_DIR}" \
  "${SAMPLE_DIR}" \
  "${CACHE_BASE}"

if [[ "${GROUP}" == *"4090"* ]]; then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1

  echo "[NCCL] RTX 4090 mode:"
  echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
else
  unset NCCL_P2P_DISABLE || true
  unset NCCL_IB_DISABLE || true
fi

# class:
# safe    = 1M 时明确适合 4090
# medium  = 1M 时为了保险用 A800
# special = RAE / SVG / SVG-T2I，1M 必须 disk cache
MODELS=(
  "SDVAE.yaml|SD-VAE|sdvae|safe"
  "SD3VAE.yaml|SD3-VAE|sd3vae|medium"
  "FLUXVAE.yaml|FLUX-VAE|fluxvae|medium"
  "DCAE.yaml|DC-AE|dcae|safe"
  "SOFTVQ.yaml|SOFT-VQ|softvq|safe"
  "VAVAE.yaml|VA-VAE|vavae|safe"
  "VAVAE64.yaml|VA-VAE-64|vavae64|medium"
  "EQVAE.yaml|EQ-VAE|eqvae|safe"
  "MAETOK.yaml|MAE-TOK|maetok|safe"
  "INVAE.yaml|IN-VAE|invae|safe"
  "REPAEVAE.yaml|REPAE|repae-sdvae|safe"
  "DETOK.yaml|DE-TOK|detok|safe"
  "QWVAE.yaml|QW-VAE|qwenimgvae|medium"
  "RAE.yaml|RAE|rae|special"
  "SVG.yaml|SVG|svg|special"
  "FLUX2VAE.yaml|FLUX2-VAE|flux2|medium"
  "SVGT2I_P_STAGE1_256.yaml|SVG-T2I|svg-t2i-p256|special"
  "DMVAE.yaml|DM-VAE|dmvae|safe"
  "VTPS.yaml|VTP-S|vtps|medium"
  "VTPB.yaml|VTP-B|vtpb|medium"
  "VTPL.yaml|VTP-L|vtpl|medium"
  "PAE_DINOv2L_d32.yaml|PAE|pae-dino|safe"
)

should_run() {
  local class="$1"

  case "${REF_SIZE}:${GROUP}:${class}" in
    # 50k:
    # 所有非 special 模型都直接放 4090。
    50000:regular4090:safe) return 0 ;;
    50000:regular4090:medium) return 0 ;;
    50000:speciala800:special) return 0 ;;

    # 1M:
    1000000:safe4090:safe) return 0 ;;
    1000000:mediuma800:medium) return 0 ;;
    1000000:speciala800:special) return 0 ;;

    # 某个 4090 模型如果意外 OOM，可手工丢到 A800。
    1000000:manuala800:safe) return 0 ;;
    1000000:manuala800:medium) return 0 ;;

    *) return 1 ;;
  esac
}

is_done() {
  local out="$1"

  if [[ -f "${out}.done" ]]; then
    return 0
  fi

  if [[ -f "${out}" ]] &&
     grep -q '^ifid=' "${out}" &&
     grep -q '^lpips=' "${out}"; then
    touch "${out}.done"
    return 0
  fi

  return 1
}

echo "======================================================================"
echo "Reference-size ablation"
echo "REF_SIZE=${REF_SIZE}"
echo "GROUP=${GROUP}"
echo "ONLY_TAG=${ONLY_TAG}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "CACHE_BASE=${CACHE_BASE}"
echo "======================================================================"

FAILED=0
DONE=0
SKIPPED=0

for idx in "${!MODELS[@]}"; do

  IFS="|" read -r CFG MODEL TAG CLASS <<< "${MODELS[$idx]}"

  if [[ "${ONLY_TAG}" != "all" && "${ONLY_TAG}" != "${TAG}" ]]; then
    continue
  fi

  if ! should_run "${CLASS}"; then
    continue
  fi

  EXP="ifid-ref${REF_SIZE}-${TAG}-slerp-a05-top10"
  OUT="${LOG_DIR}/${EXP}.out"

  if is_done "${OUT}"; then
    echo "[SKIP DONE] ${MODEL}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  rm -f \
    "${OUT}.done" \
    "${OUT}.failed"

  rm -rf \
    "${SAMPLE_DIR}/${EXP}" \
    "${SAMPLE_DIR}/${EXP}_extra"

  PORT=$((BASE_PORT + idx))

  METHOD="prealloc"

  if [[ "${REF_SIZE}" -eq 1000000 && "${CLASS}" == "special" ]]; then
    METHOD="diskcache"
  fi

  echo
  echo "======================================================================"
  echo "[START] ${MODEL}"
  echo "tag=${TAG}"
  echo "class=${CLASS}"
  echo "ref=${REF_SIZE}"
  echo "val=${VAL_SIZE}"
  echo "method=${METHOD}"
  echo "bs=${BS}"
  echo "======================================================================"

  if [[ "${METHOD}" == "prealloc" ]]; then

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4 \
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node="${NUM_GPUS}" \
      --master-port="${PORT}" \
      "${PREALLOC_SCRIPT}" \
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
      > "${OUT}" 2>&1

    STATUS=$?

  else

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=4 \
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node="${NUM_GPUS}" \
      --master-port="${PORT}" \
      "${DISKCACHE_SCRIPT}" \
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
      --cache-dtype="${CACHE_DTYPE}" \
      --metric-dtype="${METRIC_DTYPE}" \
      --metric-norm=none \
      --metric-chunk-size="${METRIC_CHUNK_SIZE}" \
      --cache-group-size="${CACHE_GROUP_SIZE}" \
      > "${OUT}" 2>&1

    STATUS=$?
  fi

  if [[ "${STATUS}" -eq 0 ]] &&
     grep -q '^ifid=' "${OUT}" &&
     grep -q '^lpips=' "${OUT}"; then

    touch "${OUT}.done"
    rm -f "${OUT}.failed"

    echo "[DONE] ${MODEL}"
    grep -E '^ifid=|^rfid=|^lpips=' "${OUT}" || true

    DONE=$((DONE + 1))

    # disk cache 成功后立刻删除。
    if [[ "${METHOD}" == "diskcache" ]]; then
      CACHE_ROOT="${CACHE_BASE}/${EXP}_small${REF_SIZE}_seed0"
      echo "[CLEAN CACHE] ${CACHE_ROOT}"
      rm -rf "${CACHE_ROOT}"
    fi

  else
    echo "${STATUS}" > "${OUT}.failed"

    echo "[FAILED] ${MODEL}, status=${STATUS}"
    tail -n 100 "${OUT}" || true

    FAILED=$((FAILED + 1))
  fi

done

echo
echo "======================================================================"
echo "DONE=${DONE}"
echo "SKIPPED=${SKIPPED}"
echo "FAILED=${FAILED}"
echo "======================================================================"

if (( FAILED > 0 )); then
  exit 1
fi
