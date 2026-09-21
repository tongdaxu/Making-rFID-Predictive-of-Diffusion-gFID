#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"
SAMPLE_ROOT="${SAMPLE_ROOT:-${ROOT}/ifid_saved_images}"

IIS_NPZ_DIR="${IIS_NPZ_DIR:-${ROOT}/iis_npz_from_intp_dist}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results_iis_from_intp_dist}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_iis_from_intp_dist}"

PACK_SCRIPT="${PACK_SCRIPT:-pack_iis_npz_from_intp_dist.py}"
IS_SCRIPT="${IS_SCRIPT:-run_openai_is_only.py}"

GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-/video_ssd/kongzishang/xutongda/git/guided-diffusion}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
TOP="${TOP:-10}"
INTP="${INTP:-slerp}"
ALPHA="${ALPHA:-0.5}"

TF_BATCH_SIZE="${TF_BATCH_SIZE:-64}"
SOFTMAX_BATCH_SIZE="${SOFTMAX_BATCH_SIZE:-512}"

GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"

mkdir -p "${IIS_NPZ_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

alpha_tag() {
  local a="$1"
  echo "a${a/./}"
}

ATAG="$(alpha_tag "${ALPHA}")"

MODELS=(
  "SDVAE.yaml|SDVAE|sdvae|regular"
  "SD3VAE.yaml|SD3VAE|sd3vae|regular"
  "FLUXVAE.yaml|FLUXVAE|fluxvae|regular"
  "DCAE.yaml|DCAE|dcae|regular"
  "SOFTVQ.yaml|SOFTVQ|softvq|regular"
  "VAVAE.yaml|VAVAE|vavae|regular"
  "VAVAE64.yaml|VAVAE64|vavae64|regular"
  "EQVAE.yaml|EQVAE|eqvae|regular"
  "MAETOK.yaml|MAETOK|maetok|regular"
  "INVAE.yaml|INVAE|invae|regular"
  "REPAEVAE.yaml|REPAEVAE|repae-sdvae|regular"
  "DETOK.yaml|DETOK|detok|regular"
  "QWVAE.yaml|QWVAE|qwenimgvae|regular"
  "RAE.yaml|RAE|rae|special"
  "SVG.yaml|SVG|svg|special"
  "FLUX2VAE.yaml|FLUX2VAE|flux2|regular"
  "SVGT2I_P_STAGE1_256.yaml|SVGT2I_P_STAGE1_256|svg-t2i-p256|special"
  "DMVAE.yaml|DMVAE|dmvae|regular"
  "VTPS.yaml|VTPS|vtps|regular"
  "VTPB.yaml|VTPB|vtpb|regular"
  "VTPL.yaml|VTPL|vtpl|regular"
  "PAE_DINOv2L_d32.yaml|PAE_DINOv2L_d32|pae-dino|regular"
)

MANIFEST="${RESULT_DIR}/iis_manifest.tsv"
: > "${MANIFEST}"

for item in "${MODELS[@]}"; do
  IFS="|" read -r YAML MODEL TAG GROUP <<< "${item}"

  EXP_NAME="ifid-iis-${TAG}-${INTP}-${ATAG}-top${TOP}-${SMALL}x${SMALL_VAL}"
  NPZ="${IIS_NPZ_DIR}/${EXP_NAME}.npz"
  JSON="${RESULT_DIR}/${EXP_NAME}.json"

  echo -e "${YAML}\t${MODEL}\t${TAG}\t${GROUP}\t${EXP_NAME}\t${NPZ}\t${JSON}" >> "${MANIFEST}"
done

echo "================================================================================"
echo "[iIS from iFID saved intp_dist]"
echo "ROOT=${ROOT}"
echo "SAMPLE_ROOT=${SAMPLE_ROOT}"
echo "IIS_NPZ_DIR=${IIS_NPZ_DIR}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR}"
echo "GPUS=${GPUS_STR}"
echo "MANIFEST=${MANIFEST}"
echo "================================================================================"

if [[ ! -f "${PACK_SCRIPT}" ]]; then
  echo "[FATAL] PACK_SCRIPT not found: ${PACK_SCRIPT}"
  exit 1
fi

if [[ ! -f "${IS_SCRIPT}" ]]; then
  echo "[FATAL] IS_SCRIPT not found: ${IS_SCRIPT}"
  exit 1
fi

if [[ ! -d "${SAMPLE_ROOT}" ]]; then
  echo "[FATAL] SAMPLE_ROOT not found: ${SAMPLE_ROOT}"
  exit 1
fi

if [[ ! -d "${GUIDED_DIFFUSION_DIR}" ]]; then
  echo "[FATAL] GUIDED_DIFFUSION_DIR not found: ${GUIDED_DIFFUSION_DIR}"
  exit 1
fi

cat "${MANIFEST}"
echo

IFS=' ' read -r -a GPU_ARR <<< "${GPUS_STR}"
NUM_GPUS="${#GPU_ARR[@]}"

run_worker() {
  local worker_id="$1"
  local gpu="$2"
  local line_no=0

  echo "[WORKER ${worker_id}] GPU=${gpu} start"

  while IFS=$'\t' read -r YAML MODEL TAG GROUP EXP_NAME NPZ JSON; do
    line_no=$((line_no + 1))

    if (( (line_no - 1) % NUM_GPUS != worker_id )); then
      continue
    fi

    PACK_LOG="${LOG_DIR}/${EXP_NAME}.pack.out"
    IS_LOG="${LOG_DIR}/${EXP_NAME}.is.out"

    if [[ -f "${JSON}" ]]; then
      echo "[WORKER ${worker_id}] [SKIP JSON DONE] ${MODEL}: ${JSON}"
      continue
    fi

    if [[ ! -s "${NPZ}" ]]; then
      echo "--------------------------------------------------------------------------------"
      echo "[WORKER ${worker_id}] [PACK] ${MODEL}"
      echo "[EXP_NAME] ${EXP_NAME}"
      echo "[NPZ] ${NPZ}"
      echo "[PACK_LOG] ${PACK_LOG}"
      echo "--------------------------------------------------------------------------------"

      python "${PACK_SCRIPT}" \
        --sample-root "${SAMPLE_ROOT}" \
        --exp-name "${EXP_NAME}" \
        --out-npz "${NPZ}" \
        --subdir-name "intp_dist" \
        --expected "${SMALL_VAL}" \
        --image-size 256 \
        > "${PACK_LOG}" 2>&1

      STATUS=$?
      if [[ "${STATUS}" -ne 0 ]]; then
        echo "[WORKER ${worker_id}] [ERROR] pack failed: ${MODEL}, status=${STATUS}"
        echo "${STATUS}" > "${PACK_LOG}.failed"
        tail -n 120 "${PACK_LOG}" || true
        continue
      fi
    else
      echo "[WORKER ${worker_id}] [SKIP NPZ EXISTS] ${MODEL}: ${NPZ}"
    fi

    echo "--------------------------------------------------------------------------------"
    echo "[WORKER ${worker_id}] [IS] ${MODEL}"
    echo "[GPU] ${gpu}"
    echo "[NPZ] ${NPZ}"
    echo "[JSON] ${JSON}"
    echo "[IS_LOG] ${IS_LOG}"
    echo "--------------------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    python "${IS_SCRIPT}" \
      --guided-diffusion-dir "${GUIDED_DIFFUSION_DIR}" \
      --sample-npz "${NPZ}" \
      --out-json "${JSON}" \
      --tf-batch-size "${TF_BATCH_SIZE}" \
      --softmax-batch-size "${SOFTMAX_BATCH_SIZE}" \
      > "${IS_LOG}" 2>&1

    STATUS=$?
    if [[ "${STATUS}" -ne 0 ]]; then
      echo "[WORKER ${worker_id}] [ERROR] IS failed: ${MODEL}, status=${STATUS}"
      echo "${STATUS}" > "${IS_LOG}.failed"
      tail -n 120 "${IS_LOG}" || true
      continue
    fi

    echo "[WORKER ${worker_id}] [DONE] ${MODEL}"
    cat "${JSON}"
    echo

  done < "${MANIFEST}"

  echo "[WORKER ${worker_id}] GPU=${gpu} finished"
}

for i in "${!GPU_ARR[@]}"; do
  run_worker "${i}" "${GPU_ARR[$i]}" > "${LOG_DIR}/worker_gpu${GPU_ARR[$i]}.out" 2>&1 &
done

wait

echo "================================================================================"
echo "[ALL IIS WORKERS FINISHED]"
echo "RESULT_DIR=${RESULT_DIR}"
echo "IIS_NPZ_DIR=${IIS_NPZ_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"
