#!/usr/bin/env bash
set -uo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

ROOT="/share/xutongda/REPA-E/exps_interpolate/runtime_sdvae_refsize_8a800"
mkdir -p "${ROOT}"

CSV="${ROOT}/sdvae_refsize_runtime_8a800.csv"

echo "ref_size,val_size,num_gpus,runtime_seconds,runtime_minutes,status,ifid" > "${CSV}"

run_one () {
    REF_SIZE="$1"
    PORT="$2"

    TAG="sdvae-ref${REF_SIZE}-8a800"
    LOG="${ROOT}/${TAG}.out"

    echo "============================================================"
    echo "Running ${TAG}"
    echo "REF_SIZE=${REF_SIZE}"
    echo "VAL_SIZE=50000"
    echo "GPUS=8"
    echo "============================================================"

    START=$(date +%s)

    STATUS=0

    torchrun \
      --nproc_per_node=8 \
      --master_port="${PORT}" \
      evalvae_fix_recon.py \
      --vae-config configs/SDVAE.yaml \
      --seed 0 \
      --bs 25 \
      --intp slerp \
      --alpha 0.5 \
      --exp-name "${TAG}" \
      --small "${REF_SIZE}" \
      --small_val 50000 \
      --top 10 \
      --dataset /video_ssd/lpm/ImageNet/val \
      --dataset-ref /video_ssd/lpm/ImageNet/train \
      --sample-dir "${ROOT}/${TAG}" \
      2>&1 | tee "${LOG}" || STATUS=$?

    END=$(date +%s)
    SECONDS=$((END - START))

    MINUTES=$(python - <<PY
print(f"{${SECONDS}/60:.3f}")
PY
)

    IFID=$(grep '^ifid=' "${LOG}" | tail -1 | sed 's/^ifid=//' | xargs || true)

    if [[ "${STATUS}" -eq 0 ]] && [[ -n "${IFID}" ]]; then
        RESULT_STATUS="success"
    else
        RESULT_STATUS="failed"
    fi

    echo "${REF_SIZE},50000,8,${SECONDS},${MINUTES},${RESULT_STATUS},${IFID}" >> "${CSV}"

    echo
    echo "[DONE] ${TAG}"
    echo "runtime = ${SECONDS} sec = ${MINUTES} min"
    echo "ifid    = ${IFID}"
    echo "status  = ${RESULT_STATUS}"
    echo
}

run_one 50000   32101
run_one 200000  32102
run_one 1000000 32103

echo
echo "================ FINAL RESULTS ================"
column -s, -t "${CSV}" || cat "${CSV}"
echo
echo "CSV: ${CSV}"
