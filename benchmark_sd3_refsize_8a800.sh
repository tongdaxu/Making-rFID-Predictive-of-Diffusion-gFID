#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

ROOT="/share/xutongda/REPA-E/exps_interpolate/runtime_sd3vae_refsize_8a800"
mkdir -p "${ROOT}"

CSV="${ROOT}/summary.csv"
echo "ref_size,runtime_sec,runtime_min,peak_mem_mib,peak_mem_gb,ifid,status" > "${CSV}"

run_one () {
    REF="$1"
    PORT="$2"

    TAG="sd3vae-ref${REF}-8a800"
    LOG="${ROOT}/${TAG}.out"
    MEMLOG="${ROOT}/${TAG}_gpu_mem.txt"

    echo "=================================================="
    echo "START ${TAG}"
    echo "=================================================="

    # 每秒记录一次 8 卡显存
    (
        while true; do
            nvidia-smi \
              --query-gpu=memory.used \
              --format=csv,noheader,nounits
            sleep 1
        done
    ) > "${MEMLOG}" 2>/dev/null &

    MON_PID=$!

    START=$(date +%s)

    STATUS=0

    torchrun \
      --nproc_per_node=8 \
      --master_port="${PORT}" \
      evalvae_fix_recon.py \
      --vae-config configs/SD3VAE.yaml \
      --seed 0 \
      --bs 25 \
      --intp slerp \
      --alpha 0.5 \
      --exp-name "${TAG}" \
      --small "${REF}" \
      --small_val 50000 \
      --top 10 \
      --dataset /video_ssd/lpm/ImageNet/val \
      --dataset-ref /video_ssd/lpm/ImageNet/train \
      --sample-dir "${ROOT}/${TAG}" \
      2>&1 | tee "${LOG}"

    STATUS=${PIPESTATUS[0]}

    END=$(date +%s)

    kill "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true

    SEC=$((END - START))
    MIN=$(awk -v s="${SEC}" 'BEGIN {printf "%.2f", s/60}')

    PEAK_MIB=$(awk '
        {
            gsub(/[[:space:]]/, "", $0)
            if ($0 ~ /^[0-9]+$/ && $0 > max) max=$0
        }
        END {print max+0}
    ' "${MEMLOG}")

    PEAK_GB=$(awk -v m="${PEAK_MIB}" 'BEGIN {printf "%.2f", m/1024}')

    IFID=$(grep '^ifid=' "${LOG}" | tail -1 | sed 's/^ifid=//' | xargs || true)

    if [[ "${STATUS}" -eq 0 && -n "${IFID}" ]]; then
        RESULT="success"
    else
        RESULT="failed"
    fi

    echo "${REF},${SEC},${MIN},${PEAK_MIB},${PEAK_GB},${IFID},${RESULT}" >> "${CSV}"

    echo
    echo "[DONE] ${TAG}"
    echo "Runtime:        ${MIN} min"
    echo "Peak GPU Mem.: ${PEAK_GB} GB/GPU"
    echo "iFID:           ${IFID}"
    echo "Status:         ${RESULT}"
    echo

    if [[ "${RESULT}" != "success" ]]; then
        echo "[STOP] ${TAG} failed."
        exit 1
    fi
}

run_one 50000   32201
run_one 200000  32202
run_one 1000000 32203

echo "================ FINAL SUMMARY ================"
cat "${CSV}"
