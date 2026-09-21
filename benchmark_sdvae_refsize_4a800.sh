#!/usr/bin/env bash
set -uo pipefail

ROOT="/share/xutongda/REPA-E/exps_interpolate"
OUT_DIR="${ROOT}/runtime_sdvae_refsize_4a800"

mkdir -p "${OUT_DIR}"

SCRIPT="evalvae_fix_recon_prealloc.py"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS=4
BASE_PORT="${BASE_PORT:-30200}"

SIZES=(
  50000
  200000
  1000000
)

CSV="${OUT_DIR}/sdvae_refsize_runtime.csv"

echo "RefImages,Seconds,Minutes,Hours,iFID" > "${CSV}"

for idx in "${!SIZES[@]}"; do

  REF_SIZE="${SIZES[$idx]}"

  EXP="runtime-sdvae-ref${REF_SIZE}-val50000"
  LOG="${OUT_DIR}/${EXP}.out"

  echo
  echo "=============================================================="
  echo "[BENCHMARK]"
  echo "SD-VAE"
  echo "reference=${REF_SIZE}"
  echo "validation=50000"
  echo "GPUs=${GPU_IDS}"
  echo "=============================================================="

  START_TIME="$(python - <<'PY'
import time
print(time.time())
PY
)"

  set +e

  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" \
    --master-port="$((BASE_PORT + idx))" \
    "${SCRIPT}" \
    --seed=0 \
    --sample-dir="${OUT_DIR}/tmp" \
    --exp-name="${EXP}" \
    --dataset="/video_ssd/lpm/ImageNet/val" \
    --dataset-ref="/video_ssd/lpm/ImageNet/train" \
    --vae-config="./configs/SDVAE.yaml" \
    --small "${REF_SIZE}" \
    --small_val 50000 \
    --bs 25 \
    --top 10 \
    --intp slerp \
    --alpha 0.5 \
    > "${LOG}" 2>&1

  STATUS=$?

  set -e

  END_TIME="$(python - <<'PY'
import time
print(time.time())
PY
)"

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[FAILED] reference=${REF_SIZE}, status=${STATUS}"
    tail -n 100 "${LOG}"
    exit 1
  fi

  if ! grep -q '^ifid=' "${LOG}"; then
    echo "[FAILED] missing iFID: reference=${REF_SIZE}"
    tail -n 100 "${LOG}"
    exit 1
  fi

  IFID="$(grep '^ifid=' "${LOG}" | tail -1 | cut -d= -f2)"

  python - \
    "${REF_SIZE}" \
    "${START_TIME}" \
    "${END_TIME}" \
    "${IFID}" \
    "${CSV}" <<'PY'
import sys

ref = int(sys.argv[1])
start = float(sys.argv[2])
end = float(sys.argv[3])
ifid = float(sys.argv[4])
csv = sys.argv[5]

sec = end - start

with open(csv, "a") as f:
    f.write(
        f"{ref},"
        f"{sec:.2f},"
        f"{sec/60:.3f},"
        f"{sec/3600:.4f},"
        f"{ifid:.8f}\n"
    )

print(
    f"[DONE] ref={ref}: "
    f"{sec:.2f}s = "
    f"{sec/60:.2f}min = "
    f"{sec/3600:.3f}h, "
    f"iFID={ifid:.4f}"
)
PY

done

echo
echo "=============================================================="
echo "[FINAL]"
cat "${CSV}"
echo "=============================================================="
