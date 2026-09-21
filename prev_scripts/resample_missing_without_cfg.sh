#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/video_ssd/kongzishang/xutongda/git/IFID/sample_part_logs"
SUMMARY_CSV="${LOG_DIR}/without_cfg_fid_summary.csv"

EXP_ROOT="/share/xutongda/REPA-E/exps"
REF_NPZ="/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29541}"

TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PPROC_BATCH_SIZE="${PPROC_BATCH_SIZE:-32}"

mkdir -p "${LOG_DIR}"

# 如果命令行手动传了模型，就跑手动传入的模型；
# 如果没传，就自动从 without_cfg_fid_summary.csv 里找 fid 缺失的模型。
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
else
  if [[ ! -f "${SUMMARY_CSV}" ]]; then
    echo "[ERROR] Missing summary CSV: ${SUMMARY_CSV}"
    echo "Please run collect_without_cfg_fid.py first, or pass model names manually."
    exit 1
  fi

  mapfile -t MODELS < <(python3 - <<PY
import csv
from pathlib import Path

p = Path("${SUMMARY_CSV}")
with p.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        model = (r.get("model") or "").strip()
        fid = (r.get("fid") or "").strip()
        if model and (fid == "" or fid.upper() == "MISSING"):
            print(model)
PY
)
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "[OK] No missing without-cfg FID models found."
  exit 0
fi

echo "Will resample without CFG for ${#MODELS[@]} models:"
printf '  %s\n' "${MODELS[@]}"
echo

for MODEL in "${MODELS[@]}"; do
  EXP="${EXP_ROOT}/${MODEL}"
  LOG_FILE="${LOG_DIR}/partresample_${MODEL}.log"

  if [[ ! -d "${EXP}" ]]; then
    echo "[ERROR] EXP dir not found: ${EXP}"
    echo "[SKIP] ${MODEL}"
    continue
  fi

  echo "================================================================================"
  echo "[RUN] MODEL=${MODEL}"
  echo "[EXP] ${EXP}"
  echo "[LOG] ${LOG_FILE}"
  echo "================================================================================"

  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    generate.py \
      --num-fid-samples "${NUM_FID_SAMPLES}" \
      --pproc-batch-size "${PPROC_BATCH_SIZE}" \
      --path-type linear \
      --mode sde \
      --num-steps 250 \
      --cfg-scale 1.0 \
      --guidance-high 1.0 \
      --guidance-low 0.0 \
      --exp-path "${EXP}" \
      --train-steps "${TRAIN_STEPS}" \
      --fid-reference-file "${REF_NPZ}" \
    > "${LOG_FILE}" 2>&1

  echo "[DONE] ${MODEL}"
  tail -n 5 "${LOG_FILE}" || true
  echo
done

echo "[ALL DONE] Re-run collect_without_cfg_fid.py to refresh the table."
