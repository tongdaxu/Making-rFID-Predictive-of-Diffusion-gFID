#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="/share/xutongda/REPA-E/exps"
LOG_DIR="/video_ssd/kongzishang/xutongda/git/IFID/sample_part_logs"
REF_NPZ="/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29561}"

TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PPROC_BATCH_SIZE="${PPROC_BATCH_SIZE:-32}"

mkdir -p "${LOG_DIR}"

MODELS=(
  "sit-xl-uae-400k"
  "sit-xl-vavae-400k"
  "sit-xl-vavae64-shift-400k"
)

CFGS=(
  1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0
  3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0
  5.25 5.5 5.75 6.0 6.25 6.5 6.75 7.0
  7.25 7.5 7.75 8.0
)

extract_fid() {
  local log_file="$1"
  grep -oE 'fid= *[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "${log_file}" \
    | tail -n 1 \
    | sed -E 's/fid= *//'
}

float_lt() {
  python3 - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print("1" if a < b else "0")
PY
}

is_valid_npz() {
  local npz="$1"
  python3 - "$npz" <<'PY'
from pathlib import Path
import sys, zipfile
import numpy as np

p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
if p.stat().st_size < 1024:
    raise SystemExit(1)
if not zipfile.is_zipfile(p):
    raise SystemExit(1)
try:
    arr = np.load(p, mmap_mode="r")["arr_0"]
    if arr.shape[0] != 50000:
        raise SystemExit(1)
    if len(arr.shape) != 4 or arr.shape[-1] != 3:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

for MODEL in "${MODELS[@]}"; do
  EXP="${EXP_ROOT}/${MODEL}"

  if [[ ! -d "${EXP}" ]]; then
    echo "[ERROR] EXP not found: ${EXP}"
    continue
  fi

  echo "================================================================================"
  echo "[MODEL] ${MODEL}"
  echo "[EXP]   ${EXP}"
  echo "================================================================================"

  BEST_CFG=""
  BEST_FID=""
  BEST_NPZ=""
  STOP_REASON="REACHED_MAX_CFG"
  STOP_CFG=""
  STOP_FID=""

  for CFG in "${CFGS[@]}"; do
    NPZ="${EXP}/${MODEL}_${TRAIN_STEPS}_cfg${CFG}-0.0-1.0.npz"
    OUT_LOG="${LOG_DIR}/rescan_${MODEL}_${TRAIN_STEPS}_cfg${CFG}.out"

    echo "--------------------------------------------------------------------------------"
    echo "[RUN] MODEL=${MODEL}, CFG=${CFG}"
    echo "[NPZ] ${NPZ}"
    echo "[LOG] ${OUT_LOG}"

    # 关键：如果目标 npz 已经存在但坏了，先删掉，避免 generate.py 读/跳过坏文件。
    if [[ -f "${NPZ}" ]]; then
      if is_valid_npz "${NPZ}"; then
        echo "[INFO] Existing valid npz found. Remove it to force fresh resampling: ${NPZ}"
        rm -f "${NPZ}"
      else
        echo "[WARN] Existing bad npz found. Remove: ${NPZ}"
        rm -f "${NPZ}"
      fi
    fi

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
        --cfg-scale "${CFG}" \
        --guidance-high 1.0 \
        --guidance-low 0.0 \
        --exp-path "${EXP}" \
        --train-steps "${TRAIN_STEPS}" \
        --fid-reference-file "${REF_NPZ}" \
      > "${OUT_LOG}" 2>&1

    FID="$(extract_fid "${OUT_LOG}" || true)"

    if [[ -z "${FID}" ]]; then
      echo "[ERROR] No fid found in log: ${OUT_LOG}"
      tail -n 80 "${OUT_LOG}" || true
      STOP_REASON="NO_FID_OR_FAILED"
      STOP_CFG="${CFG}"
      STOP_FID=""
      break
    fi

    echo "[RESULT] CFG=${CFG}, FID=${FID}"

    if [[ -z "${BEST_FID}" ]]; then
      BEST_CFG="${CFG}"
      BEST_FID="${FID}"
      BEST_NPZ="${NPZ}"
      echo "[BEST INIT] cfg=${BEST_CFG}, fid=${BEST_FID}"
      continue
    fi

    IS_BETTER="$(float_lt "${FID}" "${BEST_FID}")"

    if [[ "${IS_BETTER}" == "1" ]]; then
      BEST_CFG="${CFG}"
      BEST_FID="${FID}"
      BEST_NPZ="${NPZ}"
      echo "[BEST UPDATE] cfg=${BEST_CFG}, fid=${BEST_FID}"
    else
      STOP_REASON="FID_INCREASED_OR_EQUAL"
      STOP_CFG="${CFG}"
      STOP_FID="${FID}"
      echo "[STOP] ${STOP_REASON}: stop_cfg=${STOP_CFG}, stop_fid=${STOP_FID}, best_cfg=${BEST_CFG}, best_fid=${BEST_FID}"
      break
    fi
  done

  BEST_LOG="${LOG_DIR}/cfg_scan_best_${MODEL}_${TRAIN_STEPS}.txt"

  {
    echo "status=SUCCESS"
    echo "exp=${EXP}"
    echo "best_cfg=${BEST_CFG}"
    echo "best_fid=${BEST_FID}"
    echo "best_npz=${BEST_NPZ}"
    echo "stop_reason=${STOP_REASON}"
    echo "stop_cfg=${STOP_CFG}"
    echo "stop_fid=${STOP_FID}"
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  } > "${BEST_LOG}"

  echo "[WRITE] ${BEST_LOG}"
  cat "${BEST_LOG}"
  echo
done

echo "[DONE] rescan finished."
