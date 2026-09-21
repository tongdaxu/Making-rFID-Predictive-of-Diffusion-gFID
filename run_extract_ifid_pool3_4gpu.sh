#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
GD="${GUIDED_DIFFUSION_DIR:-$REPO/third_party/guided-diffusion}"
NPZ_DIR="${NPZ_DIR:-/share/xutongda/REPA-E/exps_interpolate/iis_npz_from_intp_dist}"
OUT_DIR="${OUT_DIR:-$REPO/ifid_r_v1/pool3}"
LOG_DIR="${LOG_DIR:-$REPO/ifid_r_v1/pool3_logs}"

# Four concurrent readers is a safer default for a shared/network filesystem.
# Override with GPU_LIST="0 1 2 3 4 5 6 7" if desired.
GPU_LIST_STR="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "$GPU_LIST_STR"
N="${#GPUS[@]}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "REPO=$REPO"
echo "GD=$GD"
echo "NPZ_DIR=$NPZ_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "GPUs=${GPUS[*]}"
echo "workers=$N"
if [[ -n "${INCEPTION_PB:-}" ]]; then
  echo "INCEPTION_PB=$INCEPTION_PB"
fi

pids=()

for wi in "${!GPUS[@]}"; do
  gpu="${GPUS[$wi]}"
  log="$LOG_DIR/worker${wi}_gpu${gpu}.out"

  cmd=(
    python "$REPO/extract_ifid_pool3_features.py"
    --guided-diffusion-dir "$GD"
    --npz-dir "$NPZ_DIR"
    --out-dir "$OUT_DIR"
    --worker-index "$wi"
    --num-workers "$N"
    --batch-size "${BATCH_SIZE:-64}"
    --skip-done
  )

  if [[ -n "${INCEPTION_PB:-}" ]]; then
    cmd+=(--inception-pb "$INCEPTION_PB")
  fi

  echo "[LAUNCH] worker=$wi gpu=$gpu log=$log"

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

echo "[DONE] all extract workers finished; rc=$rc"
exit "$rc"
