#!/usr/bin/env bash
set -euo pipefail

cd /video_ssd/kongzishang/xutongda/git/IFID

# Default = oracle-P upper-bound experiment.
# To run the fixed 7/3/1 schedule instead:
#   P_MODE=fixed bash run_sdvae_model_vs_local_score_multistep.sh

P_MODE="${P_MODE:-oracle}"
OUT="/share/xutongda/REPA-E/exps_interpolate/local_score_single/sdvae_model_vs_local_multistep_crossclass50k_${P_MODE}"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python eval_sdvae_model_vs_local_score_multistep.py \
  --exp-path /share/xutongda/REPA-E/exps/sit-xl-sdvae-400k \
  --train-steps 400000 \
  --dataset /video_ssd/lpm/ImageNet/val \
  --dataset-ref /video_ssd/lpm/ImageNet/train \
  --val-index 0 \
  --unconditional-bank \
  --unconditional-ref-size 50000 \
  --t-start 0.8 \
  --nfe 2 4 8 \
  --patch-sizes 1 3 5 7 9 13 17 25 31 \
  --p-mode "$P_MODE" \
  --fixed-high-threshold 0.7 \
  --fixed-low-threshold 0.3 \
  --fixed-high-p 7 \
  --fixed-mid-p 3 \
  --fixed-low-p 1 \
  --time-mode shifted_raw \
  --seed 12345 \
  --cache-dir /share/xutongda/REPA-E/exps_interpolate/local_score_single/cache \
  --out-dir "$OUT" \
  --save-intermediate
