#!/usr/bin/env bash
set -euo pipefail

cd /video_ssd/kongzishang/xutongda/git/IFID

OUT="/share/xutongda/REPA-E/exps_interpolate/local_score_single/sdvae_model_vs_local"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python eval_sdvae_model_vs_local_score.py \
  --exp-path /share/xutongda/REPA-E/exps/sit-xl-sdvae-400k \
  --train-steps 400000 \
  --dataset /video_ssd/lpm/ImageNet/val \
  --dataset-ref /video_ssd/lpm/ImageNet/train \
  --val-index 0 \
  --ref-per-class 1000 \
  --times 0.8 0.6 0.4 0.2 \
  --patch-sizes 3 5 7 9 13 17 25 31 \
  --time-mode shifted_raw \
  --seed 12345 \
  --cache-dir /share/xutongda/REPA-E/exps_interpolate/local_score_single/cache \
  --out-dir "$OUT"
