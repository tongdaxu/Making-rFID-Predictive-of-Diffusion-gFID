#!/usr/bin/env bash
set -euo pipefail

cd /video_ssd/kongzishang/xutongda/git/IFID

OUT="/share/xutongda/REPA-E/exps_interpolate/local_score_single/sdvae_model_vs_ls_vs_els_crossclass50k_p1"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python eval_sdvae_model_vs_els_score.py \
  --exp-path /share/xutongda/REPA-E/exps/sit-xl-sdvae-400k \
  --train-steps 400000 \
  --dataset /video_ssd/lpm/ImageNet/val \
  --dataset-ref /video_ssd/lpm/ImageNet/train \
  --val-index 0 \
  --unconditional-bank \
  --unconditional-ref-size 50000 \
  --times 0.4 0.2 \
  --patch-sizes 1 \
  --time-mode shifted_raw \
  --seed 12345 \
  --els-ref-chunk 32 \
  --cache-dir /share/xutongda/REPA-E/exps_interpolate/local_score_single/cache \
  --out-dir "$OUT"
