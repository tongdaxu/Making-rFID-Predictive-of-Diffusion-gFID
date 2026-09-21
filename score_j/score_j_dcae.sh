
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN before running}

CUDA_VISIBLE_DEVICES=0 python score_jacobian_spectrum.py \
  --vae-config ./configs/DCAE.yaml \
  --dataset-ref /video_ssd/lpm/ImageNet/train \
  --dataset /video_ssd/lpm/ImageNet/val \
  --train-size 200000 \
  --val-size 50000 \
  --topk 512 \
  --t 0.5 \
  --out-dir ./score_jacobian_dcae_t05_debug \
  --metric-norm none