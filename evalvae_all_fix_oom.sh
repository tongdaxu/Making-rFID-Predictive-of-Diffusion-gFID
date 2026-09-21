#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES=8
GPU_IDS="0,1,2,3,4,5,6,7"

SAMPLE_DIR="./samples_fix"
DATASET="/video_ssd/lpm/ImageNet/val"
DATASET_REF="/video_ssd/lpm/ImageNet/train"
CONFIG_DIR="./configs"

SMALL=200000
SAVE_EVERY=1
SAVE_MAX=100

LOG_DIR="./nohup_logs_fix"
mkdir -p "${LOG_DIR}"

# 格式：
# 显示名|config文件名|exp-name
JOBS=(
  # "SD-VAE|SDVAE.yaml|ifid-sdvae"
  # "FLUX-VAE|FLUXVAE.yaml|ifid-fluxvae"
  # "QwenImg-VAE|QWVAE.yaml|ifid-qwenimgvae"
  # "SD3-VAE|SD3VAE.yaml|ifid-sd3vae"
  # "SOFT-VQ|SOFTVQ.yaml|ifid-softvq"
  # "MAE-TOK|MAETOK.yaml|ifid-maetok"
  # "DE-TOK|DETOK.yaml|ifid-detok"
  # "DM-VAE|DMVAE.yaml|ifid-dmvae"
  # "REPAE-SDVAE|REPAEVAE.yaml|ifid-repae-sdvae"
  # "VTPB|VTPB.yaml|ifid-vtpb"
  # "VTPL|VTPL.yaml|ifid-vtpl"
  # "VTPS|VTPS.yaml|ifid-vtps"
  # "EQ-VAE|EQVAE.yaml|ifid-eqvae"
  # "IN-VAE|INVAE.yaml|ifid-invae"
  # "VA-VAE|VAVAE.yaml|ifid-vavae"
  # "VA-VAE-64|VAVAE64.yaml|ifid-vavae64"
  # "RAE|RAE.yaml|ifid-rae"
  # "FLUX2-VAE|FLUX2VAE.yaml|ifid-flux2"
  # "SVG|SVG.yaml|ifid-svg"
  # "UAE|UAE.yaml|ifid-uae"
  "ScaleRAE-SigLIP2|ScaleRAE_SIGLIP2_NONORM.yaml|ifid-scalerae-siglip2"
  # "ScaleRAE-WEBSSL|ScaleRAE_WEBSSL.yaml|ifid-scalerae-webssl"
  # "SVG-T2I-P-256|SVGT2I_P_STAGE1_256.yaml|ifid-svg-t2i-p256"
)

for job in "${JOBS[@]}"; do
  IFS="|" read -r NAME CONFIG_FILE EXP_NAME <<< "${job}"

  CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
  OUT_FILE="${LOG_DIR}/${EXP_NAME}.out"

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}"
    echo "[SKIP] ${NAME}: config not found: ${CONFIG_PATH}" > "${OUT_FILE}"
    continue
  fi

  echo "============================================================"
  echo "[START] ${NAME}"
  echo "[CONFIG] ${CONFIG_PATH}"
  echo "[EXP] ${EXP_NAME}"
  echo "[LOG] ${OUT_FILE}"
  echo "============================================================"

  nohup accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --gpu_ids="${GPU_IDS}" \
    evalvae_fix_oom.py \
    --seed=0 \
    --sample-dir="${SAMPLE_DIR}" \
    --exp-name="${EXP_NAME}" \
    --dataset="${DATASET}" \
    --dataset-ref="${DATASET_REF}" \
    --vae-config="${CONFIG_PATH}" \
    --small "${SMALL}" \
    -save \
    --save-every "${SAVE_EVERY}" \
    --save-max "${SAVE_MAX}" \
    --metric-norm "bn_channel" \
    --metric-chunk-size 128 \
    --metric-dtype fp32 \
    --cache-dtype bf16 \
    --cache-group-size 20 \
    --latent-cache-dir "./latent_cache_fix" \
    > "${OUT_FILE}" 2>&1

  echo "[DONE] ${NAME}"
  echo
done

echo "All jobs finished."