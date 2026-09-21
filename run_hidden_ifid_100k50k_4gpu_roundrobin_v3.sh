#!/usr/bin/env bash
# Run 25 SiT-XL hidden-iFID evaluations on GPUs 0,1,2,3.
# Each GPU owns a round-robin queue and runs its models sequentially:
#   GPU0: indices 0,4,8,...
#   GPU1: indices 1,5,9,...
#   GPU2: indices 2,6,10,...
#   GPU3: indices 3,7,11,...
# A failed model is marked and the worker continues to the next one.

set -uo pipefail

REPO="/video_ssd/kongzishang/xutongda/git/IFID"
PY_SCRIPT="${REPO}/eval_sit_hidden_ifid_100k50k_v3.py"
EXP_ROOT="/share/xutongda/REPA-E/exps"
OUT_ROOT="${REPO}/exps_interpolate"

DATASET_REF="/video_ssd/lpm/ImageNet/train"
DATASET_VAL="/video_ssd/lpm/ImageNet/val"

CACHE_ROOT="${OUT_ROOT}/hidden_ifid_cache_100k50k"
RESULT_ROOT="${OUT_ROOT}/hidden_ifid_results_100k50k"
LOG_ROOT="${OUT_ROOT}/logs_hidden_ifid_100k50k"
STATUS_ROOT="${OUT_ROOT}/status_hidden_ifid_100k50k"

SMALL_REF=100000
SMALL_VAL=50000
DEPTH=8
T_VALUE=0.1
ALPHA=0.5
COARSE_TOPK=64
SEED=0
DEFAULT_BATCH_SIZE=8
LARGE_MODEL_BATCH_SIZE=2
NUM_WORKERS=8

mkdir -p "$CACHE_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

MODELS=(
  "SD-VAE"
  "FLUX-VAE"
  "QwenImg-VAE"
  "SD3-VAE"
  "SOFT-VQ"
  "MAE-TOK"
  "DE-TOK"
  "DM-VAE"
  "REPAE-SDVAE"
  "VTPB"
  "VTPL"
  "VTPS"
  "EQ-VAE"
  "IN-VAE"
  "VA-VAE"
  "VA-VAE-64"
  "RAE"
  "FLUX2-VAE"
  "SVG"
  "UAE"
  "ScaleRAE-SigLIP2"
  "ScaleRAE-WEBSSL"
  "SVG-T2I-P-256"
  "DCAE"
  "PAE-DINO"
)

# Kept for logging/reference. The evaluator deliberately loads the exact VAE
# config recorded in each experiment's args.json, rather than guessing a YAML.
YAMLS=(
  "SDVAE.yaml"
  "FLUXVAE.yaml"
  "QWVAE.yaml"
  "SD3VAE.yaml"
  "SOFTVQ.yaml"
  "MAETOK.yaml"
  "DETOK.yaml"
  "DMVAE.yaml"
  "REPAEVAE.yaml"
  "VTPB.yaml"
  "VTPL.yaml"
  "VTPS.yaml"
  "EQVAE.yaml"
  "INVAE.yaml"
  "VAVAE.yaml"
  "VAVAE64.yaml"
  "RAE.yaml"
  "FLUX2VAE.yaml"
  "SVG.yaml"
  "UAE.yaml"
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE_WEBSSL.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
  "DCAE.yaml"
  "PAE_DINOv2L_d32.yaml"
)

TAGS=(
  "ifid-sdvae"
  "ifid-fluxvae"
  "ifid-qwenimgvae"
  "ifid-sd3vae"
  "ifid-softvq"
  "ifid-maetok"
  "ifid-detok"
  "ifid-dmvae"
  "ifid-repae-sdvae"
  "ifid-vtpb"
  "ifid-vtpl"
  "ifid-vtps"
  "ifid-eqvae"
  "ifid-invae"
  "ifid-vavae"
  "ifid-vavae64"
  "ifid-rae"
  "ifid-flux2"
  "ifid-svg"
  "ifid-uae"
  "ifid-scalerae-siglip2"
  "ifid-scalerae-webssl"
  "ifid-svg-t2i-p256"
  "ifid-dcae"
  "ifid-pae-dino"
)

EXPS=(
  "sit-xl-sdvae-400k"
  "sit-xl-flux-shift-400k"
  "sit-xl-qw-shift-400k"
  "sit-xl-sd3-shift-400k"
  "sit-XL-softvq-400k"
  "sit-XL-maetok-400k"
  "sit-xl-detok-400k"
  "sit-xl-dmvae-400k"
  "sit-xl-repae-400k"
  "sit-xl-vtpb-400k"
  "sit-xl-vtpl-400k"
  "sit-xl-vtps-400k"
  "sit-XL-eqvae-400k"
  "sit-xl-invae-400k"
  "sit-xl-vavae-400k"
  "sit-xl-vavae64-shift-400k"
  "sit-xl-rae-shift-400k"
  "sit-xl-flux2vae-400k"
  "sit-xl-svg-400k"
  "sit-xl-uae-400k"
  "sit-xl-raet2i_siglip2-400k"
  "sit-xl-raet2i_webssl-400k"
  "sit-xl-svgt2iP-400k"
  "sit-xl-dcae-400k"
  "sit-xl-pae-dinov2-400k"
)

N=${#MODELS[@]}
if (( ${#YAMLS[@]} != N || ${#TAGS[@]} != N || ${#EXPS[@]} != N )); then
  echo "ERROR: array lengths do not match" >&2
  exit 2
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: evaluator not found: $PY_SCRIPT" >&2
  exit 2
fi

batch_size_for_tag() {
  local tag="$1"
  case "$tag" in
    ifid-rae|ifid-svg|ifid-svg-t2i-p256|ifid-scalerae-siglip2|ifid-scalerae-webssl)
      echo "$LARGE_MODEL_BATCH_SIZE"
      ;;
    *)
      echo "$DEFAULT_BATCH_SIZE"
      ;;
  esac
}

run_gpu_worker() {
  local gpu="$1"
  local idx model yaml tag exp_dir exp_path cache_dir result_dir model_log bs status

  echo "[$(date '+%F %T')] GPU${gpu} worker started"

  for ((idx=gpu; idx<N; idx+=4)); do
    model="${MODELS[$idx]}"
    yaml="${YAMLS[$idx]}"
    tag="${TAGS[$idx]}"
    exp_dir="${EXPS[$idx]}"
    exp_path="${EXP_ROOT}/${exp_dir}"
    cache_dir="${CACHE_ROOT}/${tag}"
    result_dir="${RESULT_ROOT}/${tag}"
    model_log="${LOG_ROOT}/${tag}.out"
    bs="$(batch_size_for_tag "$tag")"

    mkdir -p "$result_dir"
    rm -f "${STATUS_ROOT}/${tag}.running" "${STATUS_ROOT}/${tag}.failed"

    if [[ -s "${result_dir}/hidden_ifid_metrics.json" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} SKIP ${model}: result already exists"
      touch "${STATUS_ROOT}/${tag}.done"
      continue
    fi

    if [[ ! -d "$exp_path" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING experiment: $exp_path" | tee "$model_log"
      echo "missing experiment: $exp_path" > "${STATUS_ROOT}/${tag}.failed"
      continue
    fi
    if [[ ! -f "${exp_path}/args.json" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING args.json: ${exp_path}/args.json" | tee "$model_log"
      echo "missing args.json" > "${STATUS_ROOT}/${tag}.failed"
      continue
    fi
    if [[ ! -f "${exp_path}/checkpoints/0400000.pt" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING checkpoint: ${exp_path}/checkpoints/0400000.pt" | tee "$model_log"
      echo "missing checkpoint" > "${STATUS_ROOT}/${tag}.failed"
      continue
    fi

    actual_sit_model="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model", ""))' "${exp_path}/args.json" 2>/dev/null || true)"
    case "$actual_sit_model" in
      SiT-XL/*|SiT_XL/*)
        ;;
      *)
        echo "[$(date '+%F %T')] GPU${gpu} SKIP non-XL model in args.json: ${actual_sit_model}" | tee "$model_log"
        echo "non-XL model: ${actual_sit_model}" > "${STATUS_ROOT}/${tag}.failed"
        continue
        ;;
    esac

    touch "${STATUS_ROOT}/${tag}.running"
    {
      echo "============================================================"
      echo "start_time=$(date '+%F %T')"
      echo "gpu=${gpu}"
      echo "index=${idx}/${N}"
      echo "model=${model}"
      echo "yaml_reference=${yaml}"
      echo "tag=${tag}"
      echo "exp_path=${exp_path}"
      echo "sit_model=${actual_sit_model}"
      echo "batch_size=${bs}"
      echo "reference=${SMALL_REF} validation=${SMALL_VAL}"
      echo "============================================================"
    } > "$model_log"

    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 \
      python -u "$PY_SCRIPT" \
        --exp-path "$exp_path" \
        --train-steps 400000 \
        --ckpt-key ema \
        --dataset-ref "$DATASET_REF" \
        --dataset-val "$DATASET_VAL" \
        --small-ref "$SMALL_REF" \
        --small-val "$SMALL_VAL" \
        --batch-size "$bs" \
        --num-workers "$NUM_WORKERS" \
        --depth "$DEPTH" \
        --t "$T_VALUE" \
        --alpha "$ALPHA" \
        --nn-metric cosine \
        --coarse-topk "$COARSE_TOPK" \
        --coarse-backend torch \
        --allow-torch-coarse-fallback \
        --seed "$SEED" \
        --device cuda:0 \
        --expected-tokens 0 \
        --expected-dim 1152 \
        --cache-dir "$cache_dir" \
        --output-dir "$result_dir" \
        --reuse-cache \
        --no-keep-cache \
        >> "$model_log" 2>&1
    status=$?

    rm -f "${STATUS_ROOT}/${tag}.running"
    if (( status == 0 )) && [[ -s "${result_dir}/hidden_ifid_metrics.json" ]]; then
      touch "${STATUS_ROOT}/${tag}.done"
      echo "[$(date '+%F %T')] GPU${gpu} DONE ${model}" | tee -a "$model_log"
    else
      printf 'exit_code=%s\nlog=%s\n' "$status" "$model_log" > "${STATUS_ROOT}/${tag}.failed"
      echo "[$(date '+%F %T')] GPU${gpu} FAILED ${model}, exit=${status}; continuing" | tee -a "$model_log"
    fi
  done

  echo "[$(date '+%F %T')] GPU${gpu} worker finished"
}

PIDS=()
for gpu in 0 1 2 3; do
  run_gpu_worker "$gpu" > "${LOG_ROOT}/gpu${gpu}_worker.out" 2>&1 &
  PIDS+=("$!")
done

printf '%s\n' "${PIDS[@]}" > "${STATUS_ROOT}/worker_pids.txt"
echo "Started workers: ${PIDS[*]}"

fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "[$(date '+%F %T')] all GPU workers finished; worker_fail=${fail}"
exit "$fail"
