#!/usr/bin/env bash
# V6 t-sweep runner for 4 GPUs.
#
# Main user parameters:
#   T_VALUES="0.05 0.10 0.25 0.50 0.75 0.90"
#   NN_SCOPE="class"   # class or global
#
# Examples:
#   T_VALUES="0.05 0.10 0.25 0.50" NN_SCOPE=class bash this_script.sh
#   T_VALUES="0.10 0.25 0.50" NN_SCOPE=global bash this_script.sh
#
# Every (t, model) task has its own cache/result/log/status directory.
# Finished JSON results are skipped on restart.

set -uo pipefail

REPO="/video_ssd/kongzishang/xutongda/git/IFID"
EXP_ROOT="/share/xutongda/REPA-E/exps"
OUT_ROOT="${REPO}/exps_interpolate"

CLASS_EVAL="${REPO}/eval_sit_hidden_ifid_classnn_v6_a800.py"
GLOBAL_EVAL="${REPO}/eval_sit_hidden_ifid_fullsketch_v6_a800.py"

DATASET_REF="/video_ssd/lpm/ImageNet/train"
DATASET_VAL="/video_ssd/lpm/ImageNet/val"

# ---------------------------------------------------------------------------
# USER-EDITABLE SWEEP PARAMETERS
# ---------------------------------------------------------------------------
T_VALUES_STRING="${T_VALUES:-0.05 0.10 0.25 0.50 0.75 0.90}"
NN_SCOPE="${NN_SCOPE:-class}"       # class | global
# ---------------------------------------------------------------------------

SMALL_REF="${SMALL_REF:-50000}"
SMALL_VAL="${SMALL_VAL:-50000}"
DEPTH="${DEPTH:-8}"
ALPHA="${ALPHA:-0.5}"
NN_METRIC="${NN_METRIC:-cosine}"

# Used only by the global v6 evaluator. The class evaluator searches all
# same-class references and therefore ignores these shortlist parameters.
COARSE_TOPK="${COARSE_TOPK:-1024}"
SKETCH_DIM="${SKETCH_DIM:-4096}"
SKETCH_SEED="${SKETCH_SEED:-9187}"
SKETCH_BUILD_BATCH="${SKETCH_BUILD_BATCH:-512}"

SEED="${SEED:-0}"
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-16}"
LARGE_MODEL_BATCH_SIZE="${LARGE_MODEL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GPU_MEMORY_RESERVE_GIB="${GPU_MEMORY_RESERVE_GIB:-20}"
PRELOAD_CHUNK="${PRELOAD_CHUNK:-512}"

case "$NN_SCOPE" in
  class)
    PY_SCRIPT="$CLASS_EVAL"
    ;;
  global)
    PY_SCRIPT="$GLOBAL_EVAL"
    ;;
  *)
    echo "ERROR: NN_SCOPE must be 'class' or 'global', got: $NN_SCOPE" >&2
    exit 2
    ;;
esac

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: evaluator not found: $PY_SCRIPT" >&2
  exit 2
fi

read -r -a T_LIST <<< "$T_VALUES_STRING"
if (( ${#T_LIST[@]} == 0 )); then
  echo "ERROR: T_VALUES is empty" >&2
  exit 2
fi

# Validate t values and reject exact duplicate strings.
declare -A SEEN_T=()
for t in "${T_LIST[@]}"; do
  if [[ -n "${SEEN_T[$t]+x}" ]]; then
    echo "ERROR: duplicate t value: $t" >&2
    exit 2
  fi
  SEEN_T[$t]=1
  python - "$t" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"Invalid t value: {sys.argv[1]}") from exc
if not math.isfinite(value) or not (0.0 <= value <= 1.0):
    raise SystemExit(f"t must be finite and in [0,1], got {value}")
PY
  if (( $? != 0 )); then
    exit 2
  fi
done

BASE_ROOT="${OUT_ROOT}/hidden_ifid_v6_tsweep/${NN_SCOPE}"
CACHE_ROOT="${BASE_ROOT}/cache"
RESULT_ROOT="${BASE_ROOT}/results"
LOG_ROOT="${BASE_ROOT}/logs"
STATUS_ROOT="${BASE_ROOT}/status"
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
NT=${#T_LIST[@]}
TOTAL=$((N * NT))

if (( ${#YAMLS[@]} != N || ${#TAGS[@]} != N || ${#EXPS[@]} != N )); then
  echo "ERROR: model metadata array lengths do not match" >&2
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

t_slug() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  echo "t_${value}"
}

run_gpu_worker() {
  local gpu="$1"
  local task_idx t_idx model_idx t t_dir
  local model yaml tag exp_dir exp_path cache_dir result_dir
  local t_log_root t_status_root model_log bs status actual_sit_model

  echo "[$(date '+%F %T')] GPU${gpu} worker started; scope=${NN_SCOPE}"

  for ((task_idx=gpu; task_idx<TOTAL; task_idx+=4)); do
    t_idx=$((task_idx / N))
    model_idx=$((task_idx % N))

    t="${T_LIST[$t_idx]}"
    t_dir="$(t_slug "$t")"
    model="${MODELS[$model_idx]}"
    yaml="${YAMLS[$model_idx]}"
    tag="${TAGS[$model_idx]}"
    exp_dir="${EXPS[$model_idx]}"
    exp_path="${EXP_ROOT}/${exp_dir}"

    cache_dir="${CACHE_ROOT}/${t_dir}/${tag}"
    result_dir="${RESULT_ROOT}/${t_dir}/${tag}"
    t_log_root="${LOG_ROOT}/${t_dir}"
    t_status_root="${STATUS_ROOT}/${t_dir}"
    model_log="${t_log_root}/${tag}.out"
    bs="$(batch_size_for_tag "$tag")"

    mkdir -p "$cache_dir" "$result_dir" "$t_log_root" "$t_status_root"
    rm -f "${t_status_root}/${tag}.running" "${t_status_root}/${tag}.failed"

    if [[ -s "${result_dir}/hidden_ifid_metrics.json" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} SKIP t=${t} ${model}: result exists"
      touch "${t_status_root}/${tag}.done"
      continue
    fi

    if [[ ! -d "$exp_path" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING experiment: $exp_path" | tee "$model_log"
      echo "missing experiment: $exp_path" > "${t_status_root}/${tag}.failed"
      continue
    fi
    if [[ ! -f "${exp_path}/args.json" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING args.json: ${exp_path}/args.json" | tee "$model_log"
      echo "missing args.json" > "${t_status_root}/${tag}.failed"
      continue
    fi
    if [[ ! -f "${exp_path}/checkpoints/0400000.pt" ]]; then
      echo "[$(date '+%F %T')] GPU${gpu} MISSING checkpoint: ${exp_path}/checkpoints/0400000.pt" | tee "$model_log"
      echo "missing checkpoint" > "${t_status_root}/${tag}.failed"
      continue
    fi

    actual_sit_model="$(
      python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model", ""))' \
        "${exp_path}/args.json" 2>/dev/null || true
    )"
    case "$actual_sit_model" in
      SiT-XL/*|SiT_XL/*)
        ;;
      *)
        echo "[$(date '+%F %T')] GPU${gpu} SKIP non-XL model: ${actual_sit_model}" | tee "$model_log"
        echo "non-XL model: ${actual_sit_model}" > "${t_status_root}/${tag}.failed"
        continue
        ;;
    esac

    touch "${t_status_root}/${tag}.running"
    {
      echo "============================================================"
      echo "start_time=$(date '+%F %T')"
      echo "gpu=${gpu}"
      echo "task_index=${task_idx}/${TOTAL}"
      echo "nn_scope=${NN_SCOPE}"
      echo "t=${t}"
      echo "model=${model}"
      echo "yaml_reference=${yaml}"
      echo "tag=${tag}"
      echo "exp_path=${exp_path}"
      echo "sit_model=${actual_sit_model}"
      echo "batch_size=${bs}"
      echo "reference=${SMALL_REF}"
      echo "validation=${SMALL_VAL}"
      echo "depth=${DEPTH}"
      echo "alpha=${ALPHA}"
      echo "nn_metric=${NN_METRIC}"
      echo "coarse_topk=${COARSE_TOPK}"
      echo "sketch_dim=${SKETCH_DIM}"
      echo "gpu_reference_preload=1"
      echo "============================================================"
    } > "$model_log"

    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=6 \
      MKL_NUM_THREADS=6 \
      OPENBLAS_NUM_THREADS=6 \
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
        --prefetch-factor "$PREFETCH_FACTOR" \
        --depth "$DEPTH" \
        --t "$t" \
        --alpha "$ALPHA" \
        --nn-metric "$NN_METRIC" \
        --coarse-topk "$COARSE_TOPK" \
        --sketch-dim "$SKETCH_DIM" \
        --sketch-seed "$SKETCH_SEED" \
        --sketch-build-batch "$SKETCH_BUILD_BATCH" \
        --seed "$SEED" \
        --device cuda:0 \
        --expected-tokens 0 \
        --expected-dim 1152 \
        --cache-dir "$cache_dir" \
        --preload-reference-gpu \
        --no-preload-reference-ram \
        --gpu-memory-reserve-gib "$GPU_MEMORY_RESERVE_GIB" \
        --preload-chunk "$PRELOAD_CHUNK" \
        --output-dir "$result_dir" \
        --reuse-cache \
        --no-keep-cache \
        --alpha0-tolerance 0.001 \
        >> "$model_log" 2>&1
    status=$?

    rm -f "${t_status_root}/${tag}.running"
    if (( status == 0 )) && [[ -s "${result_dir}/hidden_ifid_metrics.json" ]]; then
      touch "${t_status_root}/${tag}.done"
      echo "[$(date '+%F %T')] GPU${gpu} DONE t=${t} ${model}" | tee -a "$model_log"
    else
      printf 'exit_code=%s\nlog=%s\n' "$status" "$model_log" \
        > "${t_status_root}/${tag}.failed"
      echo "[$(date '+%F %T')] GPU${gpu} FAILED t=${t} ${model}, exit=${status}; continuing" \
        | tee -a "$model_log"
    fi
  done

  echo "[$(date '+%F %T')] GPU${gpu} worker finished"
}

echo "============================================================"
echo "V6 t sweep"
echo "NN_SCOPE=${NN_SCOPE}"
echo "T_VALUES=${T_LIST[*]}"
echo "models=${N}"
echo "tasks=${TOTAL}"
echo "evaluator=${PY_SCRIPT}"
echo "result_root=${RESULT_ROOT}"
echo "============================================================"

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
