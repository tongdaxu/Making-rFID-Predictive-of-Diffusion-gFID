#!/usr/bin/env bash
set -u

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
EVAL_PY="${EVAL_PY:-$REPO/eval_one_step_rfid_safe.py}"
EXP_ROOT="${EXP_ROOT:-/share/xutongda/REPA-E/exps}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"

TIME_MODE="${TIME_MODE:-shifted_raw}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-32500}"

OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/one_step_rfid_many_b_${TIME_MODE}}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
SUMMARY="$OUT_ROOT/summary.csv"
MANIFEST="$OUT_ROOT/manifest.tsv"

# Explicit mapping to the exact SiT-B baselines used in the benchmark.
# This intentionally avoids heuristic experiment discovery.
MODELS=(
  "SDVAE|sit-b-sdvae-400k"
  "SD3VAE|sit-b-sd3-400k"
  "FLUXVAE|sit-b-flux-shift-400k"
  "DCAE|sit-b-dcae-400k"
  "SOFTVQ|sit-b-softvq-400k"
  "VAVAE|sit-b-vavae-400k"
  "VAVAE64|sit-b-vavae64-400k"
  "EQVAE|sit-b-eqvae-400k"
  "MAETOK|sit-b-maetok-400k"
  "INVAE|sit-b-invae-400k"
  "REPAEVAE|sit-b-repae-400k"
  "DETOK|sit-b-detok-400k"
  "QWVAE|sit-b-qw-shift-400k"
  "RAE|sit-b-rae-shift-400k"
  "SVG|sit-b-svg-400k"
  "FLUX2VAE|sit-b-flux2vae-400k"
  "SVGT2I_P_STAGE1_256|sit-b-svgt2iP-400k"
  "DMVAE|sit-b-dmvae-400k"
  "UAE|sit-b-uae-400k"
  "VTPS|sit-b-vtps-400k"
  "VTPB|sit-b-vtpb-400k"
  "VTPL|sit-b-vtpl-400k"
  "ScaleRAE_SIGLIP2_NONORM|sit-b-raet2i_siglip2-400k"
  "ScaleRAE_WEBSSL|sit-b-raet2i_webssl-400k"
  "PAE_DINOv2L_d32|sit-b-pae-dinov2-400k"
)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

if [[ ! -f "$EVAL_PY" ]]; then
  echo "ERROR: evaluator not found: $EVAL_PY" >&2
  exit 2
fi

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
CKPT_NAME=$(printf "%07d.pt" "$TRAIN_STEPS")

echo 'model,exp_path,rfid,one_step_rfid_t0.1,one_step_rfid_t0.4,one_step_rfid_t0.8,delta_t0.1,delta_t0.4,delta_t0.8,ratio_t0.1,ratio_t0.4,ratio_t0.8,status' > "$SUMMARY"
echo -e 'model\texp_path\tmodel_arch\tvae_config\tprediction\tprediction_internal\tpath_type\tstatus' > "$MANIFEST"

validate_exp () {
  local label="$1"
  local exp="$2"

  python - "$label" "$exp" "$CKPT_NAME" <<'PY'
import json, os, sys
label, exp, ckpt_name = sys.argv[1:]

ap = os.path.join(exp, "args.json")
cp = os.path.join(exp, "checkpoints", ckpt_name)

if not os.path.isfile(ap):
    raise SystemExit(f"NO_ARGS_JSON\t{ap}")
if not os.path.isfile(cp):
    raise SystemExit(f"NO_CHECKPOINT\t{cp}")

with open(ap) as f:
    a = json.load(f)

model = str(a.get("model", ""))
model_norm = model.upper().replace("_", "-")
pred = str(a.get("prediction", "v")).lower()
pinternal = a.get("prediction_internal", "<absent>")
ptype = str(a.get("path_type", "linear")).lower()
vae_config = os.path.basename(str(a.get("vae_config", "")))

if not model_norm.startswith("SIT-B/"):
    raise SystemExit(f"BAD_MODEL\t{model}")
if pred != "v":
    raise SystemExit(f"BAD_PREDICTION\t{pred}")
if "prediction_internal" in a:
    raise SystemExit(f"NEW_INTERNAL_BRANCH\t{pinternal}")
if ptype != "linear":
    raise SystemExit(f"BAD_PATH_TYPE\t{ptype}")

print("\t".join([
    model,
    vae_config,
    pred,
    str(pinternal),
    ptype,
]))
PY
}

idx=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r LABEL EXP_NAME <<< "$spec"
  EXP="$EXP_ROOT/$EXP_NAME"
  PORT=$((BASE_PORT + idx))
  idx=$((idx + 1))

  echo "============================================================"
  echo "[$LABEL] $EXP"

  set +e
  META=$(validate_exp "$LABEL" "$EXP" 2>&1)
  VRC=$?
  set -e

  if [[ $VRC -ne 0 ]]; then
    echo "[$LABEL] VALIDATION FAILED: $META"
    echo -e "$LABEL\t$EXP\t\t\t\t\t\tVALIDATION_FAILED:$META" >> "$MANIFEST"
    echo "$LABEL,$EXP,,,,,,,,,,,VALIDATION_FAILED" >> "$SUMMARY"
    continue
  fi

  IFS=$'\t' read -r ARCH VAE_CFG PRED PINTERNAL PTYPE <<< "$META"
  echo "[$LABEL] model=$ARCH vae_config=$VAE_CFG prediction=$PRED prediction_internal=$PINTERNAL path_type=$PTYPE"
  echo -e "$LABEL\t$EXP\t$ARCH\t$VAE_CFG\t$PRED\t$PINTERNAL\t$PTYPE\tFOUND" >> "$MANIFEST"

  RUN_DIR="$OUT_ROOT/${EXP_NAME}_step$(printf '%07d' "$TRAIN_STEPS")_linear_${TIME_MODE}_t0.1-0.4-0.8"
  METRICS="$RUN_DIR/metrics.json"
  LOG="$LOG_ROOT/${LABEL}.out"

  if [[ -s "$METRICS" ]]; then
    echo "[$LABEL] existing result found; skip compute"
    STATUS="DONE"
  else
    echo "[$LABEL] running: time_mode=$TIME_MODE, batch/GPU=$BATCH_SIZE, GPUs=$GPU_IDS"

    set +e
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    torchrun \
      --nproc_per_node="$NPROC" \
      --master_port="$PORT" \
      "$EVAL_PY" \
      --exp-path "$EXP" \
      --train-steps "$TRAIN_STEPS" \
      --dataset "$DATASET" \
      --num-images "$NUM_IMAGES" \
      --batch-size "$BATCH_SIZE" \
      --t-values 0.1 0.4 0.8 \
      --time-mode "$TIME_MODE" \
      --path-type linear \
      --out-dir "$OUT_ROOT" \
      > "$LOG" 2>&1
    RC=$?
    set -e

    if [[ $RC -eq 0 && -s "$METRICS" ]]; then
      STATUS="DONE"
      echo "[$LABEL] DONE"
    else
      STATUS="FAILED"
      echo "[$LABEL] FAILED rc=$RC log=$LOG"
    fi
  fi

  if [[ "$STATUS" == "DONE" ]]; then
    python - "$LABEL" "$EXP" "$METRICS" >> "$SUMMARY" <<'PY'
import json, sys
label, exp, path = sys.argv[1:]
with open(path) as f:
    d = json.load(f)
m = d.get("metrics", {})
keys = [
    "rfid",
    "one_step_rfid_t0.1",
    "one_step_rfid_t0.4",
    "one_step_rfid_t0.8",
    "delta_rfid_t0.1",
    "delta_rfid_t0.4",
    "delta_rfid_t0.8",
    "ratio_rfid_t0.1",
    "ratio_rfid_t0.4",
    "ratio_rfid_t0.8",
]
print(",".join([label, exp] + [str(m.get(k, "")) for k in keys] + ["DONE"]))
PY
  else
    echo "$LABEL,$EXP,,,,,,,,,,,FAILED" >> "$SUMMARY"
  fi
done

echo "============================================================"
echo "FINAL SUMMARY: $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "MANIFEST: $MANIFEST"
