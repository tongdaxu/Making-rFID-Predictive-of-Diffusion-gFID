#!/usr/bin/env bash
set -u

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
EVAL_PY="${EVAL_PY:-$REPO/eval_step_then_xpred_rfid.py}"
EXP_ROOT="${EXP_ROOT:-/share/xutongda/REPA-E/exps}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"

STEP_MODE="${STEP_MODE:-ode}"
TIME_MODE="${TIME_MODE:-shifted_raw}"
T_START="${T_START:-0.8}"
MID_FRAC="${MID_FRAC:-0.5}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-32800}"

OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/step_then_xpred_xl_${STEP_MODE}_raw${T_START}_mid${MID_FRAC}}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
SUMMARY="$OUT_ROOT/summary.csv"
MANIFEST="$OUT_ROOT/manifest.tsv"

MODELS=(
  "SDVAE|sit-xl-sdvae-400k"
  "SD3VAE|sit-xl-sd3-shift-400k"
  "FLUXVAE|sit-xl-flux-shift-400k"
  "DCAE|sit-xl-dcae-400k"
  "SOFTVQ|sit-XL-softvq-400k"
  "VAVAE|sit-xl-vavae-400k"
  "VAVAE64|sit-xl-vavae64-shift-400k"
  "EQVAE|sit-XL-eqvae-400k"
  "MAETOK|sit-XL-maetok-400k"
  "INVAE|sit-xl-invae-400k"
  "REPAEVAE|sit-xl-repae-400k"
  "DETOK|sit-xl-detok-400k"
  "QWVAE|sit-xl-qw-shift-400k"
  "RAE|sit-xl-rae-shift-400k"
  "SVG|sit-xl-svg-400k"
  "FLUX2VAE|sit-xl-flux2vae-400k"
  "SVGT2I_P_STAGE1_256|sit-xl-svgt2iP-400k"
  "DMVAE|sit-xl-dmvae-400k"
  "UAE|sit-xl-uae-400k"
  "VTPS|sit-xl-vtps-400k"
  "VTPB|sit-xl-vtpb-400k"
  "VTPL|sit-xl-vtpl-400k"
  "ScaleRAE_SIGLIP2_NONORM|sit-xl-raet2i_siglip2-400k"
  "ScaleRAE_WEBSSL|sit-xl-raet2i_webssl-400k"
  "PAE_DINOv2L_d32|sit-xl-pae-dinov2-400k"
)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

if [[ ! -f "$EVAL_PY" ]]; then
  echo "ERROR: evaluator not found: $EVAL_PY" >&2
  exit 2
fi

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
CKPT_NAME=$(printf "%07d.pt" "$TRAIN_STEPS")
MID_T=$(python - "$T_START" "$MID_FRAC" <<'PY'
import sys
print(f"{float(sys.argv[1])*float(sys.argv[2]):g}")
PY
)

echo "model,exp_path,metric,status" > "$SUMMARY"
echo -e "model\texp_path\tmodel_arch\tvae_config\tprediction\tprediction_internal\tpath_type\tstatus" > "$MANIFEST"

validate_exp () {
  local exp="$1"
  python - "$exp" "$CKPT_NAME" "SIT-XL/" <<'PY'
import json, os, sys
exp, ckpt_name, expected_prefix = sys.argv[1:]
ap = os.path.join(exp, "args.json")
cp = os.path.join(exp, "checkpoints", ckpt_name)
if not os.path.isfile(ap): raise SystemExit(f"NO_ARGS_JSON\t{ap}")
if not os.path.isfile(cp): raise SystemExit(f"NO_CHECKPOINT\t{cp}")
a = json.load(open(ap))
model = str(a.get("model",""))
model_norm = model.upper().replace("_","-")
pred = str(a.get("prediction","v")).lower()
pinternal = a.get("prediction_internal","<absent>")
ptype = str(a.get("path_type","linear")).lower()
vae_cfg = os.path.basename(str(a.get("vae_config","")))
if not model_norm.startswith(expected_prefix):
    raise SystemExit(f"BAD_MODEL\t{model}")
if pred != "v": raise SystemExit(f"BAD_PREDICTION\t{pred}")
if "prediction_internal" in a: raise SystemExit(f"NEW_INTERNAL_BRANCH\t{pinternal}")
if ptype != "linear": raise SystemExit(f"BAD_PATH_TYPE\t{ptype}")
print("\t".join([model,vae_cfg,pred,str(pinternal),ptype]))
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
  META=$(validate_exp "$EXP" 2>&1)
  VRC=$?
  set -e

  if [[ $VRC -ne 0 ]]; then
    echo "[$LABEL] VALIDATION FAILED: $META"
    echo -e "$LABEL\t$EXP\t\t\t\t\t\tVALIDATION_FAILED:$META" >> "$MANIFEST"
    echo "$LABEL,$EXP,,VALIDATION_FAILED" >> "$SUMMARY"
    continue
  fi

  IFS=$'\t' read -r ARCH VAE_CFG PRED PINTERNAL PTYPE <<< "$META"
  echo -e "$LABEL\t$EXP\t$ARCH\t$VAE_CFG\t$PRED\t$PINTERNAL\t$PTYPE\tFOUND" >> "$MANIFEST"

  RUN_DIR="$OUT_ROOT/${EXP_NAME}_step$(printf '%07d' "$TRAIN_STEPS")_${STEP_MODE}_linear_${TIME_MODE}_mid${MID_FRAC}_t${T_START}"
  METRICS="$RUN_DIR/metrics.json"
  LOG="$LOG_ROOT/${LABEL}.out"

  if [[ -s "$METRICS" ]]; then
    echo "[$LABEL] existing result found; skip"
    STATUS="DONE"
  else
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
      --t-start-values "$T_START" \
      --mid-frac "$MID_FRAC" \
      --step-mode "$STEP_MODE" \
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
    python - "$LABEL" "$EXP" "$METRICS" "$STEP_MODE" "$T_START" "$MID_T" >> "$SUMMARY" <<'PY'
import json, sys
label, exp, path, mode, ts, tm = sys.argv[1:]
d = json.load(open(path))
m = d.get("metrics", {})
key = f"{mode}_step_then_xpred_rfid_t{float(ts):g}_to_t{float(tm):g}"
print(",".join([label, exp, str(m.get(key, "")), "DONE"]))
PY
  else
    echo "$LABEL,$EXP,,FAILED" >> "$SUMMARY"
  fi
done

echo "============================================================"
echo "FINAL SUMMARY: $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "MANIFEST: $MANIFEST"
