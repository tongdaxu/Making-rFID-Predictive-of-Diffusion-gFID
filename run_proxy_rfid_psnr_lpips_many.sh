#!/usr/bin/env bash
set -u

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
EVAL_PY="${EVAL_PY:-$REPO/eval_proxy_rfid_psnr_lpips.py}"
EXP_ROOT="${EXP_ROOT:-/share/xutongda/REPA-E/exps}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"

# ------------------------------------------------------------
# EDIT / OVERRIDE THESE TWO PARAMETERS MOST OF THE TIME
# ------------------------------------------------------------
BACKBONE="${BACKBONE:-xl}"   # b | xl
SETTINGS="${SETTINGS:-ode:0.1,sde:0.1,ode:0.05,sde:0.05}"
# Optional example:
# SETTINGS="ode:0.4,sde:0.4"
# SETTINGS="recon,direct:0.1,direct:0.4,direct:0.8"

TIME_MODE="${TIME_MODE:-shifted_raw}"
MID_FRAC="${MID_FRAC:-0.5}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-49000}"
LPIPS_NETWORK="${LPIPS_NETWORK:-alex}"

if [[ "$BACKBONE" != "b" && "$BACKBONE" != "xl" ]]; then
  echo "ERROR: BACKBONE must be b or xl, got: $BACKBONE" >&2
  exit 2
fi

SETTINGS_TAG=$(python - "$SETTINGS" <<'PY'
import sys
s=sys.argv[1]
print(s.replace(":", "").replace(",", "__").replace(".", "p"))
PY
)

OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/proxy_rfid_psnr_lpips_${BACKBONE}_${SETTINGS_TAG}}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
SUMMARY_LONG="$OUT_ROOT/summary_long.csv"
SUMMARY_WIDE="$OUT_ROOT/summary_wide.csv"
MANIFEST="$OUT_ROOT/manifest.tsv"

if [[ "$BACKBONE" == "b" ]]; then
  EXPECTED_PREFIX="SIT-B/"
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
else
  EXPECTED_PREFIX="SIT-XL/"
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
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

if [[ ! -f "$EVAL_PY" ]]; then
  echo "ERROR: evaluator not found: $EVAL_PY" >&2
  exit 2
fi

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
CKPT_NAME=$(printf "%07d.pt" "$TRAIN_STEPS")

echo 'model,exp_path,setting,rfid,psnr,lpips,status' > "$SUMMARY_LONG"
echo -e 'model\texp_path\tmodel_arch\tvae_config\tprediction\tprediction_internal\tpath_type\tstatus' > "$MANIFEST"

validate_exp () {
  local exp="$1"
  python - "$exp" "$CKPT_NAME" "$EXPECTED_PREFIX" <<'PY'
import json, os, sys
exp, ckpt_name, expected_prefix = sys.argv[1:]
ap = os.path.join(exp, "args.json")
cp = os.path.join(exp, "checkpoints", ckpt_name)
if not os.path.isfile(ap):
    raise SystemExit(f"NO_ARGS_JSON\t{ap}")
if not os.path.isfile(cp):
    raise SystemExit(f"NO_CHECKPOINT\t{cp}")
a = json.load(open(ap))
model = str(a.get("model", ""))
model_norm = model.upper().replace("_", "-")
pred = str(a.get("prediction", "v")).lower()
pinternal = a.get("prediction_internal", "<absent>")
ptype = str(a.get("path_type", "linear")).lower()
vae_cfg = os.path.basename(str(a.get("vae_config", "")))
if not model_norm.startswith(expected_prefix):
    raise SystemExit(f"BAD_MODEL\t{model}")
if pred != "v":
    raise SystemExit(f"BAD_PREDICTION\t{pred}")
if "prediction_internal" in a:
    raise SystemExit(f"NEW_INTERNAL_BRANCH\t{pinternal}")
if ptype != "linear":
    raise SystemExit(f"BAD_PATH_TYPE\t{ptype}")
print("\t".join([model, vae_cfg, pred, str(pinternal), ptype]))
PY
}

# Compute the evaluator's exact setting tag so we know the metrics.json path.
EVAL_TAG=$(python - "$SETTINGS" <<'PY'
import sys

def key(kind,t):
    return "recon" if kind=="recon" else f"{kind}_t{float(t):g}"
seen=set(); out=[]
for raw in sys.argv[1].split(','):
    raw=raw.strip()
    if not raw: continue
    if raw=='recon': item=('recon',None)
    else:
        k,t=raw.split(':',1); item=(k.strip().lower(),float(t))
    q=key(*item)
    if q not in seen:
        seen.add(q); out.append(q.replace('.', 'p'))
print('__'.join(out))
PY
)

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
    echo "$LABEL,$EXP,,,,$(printf %q "VALIDATION_FAILED")" >> "$SUMMARY_LONG"
    continue
  fi

  IFS=$'\t' read -r ARCH VAE_CFG PRED PINTERNAL PTYPE <<< "$META"
  echo -e "$LABEL\t$EXP\t$ARCH\t$VAE_CFG\t$PRED\t$PINTERNAL\t$PTYPE\tFOUND" >> "$MANIFEST"

  RUN_DIR="$OUT_ROOT/${EXP_NAME}_step$(printf '%07d' "$TRAIN_STEPS")_${TIME_MODE}_${EVAL_TAG}_rfid_psnr_lpips"
  METRICS="$RUN_DIR/metrics.json"
  LOG="$LOG_ROOT/${LABEL}.out"

  if [[ -s "$METRICS" ]]; then
    echo "[$LABEL] existing metrics.json found; skip compute"
    STATUS="DONE"
  else
    echo "[$LABEL] running settings=$SETTINGS"
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
      --settings "$SETTINGS" \
      --mid-frac "$MID_FRAC" \
      --time-mode "$TIME_MODE" \
      --path-type linear \
      --lpips-network "$LPIPS_NETWORK" \
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
    python - "$LABEL" "$EXP" "$METRICS" >> "$SUMMARY_LONG" <<'PY'
import json, sys
label, exp, path = sys.argv[1:]
d = json.load(open(path))
for setting, m in d["metrics"].items():
    print(f"{label},{exp},{setting},{m['rfid']},{m['psnr']},{m['lpips']},DONE")
PY
  else
    echo "$LABEL,$EXP,,,,,FAILED" >> "$SUMMARY_LONG"
  fi
done

# Wide CSV for directly pasting into the spreadsheet.
python - "$SUMMARY_LONG" "$SUMMARY_WIDE" "$SETTINGS" <<'PY'
import csv, sys
long_path, wide_path, settings_spec = sys.argv[1:]

def norm_setting(raw):
    raw=raw.strip()
    if raw=='recon': return 'recon'
    k,t=raw.split(':',1)
    return f"{k.strip().lower()}_t{float(t):g}"

settings=[]
for x in settings_spec.split(','):
    if x.strip():
        s=norm_setting(x)
        if s not in settings: settings.append(s)

rows={}
order=[]
with open(long_path, newline='') as f:
    for r in csv.DictReader(f):
        if r['status']!='DONE' or not r['setting']:
            continue
        model=r['model']
        if model not in rows:
            rows[model]={}; order.append(model)
        rows[model][r['setting']]=(r['rfid'],r['psnr'],r['lpips'])

header=['model']
for s in settings:
    header += [f'{s}_rfid', f'{s}_psnr', f'{s}_lpips']

with open(wide_path,'w',newline='') as f:
    w=csv.writer(f); w.writerow(header)
    for model in order:
        row=[model]
        for s in settings:
            rf,p,l=rows[model].get(s, ('','',''))
            row += [rf,p,l]
        w.writerow(row)
print(wide_path)
PY

echo "============================================================"
echo "LONG SUMMARY : $SUMMARY_LONG"
echo "WIDE SUMMARY : $SUMMARY_WIDE"
echo "MANIFEST     : $MANIFEST"
