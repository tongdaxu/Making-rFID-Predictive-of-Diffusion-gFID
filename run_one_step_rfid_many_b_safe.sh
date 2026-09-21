#!/usr/bin/env bash
set -u

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
EVAL_PY="${EVAL_PY:-$REPO/eval_one_step_rfid_safe.py}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
TIME_MODE="${TIME_MODE:-shifted_raw}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/one_step_rfid_many_b_${TIME_MODE}}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-32500}"
DRY_RUN="${DRY_RUN:-0}"

EXP_ROOTS=(
  "/share/xutongda/REPA-E/exps"
  "/video_ssd/kongzishang/xutongda/git/REPA-E/exps"
  "/video_ssd/kongzishang/xutongda/git/REPA-E/exps_unet"
)

# Larger current set (25); intentionally not restricted to the paper's 22-model subset.
MODELS=(
  "SDVAE.yaml|SDVAE|sdvae"
  "SD3VAE.yaml|SD3VAE|sd3vae"
  "FLUXVAE.yaml|FLUXVAE|fluxvae"
  "DCAE.yaml|DCAE|dcae"
  "SOFTVQ.yaml|SOFTVQ|softvq"
  "VAVAE.yaml|VAVAE|vavae"
  "VAVAE64.yaml|VAVAE64|vavae64"
  "EQVAE.yaml|EQVAE|eqvae"
  "MAETOK.yaml|MAETOK|maetok"
  "INVAE.yaml|INVAE|invae"
  "REPAEVAE.yaml|REPAEVAE|repae"
  "DETOK.yaml|DETOK|detok"
  "QWVAE.yaml|QWVAE|qwen"
  "RAE.yaml|RAE|rae"
  "SVG.yaml|SVG|svg"
  "FLUX2VAE.yaml|FLUX2VAE|flux2"
  "SVGT2I_P_STAGE1_256.yaml|SVGT2I_P_STAGE1_256|svg"
  "DMVAE.yaml|DMVAE|dmvae"
  "UAE.yaml|UAE|uae"
  "VTPS.yaml|VTPS|vtps"
  "VTPB.yaml|VTPB|vtpb"
  "VTPL.yaml|VTPL|vtpl"
  "ScaleRAE_SIGLIP2_NONORM.yaml|ScaleRAE_SIGLIP2_NONORM|siglip"
  "ScaleRAE_WEBSSL.yaml|ScaleRAE_WEBSSL|webssl"
  "PAE_DINOv2L_d32.yaml|PAE_DINOv2L_d32|pae"
)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$REPO"

if [[ ! -f "$EVAL_PY" ]]; then
  echo "ERROR: evaluator not found: $EVAL_PY" >&2
  exit 2
fi

NPROC=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
CKPT_NAME=$(printf "%07d.pt" "$TRAIN_STEPS")
SUMMARY="$OUT_ROOT/summary.csv"
MANIFEST="$OUT_ROOT/manifest.tsv"
CANDIDATES="$OUT_ROOT/candidates.tsv"

# Recreate summaries for this invocation; metrics.json files are not deleted.
echo 'model,exp_path,rfid,one_step_rfid_t0.1,one_step_rfid_t0.4,one_step_rfid_t0.8,delta_t0.1,delta_t0.4,delta_t0.8,ratio_t0.1,ratio_t0.4,ratio_t0.8,status' > "$SUMMARY"
echo -e 'model\tvae_config\texp_path\tmodel_arch\tprediction\tprediction_internal\tpath_type\tstatus' > "$MANIFEST"
echo -e 'model\tscore\texp_path' > "$CANDIDATES"

# Discover ONLY legacy IFID SiT-B baselines:
#   exact VAE config basename
#   SiT-B/* architecture
#   400k checkpoint exists
#   prediction=v
#   prediction_internal key ABSENT
#   path_type=linear
# Obvious ablations are rejected rather than merely penalized.
discover_exp() {
  local cfg="$1"
  local alias="$2"
  local label="$3"
  python - "$cfg" "$alias" "$label" "$TRAIN_STEPS" "$CANDIDATES" "${EXP_ROOTS[@]}" <<'PY'
import json, os, sys
cfg, alias, label, steps, candfile, *roots = sys.argv[1:]
steps = int(steps)
ckpt = f"{steps:07d}.pt"
bad_tokens = [
    "consistency", "curriculum", "navgeom", "pathgeom", "local-",
    "orthanchor", "ablation", "selfcons", "basin-", "coarse", "frozensit",
]
cands = []
for root in roots:
    if not os.path.isdir(root):
        continue
    try:
        entries = list(os.scandir(root))
    except OSError:
        continue
    for e in entries:
        if not e.is_dir():
            continue
        ap = os.path.join(e.path, "args.json")
        cp = os.path.join(e.path, "checkpoints", ckpt)
        if not (os.path.isfile(ap) and os.path.isfile(cp)):
            continue
        try:
            with open(ap) as f:
                a = json.load(f)
        except Exception:
            continue
        if os.path.basename(str(a.get("vae_config", ""))) != cfg:
            continue
        model = str(a.get("model", ""))
        # Legacy 1D checkpoints use names such as "SiT_B/1D", while
        # 2D checkpoints use "SiT-B/1", "SiT-B/2", etc.
        # Normalize "_" -> "-" before checking the backbone family.
        model_norm = model.upper().replace("_", "-")
        if not model_norm.startswith("SIT-B/"):
            continue
        if str(a.get("prediction", "v")).lower() != "v":
            continue
        # Presence itself means this is a newer semantics branch; do not mix it.
        if "prediction_internal" in a:
            continue
        if str(a.get("path_type", "linear")).lower() != "linear":
            continue
        name = os.path.basename(e.path).lower()
        if any(tok in name for tok in bad_tokens):
            continue

        score = 0
        if name.startswith("sit-b-"):
            score += 40
        if name.endswith("-400k"):
            score += 40
        if alias.lower() and alias.lower() in name:
            score += 30
        # Prefer shorter canonical baseline names after hard semantic filtering.
        cands.append((score, -len(name), e.path, model))

with open(candfile, "a") as f:
    for score, neglen, path, model in sorted(cands, reverse=True):
        f.write(f"{label}\t{score}\t{path}\n")

if not cands:
    sys.exit(1)
cands.sort(reverse=True)
print(cands[0][2])
PY
}

validate_exp() {
  local exp="$1"
  python - "$exp" <<'PY'
import json, os, sys
exp = sys.argv[1]
with open(os.path.join(exp, "args.json")) as f:
    a = json.load(f)
print("\t".join([
    str(a.get("model", "")),
    str(a.get("prediction", "v")),
    str(a.get("prediction_internal", "<absent>")),
    str(a.get("path_type", "linear")),
]))
PY
}

idx=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r CFG LABEL ALIAS <<< "$spec"
  PORT=$((BASE_PORT + idx))
  idx=$((idx + 1))

  echo "============================================================"
  echo "[$LABEL] discovering legacy paired SiT-B checkpoint ..."

  EXP=$(discover_exp "$CFG" "$ALIAS" "$LABEL" 2>/dev/null || true)
  if [[ -z "$EXP" ]]; then
    echo "[$LABEL] SKIP: no safe legacy SiT-B $TRAIN_STEPS checkpoint found"
    echo -e "$LABEL\t$CFG\t\t\t\t\t\tMISSING_EXP" >> "$MANIFEST"
    echo "$LABEL,,,,,,,,,,,,MISSING_EXP" >> "$SUMMARY"
    continue
  fi

  META=$(validate_exp "$EXP")
  IFS=$'\t' read -r ARCH PRED PINTERNAL PTYPE <<< "$META"
  echo "[$LABEL] exp: $EXP"
  echo "[$LABEL] args: model=$ARCH prediction=$PRED prediction_internal=$PINTERNAL path_type=$PTYPE"
  echo -e "$LABEL\t$CFG\t$EXP\t$ARCH\t$PRED\t$PINTERNAL\t$PTYPE\tFOUND" >> "$MANIFEST"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$LABEL] DRY_RUN: mapping only; not computing"
    echo "$LABEL,$EXP,,,,,,,,,,,READY" >> "$SUMMARY"
    continue
  fi

  EXP_BASE=$(basename "$EXP")
  RUN_DIR="$OUT_ROOT/${EXP_BASE}_step$(printf '%07d' "$TRAIN_STEPS")_linear_${TIME_MODE}_t0.1-0.4-0.8"
  METRICS="$RUN_DIR/metrics.json"
  LOG="$LOG_ROOT/${LABEL}.out"

  if [[ -s "$METRICS" ]]; then
    echo "[$LABEL] existing result found, skip compute: $METRICS"
    STATUS="DONE"
  else
    echo "[$LABEL] running on GPUs $GPU_IDS, batch/GPU=$BATCH_SIZE, time_mode=$TIME_MODE"
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
      echo "[$LABEL] FAILED rc=$RC; log=$LOG"
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
    "one_step_rfid_t0.1", "one_step_rfid_t0.4", "one_step_rfid_t0.8",
    "delta_rfid_t0.1", "delta_rfid_t0.4", "delta_rfid_t0.8",
    "ratio_rfid_t0.1", "ratio_rfid_t0.4", "ratio_rfid_t0.8",
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
echo "CANDIDATES: $CANDIDATES"
