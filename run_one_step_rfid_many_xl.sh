#!/usr/bin/env bash
set -u

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
EVAL_PY="${EVAL_PY:-$REPO/eval_one_step_rfid.py}"
DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/one_step_rfid_many_xl}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-32400}"

# Experiment roots to inspect. Only direct children are scanned.
EXP_ROOTS=(
  "/share/xutongda/REPA-E/exps"
  "/video_ssd/kongzishang/xutongda/git/REPA-E/exps"
  "/video_ssd/kongzishang/xutongda/git/REPA-E/exps_unet"
)

# Current larger VAE/tokenizer set (25), intentionally not tied to the 22-model paper subset.
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

echo 'model,exp_path,one_step_rfid_t0.1,one_step_rfid_t0.4,one_step_rfid_t0.8,status' > "$SUMMARY"
echo -e 'model\tvae_config\texp_path\tstatus' > "$MANIFEST"

# Select the most likely paired SiT-XL baseline by reading args.json, rather than
# assuming every experiment directory follows exactly the same naming convention.
discover_exp() {
  local cfg="$1"
  local alias="$2"
  python - "$cfg" "$alias" "$TRAIN_STEPS" "${EXP_ROOTS[@]}" <<'PY'
import json, os, sys
cfg, alias, steps, *roots = sys.argv[1:]
steps = int(steps)
ckpt = f"{steps:07d}.pt"
cs = []
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
            a = json.load(open(ap))
        except Exception:
            continue
        vc = str(a.get("vae_config", ""))
        if os.path.basename(vc) != cfg:
            continue
        model = str(a.get("model", ""))
        if "XL" not in model.upper():
            continue
        name = os.path.basename(e.path).lower()
        al = alias.lower()
        # Prefer the canonical short 400k SiT-XL experiment name. The config match
        # is the hard constraint; name matching is only a ranking signal.
        score = 0
        if name.startswith("sit-xl-"):
            score += 40
        if name.endswith("-400k"):
            score += 40
        if al and al in name:
            score += 30
        # Avoid selecting obvious unrelated ablation branches when a baseline exists.
        for bad in ["consistency", "curriculum", "navgeom", "pathgeom", "local-", "orthanchor", "ablation"]:
            if bad in name:
                score -= 100
        cs.append((score, -len(name), e.path))
if not cs:
    sys.exit(1)
cs.sort(reverse=True)
print(cs[0][2])
PY
}

idx=0
for spec in "${MODELS[@]}"; do
  IFS='|' read -r CFG LABEL ALIAS <<< "$spec"
  PORT=$((BASE_PORT + idx))
  idx=$((idx + 1))

  echo "============================================================"
  echo "[$LABEL] discovering paired SiT-XL checkpoint ..."

  EXP=$(discover_exp "$CFG" "$ALIAS" 2>/dev/null || true)
  if [[ -z "$EXP" ]]; then
    echo "[$LABEL] SKIP: no paired SiT-XL $TRAIN_STEPS checkpoint found"
    echo -e "$LABEL\t$CFG\t\tMISSING_EXP" >> "$MANIFEST"
    echo "$LABEL,,,,,MISSING_EXP" >> "$SUMMARY"
    continue
  fi

  EXP_BASE=$(basename "$EXP")
  RUN_DIR="$OUT_ROOT/${EXP_BASE}_step$(printf '%07d' "$TRAIN_STEPS")_linear_t0.1-0.4-0.8"
  METRICS="$RUN_DIR/metrics.json"
  LOG="$LOG_ROOT/${LABEL}.out"

  echo "[$LABEL] exp: $EXP"
  echo -e "$LABEL\t$CFG\t$EXP\tFOUND" >> "$MANIFEST"

  if [[ -s "$METRICS" ]]; then
    echo "[$LABEL] existing result found, skip compute: $METRICS"
    STATUS="DONE"
  else
    echo "[$LABEL] running on GPUs $GPU_IDS, batch/GPU=$BATCH_SIZE"
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
      --path-type linear \
      --out-dir "$OUT_ROOT" \
      --no-include-rfid \
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
d = json.load(open(path))
m = d.get("metrics", {})
print(",".join([
    label,
    exp,
    str(m.get("one_step_rfid_t0.1", "")),
    str(m.get("one_step_rfid_t0.4", "")),
    str(m.get("one_step_rfid_t0.8", "")),
    "DONE",
]))
PY
  else
    echo "$LABEL,$EXP,,,,FAILED" >> "$SUMMARY"
  fi
done

echo "============================================================"
echo "FINAL SUMMARY: $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "MANIFEST: $MANIFEST"
