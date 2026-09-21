#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate/one_step_rfid_many_xl}"
MANIFEST="${MANIFEST:-$ROOT/manifest.tsv}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 2
fi

python - "$MANIFEST" <<'PY'
import csv, json, os, sys
p = sys.argv[1]
print("label\tmodel\tprediction\tprediction_internal\tpath_type\tvae_config\texp\tverdict")
with open(p, newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        exp = (r.get('exp_path') or '').strip()
        label = (r.get('model') or '').strip()
        if not exp:
            print(f"{label}\t\t\t\t\t\t\tNO_EXP")
            continue
        ap = os.path.join(exp, 'args.json')
        try:
            with open(ap) as g:
                a = json.load(g)
        except Exception as e:
            print(f"{label}\t\t\t\t\t\t{exp}\tBAD_ARGS:{e}")
            continue
        model = str(a.get('model',''))
        pred = str(a.get('prediction','v'))
        pi = '<absent>' if 'prediction_internal' not in a else str(a.get('prediction_internal'))
        pt = str(a.get('path_type','linear'))
        vc = os.path.basename(str(a.get('vae_config','')))
        bad = []
        if 'XL' not in model.upper(): bad.append('NOT_XL')
        if pred.lower() != 'v': bad.append('NOT_V')
        if 'prediction_internal' in a: bad.append('HAS_PRED_INTERNAL')
        if pt.lower() != 'linear': bad.append('NOT_LINEAR')
        name = os.path.basename(exp).lower()
        for tok in ['consistency','curriculum','navgeom','pathgeom','local-','orthanchor','ablation','selfcons','basin-','coarse','frozensit']:
            if tok in name:
                bad.append('ABLATION_NAME')
                break
        verdict = 'OK_LEGACY_BASELINE_ARGS' if not bad else '|'.join(bad)
        print(f"{label}\t{model}\t{pred}\t{pi}\t{pt}\t{vc}\t{exp}\t{verdict}")
PY

echo
echo "Current repository source checks:"
if grep -R -n --include='*.py' 'prediction_internal' "$REPO/ifid/sit" 2>/dev/null | head -20; then
  true
fi

echo
echo "Legacy BN normalization occurrences:"
grep -R -n --include='*.py' -E 'normalized_x = self\.bn|self\.bn\(x\.transpose' "$REPO/ifid/sit" 2>/dev/null | head -30 || true
