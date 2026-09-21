#!/usr/bin/env bash
set -euo pipefail

# Direct OpenAI guided-diffusion evaluator batch runner.
# It does NOT convert images to npz.
# It directly evaluates existing npz files under /share/xutongda/REPA-E/exps.

EXP_ROOT="${EXP_ROOT:-/share/xutongda/REPA-E/exps}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-./third_party/guided-diffusion}"
REF_NPZ="${REF_NPZ:-/video_ssd/kongzishang/xutongda/git/IFID/guided_eval_direct/VIRTUAL_imagenet256_labeled.npz}"

OUT_ROOT="${OUT_ROOT:-./guided_eval_direct}"
MANIFEST="${OUT_ROOT}/guided_npz_manifest.tsv"
RESULT_DIR="${OUT_ROOT}/results"
LOG_DIR="${OUT_ROOT}/logs"

# all: collect every cfg>=1.0 npz
# cfg1_best: collect cfg1.0 plus best_npz from cfg_scan_best logs
COLLECT_MODE="${COLLECT_MODE:-all}"

# Re-run even when result json already exists.
FORCE="${FORCE:-0}"

mkdir -p "${OUT_ROOT}" "${RESULT_DIR}" "${LOG_DIR}"

if [[ ! -f "${REF_NPZ}" ]]; then
  echo "[ERROR] REF_NPZ not found: ${REF_NPZ}"
  echo "Try:"
  echo "  find /share/xutongda/REPA-E -name 'VIRTUAL_imagenet256_labeled.npz'"
  exit 1
fi

if [[ ! -f "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py" ]]; then
  echo "[ERROR] evaluator.py not found: ${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py"
  exit 1
fi

echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR}"
echo "[INFO] REF_NPZ=${REF_NPZ}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] COLLECT_MODE=${COLLECT_MODE}"

python3 - <<PY
from pathlib import Path
import re
import csv

EXP_ROOT = Path("${EXP_ROOT}")
OUT_ROOT = Path("${OUT_ROOT}")
RESULT_DIR = Path("${RESULT_DIR}")
LOG_DIR = Path("${LOG_DIR}")
MANIFEST = Path("${MANIFEST}")
COLLECT_MODE = "${COLLECT_MODE}"

LOG_BEST_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/sample_part_logs")

npz_re = re.compile(r"^(sit-.+?-400k)_0400000_cfg([0-9]+(?:\\.[0-9]+)?)-0\\.0-1\\.0\\.npz$")

def safe_name(model, cfg):
    return f"{model}_cfg{cfg}".replace("/", "_")

def parse_best_npz(model):
    p = LOG_BEST_DIR / f"cfg_scan_best_{model}_0400000.txt"
    if not p.exists():
        return None
    d = {}
    for line in p.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    best_npz = d.get("best_npz", "")
    if best_npz and Path(best_npz).exists():
        return Path(best_npz)
    return None

rows = []
seen = set()

for d in sorted(EXP_ROOT.iterdir()):
    if not d.is_dir():
        continue

    model = d.name
    if not (model.startswith("sit-") and model.endswith("-400k")):
        continue

    candidates = []

    if COLLECT_MODE == "all":
        candidates = sorted(d.glob(f"{model}_0400000_cfg*-0.0-1.0.npz"))

    elif COLLECT_MODE == "cfg1_best":
        cfg1 = d / f"{model}_0400000_cfg1.0-0.0-1.0.npz"
        if cfg1.exists():
            candidates.append(cfg1)
        best_npz = parse_best_npz(model)
        if best_npz is not None:
            candidates.append(best_npz)

    else:
        raise ValueError(f"Unknown COLLECT_MODE: {COLLECT_MODE}")

    for p in candidates:
        m = npz_re.match(p.name)
        if not m:
            continue

        name_model = m.group(1)
        cfg = float(m.group(2))

        # Ensure folder name and filename model prefix match exactly.
        if name_model != model:
            continue

        if cfg < 1.0:
            continue

        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)

        cfg_str = m.group(2)
        out_name = safe_name(model, cfg_str)
        result_json = RESULT_DIR / f"{out_name}.json"
        log_file = LOG_DIR / f"{out_name}.out"

        rows.append({
            "model": model,
            "cfg": cfg_str,
            "sample_npz": str(p),
            "result_json": str(result_json),
            "log_file": str(log_file),
        })

rows.sort(key=lambda r: (r["model"], float(r["cfg"])))

with MANIFEST.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "cfg", "sample_npz", "result_json", "log_file"],
        delimiter="\\t",
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 100)
print(f"Collected npz files: {len(rows)}")
print(f"Manifest saved to: {MANIFEST}")
print("=" * 100)
for r in rows:
    print(f"{r['model']:45s} cfg={r['cfg']:>6s}  {r['sample_npz']}")
PY

if [[ "${1:-list}" == "list" ]]; then
  echo
  echo "[DONE] List-only mode."
  echo "Check manifest:"
  echo "  column -t -s \$'\\t' ${MANIFEST} | less -S"
  exit 0
fi

echo
echo "[RUN] Start evaluating guided-diffusion metrics from manifest..."
echo

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r MODEL CFG SAMPLE_NPZ OUT_JSON OUT_LOG; do
  echo "================================================================================"
  echo "[MODEL] ${MODEL}"
  echo "[CFG]   ${CFG}"
  echo "[NPZ]   ${SAMPLE_NPZ}"
  echo "[JSON]  ${OUT_JSON}"
  echo "[LOG]   ${OUT_LOG}"
  echo "================================================================================"

  if [[ "${FORCE}" != "1" && -f "${OUT_JSON}" ]]; then
    if grep -q '"inception_score"' "${OUT_JSON}" && grep -q '"precision"' "${OUT_JSON}" && grep -q '"recall"' "${OUT_JSON}"; then
      echo "[SKIP] result exists: ${OUT_JSON}"
      continue
    fi
  fi

  if [[ ! -f "${SAMPLE_NPZ}" ]]; then
    echo "[ERROR] sample npz missing: ${SAMPLE_NPZ}" | tee "${OUT_LOG}"
    continue
  fi

  set +e
  python "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py" \
    "${REF_NPZ}" \
    "${SAMPLE_NPZ}" \
    > "${OUT_LOG}" 2>&1
  STATUS=$?
  set -e

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[ERROR] evaluator failed for ${MODEL} cfg=${CFG}, status=${STATUS}"
    tail -n 40 "${OUT_LOG}" || true
    continue
  fi

  python3 - "${OUT_LOG}" "${OUT_JSON}" "${MODEL}" "${CFG}" "${SAMPLE_NPZ}" "${REF_NPZ}" <<'PY'
from pathlib import Path
import sys
import re
import json
from datetime import datetime

log_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
model = sys.argv[3]
cfg = sys.argv[4]
sample_npz = sys.argv[5]
ref_npz = sys.argv[6]

text = log_path.read_text(errors="ignore")

patterns = {
    "inception_score": r"Inception Score:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    "fid": r"FID:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    "sfid": r"sFID:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    "precision": r"Precision:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    "recall": r"Recall:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
}

data = {
    "model": model,
    "cfg": cfg,
    "sample_npz": sample_npz,
    "ref_npz": ref_npz,
    "log_file": str(log_path),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "evaluator": "openai/guided-diffusion/evaluations/evaluator.py",
}

missing = []
for k, pat in patterns.items():
    m = re.search(pat, text)
    if m:
        data[k] = float(m.group(1))
    else:
        data[k] = None
        missing.append(k)

data["parse_ok"] = len(missing) == 0
data["missing_metrics"] = missing

out_json.write_text(json.dumps(data, indent=2, sort_keys=True))
print(json.dumps(data, indent=2, sort_keys=True))

if missing:
    raise SystemExit(f"Missing metrics in log: {missing}")
PY

done

echo
echo "[DONE] All guided-diffusion direct evaluations finished."
echo "Results: ${RESULT_DIR}"
echo "Logs:    ${LOG_DIR}"
echo "Manifest:${MANIFEST}"
