#!/usr/bin/env bash
set -uo pipefail

# Dynamic 8-GPU runner for OpenAI guided-diffusion evaluator.
# Each GPU runs one evaluator process at a time.
# When one finishes, it immediately pulls the next npz from the shared queue.

GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-./third_party/guided-diffusion}"
REF_NPZ="${REF_NPZ:-/video_ssd/kongzishang/xutongda/git/IFID/VIRTUAL_imagenet256_labeled.npz}"

OUT_ROOT="${OUT_ROOT:-./guided_eval_direct}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/guided_npz_manifest.tsv}"
RESULT_DIR="${RESULT_DIR:-${OUT_ROOT}/results}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
WORK_DIR="${WORK_DIR:-${OUT_ROOT}/work_8gpu}"

GPU_LIST_STR="${GPU_LIST:-0,1,2,3,4,5,6,7}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${WORK_DIR}"

TASK_FILE="${WORK_DIR}/tasks.tsv"
IDX_FILE="${WORK_DIR}/next_idx.txt"
LOCK_FILE="${WORK_DIR}/queue.lock"
WORKER_LOG_DIR="${WORK_DIR}/worker_logs"
mkdir -p "${WORKER_LOG_DIR}"

if [[ ! -f "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py" ]]; then
  echo "[ERROR] evaluator.py not found: ${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py"
  exit 1
fi

if [[ ! -f "${REF_NPZ}" ]]; then
  echo "[ERROR] REF_NPZ not found: ${REF_NPZ}"
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST}"
  echo "Run first:"
  echo "  COLLECT_MODE=cfg1_best bash batch_eval_guided_npz_direct.sh list"
  exit 1
fi

# Remove header from manifest.
tail -n +2 "${MANIFEST}" > "${TASK_FILE}"
echo 0 > "${IDX_FILE}"

TOTAL_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')
echo "================================================================================"
echo "[START] guided-diffusion 8-GPU dynamic evaluator"
echo "[GUIDED_DIFFUSION_DIR] ${GUIDED_DIFFUSION_DIR}"
echo "[REF_NPZ] ${REF_NPZ}"
echo "[MANIFEST] ${MANIFEST}"
echo "[TASKS] ${TOTAL_TASKS}"
echo "[GPU_LIST] ${GPU_LIST_STR}"
echo "[FORCE] ${FORCE}"
echo "================================================================================"

parse_to_json() {
  local OUT_LOG="$1"
  local OUT_JSON="$2"
  local MODEL="$3"
  local CFG="$4"
  local SAMPLE_NPZ="$5"

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

text = log_path.read_text(errors="ignore") if log_path.exists() else ""

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

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(data, indent=2, sort_keys=True))

if missing:
    print(f"[PARSE_ERROR] missing={missing} log={log_path}")
else:
    print(f"[PARSE_OK] {model} cfg={cfg} IS={data['inception_score']} FID={data['fid']} sFID={data['sfid']} P={data['precision']} R={data['recall']}")
PY
}

run_one_task() {
  local GPU="$1"
  local LINE="$2"

  IFS=$'\t' read -r MODEL CFG SAMPLE_NPZ OUT_JSON OUT_LOG <<< "${LINE}"

  echo "[GPU ${GPU}] START model=${MODEL} cfg=${CFG}"
  echo "[GPU ${GPU}] NPZ=${SAMPLE_NPZ}"
  echo "[GPU ${GPU}] JSON=${OUT_JSON}"
  echo "[GPU ${GPU}] LOG=${OUT_LOG}"

  if [[ "${FORCE}" != "1" && -f "${OUT_JSON}" ]]; then
    if grep -q '"parse_ok": true' "${OUT_JSON}" 2>/dev/null; then
      echo "[GPU ${GPU}] SKIP existing parse_ok result: ${OUT_JSON}"
      return 0
    fi
  fi

  if [[ ! -f "${SAMPLE_NPZ}" ]]; then
    echo "[GPU ${GPU}] ERROR sample npz missing: ${SAMPLE_NPZ}" | tee "${OUT_LOG}"
    return 0
  fi

  # Important:
  # CUDA_VISIBLE_DEVICES makes this process see only one GPU as cuda:0.
  # TF_FORCE_GPU_ALLOW_GROWTH prevents TensorFlow from pre-allocating all visible GPU memory.
  CUDA_VISIBLE_DEVICES="${GPU}" \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  "${PYTHON_BIN}" "${GUIDED_DIFFUSION_DIR}/evaluations/evaluator.py" \
    "${REF_NPZ}" \
    "${SAMPLE_NPZ}" \
    > "${OUT_LOG}" 2>&1

  local STATUS=$?

  if [[ "${STATUS}" -ne 0 ]]; then
    echo "[GPU ${GPU}] ERROR evaluator failed, status=${STATUS}, model=${MODEL}, cfg=${CFG}"
    tail -n 40 "${OUT_LOG}" || true
    return 0
  fi

  parse_to_json "${OUT_LOG}" "${OUT_JSON}" "${MODEL}" "${CFG}" "${SAMPLE_NPZ}"

  echo "[GPU ${GPU}] DONE model=${MODEL} cfg=${CFG}"
}

worker_loop() {
  local GPU="$1"
  local WLOG="${WORKER_LOG_DIR}/gpu${GPU}.out"

  {
    echo "[GPU ${GPU}] worker started"

    while true; do
      local IDX
      local LINE

      {
        flock 200
        IDX=$(cat "${IDX_FILE}")
        IDX=$((IDX + 1))
        echo "${IDX}" > "${IDX_FILE}"
      } 200>"${LOCK_FILE}"

      LINE=$(sed -n "${IDX}p" "${TASK_FILE}" || true)

      if [[ -z "${LINE}" ]]; then
        echo "[GPU ${GPU}] no more tasks, exit"
        break
      fi

      echo "[GPU ${GPU}] picked task #${IDX}/${TOTAL_TASKS}"
      run_one_task "${GPU}" "${LINE}"
      echo
    done
  } > "${WLOG}" 2>&1
}

IFS=',' read -r -a GPU_LIST_ARR <<< "${GPU_LIST_STR}"

PIDS=()
for GPU in "${GPU_LIST_ARR[@]}"; do
  worker_loop "${GPU}" &
  PIDS+=("$!")
  echo "[LAUNCH] worker GPU=${GPU}, pid=${PIDS[-1]}"
done

echo
echo "[INFO] Worker logs:"
for GPU in "${GPU_LIST_ARR[@]}"; do
  echo "  tail -f ${WORKER_LOG_DIR}/gpu${GPU}.out"
done
echo

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || FAIL=1
done

echo "================================================================================"
echo "[DONE] all workers finished"
echo "[RESULT_DIR] ${RESULT_DIR}"
echo "[LOG_DIR] ${LOG_DIR}"
echo "[WORKER_LOG_DIR] ${WORKER_LOG_DIR}"
echo "================================================================================"

python3 - <<PY
from pathlib import Path
import json
import csv

result_dir = Path("${RESULT_DIR}")
out_csv = Path("${OUT_ROOT}") / "guided_metrics_summary.csv"
rows = []

for p in sorted(result_dir.glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    rows.append({
        "model": d.get("model", ""),
        "cfg": d.get("cfg", ""),
        "inception_score": d.get("inception_score", ""),
        "fid": d.get("fid", ""),
        "sfid": d.get("sfid", ""),
        "precision": d.get("precision", ""),
        "recall": d.get("recall", ""),
        "parse_ok": d.get("parse_ok", ""),
        "sample_npz": d.get("sample_npz", ""),
        "json": str(p),
        "log_file": d.get("log_file", ""),
    })

def key(r):
    try:
        cfg = float(r["cfg"])
    except Exception:
        cfg = 999
    return (r["model"], cfg)

rows.sort(key=key)

fields = [
    "model", "cfg", "inception_score", "fid", "sfid",
    "precision", "recall", "parse_ok", "sample_npz", "json", "log_file"
]

with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"[SUMMARY] saved: {out_csv}")
print(f"[SUMMARY] json files: {len(rows)}")
PY

exit "${FAIL}"
