#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/xutongda/REPA-E/exps"
CONFIG_DIR="/video_ssd/kongzishang/xutongda/git/IFID/configs"
TRAIN_STEPS=400000
CKPT="0400000.pt"

# Usage:
#   PART_ID=1 bash sample_part4.sh
#   PART_ID=2 bash sample_part4.sh
#   PART_ID=3 bash sample_part4.sh
#   PART_ID=4 bash sample_part4.sh
#
# Or:
#   bash sample_part4.sh 1
#   bash sample_part4.sh 2
#   bash sample_part4.sh 3
#   bash sample_part4.sh 4

PART_ID="${1:-${PART_ID:-1}}"
SHARD_TOTAL=4

if ! [[ "${PART_ID}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] PART_ID must be one of: 1, 2, 3, 4"
    echo "Example: PART_ID=1 bash sample_part4.sh"
    exit 1
fi

PART_NAME="part${PART_ID}"
SHARD_INDEX=$((PART_ID - 1))

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-$((29504 + PART_ID))}"

LOG_DIR="${LOG_DIR:-sample_part_logs}"
mkdir -p "$LOG_DIR"

echo "[Info] SCRIPT=${PART_NAME}"
echo "[Info] SHARD_INDEX=${SHARD_INDEX}"
echo "[Info] SHARD_TOTAL=${SHARD_TOTAL}"
echo "[Info] GPUS=${GPUS}"
echo "[Info] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Info] MASTER_PORT=${MASTER_PORT}"
echo "[Info] LOG_DIR=${LOG_DIR}"

check_log_status() {
    local log_file="$1"

    if [[ ! -f "${log_file}" ]]; then
        echo "NOT_STARTED"
        return 0
    fi

    python3 - "${log_file}" <<'PY'
import re
import sys

log_file = sys.argv[1]

bad_markers = [
    "Traceback",
    "DistBackendError",
    "ChildFailedError",
    "Watchdog caught collective operation timeout",
    "Some NCCL operations have failed or timed out",
    "NCCL timeout",
    "CUDA out of memory",
    "RuntimeError",
    "Exception",
    "terminate called after throwing",
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError",
]

with open(log_file, "r", errors="ignore") as f:
    lines = f.read().splitlines()

nonempty = [line.strip() for line in lines if line.strip()]

if not nonempty:
    print("EMPTY")
    raise SystemExit(0)

last = nonempty[-1]

# Only this means full success.
# Examples:
# fid= 14.102678724892996
# fid=14.102678724892996
# fid= 1.23e+01
if re.fullmatch(
    r"fid=\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
    last,
):
    print("SUCCESS")
    raise SystemExit(0)

text = "\n".join(lines)

if any(marker in text for marker in bad_markers):
    print("FAILED")
else:
    print("INCOMPLETE")
PY
}

mapfile -d '' ALL_EXPS < <(
  python3 - <<'PY'
from pathlib import Path
import sys

ROOT = Path("/share/xutongda/REPA-E/exps")

keys = [
    "raet2i_siglip2",
    "raet2i_webssl",
    "pae-dinov2",
    "svgt2ip",
    "vavae64",
    "flux2",
    "flux-shift",
    "qw-shift",
    "rae-shift",
    "sdvae",
    "sd3",
    "dcae",
    "softvq",
    "vavae",
    # "eqvae",
    "maetok",
    "invae",
    "repae",
    "detok",
    "dmvae",
    "uae",
    "vtps",
    "vtpb",
    "vtpl",
    "svg",
]

for exp in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    name = exp.name.lower()

    if not (exp / "args.json").exists():
        continue

    if not (exp / "checkpoints" / "0400000.pt").exists():
        continue

    if any(k in name for k in keys):
        sys.stdout.buffer.write(str(exp).encode() + b"\0")
PY
)

echo "Found ${#ALL_EXPS[@]} total whitelisted experiments"

EXPS=()
for i in "${!ALL_EXPS[@]}"; do
    if (( i % SHARD_TOTAL == SHARD_INDEX )); then
        EXPS+=("${ALL_EXPS[$i]}")
    fi
done

echo "${PART_NAME} will run ${#EXPS[@]} experiments"
for EXP in "${EXPS[@]}"; do
    printf '  [%s] %s\n' "${PART_NAME}" "$EXP"
done

if (( ${#EXPS[@]} == 0 )); then
    echo "[ERROR] No experiments assigned to ${PART_NAME}."
    exit 1
fi

python3 - "${EXPS[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

CONFIG_DIR = Path("/video_ssd/kongzishang/xutongda/git/IFID/configs")

rules = [
    ("raet2i_siglip2", "ScaleRAE_SIGLIP2.yaml"),
    ("raet2i_webssl", "ScaleRAE_WEBSSL.yaml"),
    ("pae-dinov2", "PAE_DINOv2L_d32.yaml"),
    ("svgt2ip", "SVGT2I_P_STAGE1_256.yaml"),
    ("vavae64", "VAVAE64.yaml"),
    ("flux2", "FLUX2VAE.yaml"),
    ("flux-shift", "FLUXVAE.yaml"),
    ("qw-shift", "QWVAE.yaml"),
    ("rae-shift", "RAE.yaml"),
    ("sdvae", "SDVAE.yaml"),
    ("sd3", "SD3VAE.yaml"),
    ("dcae", "DCAE.yaml"),
    ("softvq", "SOFTVQ.yaml"),
    ("vavae", "VAVAE.yaml"),
    # ("eqvae", "EQVAE.yaml"),
    ("maetok", "MAETOK.yaml"),
    ("invae", "INVAE.yaml"),
    ("repae", "REPAEVAE.yaml"),
    ("detok", "DETOK.yaml"),
    ("dmvae", "DMVAE.yaml"),
    ("uae", "UAE.yaml"),
    ("vtps", "VTPS.yaml"),
    ("vtpb", "VTPB.yaml"),
    ("vtpl", "VTPL.yaml"),
    ("svg", "SVG.yaml"),
]

for exp_arg in sys.argv[1:]:
    exp = Path(exp_arg)
    name = exp.name.lower()
    yaml_path = None

    for key, yaml_name in rules:
        if key in name:
            yaml_path = CONFIG_DIR / yaml_name
            break

    if yaml_path is None:
        print(f"[SKIP not in whitelist] {exp.name}")
        continue

    if not yaml_path.exists():
        print(f"[SKIP missing yaml] {exp.name} -> {yaml_path}")
        continue

    args_path = exp / "args.json"
    if not args_path.exists():
        print(f"[SKIP no args.json] {exp.name}")
        continue

    data = json.loads(args_path.read_text())
    data["vae_config"] = str(yaml_path)

    bak = exp / "args.json.bak"
    if not bak.exists():
        shutil.copy2(args_path, bak)

    args_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[UPDATED] {exp.name} -> {yaml_path}")
PY

for EXP in "${EXPS[@]}"; do
    CFG=1.0
    GUIDANCE_LOW=0.0
    GUIDANCE_HIGH=1.0

    EXP_NAME="$(basename "$EXP")"
    LOG_FILE="${LOG_DIR}/${PART_NAME}_${EXP_NAME}.log"

    TRAIN_STEP_STR="$(printf "%07d" "${TRAIN_STEPS}")"

    SAMPLE_FOLDER="${EXP}/${EXP_NAME}_${TRAIN_STEP_STR}_cfg${CFG}-${GUIDANCE_LOW}-${GUIDANCE_HIGH}"
    NPZ_FILE="${SAMPLE_FOLDER}.npz"

    echo
    echo "============================================================"
    echo "[${PART_NAME}] Checking EXP=${EXP}"
    echo "[${PART_NAME}] LOG=${LOG_FILE}"
    echo "[${PART_NAME}] SAMPLE_FOLDER=${SAMPLE_FOLDER}"
    echo "[${PART_NAME}] NPZ_FILE=${NPZ_FILE}"
    echo "============================================================"

    # Because you previously used part1/part2 logs, check all part*_EXP logs.
    # If any old part log says SUCCESS, skip this experiment.
    SUCCESS_LOG=""
    ANY_LOG_FOUND=0

    for CANDIDATE_LOG in "${LOG_FILE}" "${LOG_DIR}"/part*_"${EXP_NAME}".log; do
        if [[ ! -f "${CANDIDATE_LOG}" ]]; then
            continue
        fi

        ANY_LOG_FOUND=1
        CANDIDATE_STATUS="$(check_log_status "${CANDIDATE_LOG}")"
        echo "[${PART_NAME}] CANDIDATE_LOG=${CANDIDATE_LOG}, STATUS=${CANDIDATE_STATUS}"

        if [[ "${CANDIDATE_STATUS}" == "SUCCESS" ]]; then
            SUCCESS_LOG="${CANDIDATE_LOG}"
            break
        fi
    done

    if [[ -n "${SUCCESS_LOG}" ]]; then
        echo "[${PART_NAME}] SKIP successful run according to log: ${SUCCESS_LOG}"
        continue
    fi

    if (( ANY_LOG_FOUND == 0 )); then
        echo "[${PART_NAME}] No previous log found. Running from scratch."
    else
        echo "[${PART_NAME}] Previous logs exist but none is successful. Cleaning residual files."

        if [[ -d "${SAMPLE_FOLDER}" ]]; then
            echo "[${PART_NAME}] Removing residual sample folder: ${SAMPLE_FOLDER}"
            rm -rf "${SAMPLE_FOLDER}"
        fi

        if [[ -f "${NPZ_FILE}" ]]; then
            echo "[${PART_NAME}] Removing residual npz: ${NPZ_FILE}"
            rm -f "${NPZ_FILE}"
        fi
    fi

    echo
    echo "============================================================"
    echo "[${PART_NAME}] Running EXP=${EXP}"
    echo "[${PART_NAME}] CFG=${CFG}, STEPS=${TRAIN_STEPS}, CKPT=${CKPT}"
    echo "[${PART_NAME}] GPUS=${GPUS}, NPROC_PER_NODE=${NPROC_PER_NODE}, PORT=${MASTER_PORT}"
    echo "[${PART_NAME}] LOG=${LOG_FILE}"
    echo "============================================================"

    set +e

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port "${MASTER_PORT}" \
        generate.py \
        --num-fid-samples 50000 \
        --pproc-batch-size 16 \
        --path-type linear \
        --mode sde \
        --num-steps 250 \
        --cfg-scale "${CFG}" \
        --guidance-high "${GUIDANCE_HIGH}" \
        --guidance-low "${GUIDANCE_LOW}" \
        --exp-path "${EXP}" \
        --sample-dir "${EXP}" \
        --train-steps "${TRAIN_STEPS}" \
        --fid-reference-file "/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz" \
        2>&1 | tee "${LOG_FILE}"

    RUN_STATUS=${PIPESTATUS[0]}

    set -e

    if [[ "${RUN_STATUS}" -ne 0 ]]; then
        echo "[${PART_NAME}] FAILED EXP=${EXP}, exit_code=${RUN_STATUS}"
        echo "[${PART_NAME}] Continue to next experiment."
        continue
    fi

    FINAL_STATUS="$(check_log_status "${LOG_FILE}")"
    echo "[${PART_NAME}] FINAL_LOG_STATUS=${FINAL_STATUS}"

    if [[ "${FINAL_STATUS}" != "SUCCESS" ]]; then
        echo "[${PART_NAME}] WARNING: torchrun exited 0 but log is not successful: ${FINAL_STATUS}"
        echo "[${PART_NAME}] Continue to next experiment."
        continue
    fi

    echo "[${PART_NAME}] Finished EXP=${EXP}"
done

echo "[DONE] ${PART_NAME} finished all assigned experiments."