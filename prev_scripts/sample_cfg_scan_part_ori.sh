#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/xutongda/REPA-E/exps"
CONFIG_DIR="/video_ssd/kongzishang/xutongda/git/IFID/configs"
TRAIN_STEPS=400000
CKPT="0400000.pt"

# 4-part by default.
# Usage:
#   PART_ID=1 SHARD_TOTAL=4 bash sample_cfg_scan_part.sh
#   PART_ID=2 SHARD_TOTAL=4 bash sample_cfg_scan_part.sh
#   PART_ID=3 SHARD_TOTAL=4 bash sample_cfg_scan_part.sh
#   PART_ID=4 SHARD_TOTAL=4 bash sample_cfg_scan_part.sh

PART_ID="${1:-${PART_ID:-1}}"
SHARD_TOTAL="${SHARD_TOTAL:-4}"

if ! [[ "${PART_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PART_ID must be an integer, got: ${PART_ID}"
    exit 1
fi

if ! [[ "${SHARD_TOTAL}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SHARD_TOTAL must be an integer, got: ${SHARD_TOTAL}"
    exit 1
fi

if (( PART_ID < 1 || PART_ID > SHARD_TOTAL )); then
    echo "[ERROR] PART_ID must be in [1, ${SHARD_TOTAL}], got: ${PART_ID}"
    exit 1
fi

PART_NAME="part${PART_ID}"
SHARD_INDEX=$((PART_ID - 1))

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-$((29504 + PART_ID))}"

LOG_DIR="${LOG_DIR:-sample_part_logs}"
mkdir -p "$LOG_DIR"

# CFG scan settings.
CFG_START="${CFG_START:-1.25}"
CFG_STEP="${CFG_STEP:-0.25}"
MAX_CFG="${MAX_CFG:-8.0}"

GUIDANCE_LOW="${GUIDANCE_LOW:-0.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-1.0}"

NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
PPROC_BATCH_SIZE="${PPROC_BATCH_SIZE:-16}"
NUM_STEPS="${NUM_STEPS:-250}"
MODE="${MODE:-sde}"
PATH_TYPE="${PATH_TYPE:-linear}"

FID_REFERENCE_FILE="${FID_REFERENCE_FILE:-/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz}"

RESULT_FILE="${LOG_DIR}/cfg_scan_results_${PART_NAME}.tsv"

if [[ ! -f "${RESULT_FILE}" ]]; then
    printf "exp\tbest_cfg\tbest_fid\tbest_npz\tstop_reason\tstop_cfg\tstop_fid\n" > "${RESULT_FILE}"
fi

echo "[Info] SCRIPT=${PART_NAME}"
echo "[Info] SHARD_INDEX=${SHARD_INDEX}"
echo "[Info] SHARD_TOTAL=${SHARD_TOTAL}"
echo "[Info] GPUS=${GPUS}"
echo "[Info] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Info] MASTER_PORT=${MASTER_PORT}"
echo "[Info] LOG_DIR=${LOG_DIR}"
echo "[Info] CFG_START=${CFG_START}"
echo "[Info] CFG_STEP=${CFG_STEP}"
echo "[Info] MAX_CFG=${MAX_CFG}"
echo "[Info] RESULT_FILE=${RESULT_FILE}"

cfg_at_index() {
    local idx="$1"
    python3 - "${CFG_START}" "${CFG_STEP}" "${idx}" <<'PY'
from decimal import Decimal
import sys

start = Decimal(sys.argv[1])
step = Decimal(sys.argv[2])
idx = int(sys.argv[3])

val = start + step * idx

# Match argparse float -> str(float) style used by generate.py:
# 1.50 -> 1.5, 2.00 -> 2.0
print(str(float(val)))
PY
}

cfg_leq_max() {
    local cfg="$1"
    python3 - "${cfg}" "${MAX_CFG}" <<'PY'
import sys

cfg = float(sys.argv[1])
max_cfg = float(sys.argv[2])

raise SystemExit(0 if cfg <= max_cfg + 1e-12 else 1)
PY
}

fid_lt() {
    local a="$1"
    local b="$2"
    python3 - "${a}" "${b}" <<'PY'
import sys

a = float(sys.argv[1])
b = float(sys.argv[2])

# Lower FID is better.
raise SystemExit(0 if a < b else 1)
PY
}

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

extract_fid_from_log() {
    local log_file="$1"

    python3 - "${log_file}" <<'PY'
import re
import sys

log_file = sys.argv[1]

with open(log_file, "r", errors="ignore") as f:
    lines = f.read().splitlines()

nonempty = [line.strip() for line in lines if line.strip()]
if not nonempty:
    raise SystemExit(1)

last = nonempty[-1]
m = re.fullmatch(
    r"fid=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    last,
)

if not m:
    raise SystemExit(1)

print(m.group(1))
PY
}

write_best_marker() {
    local marker_file="$1"
    local exp="$2"
    local best_cfg="$3"
    local best_fid="$4"
    local best_npz="$5"
    local stop_reason="$6"
    local stop_cfg="$7"
    local stop_fid="$8"

    cat > "${marker_file}" <<EOF
status=SUCCESS
exp=${exp}
best_cfg=${best_cfg}
best_fid=${best_fid}
best_npz=${best_npz}
stop_reason=${stop_reason}
stop_cfg=${stop_cfg}
stop_fid=${stop_fid}
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
EOF

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${exp}" \
        "${best_cfg}" \
        "${best_fid}" \
        "${best_npz}" \
        "${stop_reason}" \
        "${stop_cfg}" \
        "${stop_fid}" \
        >> "${RESULT_FILE}"
}

run_one_cfg() {
    local exp="$1"
    local exp_name="$2"
    local cfg="$3"

    local train_step_str
    train_step_str="$(printf "%07d" "${TRAIN_STEPS}")"

    local sample_folder
    local npz_file
    local log_file

    sample_folder="${exp}/${exp_name}_${train_step_str}_cfg${cfg}-${GUIDANCE_LOW}-${GUIDANCE_HIGH}"
    npz_file="${sample_folder}.npz"
    log_file="${LOG_DIR}/${PART_NAME}_${exp_name}_cfg${cfg}.log"

    RUN_CFG_FID=""
    RUN_CFG_NPZ="${npz_file}"
    RUN_CFG_LOG="${log_file}"

    echo
    echo "------------------------------------------------------------"
    echo "[${PART_NAME}] CFG check"
    echo "[${PART_NAME}] EXP=${exp}"
    echo "[${PART_NAME}] CFG=${cfg}"
    echo "[${PART_NAME}] LOG=${log_file}"
    echo "[${PART_NAME}] SAMPLE_FOLDER=${sample_folder}"
    echo "[${PART_NAME}] NPZ=${npz_file}"
    echo "------------------------------------------------------------"

    # Reuse a successful previous log only if the corresponding npz still exists.
    # This lets interrupted jobs resume when possible.
    local candidate_log
    shopt -s nullglob
    for candidate_log in "${log_file}" "${LOG_DIR}"/part*_"${exp_name}"_cfg"${cfg}".log; do
        if [[ ! -f "${candidate_log}" ]]; then
            continue
        fi

        local candidate_status
        candidate_status="$(check_log_status "${candidate_log}")"
        echo "[${PART_NAME}] CANDIDATE_LOG=${candidate_log}, STATUS=${candidate_status}"

        if [[ "${candidate_status}" == "SUCCESS" && -s "${npz_file}" ]]; then
            RUN_CFG_FID="$(extract_fid_from_log "${candidate_log}")"
            RUN_CFG_LOG="${candidate_log}"
            echo "[${PART_NAME}] Reuse successful CFG=${cfg}, FID=${RUN_CFG_FID}, NPZ=${npz_file}"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob

    # Existing log is not sufficient; clean current cfg residual files and rerun.
    if [[ -d "${sample_folder}" ]]; then
        echo "[${PART_NAME}] Removing residual sample folder for CFG=${cfg}: ${sample_folder}"
        rm -rf "${sample_folder}"
    fi

    if [[ -f "${npz_file}" ]]; then
        echo "[${PART_NAME}] Removing residual npz for CFG=${cfg}: ${npz_file}"
        rm -f "${npz_file}"
    fi

    echo
    echo "============================================================"
    echo "[${PART_NAME}] Running EXP=${exp}"
    echo "[${PART_NAME}] CFG=${cfg}, STEPS=${TRAIN_STEPS}, CKPT=${CKPT}"
    echo "[${PART_NAME}] GPUS=${GPUS}, NPROC_PER_NODE=${NPROC_PER_NODE}, PORT=${MASTER_PORT}"
    echo "[${PART_NAME}] LOG=${log_file}"
    echo "============================================================"

    set +e

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port "${MASTER_PORT}" \
        generate.py \
        --num-fid-samples "${NUM_FID_SAMPLES}" \
        --pproc-batch-size "${PPROC_BATCH_SIZE}" \
        --path-type "${PATH_TYPE}" \
        --mode "${MODE}" \
        --num-steps "${NUM_STEPS}" \
        --cfg-scale "${cfg}" \
        --guidance-high "${GUIDANCE_HIGH}" \
        --guidance-low "${GUIDANCE_LOW}" \
        --exp-path "${exp}" \
        --sample-dir "${exp}" \
        --train-steps "${TRAIN_STEPS}" \
        --fid-reference-file "${FID_REFERENCE_FILE}" \
        2>&1 | tee "${log_file}"

    local run_status
    run_status=${PIPESTATUS[0]}

    set -e

    if [[ "${run_status}" -ne 0 ]]; then
        echo "[${PART_NAME}] FAILED EXP=${exp}, CFG=${cfg}, exit_code=${run_status}"
        echo "[${PART_NAME}] Continue to next experiment."

        if [[ -d "${sample_folder}" ]]; then
            echo "[${PART_NAME}] Removing failed sample folder: ${sample_folder}"
            rm -rf "${sample_folder}"
        fi

        if [[ -f "${npz_file}" ]]; then
            echo "[${PART_NAME}] Removing failed npz: ${npz_file}"
            rm -f "${npz_file}"
        fi

        return 1
    fi

    local final_status
    final_status="$(check_log_status "${log_file}")"
    echo "[${PART_NAME}] FINAL_LOG_STATUS=${final_status}"

    if [[ "${final_status}" != "SUCCESS" ]]; then
        echo "[${PART_NAME}] WARNING: torchrun exited 0 but log is not successful: ${final_status}"
        echo "[${PART_NAME}] Continue to next experiment."

        if [[ -d "${sample_folder}" ]]; then
            echo "[${PART_NAME}] Removing incomplete sample folder: ${sample_folder}"
            rm -rf "${sample_folder}"
        fi

        if [[ -f "${npz_file}" ]]; then
            echo "[${PART_NAME}] Removing incomplete npz: ${npz_file}"
            rm -f "${npz_file}"
        fi

        return 1
    fi

    if [[ ! -s "${npz_file}" ]]; then
        echo "[${PART_NAME}] ERROR: log says SUCCESS but npz missing or empty: ${npz_file}"
        return 1
    fi

    RUN_CFG_FID="$(extract_fid_from_log "${log_file}")"
    RUN_CFG_NPZ="${npz_file}"
    RUN_CFG_LOG="${log_file}"

    echo "[${PART_NAME}] SUCCESS EXP=${exp}, CFG=${cfg}, FID=${RUN_CFG_FID}, NPZ=${RUN_CFG_NPZ}"
    return 0
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
    EXP_NAME="$(basename "$EXP")"
    TRAIN_STEP_STR="$(printf "%07d" "${TRAIN_STEPS}")"
    BEST_MARKER="${LOG_DIR}/cfg_scan_best_${EXP_NAME}_${TRAIN_STEP_STR}.txt"

    echo
    echo "############################################################"
    echo "[${PART_NAME}] Start CFG scan for EXP=${EXP}"
    echo "[${PART_NAME}] BEST_MARKER=${BEST_MARKER}"
    echo "############################################################"

    if [[ -f "${BEST_MARKER}" ]]; then
        EXISTING_BEST_NPZ="$(grep '^best_npz=' "${BEST_MARKER}" | head -n 1 | cut -d '=' -f 2- || true)"
        if [[ -n "${EXISTING_BEST_NPZ}" && -s "${EXISTING_BEST_NPZ}" ]]; then
            echo "[${PART_NAME}] SKIP already finished CFG scan:"
            cat "${BEST_MARKER}"
            continue
        else
            echo "[${PART_NAME}] Existing marker is invalid or npz missing. Removing marker and rerun."
            rm -f "${BEST_MARKER}"
        fi
    fi

    BEST_CFG=""
    BEST_FID=""
    BEST_NPZ=""

    PREV_CFG=""
    PREV_FID=""
    PREV_NPZ=""

    STOP_REASON=""
    STOP_CFG=""
    STOP_FID=""

    CFG_IDX=0

    while true; do
        CFG="$(cfg_at_index "${CFG_IDX}")"

        if ! cfg_leq_max "${CFG}"; then
            echo "[${PART_NAME}] Reached MAX_CFG=${MAX_CFG} before FID increased."
            STOP_REASON="MAX_CFG_REACHED"
            STOP_CFG="${CFG}"
            STOP_FID="NA"

            if [[ -n "${BEST_CFG}" && -n "${BEST_FID}" && -n "${BEST_NPZ}" && -s "${BEST_NPZ}" ]]; then
                write_best_marker \
                    "${BEST_MARKER}" \
                    "${EXP}" \
                    "${BEST_CFG}" \
                    "${BEST_FID}" \
                    "${BEST_NPZ}" \
                    "${STOP_REASON}" \
                    "${STOP_CFG}" \
                    "${STOP_FID}"
            fi

            break
        fi

        if ! run_one_cfg "${EXP}" "${EXP_NAME}" "${CFG}"; then
            echo "[${PART_NAME}] CFG=${CFG} failed."

            if [[ -n "${BEST_CFG}" && -n "${BEST_FID}" && -n "${BEST_NPZ}" && -s "${BEST_NPZ}" ]]; then
                echo "[${PART_NAME}] Keep best so far and move to next VAE."
                STOP_REASON="CFG_FAILED"
                STOP_CFG="${CFG}"
                STOP_FID="NA"

                write_best_marker \
                    "${BEST_MARKER}" \
                    "${EXP}" \
                    "${BEST_CFG}" \
                    "${BEST_FID}" \
                    "${BEST_NPZ}" \
                    "${STOP_REASON}" \
                    "${STOP_CFG}" \
                    "${STOP_FID}"
            else
                echo "[${PART_NAME}] No valid best CFG found for EXP=${EXP}. Move to next VAE."
            fi

            break
        fi

        CUR_CFG="${CFG}"
        CUR_FID="${RUN_CFG_FID}"
        CUR_NPZ="${RUN_CFG_NPZ}"

        if [[ -z "${PREV_CFG}" ]]; then
            echo "[${PART_NAME}] First CFG result: CFG=${CUR_CFG}, FID=${CUR_FID}"
            BEST_CFG="${CUR_CFG}"
            BEST_FID="${CUR_FID}"
            BEST_NPZ="${CUR_NPZ}"

            PREV_CFG="${CUR_CFG}"
            PREV_FID="${CUR_FID}"
            PREV_NPZ="${CUR_NPZ}"

            CFG_IDX=$((CFG_IDX + 1))
            continue
        fi

        if fid_lt "${CUR_FID}" "${PREV_FID}"; then
            echo "[${PART_NAME}] FID improved: ${CUR_FID} < ${PREV_FID}"
            echo "[${PART_NAME}] Keep current CFG=${CUR_CFG}, delete previous npz CFG=${PREV_CFG}: ${PREV_NPZ}"

            if [[ -f "${PREV_NPZ}" && "${PREV_NPZ}" != "${CUR_NPZ}" ]]; then
                rm -f "${PREV_NPZ}"
            fi

            BEST_CFG="${CUR_CFG}"
            BEST_FID="${CUR_FID}"
            BEST_NPZ="${CUR_NPZ}"

            PREV_CFG="${CUR_CFG}"
            PREV_FID="${CUR_FID}"
            PREV_NPZ="${CUR_NPZ}"

            CFG_IDX=$((CFG_IDX + 1))
            continue
        else
            echo "[${PART_NAME}] FID stopped improving: current ${CUR_FID} >= previous ${PREV_FID}"
            echo "[${PART_NAME}] Keep previous/best CFG=${PREV_CFG}, FID=${PREV_FID}"
            echo "[${PART_NAME}] Delete current worse npz CFG=${CUR_CFG}: ${CUR_NPZ}"

            if [[ -f "${CUR_NPZ}" && "${CUR_NPZ}" != "${PREV_NPZ}" ]]; then
                rm -f "${CUR_NPZ}"
            fi

            BEST_CFG="${PREV_CFG}"
            BEST_FID="${PREV_FID}"
            BEST_NPZ="${PREV_NPZ}"

            STOP_REASON="FID_INCREASED_OR_EQUAL"
            STOP_CFG="${CUR_CFG}"
            STOP_FID="${CUR_FID}"

            write_best_marker \
                "${BEST_MARKER}" \
                "${EXP}" \
                "${BEST_CFG}" \
                "${BEST_FID}" \
                "${BEST_NPZ}" \
                "${STOP_REASON}" \
                "${STOP_CFG}" \
                "${STOP_FID}"

            break
        fi
    done

    echo "[${PART_NAME}] Done CFG scan for EXP=${EXP}"
done

echo "[DONE] ${PART_NAME} finished all assigned CFG scans."