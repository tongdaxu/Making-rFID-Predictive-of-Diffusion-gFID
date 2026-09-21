#!/usr/bin/env bash
set -euo pipefail

TRAIN_STEPS=400000
CFG=1.0
GUIDANCE_LOW=0.0
GUIDANCE_HIGH=1.0

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29506}"

LOG_DIR="${LOG_DIR:-sample_part_logs}"
mkdir -p "${LOG_DIR}"

FID_REFERENCE_FILE="${FID_REFERENCE_FILE:-/video_ssd/kongzishang/xutongda/git/REPA-E/exps/VIRTUAL_imagenet256_labeled.npz}"

EXPS=(
    /share/xutongda/REPA-E/exps/sit-xl-vtpb-400k
    /share/xutongda/REPA-E/exps/sit-xl-vtpl-400k
)

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

echo "[Info] rerun without CFG"
echo "[Info] GPUS=${GPUS}"
echo "[Info] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[Info] MASTER_PORT=${MASTER_PORT}"
echo "[Info] LOG_DIR=${LOG_DIR}"
echo "[Info] FID_REFERENCE_FILE=${FID_REFERENCE_FILE}"

for EXP in "${EXPS[@]}"; do
    EXP_NAME="$(basename "${EXP}")"
    TRAIN_STEP_STR="$(printf "%07d" "${TRAIN_STEPS}")"

    SAMPLE_FOLDER="${EXP}/${EXP_NAME}_${TRAIN_STEP_STR}_cfg${CFG}-${GUIDANCE_LOW}-${GUIDANCE_HIGH}"
    NPZ_FILE="${SAMPLE_FOLDER}.npz"
    LOG_FILE="${LOG_DIR}/rerun_cfg${CFG}_${EXP_NAME}.log"

    echo
    echo "============================================================"
    echo "[Check] EXP=${EXP}"
    echo "[Check] LOG=${LOG_FILE}"
    echo "[Check] SAMPLE_FOLDER=${SAMPLE_FOLDER}"
    echo "[Check] NPZ_FILE=${NPZ_FILE}"
    echo "============================================================"

    STATUS="$(check_log_status "${LOG_FILE}")"
    echo "[Check] LOG_STATUS=${STATUS}"

    if [[ "${STATUS}" == "SUCCESS" && -s "${NPZ_FILE}" ]]; then
        echo "[SKIP] Already successful and npz exists: ${NPZ_FILE}"
        continue
    fi

    if [[ -s "${NPZ_FILE}" ]]; then
        echo "[SKIP] npz already exists even though log is not success: ${NPZ_FILE}"
        echo "[SKIP] If you want to force rerun, delete this npz and rerun."
        continue
    fi

    if [[ "${STATUS}" != "NOT_STARTED" ]]; then
        echo "[Clean] Previous log exists but not successful. Cleaning residual files."

        if [[ -d "${SAMPLE_FOLDER}" ]]; then
            echo "[Clean] Removing residual sample folder: ${SAMPLE_FOLDER}"
            rm -rf "${SAMPLE_FOLDER}"
        fi

        if [[ -f "${NPZ_FILE}" ]]; then
            echo "[Clean] Removing residual npz: ${NPZ_FILE}"
            rm -f "${NPZ_FILE}"
        fi
    fi

    echo
    echo "============================================================"
    echo "[Run] EXP=${EXP}"
    echo "[Run] CFG=${CFG}, STEPS=${TRAIN_STEPS}"
    echo "[Run] GPUS=${GPUS}, NPROC_PER_NODE=${NPROC_PER_NODE}, PORT=${MASTER_PORT}"
    echo "[Run] LOG=${LOG_FILE}"
    echo "============================================================"

    set +e

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    torchrun \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port "${MASTER_PORT}" \
        generate.py \
        --num-fid-samples 50000 \
        --pproc-batch-size 32 \
        --path-type linear \
        --mode sde \
        --num-steps 250 \
        --cfg-scale "${CFG}" \
        --guidance-high "${GUIDANCE_HIGH}" \
        --guidance-low "${GUIDANCE_LOW}" \
        --exp-path "${EXP}" \
        --sample-dir "${EXP}" \
        --train-steps "${TRAIN_STEPS}" \
        --fid-reference-file "${FID_REFERENCE_FILE}" \
        2>&1 | tee "${LOG_FILE}"

    RUN_STATUS=${PIPESTATUS[0]}

    set -e

    if [[ "${RUN_STATUS}" -ne 0 ]]; then
        echo "[FAILED] EXP=${EXP}, exit_code=${RUN_STATUS}"
        echo "[FAILED] Continue to next experiment."

        if [[ -d "${SAMPLE_FOLDER}" ]]; then
            echo "[Clean] Removing failed sample folder: ${SAMPLE_FOLDER}"
            rm -rf "${SAMPLE_FOLDER}"
        fi

        if [[ -f "${NPZ_FILE}" ]]; then
            echo "[Clean] Removing failed npz: ${NPZ_FILE}"
            rm -f "${NPZ_FILE}"
        fi

        continue
    fi

    FINAL_STATUS="$(check_log_status "${LOG_FILE}")"
    echo "[Done] FINAL_LOG_STATUS=${FINAL_STATUS}"

    if [[ "${FINAL_STATUS}" != "SUCCESS" ]]; then
        echo "[WARNING] torchrun exited 0 but final log is not fid=..., status=${FINAL_STATUS}"
        echo "[WARNING] Continue to next experiment."
        continue
    fi

    if [[ ! -s "${NPZ_FILE}" ]]; then
        echo "[WARNING] log says success but npz missing or empty: ${NPZ_FILE}"
        echo "[WARNING] Continue to next experiment."
        continue
    fi

    echo "[SUCCESS] EXP=${EXP}"
    echo "[SUCCESS] NPZ=${NPZ_FILE}"
done

echo "[DONE] rerun without-CFG missing experiments finished."