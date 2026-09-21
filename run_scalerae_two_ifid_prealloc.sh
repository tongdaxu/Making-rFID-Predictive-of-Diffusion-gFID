#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/share/xutongda/REPA-E/exps_interpolate}"

SAMPLE_ROOT="${SAMPLE_ROOT:-${ROOT}/ifid_saved_images}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_ifid_extra3}"

SCRIPT="${SCRIPT:-evalvae_fix_recon_prealloc.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

DATASET="${DATASET:-/video_ssd/lpm/ImageNet/val}"
DATASET_REF="${DATASET_REF:-/video_ssd/lpm/ImageNet/train}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
BASE_PORT="${BASE_PORT:-29984}"

SMALL="${SMALL:-200000}"
SMALL_VAL="${SMALL_VAL:-50000}"
BS="${BS:-1}"
TOP="${TOP:-10}"
ALPHA="${ALPHA:-0.5}"

OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${SAMPLE_ROOT}" "${LOG_DIR}"

JOBS=(
  "ScaleRAE-SigLIP2|scalerae-siglip2|ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE-WEBSSL|scalerae-webssl|ScaleRAE_WEBSSL.yaml"
)

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[FATAL] Missing evaluator: ${SCRIPT}"
  exit 1
fi

python -m py_compile "${SCRIPT}" || exit 1

echo "================================================================================"
echo "[ScaleRAE two-model iFID image generation]"
echo "SCRIPT=${SCRIPT}"
echo "CUDA_DEVICES=${CUDA_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "SMALL=${SMALL}"
echo "SMALL_VAL=${SMALL_VAL}"
echo "BS=${BS}"
echo "TOP=${TOP}"
echo "ALPHA=${ALPHA}"
echo "SAMPLE_ROOT=${SAMPLE_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "================================================================================"

echo
echo "[VISIBLE GPUS]"
nvidia-smi \
  --query-gpu=index,uuid,pci.bus_id,name \
  --format=csv,noheader || true

echo
echo "[PYTORCH CUDA COUNT]"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python - <<'PY'
import torch

print("torch.cuda.device_count() =", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

VISIBLE_COUNT="$(
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"

if [[ "${VISIBLE_COUNT}" -ne "${NUM_PROCESSES}" ]]; then
  echo "[FATAL] CUDA visible device count is ${VISIBLE_COUNT}, expected ${NUM_PROCESSES}"
  exit 1
fi

FAILED=0
DONE=0
SKIPPED=0
JOB_INDEX=0

run_one() {
  local model="$1"
  local tag="$2"
  local config_file="$3"
  local port="$4"

  local config_path="${CONFIG_DIR}/${config_file}"
  local exp_name="ifid-iis-${tag}-slerp-a05-top${TOP}-${SMALL}x${SMALL_VAL}"

  local out_file="${LOG_DIR}/${exp_name}.out"
  local master_file="${LOG_DIR}/master_${tag}_prealloc.out"

  local sample_dir="${SAMPLE_ROOT}/${exp_name}"
  local extra_dir="${SAMPLE_ROOT}/${exp_name}_extra"

  echo
  echo "================================================================================"
  echo "[MODEL] ${model}"
  echo "[TAG] ${tag}"
  echo "[CONFIG] ${config_path}"
  echo "[EXP_NAME] ${exp_name}"
  echo "[PORT] ${port}"
  echo "[LOG] ${out_file}"
  echo "================================================================================"

  if [[ ! -f "${config_path}" ]]; then
    echo "[FAILED] Missing config: ${config_path}"
    echo "MISSING_CONFIG" > "${out_file}.failed"
    return 1
  fi

  if [[ -f "${out_file}.done" && "${OVERWRITE}" -eq 0 ]]; then
    echo "[SKIP DONE] ${model}"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  rm -f \
    "${out_file}.done" \
    "${out_file}.failed"

  # 没有 done 标记时，删除可能存在的不完整图片，
  # 避免本次输出与上次失败残留混合。
  rm -rf \
    "${sample_dir}" \
    "${extra_dir}"

  {
    echo "[START TIME] $(date '+%F %T')"
    echo "[MODEL] ${model}"
    echo "[TAG] ${tag}"
    echo "[CONFIG] ${config_path}"
    echo "[PORT] ${port}"
  } > "${master_file}"

  set +e

  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_DEBUG=WARN \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_PROCESSES}" \
    --master-port="${port}" \
    "${SCRIPT}" \
    --seed=0 \
    --sample-dir="${SAMPLE_ROOT}" \
    --exp-name="${exp_name}" \
    --dataset="${DATASET}" \
    --dataset-ref="${DATASET_REF}" \
    --vae-config="${config_path}" \
    --small "${SMALL}" \
    --small_val "${SMALL_VAL}" \
    --bs "${BS}" \
    --top "${TOP}" \
    --intp slerp \
    --alpha "${ALPHA}" \
    -save \
    --save-every 1 \
    --save-max "${SMALL_VAL}" \
    > "${out_file}" 2>&1

  status=$?

  set -e

  echo "[EXIT STATUS] ${status}" >> "${master_file}"
  echo "[END TIME] $(date '+%F %T')" >> "${master_file}"

  if [[ "${status}" -eq 0 ]] &&
     grep -q '^ifid=' "${out_file}" &&
     grep -q '^rfid=' "${out_file}" &&
     grep -q '^lpips=' "${out_file}"; then

    touch "${out_file}.done"
    rm -f "${out_file}.failed"

    intp_dir="${extra_dir}/intp_dist"

    if [[ -d "${intp_dir}" ]]; then
      image_count="$(
        find "${intp_dir}" \
          -type f \
          \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
          | wc -l
      )"
    else
      image_count=0
    fi

    echo "[DONE] ${model}"
    echo "[INTP IMAGE COUNT] ${image_count}"

    {
      echo "[DONE] ${model}"
      echo "[INTP IMAGE COUNT] ${image_count}"
      tail -n 20 "${out_file}" || true
    } >> "${master_file}"

    if [[ "${image_count}" -ne "${SMALL_VAL}" ]]; then
      echo "[WARNING] Expected ${SMALL_VAL} intp images, found ${image_count}"
      echo "IMAGE_COUNT_${image_count}" > "${out_file}.image_warning"
    else
      rm -f "${out_file}.image_warning"
    fi

    DONE=$((DONE + 1))
    return 0
  fi

  echo "${status}" > "${out_file}.failed"

  echo "[FAILED] ${model}, status=${status}"
  tail -n 120 "${out_file}" || true

  {
    echo "[FAILED] ${model}"
    tail -n 120 "${out_file}" || true
  } >> "${master_file}"

  FAILED=$((FAILED + 1))
  return 1
}

for item in "${JOBS[@]}"; do
  IFS="|" read -r MODEL TAG CONFIG_FILE <<< "${item}"

  PORT=$((BASE_PORT + JOB_INDEX))

  # 即使第一个失败，也继续尝试第二个。
  run_one \
    "${MODEL}" \
    "${TAG}" \
    "${CONFIG_FILE}" \
    "${PORT}" || true

  JOB_INDEX=$((JOB_INDEX + 1))

  # 等待前一个 torchrun 完全退出并释放 NCCL/显存。
  sleep 10
done

echo
echo "================================================================================"
echo "[ALL FINISHED]"
echo "DONE=${DONE}"
echo "SKIPPED=${SKIPPED}"
echo "FAILED=${FAILED}"
echo
echo "[DONE FILES]"
find "${LOG_DIR}" \
  -maxdepth 1 \
  -type f \
  -name "ifid-iis-scalerae-*.out.done" \
  -print
echo
echo "[FAILED FILES]"
find "${LOG_DIR}" \
  -maxdepth 1 \
  -type f \
  -name "ifid-iis-scalerae-*.out.failed" \
  -print
echo "================================================================================"

if (( FAILED > 0 )); then
  exit 1
fi
