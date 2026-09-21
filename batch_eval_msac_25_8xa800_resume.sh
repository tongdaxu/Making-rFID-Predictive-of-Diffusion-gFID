#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_msac_25_a800.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
HIGH_GPU_LIST=(${HIGH_GPU_LIST:-0 1 2 3})

NUM_WORKERS="${NUM_WORKERS:-8}"
REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"

SEMANTIC_CACHE="${SEMANTIC_CACHE:-$REPO/msac_results/dino_semantics_dinov2b_ref50000_val50000}"
LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache}"
ENCODER_CODE_ROOT="${ENCODER_CODE_ROOT:-/share/xutongda/REPA-E}"

DEFAULT_BS="${DEFAULT_BS:-64}"
HIGH_BS="${HIGH_BS:-16}"

DEFAULT_REF_CHUNK="${DEFAULT_REF_CHUNK:-4096}"
HIGH_REF_CHUNK="${HIGH_REF_CHUNK:-512}"

ALPHA="${ALPHA:-0.5}"

OUT_ROOT="${OUT_ROOT:-$REPO/msac_results/msac_hardcos_slerp_a05_50k50k}"
LOG_DIR="${OUT_ROOT}/logs"

SKIP_DONE="${SKIP_DONE:-1}"
ONLY_CONFIG="${ONLY_CONFIG:-}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# High-dimensional reference banks still fit on A800 80GB, but use smaller
# search chunks and run these four in a separate phase to avoid unnecessary
# host-I/O/RAM pressure while their 20-30GB banks are being loaded.
NORMAL_CONFIGS=(
  "SDVAE.yaml"
  "SD3VAE.yaml"
  "FLUXVAE.yaml"
  "DCAE.yaml"
  "SOFTVQ.yaml"
  "VAVAE.yaml"
  "VAVAE64.yaml"
  "EQVAE.yaml"
  "MAETOK.yaml"
  "INVAE.yaml"
  "REPAEVAE.yaml"
  "DETOK.yaml"
  "QWVAE.yaml"
  "SVG.yaml"
  "FLUX2VAE.yaml"
  "SVGT2I_P_STAGE1_256.yaml"
  "DMVAE.yaml"
  "VTPS.yaml"
  "VTPB.yaml"
  "VTPL.yaml"
  "PAE_DINOv2L_d32.yaml"
)

HIGH_CONFIGS=(
  "RAE.yaml"
  "UAE.yaml"
  "ScaleRAE_SIGLIP2_NONORM.yaml"
  "ScaleRAE_WEBSSL.yaml"
)

run_phase() {
  local phase="$1"
  local gpu_string="$2"
  shift 2
  local -a configs=("$@")
  local -a GPUS
  read -ra GPUS <<< "${gpu_string}"

  local -a filtered=()
  for cfg in "${configs[@]}"; do
    if [[ -z "${ONLY_CONFIG}" || "${cfg}" == "${ONLY_CONFIG}" ]]; then
      filtered+=("${cfg}")
    fi
  done
  configs=("${filtered[@]}")

  (( ${#configs[@]} == 0 )) && return 0

  echo "================ ${phase} phase ================"
  echo "GPUs: ${GPUS[*]}"
  echo "Models: ${configs[*]}"

  declare -a PID CFG LOG STATUS PHYS
  local next=0
  local failed=0

  launch() {
    local slot="$1"
    local gpu="$2"
    local cfg="$3"

    local name="${cfg%.yaml}"
    local lower
    lower="$(echo "${name}" | tr '[:upper:]' '[:lower:]')"

    local bs ref_chunk
    if [[ "${phase}" == "HIGH" ]]; then
      bs="${HIGH_BS}"
      ref_chunk="${HIGH_REF_CHUNK}"
    else
      bs="${DEFAULT_BS}"
      ref_chunk="${DEFAULT_REF_CHUNK}"
    fi

    local metrics="${OUT_ROOT}/${name}/metrics.json"
    if [[ "${SKIP_DONE}" == "1" && -s "${metrics}" ]]; then
      echo "[SKIP DONE] ${cfg}: ${metrics}"
      return 2
    fi

    local log="${LOG_DIR}/${lower}.out"
    local status="${LOG_DIR}/${lower}.status"

    echo "Launching ${cfg} on A800 GPU ${gpu}: bs=${bs}, ref_chunk=${ref_chunk}"

    (
      set +e
      CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
        --vae-config "${CONFIG_DIR}/${cfg}" \
        --exp-name "${name}" \
        --ref-size "${REF_SIZE}" \
        --val-size "${VAL_SIZE}" \
        --bs "${bs}" \
        --num-workers "${NUM_WORKERS}" \
        --latent-cache-dir "${LATENT_CACHE_ROOT}" \
        --bank-dtype fp16 \
        --ref-chunk-size "${ref_chunk}" \
        --semantic-cache "${SEMANTIC_CACHE}" \
        --dino-enc-type dinov2-vit-b \
        --encoder-code-root "${ENCODER_CODE_ROOT}" \
        --alpha "${ALPHA}" \
        --no-tf32 \
        --out-dir "${OUT_ROOT}" \
        > "${log}" 2>&1

      code=$?
      echo "${code}" > "${status}"
      exit "${code}"
    ) &

    PID["${slot}"]="$!"
    CFG["${slot}"]="${cfg}"
    LOG["${slot}"]="${log}"
    STATUS["${slot}"]="${status}"
    PHYS["${slot}"]="${gpu}"
    return 0
  }

  launch_next() {
    local slot="$1"
    local gpu="$2"

    while (( next < ${#configs[@]} )); do
      local cfg="${configs[$next]}"
      next=$((next + 1))
      if launch "${slot}" "${gpu}" "${cfg}"; then
        return
      fi
    done
    PID["${slot}"]=""
  }

  for ((slot=0; slot<${#GPUS[@]}; slot++)); do
    launch_next "${slot}" "${GPUS[$slot]}"
  done

  while true; do
    local active=0

    for ((slot=0; slot<${#GPUS[@]}; slot++)); do
      local pid="${PID[$slot]:-}"
      [[ -z "${pid}" ]] && continue
      active=$((active + 1))

      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || true

        local cfg="${CFG[$slot]}"
        local code=999
        [[ -f "${STATUS[$slot]}" ]] && code="$(cat "${STATUS[$slot]}")"

        if [[ "${code}" != "0" ]]; then
          echo "[FAIL] ${cfg} exit=${code}; log=${LOG[$slot]}" >&2
          failed=$((failed + 1))
        else
          echo "[DONE] ${cfg}"
        fi

        launch_next "${slot}" "${PHYS[$slot]}"
      fi
    done

    (( active == 0 )) && break
    sleep "${POLL_INTERVAL}"
  done

  echo "${phase} phase finished. failures=${failed}"
  return "${failed}"
}

set +e
run_phase "NORMAL" "${GPU_LIST[*]}" "${NORMAL_CONFIGS[@]}"
normal_rc=$?

run_phase "HIGH" "${HIGH_GPU_LIST[*]}" "${HIGH_CONFIGS[@]}"
high_rc=$?
set -e

total_fail=$((normal_rc + high_rc))
echo "All phases finished. failures=${total_fail}"
exit "${total_fail}"
