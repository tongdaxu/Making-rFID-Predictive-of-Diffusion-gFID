#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/video_ssd/kongzishang/xutongda/git/IFID}"
cd "${REPO}"

SCRIPT="${SCRIPT:-eval_hybrid_dino_vae_ifid_25_4090.py}"
CONFIG_DIR="${CONFIG_DIR:-./configs}"

GPU_LIST=(${GPU_LIST:-0 1 2 3 4 5 6 7})
HIGH_GPU_LIST=(${HIGH_GPU_LIST:-0 1 2 3})

NUM_WORKERS="${NUM_WORKERS:-4}"
REF_SIZE="${REF_SIZE:-50000}"
VAL_SIZE="${VAL_SIZE:-50000}"

DINO_CACHE="${DINO_CACHE:-/share/xutongda/REPA-E/exps_interpolate/dino_ifid_25/dinov2_vitb_global_ref50000_val50000}"
LATENT_CACHE_ROOT="${LATENT_CACHE_ROOT:-/share/xutongda/REPA-E/exps_interpolate/local_score_rfid_25/cache}"

DEFAULT_BS="${DEFAULT_BS:-16}"
HIGH_BS="${HIGH_BS:-4}"

DEFAULT_QUERY_BLOCK="${DEFAULT_QUERY_BLOCK:-256}"
HIGH_QUERY_BLOCK="${HIGH_QUERY_BLOCK:-256}"

DEFAULT_CAND_CHUNK="${DEFAULT_CAND_CHUNK:-16}"
HIGH_CAND_CHUNK="${HIGH_CAND_CHUNK:-1}"

DEFAULT_GLOBAL_REF_CHUNK="${DEFAULT_GLOBAL_REF_CHUNK:-512}"
HIGH_GLOBAL_REF_CHUNK="${HIGH_GLOBAL_REF_CHUNK:-32}"

GPU_BANK_MAX_GB="${GPU_BANK_MAX_GB:-8}"

OUT_TAG="${OUT_TAG:-dinoK1_10_50_200_1000_globalhard_50k50k}"
OUT_ROOT="${OUT_ROOT:-/share/xutongda/REPA-E/exps_interpolate/hybrid_dino_vae_ifid_25/${OUT_TAG}}"
LOG_DIR="${OUT_ROOT}/logs"

SKIP_DONE="${SKIP_DONE:-1}"
ONLY_CONFIG="${ONLY_CONFIG:-}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# High-dimensional banks are delayed to a second phase so 8 processes do not
# simultaneously consume tens of GB of CPU RAM each.
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
  shift
  local -a phase_gpus=("$1")
  shift
  local -a configs=("$@")

  # The array-above arrives space-delimited in one item; expand it here.
  read -ra GPUS <<< "${phase_gpus[0]}"

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

    local bs qblock cchunk gchunk
    if [[ "${phase}" == "HIGH" ]]; then
      bs="${HIGH_BS}"
      qblock="${HIGH_QUERY_BLOCK}"
      cchunk="${HIGH_CAND_CHUNK}"
      gchunk="${HIGH_GLOBAL_REF_CHUNK}"
    else
      bs="${DEFAULT_BS}"
      qblock="${DEFAULT_QUERY_BLOCK}"
      cchunk="${DEFAULT_CAND_CHUNK}"
      gchunk="${DEFAULT_GLOBAL_REF_CHUNK}"
    fi

    local metrics="${OUT_ROOT}/${name}/metrics.json"
    if [[ "${SKIP_DONE}" == "1" && -s "${metrics}" ]]; then
      echo "[SKIP DONE] ${cfg}: ${metrics}"
      return 2
    fi

    local log="${LOG_DIR}/${lower}.out"
    local status="${LOG_DIR}/${lower}.status"

    echo "Launching ${cfg} on RTX4090 GPU ${gpu}: bs=${bs}, qblock=${qblock}, cand_chunk=${cchunk}, global_ref_chunk=${gchunk}"

    (
      set +e
      CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
        --vae-config "${CONFIG_DIR}/${cfg}" \
        --exp-name "${name}" \
        --ref-size "${REF_SIZE}" \
        --val-size "${VAL_SIZE}" \
        --dino-cache "${DINO_CACHE}" \
        --dino-topk 1000 \
        --bs "${bs}" \
        --ref-bs 128 \
        --num-workers "${NUM_WORKERS}" \
        --query-block-size "${qblock}" \
        --candidate-chunk "${cchunk}" \
        --global-ref-chunk "${gchunk}" \
        --gpu-bank-max-gb "${GPU_BANK_MAX_GB}" \
        --latent-cache-dir "${LATENT_CACHE_ROOT}" \
        --cache-group-batches 20 \
        --cache-dtype fp16 \
        --reuse-cache \
        --intp slerp \
        --alpha 0.5 \
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

# Bash function arguments cannot carry arrays directly, so pass GPU lists as strings.
set +e
run_phase "NORMAL" "${GPU_LIST[*]}" "${NORMAL_CONFIGS[@]}"
normal_rc=$?

run_phase "HIGH" "${HIGH_GPU_LIST[*]}" "${HIGH_CONFIGS[@]}"
high_rc=$?
set -e

total_fail=$((normal_rc + high_rc))
echo "All phases finished. failures=${total_fail}"
exit "${total_fail}"
