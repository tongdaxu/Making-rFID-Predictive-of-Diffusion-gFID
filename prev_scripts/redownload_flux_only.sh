#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/share/xutongda/hfhome}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/share/xutongda/hfhome/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "[Info] HF_ENDPOINT=$HF_ENDPOINT"
echo "[Info] HF_HOME=$HF_HOME"
echo "[Info] HF_HUB_CACHE=$HF_HUB_CACHE"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN is empty. Gated models may fail."
  echo "[WARN] Run: export HF_TOKEN=your_token"
fi

FAILED=()

validate_safetensors() {
  local f="$1"
  python3 - "$f" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.isfile(p):
    raise SystemExit(f"missing: {p}")
size = os.path.getsize(p)
if size < 16:
    raise SystemExit(f"too small: {p}")
with open(p, "rb") as f:
    n = int.from_bytes(f.read(8), "little")
    if n <= 0 or n > size - 8:
        raise SystemExit(f"bad safetensors header length: {p}, n={n}, size={size}")
    json.loads(f.read(n).decode("utf-8"))
print("[VALID]", p, size)
PY
}

download_flat() {
  local repo="$1"
  local file="$2"
  local local_dir="$3"
  local out="$local_dir/$file"

  echo
  echo "============================================================"
  echo "[Download flat] $repo :: $file"
  echo "[Out] $out"
  echo "============================================================"

  mkdir -p "$local_dir"

  for i in 1 2 3 4 5; do
    echo "[Attempt $i/5]"
    if hf download "$repo" "$file" \
      --repo-type model \
      --local-dir "$local_dir"; then

      if validate_safetensors "$out"; then
        echo "[OK] $(du -hL "$out" | awk '{print $1}')  $out"
        return 0
      fi
    fi

    echo "[WARN] failed attempt $i: $repo :: $file"
    sleep $((i * 10))
  done

  echo "[FAIL] $repo :: $file" >&2
  FAILED+=("$repo::$file")
  return 0
}

# SKIP: DINOv3 gated access not authorized for current HF account.
# download_flat "facebook/dinov3-vits16plus-pretrain-lvd1689m" \
#   "model.safetensors" \
#   "/share/xutongda/hfhome/hub/dinov3-vits16plus-pretrain-lvd1689m"

download_flat "black-forest-labs/FLUX.1-dev" \
  "text_encoder/model.safetensors" \
  "/share/xutongda/hfhome/hub/FLUX.1-dev"

download_flat "black-forest-labs/FLUX.1-schnell" \
  "text_encoder/model.safetensors" \
  "/share/xutongda/hfhome/hub/FLUX.1-schnell"

download_flat "black-forest-labs/FLUX.1-Redux-dev" \
  "image_encoder/model.safetensors" \
  "/share/xutongda/hfhome/hub/FLUX.1-Redux-dev"

echo
echo "=============================="
echo "[Summary]"
echo "=============================="

if (( ${#FAILED[@]} > 0 )); then
  echo "[WARN] Failed items:"
  for x in "${FAILED[@]}"; do
    echo "  - $x"
  done
  echo
  echo "Most likely causes:"
  echo "  1. HF_TOKEN not set"
  echo "  2. You have not accepted the gated model terms on Hugging Face"
  echo "  3. Network/proxy issue"
  exit 2
fi

echo "[DONE] remaining 4 flat files downloaded"
