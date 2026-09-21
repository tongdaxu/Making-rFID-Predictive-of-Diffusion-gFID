#!/usr/bin/env bash
OUT="$1"

echo "timestamp,gpu0,gpu1,gpu2,gpu3,gpu4,gpu5,gpu6,gpu7" > "$OUT"

while true; do
    TS=$(date +%s)

    MEMS=$(nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits | paste -sd, -)

    echo "${TS},${MEMS}" >> "$OUT"

    sleep 1
done
