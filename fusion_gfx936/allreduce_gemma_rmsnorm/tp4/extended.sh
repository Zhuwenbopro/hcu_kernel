#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export EXTENDED_OUT="${1:-$(mktemp -d "$PWD/extended-XXXXXXXX")}"
mkdir -p "$EXTENDED_OUT"
export PYTHON="${PYTHON:-python3}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu
export AITER_AR_TRANSPORT=auto AITER_AR_MAX_SIZE_MB=128 AITER_AR_SMALL_MESSAGE=off
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1 WORLD_SIZE=4
export MASTER_PORT="${MASTER_PORT:-$("$PYTHON" -S -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.6-27B safetensors checkpoint directory}"
ulimit -c 0
echo "$EXTENDED_OUT" > extended-last.txt
sha256sum extended.py tp4_ar.py native/libtp4.so > "$EXTENDED_OUT/manifest.txt"
pids=()
cleanup() {
    if [[ ${#pids[@]} -gt 0 ]]; then
        for pid in "${pids[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${pids[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        wait || true
    fi
}
trap 'cleanup; echo interrupted > "$EXTENDED_OUT/status"; exit 130' INT TERM
for i in 0 1 2 3; do
    RANK=$i LOCAL_RANK=$i "$PYTHON" -P "$PWD/extended.py" > "$EXTENDED_OUT/rank$i.log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$EXTENDED_OUT/rank-pids.txt"
remaining=4
while (( remaining )); do
    if wait -n; then
        remaining=$((remaining-1))
    else
        echo failed > "$EXTENDED_OUT/status"
        cleanup
        exit 1
    fi
done
echo complete > "$EXTENDED_OUT/status"
