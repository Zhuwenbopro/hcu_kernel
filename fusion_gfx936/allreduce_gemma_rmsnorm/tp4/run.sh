#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
stage=${1:-smoke}
case "$stage" in smoke|pilot|full) ;; *) echo 'Usage: bash run.sh {smoke|pilot|full} [output_dir]' >&2; exit 2;; esac
out=${2:-$(mktemp -d "$PWD/run-XXXXXXXX")}
mkdir -p "$out/jobs"
export OUTDIR=$(cd "$out" && pwd)
if compgen -G "$OUTDIR/jobs/*.json" >/dev/null; then
    echo "Use a fresh output directory: $OUTDIR" >&2
    exit 2
fi
export PYTHON="${PYTHON:-python3}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu
export AITER_AR_TRANSPORT=auto AITER_AR_MAX_SIZE_MB=128 AITER_AR_SMALL_MESSAGE=off
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export MASTER_ADDR=127.0.0.1 WORLD_SIZE=4
export MASTER_PORT="${MASTER_PORT:-$("$PYTHON" -S -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.6-27B safetensors checkpoint directory}"
if [[ $stage == full ]]; then
    for job in smoke pilot full confirm; do "$PYTHON" -S enqueue_benchmark.py "$job" --run "$OUTDIR"; done
else
    "$PYTHON" -S enqueue_benchmark.py "$stage" --run "$OUTDIR"
fi
"$PYTHON" -S enqueue_benchmark.py stop --run "$OUTDIR"
echo "Writing rank logs and measurements to $OUTDIR"
pids=()
for (( i=0; i<WORLD_SIZE; i++ )); do
    RANK=$i LOCAL_RANK=$i "$PYTHON" worker.py > "$OUTDIR/rank$i.log" 2>&1 &
    pids+=("$!")
done
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
}
trap 'cleanup; exit 130' INT TERM
for (( i=0; i<WORLD_SIZE; i++ )); do
    if ! wait -n; then
        cleanup
        echo failed > "$OUTDIR/status"
        exit 1
    fi
done
echo complete > "$OUTDIR/status"
