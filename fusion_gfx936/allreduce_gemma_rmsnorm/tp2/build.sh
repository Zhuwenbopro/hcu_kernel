#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lib
for variant in fixed tuned; do
    hipcc -O3 -std=c++17 -ffp-contract=off --offload-arch=gfx936 -D__gfx936__ -fPIC -shared \
        "native/$variant.cpp" -o "lib/libosar_$variant.pending.so"
    mv "lib/libosar_$variant.pending.so" "lib/libosar_$variant.so"
done
