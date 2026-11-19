#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
llvm="${DTK:-/opt/dtk}/aillvm/bin"
mkdir -p build
while read -r name cache batch serial; do
  "$llvm/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx936 -mattr=+sramecc \
    -filetype=obj --defsym="CACHE=$cache" --defsym="BATCH=$batch" --defsym="SERIAL=$serial" \
    read_xr.s -o "build/$name.o"
  "$llvm/ld.lld" -shared "build/$name.o" -o "build/$name.hsaco"
done <<'MODES'
xr_plain_window 0 0 0
xr_c3_window 3 0 0
xr_c3_batch 3 1 0
xr_c3_serial 3 0 1
MODES
