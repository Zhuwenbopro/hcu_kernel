#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:-"$source_dir/build"}
mkdir -p -- "$output_dir"
compiler=${HIPCC:-hipcc}

build_backend() {
    local name=$1
    shift
    "$compiler" -x hip -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC \
        "-Dlaunch_silu_quant=launch_$name" "-Dinit_silu_gate_lut=init_$name" "$@" \
        "$source_dir/kernels.hip" -o "$output_dir/libsilu_backend_$name.so"
}

build_backend split8
build_backend pair4 -DSILU_PAIR4
build_backend pair8 -DSILU_PAIR8
build_backend tp2_split8 -DSILU_TP2
build_backend tp2_pair4 -DSILU_TP2 -DSILU_PAIR4
"$compiler" -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC \
    "$source_dir/silu_mul_int8.cpp" -L"$output_dir" -Wl,-rpath,'$ORIGIN' \
    -lsilu_backend_split8 -lsilu_backend_pair4 -lsilu_backend_pair8 \
    -lsilu_backend_tp2_split8 -lsilu_backend_tp2_pair4 -o "$output_dir/libsilu_mul_int8.so"
