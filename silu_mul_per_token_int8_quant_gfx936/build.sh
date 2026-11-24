#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:-"$source_dir/build"}
mkdir -p -- "$output_dir"
"${HIPCC:-hipcc}" -x hip -O3 -std=c++17 --offload-arch=gfx936 -shared -fPIC \
    "$source_dir/silu_mul_int8.hip" -o "$output_dir/libsilu_mul_int8.so"
