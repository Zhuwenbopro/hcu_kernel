#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -std=c++17 -ffp-contract=off --offload-arch=gfx936 -fPIC -shared tp4.cpp -o libtp4.pending.so
mv libtp4.pending.so libtp4.so
