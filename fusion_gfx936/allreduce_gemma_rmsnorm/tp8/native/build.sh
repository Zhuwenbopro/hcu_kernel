#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
hipcc -O3 -std=c++17 -ffp-contract=off --offload-arch=gfx936 -fPIC -shared tp8.cpp -o libtp8.pending.so
mv libtp8.pending.so libtp8.so
