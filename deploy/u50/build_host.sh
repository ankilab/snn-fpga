#!/usr/bin/env bash
set -euo pipefail
source "${XILINX_XRT:-/opt/xilinx/xrt}/setup.sh"
out=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/xrt-register-server"}
g++ -O2 -std=c++17 "$(dirname "${BASH_SOURCE[0]}")/xrt_register_server.cpp" \
  -I"$XILINX_XRT/include" -L"$XILINX_XRT/lib" -Wl,-rpath,"$XILINX_XRT/lib" \
  -lxrt_coreutil -pthread -o "$out"
echo "Created $out"
