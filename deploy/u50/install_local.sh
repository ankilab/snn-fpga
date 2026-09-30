#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
prefix=${PREFIX:-"$HOME/.local"}
mkdir -p "$prefix/bin"
bash "$repo_root/deploy/u50/build_host.sh" "$prefix/bin/xrt-register-server"
echo "source /opt/xilinx/xrt/setup.sh"
echo "export PATH=\"$prefix/bin:\$PATH\""
echo "export PYTHONPATH=\"$repo_root/python:\${PYTHONPATH:-}\""
