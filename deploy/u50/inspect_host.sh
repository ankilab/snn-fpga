#!/usr/bin/env bash
set -euo pipefail
command -v lspci >/dev/null || { echo "pciutils/lspci is missing" >&2; exit 2; }
[[ -r /opt/xilinx/xrt/setup.sh ]] || { echo "/opt/xilinx/xrt/setup.sh is missing" >&2; exit 2; }
source /opt/xilinx/xrt/setup.sh
echo '=== PCIe ==='
lspci -Dnn | grep -Ei 'xilinx|10ee:' || true
echo '=== XRT management ==='
sudo xbmgmt examine
echo '=== XRT user devices ==='
xrt-smi --batch examine
echo '=== Installed U50 platforms ==='
find /opt/xilinx/platforms /opt/xilinx/firmware -maxdepth 4 \
  \( -iname '*u50*' -o -iname '*.xpfm' \) -print 2>/dev/null | sort || true
echo '=== Validation ==='
xrt-smi --batch validate
