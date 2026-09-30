#!/usr/bin/env bash
# Run from an environment with Vivado/Vitis/XRT already activated.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python=${PYTHON:-python3}
[[ $# -ge 1 ]] || { echo 'Usage: bash deploy/u50/start_reference_qualification.sh --platform FILE --data-dir DIR --run-dir DIR --device INDEX'; exit 2; }
"$python" "$repo_root/tools/qualify_references.py" "$@" --preflight-only
mkdir -p "$repo_root/build/qualification-launchers"
log=$(mktemp "$repo_root/build/qualification-launchers/launcher.XXXXXXXX.log")
nohup "$python" -u "$repo_root/tools/qualify_references.py" "$@" > "$log" 2>&1 < /dev/null &
pid=$!
printf 'PID=%s\nLOG=%s\n' "$pid" "$log"
printf 'Resume with the identical command and --run-dir. Inspect summary.json and status.json in that directory.\n'
