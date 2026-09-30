#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
kernel=${KERNEL:-snn}
[[ "$kernel" == snn ]] || { echo "Only the SNN kernel is included" >&2; exit 2; }
platform=${PLATFORM:-}
mode=${MODE:-xo}
build_dir=${BUILD_DIR:-"$repo_root/build/u50/$kernel"}
part=${PART:-xcu50-fsvh2104-2-e}
legacy_clock_adapter_compat=${LEGACY_U50_CLOCK_ADAPTER_COMPAT:-0}
case "$legacy_clock_adapter_compat" in
  0|1) ;;
  *) echo "LEGACY_U50_CLOCK_ADAPTER_COMPAT must be 0 or 1 (explicit opt-in)" >&2; exit 2 ;;
esac

command -v vivado >/dev/null || { echo "vivado is not on PATH" >&2; exit 2; }
mkdir -p "$build_dir/artifacts"
xo="$build_dir/artifacts/${kernel}_accelerator.xo"
vivado -mode batch -notrace -source "$repo_root/vivado/u50/package_u50_kernel.tcl" -tclargs \
  -kernel "$kernel" -output "$xo" -build_dir "$build_dir" -part "$part" \
  -hidden_neuron_model "${HIDDEN_NEURON_MODEL:-0}" \
  -output_neuron_model "${OUTPUT_NEURON_MODEL:-0}" \
  -input_neurons "${INPUT_NEURONS:-784}" -hidden_neurons "${HIDDEN_NEURONS:-256}" \
  -output_neurons "${OUTPUT_NEURONS:-10}" -num_events "${NUM_EVENTS:-12544}" \
  -hidden_executor_count "${HIDDEN_EXECUTOR_COUNT:-0}" \
  -output_executor_count "${OUTPUT_EXECUTOR_COUNT:-0}" \
  -interface_minor "${INTERFACE_MINOR:-0}"

if [[ "$mode" == xo ]]; then exit 0; fi
[[ -n "$platform" ]] || { echo "PLATFORM is required for MODE=xclbin" >&2; exit 2; }
command -v v++ >/dev/null || { echo "v++ is not on PATH" >&2; exit 2; }
mkdir -p "$build_dir/vpp"
link_args=(
  --link --target hw --platform "$platform" --kernel_frequency 100
  --save-temps
  --temp_dir "$build_dir/vpp"
  --output "$build_dir/artifacts/${kernel}_accelerator.xclbin"
  "$xo"
)

if [[ "$legacy_clock_adapter_compat" == 1 ]]; then
  echo "Explicit clock-adapter compatibility replacement enabled (tested: U50 202210_1, Vivado/Vitis 2026.1)."
  # The 2022.1 U50 platform contains a clk_metadata_adapter encrypted by the
  # 2019.1 tool. Vivado 2026.1 can build the platform but can no longer decrypt
  # this synthesis envelope. Stop after target generation, replace only that
  # metadata-only adapter with its transparent HDL equivalent, then resume.
  v++ "${link_args[@]}" --to_step vpl.generate_target
  adapter_source="$repo_root/deploy/u50/compat/clk_metadata_adapter_v1_0_vl_rfs.v"
  mapfile -t adapter_targets < <(
    find "$build_dir/vpp/link/vivado/vpl/prj" -type f \
      -name clk_metadata_adapter_v1_0_vl_rfs.v -print
  )
  ((${#adapter_targets[@]} > 0)) || {
    echo "Legacy U50 clock adapter was not found below $build_dir/vpp" >&2
    exit 2
  }
  for adapter_target in "${adapter_targets[@]}"; do
    if ! cmp -s "$adapter_source" "$adapter_target"; then
      if [[ ! -e "$adapter_target.encrypted-original" ]]; then
        cp "$adapter_target" "$adapter_target.encrypted-original"
      fi
      cp "$adapter_source" "$adapter_target"
    fi
  done
  v++ "${link_args[@]}" --from_step vpl.config_hw_runs
else
  v++ "${link_args[@]}"
fi
