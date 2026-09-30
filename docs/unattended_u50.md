# Unattended reference run on a Linux build machine

The runner **rebuilds LIF, IZH and HH sequentially** from
`configs/references.json` and tests each model on the specified U50. It uses
the existing Linux build flow with explicitly enabled optional clock adapter
handling. It does not use old bitstreams or old model paths. Reserve the
card for the entire run; every probe/inference command reloads the user region.

## Prepare and start

Copy the entire repository to the Linux build machine, including
the six weight files and the LUTs. Activate Vivado/Vitis and XRT as described
in the [U50 tutorial](u50.md). The four extracted MNIST files must be present;
their hashes are checked before starting. Python 3.11+ is sufficient; PyTorch
is not required. The flow performs no automatic `sudo`, password prompt,
download or base-shell modification.

In the example below, adjust the three paths to your installation and use
the device index reported by `xrt-smi --batch examine`:

```bash
cd /path/to/snn-fpga-repo
source /path/to/Vitis/2026.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
# If required, configure your existing license:
# export XILINXD_LICENSE_FILE=/path/to/your/license
# Only for the tested workaround: platform xilinx_u50_gen3x16_xdma_5_202210_1
# with Vivado/Vitis 2026.1; without this line, replacement stays disabled.
export LEGACY_U50_CLOCK_ADAPTER_COMPAT=1

python3 tools/qualify_references.py \
  --platform /path/to/u50_platform.xpfm \
  --data-dir /path/to/MNIST/raw \
  --run-dir "$PWD/build/qualification/reference-01" \
  --device 0 --preflight-only

bash deploy/u50/start_reference_qualification.sh \
  --platform /path/to/u50_platform.xpfm \
  --data-dir /path/to/MNIST/raw \
  --run-dir "$PWD/build/qualification/reference-01" \
  --device 0
```

The launcher checks prerequisites in the foreground, then starts with
`nohup`. It prints the PID and log path. You can then disconnect SSH.
A successful launch does not yet mean a successfully completed experiment.
To run in the foreground, invoke `python3 tools/qualify_references.py`
directly with the same arguments.

The documented hardware run on 2026-09-23 used this replacement for all three
models. It does not validate other platform/tool combinations. See
[clock adapter](clock_adapter.md) for details, provenance and licensing.
The runner records the option in `identity.json`; changing it requires a
new run directory. `0` disables replacement; `auto` is rejected.

## Procedure and success criteria

1. Check dataset and model hashes and local reference examples; record source
   hashes, platform files, tool versions and device index.
2. Save the XRT inventory and build the register helper locally in the run directory.
3. Rebuild XO and XCLBIN for each model. Failed builds restart in a new
   subdirectory when resuming.
4. Open the linked `*_routed.dcp` in Vivado: no nets marked `UNROUTED` or
   `PARTIAL`; existing setup and hold paths must have nonnegative worst slack.
   Save routing and timing reports. If the checkpoint cannot be identified
   unambiguously, the variant is marked as failed.
5. Check the register contract, dimensions, neuron models and model compatibility.
6. Check both synthetic images, including automatic weight upload/CRC:
   `blank` → `null`, `vertical_bar` → `1`.
7. Compare ten MNIST images against the bit-accurate Python reference.
   Every prediction must match. Increase this scope with `--gate-samples N`;
   this can substantially increase runtime, especially for HH.
8. Run all 10,000 MNIST test images on hardware. Records, order, accuracy and
   missing-prediction counts are independently checked. Expected accuracies:
   LIF 0.9738; IZH 0.9768; HH 0.84. Deviations require investigation and are
   not silently accepted.

The gate compares predictions, not complete internal neuron states.
Automated timing checks do not replace final assessment of constraints,
unclocked paths, pulse-width checks and DRC reports. Review the retained
build reports before hardware sign-off.

## Monitor, resume and handle failures

```bash
tail -f build/qualification-launchers/launcher.YOUR_ID.log
cat build/qualification/reference-01/summary.json
```

Each variant's `status.json` contains start/end times, commands, status,
errors and checksums. Each stage has separate stdout and stderr logs.
`summary.json` is updated after each variant. The runner exits with status 0
only when all three variants report `passed`. After a failure, dependent
steps for that variant are skipped and the next variant is attempted.
A failed global preflight prevents all builds.

To resume, use **the same launch command and run directory**. Only successful
stages with unchanged inputs and output files are reused; the physical probe
is repeated. Changes to tools, sources, models, platform or dataset require
a new run directory. An interrupted MNIST stage restarts from image 0.
Old failed build directories are retained for diagnosis.

By default, each stage times out after 24 hours. Adjust this with
`--stage-timeout-hours 48`. A timeout terminates the entire process group.
The device lock prevents additional instances of this runner for the same
user/device index; it does not lock out unrelated XRT applications or other
users. Therefore, also reserve the card operationally.

Local regression tests check rejection of corrupted resume data and incorrect
or incomplete qualification results. The complete flow has finished for LIF,
IZH and HH on the Linux build machine. Results are recorded in the
[hardware evidence](../evidence/u50-reference-2026-09-23/summary.json); unresolved
timing and CDC findings are described in the [timing review](timing_review.md).
A successful run does not replace revalidation after changes to the platform,
hardware, sources or model weights.
