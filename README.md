# Applied Neuromorphic Computing

**Spiking Neural Networks (SNN) on Reconfigurable Hardware (FPGA)**

In this project, we developed a two-layer, event-driven SystemVerilog spiking
neural network architecture supporting LIF, Izhikevich and Hodgkin–Huxley
neurons. The references use 784 → 256 → 10 neurons,
T = 8 and one executor per layer. The implementation and hardware validation
target the AMD Xilinx Alveo U50.

## Start here

1. **Understand:** [Architecture and parameters](docs/architecture.md).
2. **Explore:** [Open the training explorer](https://ankilab.github.io/snn-fpga/explorer/training/):
   66 fits from the 50-epoch campaign, filters, curves and data export.
   Also available: the [neuron state explorer](https://ankilab.github.io/snn-fpga/explorer/neuron_state/)
   for spike frequencies and membrane potentials.
3. **Try it:** `python tools/check_reference.py` checks the three included reference
   model packages and two synthetic images in `examples/` without an FPGA or PyTorch.
4. **Build and run:** [U50 tutorial](docs/u50.md), including an
   [unattended run for all three models (LIF, IZH and HH)](docs/unattended_u50.md).
5. **Train new models:** [Training, data and export](docs/training.md).

The links above open the hosted explorers. For offline use, download or clone
the repository and open the explorers' `index.html` files in a browser. Keep
the complete `explorer/neuron_state/` folder together.

## Contents

| Area | Contents |
|---|---|
| `rtl/` | 21 RTL files defining the system architecture |
| `tb/`, `python/tests/` | Directed tests and reference comparisons |
| `vivado/`, `deploy/u50/` | Kernel packaging, Linux linking flow and XRT host |
| `tools/` | Build, test and training launchers, including Windows entry points |
| `LICENSES/` | Full license texts and machine-readable file assignments |
| `python/snn_reference/` | Bit-accurate numerical reference and floating-point HH model |
| `python/snn_host/` | Model/event upload and inference via XRT |
| `python/snn_training/`, `configs/` | QAT, INT8 export and central training matrix |
| `data/hh_luts/` | Six required HH ROM tables |
| `results/` | Measurements from one training campaign and three model packages; personal paths anonymized |
| `examples/` | Project-generated test patterns and expected reference outputs |
| `evidence/` | U50 reference run, timing evidence and checksums |

Metadata may contain anonymized `/historical-workspace` paths.
These record provenance and are not local working directories.

## Run local checks

Python 3.11+ is sufficient for the host and reference. The full core checks
also require Icarus Verilog (`iverilog`, `vvp`) and Verilator on the search path:

```bash
python tools/run_checks.py
```

On Windows, `tools/run_tests.cmd` is an alternative. Optional training checks
in a PyTorch environment: `python tools/run_checks.py --suite python --training`.

All three reference models were tested on the U50 with freshly generated
bitstreams: LIF 97.38%, IZH 97.68%, HH 84.00%, each on 10,000 MNIST images.
The [current evidence](evidence/u50-reference-2026-09-23/summary.json) records
the hardware results. The
[timing review](docs/timing_review.md) describes unresolved reset findings
in the project RTL and platform clock/CDC findings. Full methodology sign-off
has not yet been established; resolving these findings is intentionally left
for future development.

## Licensing

Hardware design: **CERN-OHL-S-2.0**. Software and other original content:
**GPL-3.0-only**. The [license overview](LICENSE.md) defines the file assignments.
See [NOTICE.md](NOTICE.md) for provenance and third-party components.
