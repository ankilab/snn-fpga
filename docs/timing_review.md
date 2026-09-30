# Investigation of unresolved timing findings

Reviewed on 2026-09-24. Additional reports were generated on 2026-09-23 with
Vivado 2026.1 from the three existing routed checkpoints of the confirmed
U50 reference run. RTL, constraints, checkpoints and bitstreams were unchanged.
All three analysis runs completed successfully: this means the reports were
generated, not that they were free of warnings.

**Result: the eight missing I/O delays concern platform ports. There are also
specific reset findings in the project RTL and unresolved platform/CDC
findings. Full methodology sign-off has therefore not been established.**
This does not retroactively invalidate the inference tests that passed.

## 1. Which I/O delays are missing?

The detailed `check_timing` check lists the same ports for LIF, IZH and HH:

| Direction | Port | Assignment |
|---|---|---|
| Input | `io_data_satellite_ctrl_00[0]`, `[1]` | Platform satellite control interface |
| Input | `io_perst_n_00[0]` | PCIe reset |
| Input | `io_uart_cmc_00_rxd` | Platform management UART |
| Input | `io_uart_satellite_ctrl_00_rxd` | Satellite control UART |
| Output | `io_hbm_cattrip_00[0]` | Platform HBM CATTRIP signal |
| Output | `io_uart_cmc_00_txd` | Platform management UART |
| Output | `io_uart_satellite_ctrl_00_txd` | Satellite control UART |

Complete timing paths assign the UARTs to `level0_i/blp/.../ss_cmc`, the reset
to `level0_i/blp/.../ss_hif/.../pcie`, and one satellite signal to platform
pipeline `level0_i/ii_level0_pipe`. The SNN kernel's AXI-Lite interface is
inside the platform; the kernel does not expose these board ports itself.
The remaining port functions are assigned based on their platform names;
a full electrical board assessment was not performed.

According to `check_timing`, there are still no missing register clocks or
unconstrained internal endpoints. This does not establish sign-off for
external port paths. No blanket zero delays or false paths were added.
Appropriate constraints require the platform's interface assumptions.

## 2. DRC and methodology

| Design | DRC errors / critical warnings | DRC warnings | DRC advisory | Methodology critical warnings | Methodology warnings | Methodology advisory |
|---|---:|---:|---:|---:|---:|---:|
| LIF | 0 / 0 | 60 | 12 | 10 | 159 | 53 |
| IZH | 0 / 0 | 76 | 12 | 10 | 347 | 53 |
| HH | 0 / 0 | 380 | 12 | 10 | 2,609 | 53 |

These counts represent reported instances, not independent root causes.
DRC used the Vivado default rule deck on the fully routed design. It does not
replace every possible release or board check.

The ten critical methodology warnings are identical across variants:

- **TIMING-1 (1):** The declared kernel clock has a 10 ns period (100 MHz);
  static MMCM parameters expect 3.333 ns (300 MHz).
- **TIMING-7 (1):** `io_clk_pcie_user_00` and `dma_ip_axi_aclk_1` are analyzed
  as related, but the tool reports that they have no common node.
- **TIMING-14 (2):** LUTs in clock paths of management/BSCAN debug logic.
- **TIMING-54 (6):** Scoped clock exceptions may have effects beyond their scope.

The linker explicitly generated `_user_impl_clk.xdc` with a kernel clock
override. In the HH checkpoint, kernel input `s_axi_aclk` was assigned to
`clk_kernel_00_unbuffered_net`. XCLBIN metadata and the read-only XRT platform
report list DATA_CLK = 100 MHz, KERNEL_CLK = 500 MHz and HBM = 450 MHz.
The name KERNEL_CLK in the XRT table therefore does not denote the SNN clock
connected here. These values are metadata or XRT output, not independent
physical frequency measurements.

Vitis supports scalable platform clocks that XRT can adjust at runtime.
This is a plausible explanation for the difference between static MMCM
settings and the linker setting; it is not yet specific clearance of this
TIMING-1 warning for this platform/tool combination.
[AMD: Managing Clock Frequencies](https://docs.amd.com/r/2023.2-English/ug1393-vitis-application-acceleration/Managing-Clock-Frequencies?contentId=Ii6UxwDCHNN0i_Vimbwlmg).

All variants also report **122 TIMING-16 instances** in the PCIe/shell area,
with reported large setup violations from −1.001 to −1.850 ns. Meanwhile,
the regular timing summary remains positive: WNS +0.055 ns, WHS +0.009 ns.
It was regenerated after `report_methodology` on the HH checkpoint and
showed the same values. The difference has not yet been fully traced to
specific path exceptions or analysis conditions; TIMING-16 is therefore
not marked as harmless or resolved.
[AMD: TIMING-16](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/TIMING-16-Large-Setup-Violation).

## 3. Specific findings in the project RTL

**LUTAR-1: logic before asynchronous reset inputs.**

- In all three models, this affects the output decoder. The reported LUT
  drives asynchronous clear pins of the class counters.
- IZH and HH each add two instances, in the serial neuron cores of the two
  layers and their dividers.

The source provides specific starting points: `output_decoder.sv` handles
`if (rst || clear)` in a block with asynchronous `rst`.
`izhikevich_neuron.sv` and `hodgkin_huxley_neuron.sv` also explicitly connect
divider resets to `.rst(rst || clear)`. `signed_round_divider.sv` handles
reset asynchronously. Thus, an operational clear signal reaches asynchronous
reset paths through logic. This is a structural/robustness finding; by itself,
it does not demonstrate an incorrect MNIST output.

**Suggested correction:** explicitly separate global reset and synchronous
clear. Add a synchronous clear/abort input to the dividers and adapt callers.
Then check clear during idle and active computation, reset release and restart;
repeat RTL/reference tests, fresh builds, methodology analysis and hardware
qualification for all three models. This investigation does not implement
the correction.

Other kernel-related reports concern DSP input/output pipelines and
asynchronously reset registers feeding DSPs (DPIP/DPOP/DPIR). They limit
optimization opportunities but do not independently demonstrate a functional
error. Pipeline changes would affect latency and control and should form
part of a separately verified architecture change.

## 4. Additional CDC check on the HH design

A targeted timing report and a CDC report were generated for the direction
`io_clk_pcie_user_00` → `dma_ip_axi_aclk_1`. The CDC report contains:

- 14 × **CDC-7 Critical**: safe asynchronous reset crossing not recognized;
- 2 × **CDC-10 Critical**: combinational logic before a synchronizer;
- 9 × **CDC-15 Warning**: clock-enable-controlled crossing.

The listed paths are in PCIe/debug logic under `level0_i/blp`. This covers
only one clock direction in the HH design, not full CDC verification of all
variants. The tool report explicitly notes that ports without input delay
are excluded from this CDC analysis.

## 5. Implications for the handover

The conclusion remains: **all planned inference tests passed; automated
setup/hold/routing checks passed; full methodology/CDC sign-off remains open.**
The eight port reports have been classified but not fixed by new constraints.
There is no newly tested RTL version.

The timing findings remain tasks for further development. The clearly scoped
reset/clear correction is a suitable future change. Platform topics require
comparison against the unmodified platform and its clock/CDC constraints,
or reliable vendor guidance for the combination of platform 2022.1 and tools
2026.1. Until then, warnings are neither suppressed nor given blanket clearance.

## Evidence

The [machine-readable summary](../evidence/timing-review-2026-09-23/summary.json),
complete port checks and selected DRC/methodology extracts are under
`evidence/timing-review-2026-09-23/`. The extracts include the full rule
summary, all critical methodology instances, all kernel-related LUTAR
instances and at least one example of each remaining rule. Additional
HH clock/CDC reports are also included.

Personal path prefixes are anonymized in the extracts; original and
publication hashes are recorded in the summary.
