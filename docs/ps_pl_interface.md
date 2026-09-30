# U50 SNN control-register interface

`snn_axi_lite_wrapper.sv` exposes the accelerator as a 32-bit AXI4-Lite slave
inside the user-managed U50 Vitis RTL kernel. XRT supplies the kernel address;
register offsets below are relative to that kernel. The initial interface is
optimized for correctness and straightforward software, not peak upload
bandwidth. AXI DMA can be added later without changing the model/event formats.

The implementation accepts write address and write data independently, respects
the byte-enable mask supplied with each write, holds read/write responses
under backpressure, and returns
`SLVERR` or `DECERR` for rejected accesses. This follows the separate AXI4-Lite
address/data/response channels described by AMD's AXI interface documentation:
<https://docs.amd.com/r/en-US/pg115-mdm/AXI4-Lite-Slave-Interface-Signals>.

The byte-enable mask is the four-bit AXI signal `WSTRB`, also called the write
strobes. Bit `i` enables byte `WDATA[8*i +: 8]`; it is not a separate clock.
For ordinary writable registers, enabled bytes are updated and disabled bytes
retain their previous values. For example, writing `0xAABBCCDD` with
`WSTRB=0001` to a register containing `0x11223344` produces `0x112233DD`.
`WSTRB=1111` enables all four bytes.

Command and upload registers impose additional requirements: `CONTROL` and
`WEIGHT_DATA` require byte 0, while `EVENT_NEURON` requires both bytes 0 and 1.
If the required strobes are missing, the wrapper returns `SLVERR` and records
`ERROR_BAD_STROBE` rather than executing an incomplete command or upload.

## Register map

All registers are 32 bits and word-aligned.

| Offset | Name | Access | Description |
| ---: | --- | --- | --- |
| `0x00` | `ID` | RO | ASCII `SNN1` (`0x534E4E31`) |
| `0x04` | `VERSION` | RO | Interface version; handover reference `1.6` with explicit executor architecture and phase telemetry |
| `0x08` | `CONTROL` | WO | Command bits described below |
| `0x0C` | `STATUS` | RO | Runtime/status bits described below |
| `0x10` | `RESULT` | RO | Final class index |
| `0x14` | `WEIGHT_CURSOR` | RW | Layer/pre/post cursor |
| `0x18` | `WEIGHT_DATA` | WO | INT8 write/commit and cursor advance |
| `0x1C` | `EVENT_CURSOR` | RW | Event-memory address |
| `0x20` | `EVENT_TIMESTAMP` | RW | Staged unsigned timestamp |
| `0x24` | `EVENT_NEURON` | WO | Neuron ID commit and cursor advance |
| `0x28` | `DIMENSIONS_0` | RO | `[31:16] hidden`, `[15:0] input` |
| `0x2C` | `DIMENSIONS_1` | RO | `[31:16] event-memory capacity`, `[15:0] output` |
| `0x30` | `NUMERIC_0` | RO | count/current/membrane/weight widths, one byte each |
| `0x34` | `NUMERIC_1` | RO | `[15:8] leak shift`, `[7:0] time width` |
| `0x38` | `THRESHOLD` | RO | Signed threshold parameter |
| `0x3C` | `RESET_POTENTIAL` | RO | Signed reset-potential parameter |
| `0x40` | `WEIGHT_CRC32` | RO | CRC32 of all accepted weight bytes |
| `0x44` | `EVENT_CRC32` | RO | CRC32 of all accepted event records |
| `0x48` | `WEIGHT_WRITE_COUNT` | RO | Number of accepted weights |
| `0x4C` | `EVENT_WRITE_COUNT` | RO | Number of accepted events |
| `0x50` | `ERROR` | RO | Sticky error code |
| `0x54` | `NEURON_MODELS` | RO | `[31:24]` Izh Q fraction bits, `[15:8]` output model, `[7:0]` hidden model |
| `0x58` | `IZH_AB` | RO | One byte each: `b_den`, `b_num`, `a_den`, `a_num` |
| `0x5C` | `IZH_RESET_VOLTAGE` | RO | Signed Izhikevich reset voltage `c` |
| `0x60` | `IZH_RECOVERY_INCREMENT` | RO | Signed recovery increment `d` |
| `0x64` | `IZH_SPIKE_THRESHOLD` | RO | Signed spike cutoff |
| `0x68` | `HH_Q_FORMAT` | RO | One byte each: LUT-step Q, rate Q, gate Q, voltage Q |
| `0x6C` | `HH_WIDTHS_STEPS` | RO | One byte each: internal steps, rate width, gate width, voltage width |
| `0x70` | `HH_LUT_RANGE` | RO | Signed 16-bit maximum/minimum LUT voltage |
| `0x74` | `HH_SPIKE_THRESHOLD` | RO | Signed HH event threshold in mV |
| `0x78` | `INFERENCE_EVENT_COUNT` | RW | Valid events for the next inference, from zero through capacity |
| `0x7C` | `TIMESTEP_COUNT` | RW | Positive number of network timesteps for the next inference |
| `0x80` | `INFERENCE_CYCLES_LO` | RO | Low 32 bits of exact `START`-to-`DONE` core cycles |
| `0x84` | `INFERENCE_CYCLES_HI` | RO | High 32 bits of exact `START`-to-`DONE` core cycles |
| `0x88` | `CONSUMED_INPUT_EVENTS` | RO | Input events accepted by layer 0 in the last inference |
| `0x8C` | `EMITTED_HIDDEN_SPIKES` | RO | Hidden events accepted by layer 1 in the last inference |
| `0x90` | `EMITTED_OUTPUT_SPIKES` | RO | Output spikes decoded in the last inference |
| `0x94` | `EXECUTOR_COUNTS` | RO | Interface 1.5+: `[31:16]` hidden and `[15:0]` output physical executor counts |
| `0x98` / `0x9C` | `PHASE_SYNAPSE_CYCLES_LO/HI` | RO | Interface 1.6+: input-to-hidden synapse/read/drain cycles |
| `0xA0` / `0xA4` | `PHASE_HIDDEN_UPDATE_CYCLES_LO/HI` | RO | Interface 1.6+: hidden population update cycles |
| `0xA8` / `0xAC` | `PHASE_HIDDEN_ROUTING_CYCLES_LO/HI` | RO | Interface 1.6+: hidden spike encoding and hidden-to-output accumulation cycles |
| `0xB0` / `0xB4` | `PHASE_OUTPUT_UPDATE_CYCLES_LO/HI` | RO | Interface 1.6+: output population update cycles |
| `0xB8` / `0xBC` | `PHASE_CONTROL_OVERHEAD_CYCLES_LO/HI` | RO | Interface 1.6+: output decoding, timestep advance and other controller cycles |

The Izhikevich parameter registers describe the synthesis-time parameter set
shared by both layers. A version-2 model package may select LIF, Izhikevich or
Hodgkin-Huxley independently, but every selected Izhikevich layer must match
these registers.
Interface 1.2 likewise exposes the fixed Hodgkin-Huxley numerical contract.
Interface 1.3 adds runtime stream length and timestep-horizon registers.
Interface 1.4 adds per-inference cycle/activity telemetry and replay-safe
`START` behavior. Interface 1.5 adds `EXECUTOR_COUNTS`, making the physical
population architecture auditable without changing numerical or control
semantics. Interface 1.6 adds five disjoint 64-bit phase counters. Their sum
must equal `INFERENCE_CYCLES`; the host rejects a snapshot that violates this
invariant. The host does not access newer registers on older images.
The host rejects an HH package against an older interface because model ID 2
alone cannot prove matching Q formats, LUT range or integration step count.

Unsupported writes return `DECERR` and set the sticky bad-register error.
Unsupported reads return `DECERR` without mutating state. Reads of write-only
registers return zero.

## Control and status

`CONTROL` uses write-one command bits in byte lane zero:

| Bit | Name | Effect |
| ---: | --- | --- |
| 0 | `START` | Start inference if the core is idle; on success acknowledge stale `DONE`/`RESULT_VALID` without clearing uploaded events |
| 1 | `CLEAR` | Clear result/error flags, CRC states and upload counters |

`CLEAR` does not erase BRAM weights or event memory. Starting inference clears
the transient neuron/counter state through the existing soft-reset path.

`STATUS` bits are:

| Bit | Name | Meaning |
| ---: | --- | --- |
| 0 | `IDLE` | Configuration/start is currently allowed |
| 1 | `BUSY` | Inference is active |
| 2 | `DONE` | Inference completion occurred; sticky until `CLEAR` |
| 3 | `RESULT_VALID` | `RESULT` is valid; sticky until `CLEAR` |
| 4 | `ERROR` | An error occurred; sticky until `CLEAR` |
| 5 | `WEIGHT_READY` | Core can accept a weight commit |
| 6 | `EVENT_READY` | Core can accept an event commit |

`irq` is asserted while `RESULT_VALID` or `ERROR` is set. Software acknowledges
it with `CONTROL.CLEAR`.

Starting another inference in interface 1.4 clears only stale completion and
result flags. It preserves the event-memory contents, event count/CRC,
configured horizon and weights. Consequently, software may upload one stream
once and issue repeated `START` commands for a sustained measurement batch.
`CONTROL.CLEAR` retains its stronger upload-verification reset semantics and
therefore requires a fresh event upload before another start.

Error codes are `1=bad register`, `2=not ready`, `3=bad index`,
`4=bad write strobe`, and `5=bad inference configuration`. A failed write
receives `SLVERR` except for an unsupported register, which receives `DECERR`.

## Weight upload

`WEIGHT_CURSOR` is packed as:

```text
bit 31      layer (0=input→hidden, 1=hidden→output)
bits 30:16  presynaptic index
bits 15:0   postsynaptic index
```

Write a cursor once and stream signed bytes to `WEIGHT_DATA`. Each accepted byte
is committed to the core and advances the post index; at the row boundary the
post index returns to zero and the pre index increments. This matches the
pre-major/post-minor order of `weights_l0.bin` and `weights_l1.bin`.

For the default MNIST topology, software must observe:

```text
WEIGHT_WRITE_COUNT = 784*256 + 256*10 = 203264
```

The combined expected CRC is stored as `upload.weight_crc32` in the
`snn-int8-v2` manifest. It is the standard reflected IEEE CRC32 returned by
Python `zlib.crc32(weights_l0 + weights_l1)`.

## Event upload

Set `EVENT_CURSOR`, write `EVENT_TIMESTAMP`, then write the neuron ID to
`EVENT_NEURON` using byte lanes zero and one. The neuron write commits the event
and advances the cursor. `NUM_EVENTS` is the physical memory capacity, not the
runtime stream length. Software writes the actual number of committed events to
`INFERENCE_EVENT_COUNT`; zero is valid. It also writes the positive simulation
horizon to `TIMESTEP_COUNT`, so trailing timesteps without input spikes are
still evaluated. `CONTROL.START` is rejected unless the committed event count
equals `INFERENCE_EVENT_COUNT`, the count fits the capacity, and the horizon is
nonzero.

The event CRC processes six little-endian bytes per accepted event:

```text
timestamp[7:0], timestamp[15:8], timestamp[23:16], timestamp[31:24],
neuron_id[7:0], neuron_id[15:8]
```

The canonical Python helpers live in `python/snn_host/registers.py`.

## Required software sequence

1. Read `ID`, `VERSION`, dimensions, numerical and neuron-model registers.
2. Reject a model whose manifest does not exactly match those values.
3. Write `CONTROL.CLEAR`.
4. Set the layer-0 cursor and stream `weights_l0.bin`.
5. Set the layer-1 cursor and stream `weights_l1.bin`.
6. Compare weight count and CRC with the manifest.
7. Write `INFERENCE_EVENT_COUNT` and `TIMESTEP_COUNT`.
8. Set the event cursor and upload the sorted encoded events.
9. Compare event count and CRC with the locally encoded stream.
10. Write `CONTROL.START`.
11. Wait for `irq` or poll `STATUS.DONE`/`STATUS.ERROR`.
12. Read `RESULT` when `RESULT_VALID` is set, then acknowledge with `CONTROL.CLEAR`.

For an interface-1.4 replay measurement, replace the final `CLEAR` with another
`START`, wait for the new `DONE`, and read registers `0x80` through `0x90`.
The counters reset on every accepted `START` and remain stable after `DONE`.

Any upload error requires `CLEAR` and a complete upload restart, because CRC and
count state intentionally describe one uninterrupted stream.

## Current limitations

- The wrapper and SNN core share one clock/reset domain. No clock-domain-crossing
  (CDC) bridge is provided between them. A future design with separate clocks
  must add appropriate synchronization, handshaking or asynchronous FIFOs to
  prevent metastability and inconsistent data transfer. This statement applies
  only to the wrapper/core boundary; the surrounding U50 platform has multiple
  clock domains and separate [CDC findings](timing_review.md).
- AXI4-Lite serial loading is functional but relatively slow for 203,264
  weights. DMA or a burst-capable memory window is a later optimization.
- Weight readback is not exposed; accepted-write count plus CRC32 provides
  transfer verification.
- The Vitis RTL kernel is packaged by `vivado/u50/package_u50_kernel.tcl`.
  XRT resolves the kernel address; `python/snn_host` uses the persistent
  `xrt-register-server` bridge. Final qualification requires the physical U50.
