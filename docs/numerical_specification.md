# Numerical specification of the RTL SNN

This document is the software/RTL contract for the fixed two-layer inference
network implemented by `snn_top.sv`. The executable reference is
`python/snn_reference/model.py`.

## Neuron-model boundary

Hidden and output layers have independent, synthesis-time neuron-model IDs:

| ID | Model | Status |
| ---: | --- | --- |
| 0 | LIF | Implemented and default |
| 1 | Izhikevich | Bit-exact Python and complete two-layer RTL inference |
| 2 | Hodgkin-Huxley | Bit-exact Python and complete two-layer RTL inference |

The RTL `neuron_core` contract is `start`, `syn_current`, `spike`, `done`, and
`clear`. A model may take any number of clock cycles between `start` and
`done`; `spike` is a one-cycle pulse. The Python network runner uses the
corresponding population contract `step(currents) -> spikes`. Unsupported
model IDs fail explicitly instead of silently falling back to LIF.

Model ID 1 is qualified in the population reference, differentiable QAT path,
version-2 model export, host/hardware compatibility checks, individual RTL
neuron and complete two-layer RTL network.

Model ID 2 is supported in the fixed-point population reference, HH
quantization-aware training (QAT), version-2 model export, host/hardware
compatibility checks, individual RTL neuron and complete two-layer RTL network.
The handover references use interface 1.6.

HH QAT trains synaptic weights using a forward pass that follows the fixed-point
inference contract. The backward pass uses surrogate spike gradients,
straight-through estimators for discrete arithmetic, and interpolated LUT
gradients. These are gradient approximations, not exact derivatives of the
quantized operations. The implementation is
[`hh_qat.py`](../python/snn_training/hh_qat.py); see [training.md](training.md)
for the training workflow. The LUTs, Euler update and Q12-voltage/Q18-gate
contract are described below.

## Hodgkin-Huxley numerical contract

The original 6.3 °C squid-axon parameters use signed 24/Q12 voltage and
unsigned 20/Q18 gates. Six Q18 rate ROMs cover -32 through 128 mV in 0.25 mV
increments. A network timestep evaluates 100 explicit-Euler substeps, each
0.01 ms, with a constant integer current in uA/cm². Intermediate divisions use
nearest rounding with ties away from zero. Voltage saturates to its signed
24-bit range and gates saturate to `[0, 1]`; HH therefore deliberately differs
from the wrapping LIF and Izhikevich state contracts. A 20 mV upward crossing
creates an event but does not reset the four continuous states.

The hardware evaluates the same arithmetic order across registered phases per
0.01 ms substep. Exact divisions by 100 and 10 use iterative signed dividers
with nearest rounding and ties away from zero. A layer has one shared HH
executor and serialized V/m/h/n state storage; serialization changes clock
latency but not numerical timestep semantics or neuron independence.

## Izhikevich numerical contract

The initial implementation is the regular-spiking parameterization from the
2003 model: `a=0.02`, `b=0.2`, `c=-65 mV`, `d=8`, and spike cutoff `30 mV`.
To avoid coefficient-dependent floating-point behavior, `a` and `b` are stored
as rational numerator/denominator pairs (`1/50` and `1/5`). The coefficient
`0.04` is represented as the exact fraction `1/25`, avoiding a separately
quantized approximation of that coefficient. The resulting division still
truncates; this does not make the entire fixed-point update exact in real
arithmetic.

`v` and `u` use signed Q8 storage by default. Thus `-65 mV` is stored as
`-65 * 256 = -16640`. Synaptic current remains a signed integer at the common
layer interface and is multiplied by 256 inside the neuron. Every division is
signed and truncates toward zero. State assignments wrap to `MEM_WIDTH`.

In the pseudocode below, `v` and `u` are stored integers, `SCALE = 256`, and
`I` is the integer synaptic current. Squaring `v` produces a value with scale
`SCALE^2`, so the quadratic term must be divided by `25*SCALE` to return to
the state scale. For example, physical voltage -65 mV is stored as -16640;
its quadratic term is 43264 in Q8, representing 169. Other inputs can require
truncation. The quadratic quotient is truncated before forming the derivative,
and the half-step division is truncated separately.

One network timestep is evaluated in this exact order:

```text
q = trunc(v^2 / (25*SCALE))
v = wrap(v + trunc((q + 5*v + 140*SCALE - u + I*SCALE) / 2))
q = trunc(v^2 / (25*SCALE))
v = wrap(v + trunc((q + 5*v + 140*SCALE - u + I*SCALE) / 2))
u = wrap(u + trunc(a_num * (trunc(b_num*v/b_den) - u) / a_den))

if v >= 30*SCALE:
    spike = 1
    v = -65*SCALE
    u = wrap(u + 8*SCALE)
```

The voltage is advanced through two explicit Euler half-steps of 0.5 ms,
holding `u` and `I` fixed during both. The second half-step uses the updated
voltage from the first. Recovery `u` then advances through one 1 ms Euler step
using the voltage after both half-steps. Thus both states advance over the same
simulated millisecond; these durations do not describe FPGA execution time.

This integration scheme follows the two-half-step voltage pattern in
[Izhikevich's published example](https://izhikevich.org/publications/net.m).
Voltage changes rapidly near spike initiation, whereas recovery evolves more
slowly for the selected `a=0.02`. The smaller voltage steps improve numerical
behavior compared with one coarse voltage step. This is a numerical design
choice, not a requirement of the differential equations or a guarantee of
stability for arbitrary inputs. Changing the step sizes or update order would
change the numerical contract and require coordinated changes to training,
reference and RTL implementations.

Reset initializes `v=c` and `u=b*c`. The three constant divisions
use iterative signed dividers in truncation mode. This preserves the numerical
contract while limiting each divider timing path to a small
compare/subtract stage. `spike` is pulsed during the threshold/reset phase.

Each layer uses one shared Izhikevich executor and serialized per-neuron `v/u`
state storage. A population update takes approximately 280 clocks per neuron
plus fixed load/completion overhead. Serialization changes only clock latency:
every neuron still receives exactly one update in index order with its own
state and current. The array retains a threshold-phase spike until the shared
executor asserts `done`, so the serialized handshake is bit-exact with
independent parallel neuron instances.

## Network and storage layout

The implemented topology is fully connected and has exactly two trainable
weight matrices:

1. input to hidden: `INPUT_NEURONS × HIDDEN_NEURONS`
2. hidden to output: `HIDDEN_NEURONS × OUTPUT_NEURONS`

Matrices use pre-major order. In mathematical notation, the value stored at
`weights[pre][post]` connects presynaptic neuron `pre` to postsynaptic neuron
`post`. There are no biases.

The default RTL dimensions and widths are:

| Quantity | Default | Interpretation |
| --- | ---: | --- |
| Input/hidden/output neurons | 784 / 256 / 10 | Fixed MNIST topology |
| Weight | 8 bit | Signed two's complement |
| Synaptic current | 24 bit | Signed two's complement |
| Membrane | 24 bit | Signed two's complement |
| Class counter | 16 bit | Unsigned |
| Timestamp | 32 bit | Unsigned |
| Threshold | 100 | Signed membrane value |
| Reset potential | 0 | Signed membrane value |
| Leak shift | 3 | Arithmetic right shift amount |

The current RTL requires membrane and current widths to be equal.

## Event and timestep semantics

Input events are pairs `(timestamp, neuron_id)`. Event memory has a fixed
capacity, while `INFERENCE_EVENT_COUNT` specifies the valid entries for each
inference, from zero through that capacity. Entries must be sorted by
nondecreasing timestamp; multiple events at the same timestamp are allowed.
Neuron IDs must be within the input layer, and timestamps must lie within
the configured inference horizon.

Inference starts at timestep zero and processes `TIMESTEP_COUNT` timesteps,
including those with no input events and any silent timesteps after the last
event. The references use `TIMESTEP_COUNT = 8`. The Python reference accepts
the same explicit horizon through `timesteps`; only its legacy call without
that argument derives the horizon from the final event and requires a non-empty
stream. At each timestep, the network performs, in order:

1. accumulate all input-event weights into the hidden current vector;
2. update every hidden neuron once;
3. serialize all hidden spikes and accumulate their output-layer weights;
4. update every output neuron once;
5. increment the counters corresponding to output spikes.

Starting an inference clears membrane potentials, accumulated currents, spike
latches and class counters. It does not clear weights or input-event memory.

## Fixed-width arithmetic

For LIF and Izhikevich, signed values use two's-complement wrap-around. There
is no saturation in those two models. Hodgkin-Huxley uses the saturation rules
defined above.
Weight accumulation wraps to `CURRENT_WIDTH` after each addition. The membrane
equation is evaluated at `MEM_WIDTH`:

```text
leak       = membrane >>> LEAK_SHIFT
next_mem   = wrap_signed(membrane - leak + synaptic_current, MEM_WIDTH)
```

`>>>` is an arithmetic right shift. Negative values are sign-extended; this is
equivalent to division by `2**LEAK_SHIFT` rounded toward negative infinity.

The spike decision uses a signed comparison after wrapping:

```text
if next_mem >= THRESHOLD:
    spike = 1
    membrane = RESET_POT
else:
    spike = 0
    membrane = next_mem
```

Thus a positive arithmetic overflow can wrap to a negative value before the
threshold comparison and suppress a spike. Training and quantization must avoid
unintended overflow or deliberately model it.

## Output decoding

Output neuron index equals class index. Each output spike increments its
unsigned `COUNT_WIDTH` counter with wrap-around. The final prediction is the
index with the largest counter. Equal counts select the lowest class index.

If every final class counter is zero, `prediction_valid` remains low and no
prediction transaction is emitted. This also applies if counters have wrapped
back to zero.

## Reference-model boundary

The Python model is bit-accurate at timestep boundaries and for all numerical
state. It intentionally does not model clock-by-clock FSM latency, BRAM latency,
or valid/ready timing. Those protocol details remain the responsibility of the
RTL testbenches and later co-simulation tests.
