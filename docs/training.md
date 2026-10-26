# Exploring training data and creating new models

## Training explorer

Open the [hosted training explorer](https://ankilab.github.io/snn-fpga/explorer/training/)
in your browser. For offline use, download or clone the repository and open
`explorer/training/index.html`. The HTML file contains all data and requires
neither a server nor internet for offline viewing.
Its colors automatically follow the system light/dark setting, including
changes made while the page is open.
It shows **66 fits and 3,366 epoch records including baselines**: LIF, IZH and
HH with 22 encoding variants each, 50 epochs and training seed 14.
The three reference models are preselected; up to six fits can be compared.

Features include filters by neuron model and encoding, learning curves,
diagnostics, timing components, parameter inspection and CSV/JSON export.
Test values are final endpoints. A single seed cannot estimate variability
across training initializations. Different training recipes prevent a
comparison that isolates the effect of the neuron model alone.

```bash
python explorer/training/build_dashboard.py
node --check explorer/training/dashboard.js
```

The builder reads only the supplied manifests and epoch metrics. It compares
both representations of the history and requires exactly 66 fits with a
baseline plus 50 epochs. It performs no training and changes no original data.

## Neuron states and spike frequencies

The [neuron state explorer](https://ankilab.github.io/snn-fpga/explorer/neuron_state/) displays
recorded states from bit-accurate software inference and spike frequencies
using Viridis. Its [guide](../explorer/neuron_state/README.md) explains the
data scope, online and offline use, and customization.

## Data scope and selection

Evidence is stored in `results/full-encoding-50epoch-2026-09-11/`.
Anonymized `/historical-workspace` paths record provenance and are not local
commands. `configs/references.json` contains checksums for the supplied model
packages. The [hardware results](../evidence/u50-reference-2026-09-23/summary.json)
map tested model hashes to published model hashes; weight files and numerical
measurements are unchanged.

MNIST was split into 55,000 training, 5,000 validation and 10,000 test images.
The checkpoint and reference package are selected using validation results.
The entire campaign runs for 50 epochs; the selected checkpoint may be earlier.

| Reference | Encoding | Checkpoint | Validation | Reported training test |
|---|---|---:|---:|---:|
| LIF | deterministic rate, scale 1 | 50 | 97.80% | 97.38% |
| IZH | Bernoulli rate, scale 1, encoding seed 1 | 34 | 98.30% | 97.68% |
| HH | deterministic rate, scale 0.25 | 47 | 86.38% | 84.00% |

The six corresponding INT8 weight files are included. Other fits contain
evaluation results but no weights. Recovery states and complete per-epoch
weight archives are intentionally excluded from this repository; the supplied
campaign cannot be resumed at an old intermediate state from these files.
New training runs create their own resume files.

## Training environment and MNIST

Python 3.11+ is sufficient for the host/reference. Training was tested locally
with Python 3.12, Torch 2.13.0 and Torchvision 0.28.0; the CPU versions suffice
for the small tests. The full HH campaign requires Linux, a suitable NVIDIA
GPU, a matching CUDA-enabled PyTorch installation and the compiler backend
path. CPU checks do not validate this GPU environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements-training.txt
python tools/run_checks.py --suite python --training
```

Select a PyTorch distribution matching your CUDA version; version pins alone
do not guarantee a CUDA backend. Do not replace an existing CUDA installation
with a CPU distribution without checking. For the 50-epoch campaign,
`torch.cuda.is_available()` must return true.

Download MNIST separately and check the extracted IDX files:

```bash
python -c "from torchvision.datasets import MNIST; MNIST('data', train=True, download=True); MNIST('data', train=False, download=True)"
python tools/check_dataset.py data/MNIST/raw
```

## Starting a new run

The neutral training matrix is `configs/full_encoding_50epoch_matrix.json`.
Use a new output directory rather than the supplied results archive:

```bash
python tools/train_campaign.py --output runs/my-campaign --data-dir data --mode prepare
python tools/train_campaign.py --output runs/my-campaign --data-dir data --mode pilot
python tools/train_campaign.py --output runs/my-campaign --data-dir data --mode run
```

`prepare` freezes the plan, source fingerprint and dataset hashes. `pilot`
checks a short HH run. `run` executes all 66 fits and resumes existing
checkpoints from your run. After changing the sources, use a new run directory.
The full sweep is a GPU workload lasting several days, not an introductory
test. The controller records temperature and runtime and executes serially.

For a single model, use `python -m snn_training.train_mnist --help` with
`PYTHONPATH=python`. The training CLI, QAT model and INT8 export are included.
`python -m snn_training.evaluate_export MODEL --data-dir data --samples 10`
evaluates exported weights with the bit-accurate reference and requires the
training libraries to access MNIST. Before FPGA use, always check the model
contract and then the hardware gate in the [U50 guide](u50.md).
