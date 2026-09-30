# Neuron state explorer

Open [index.html](index.html) directly in a browser after downloading.
**Keep the entire folder together:** `data.js` and `states/` are required.
Viewing requires no server, Python installation or internet connection.
A modern browser supporting `DecompressionStream('gzip')` is required for
the state display. GitHub's source view does not execute HTML; download or
clone the repository first.

The explorer shows the 50-epoch campaign: 22 encoding variants, LIF/IZH/HH and
ten digits. The data contains 880 traces, including 660 with neuron states
from bit-accurate software inference. These states are not measurements of
internal FPGA signals. Large state files load only when a model and digit
are selected; the most recently decoded selection is cached in memory.

Spike frequency uses Viridis: dark purple represents zero spikes and yellow
represents a spike at every time step. For all layers, the scale is normalized
to `spikes / time steps`, not the observed maximum. The legend and PNG export
use the same scale. Grayscale is also available. The Viridis color values
match the Matplotlib colormap.

## Customizing the view

`viewer.template.html` contains the editable presentation and JavaScript.
Rebuild the HTML from this template with Python 3.11+:

```bash
python explorer/neuron_state/build_viewer.py
```

The supplied data remains unchanged. `data_manifest.json` contains its
checksums. To import a complete standalone explorer HTML file, additionally
pass `--source /path/to/neuron_state_explorer.html`. Data is split losslessly
into local JavaScript files so that no file exceeds GitHub's 100 MiB limit.
The total size remains large because all recorded states are retained.
Git LFS and a separate release download are not required.

Fully recomputing the states requires all 66 model packages. This repository
includes weights for only the three reference models; the build script
processes recorded data and does not rerun inference.
