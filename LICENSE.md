# Project licenses

Copyright (c) 2026 the SNN U50 project author and contributors.

Original project components are offered under the following licenses as of
this release. Licenses are assigned by file; there is **no general
choice between the two licenses**. The full English license texts govern.

## Hardware design: CERN-OHL-S-2.0

The following files are Covered Source under the
[CERN Open Hardware Licence Version 2 – Strongly Reciprocal](LICENSES/CERN-OHL-S-2.0.txt):

- All `*.sv` files in `rtl/` and all SystemVerilog testbenches in `tb/`.
- `data/hh_luts/*.mem`.
- `deploy/u50/compat/*.v`, to the extent that original rights exist.
- `vivado/u50/snn_kernel.xml` as the hardware interface description.
- `docs/architecture.md`, `docs/numerical_specification.md`,
  `docs/ps_pl_interface.md`, and the diagram sources and SVGs in `docs/images/`.

This explicitly means **version 2, variant S**; this project does not grant
an additional “or any later version” option.

## Software and other original content: GPL-3.0-only

All remaining original project files are licensed under the
[GNU General Public License, Version 3](LICENSES/GPL-3.0-only.txt),
**version 3 only**. These include:

- Python reference, training, host software, tests and utilities;
- C++ XRT helpers, shell, PowerShell, CMD and Tcl scripts;
- Explorers, including generators, JavaScript, HTML and prepared data;
- Configurations and remaining documentation;
- Original model weights, training metrics, manifests and synthetic examples,
  in each case to the extent that corresponding rights exist.

The preferred editable form of the diagrams is provided as `.mmd` files;
the explorers include their generator, template and JavaScript sources.
Model provenance, training recipes and export are described in the documentation.
These license assignments do not assert that individual measurements or
AI-generated components are independently protected by copyright.

## Practical effect of this choice

Both licenses permit use, modification and commercial redistribution under
their terms. Their copyleft requirements help keep distributed developments
available for others to study, reproduce and modify.

- **CERN-OHL-S-2.0** addresses hardware design sources and products made from
  them. Its strong reciprocity can extend to a larger design incorporating
  Covered Source, subject to the Available Component provisions. Recipients
  of covered products must receive Complete Source or notice of its Source
  Location as required by the license.
- **GPL-3.0-only** protects the same freedoms for software. Distributing a
  covered derivative requires complying with GPLv3, including the applicable
  Corresponding Source obligations for binaries. This can prevent distribution
  of a combined derivative as proprietary software; it does not automatically
  apply to independent programs merely used alongside it.

These obligations can make integration into proprietary products more
demanding than with permissive licenses. Neither license prohibits selling
the project or requires contributors to submit changes to this repository.
Private modifications generally need not be published. GPLv3 does not impose
a source-sharing obligation solely because modified software is used to run
a network service; distributing client-side code is a separate matter.
The licenses provide no general promise of maintenance or support.

The intended benefit is openness of distributed derivatives, not mandatory
publication of every private experiment. The exact obligations and exceptions
are defined by the full license texts.

## Scope and redistribution

License texts under `LICENSES/` retain their own copyright and redistribution
notices. The project licenses do not relicense those texts. Existing
third-party rights remain unaffected. In particular, this declaration does not
place AMD tools, platform IP, XRT, Python libraries or MNIST under a project
license. Historical tool reports also contain outputs and notices from
third-party tools; this declaration covers only original contributions and
original rights within them. See [NOTICE.md](NOTICE.md).

Copyleft does not require every private modification to be published on GitHub.
When distributing material, the source, notice and other obligations of the
applicable license apply. Hardware products and bitstream releases require
particular attention to providing Complete Source under CERN-OHL-S; software
binaries require Corresponding Source under GPLv3. This overview does not
replace the license texts.

Before distributing a corresponding product or bitstream, assess which parts
of the external AMD platform qualify as Available Components under
CERN-OHL-S 1.7/1.8 and which sources, procurement details and interface
information must accompany them. Merely using external tools does not provide
blanket compatibility clearance for incorporated IP.

The machine-readable file assignments are in [licensing.json](LICENSES/licensing.json).
Include that file and this overview when redistributing the sources so that
the intentionally unchanged RTL and result files remain clearly assigned.
A public Source Location link for releases will be added once the actual
repository destination has been selected.
