# Provenance and third-party components

Original hardware contributions are licensed under CERN-OHL-S-2.0; software
and other original content under GPL-3.0-only. File assignments are documented
in [LICENSE.md](LICENSE.md) and [licensing.json](LICENSES/licensing.json).

This project was developed as coursework with assistance from Codex.
The architecture, host software, training code and documentation are project
contributions. The HH LUTs can be generated with
`python/snn_verification/generate_hh_luts.py`; the diagrams include editable
Mermaid sources. Model weights and measurements originate from the documented
training campaign and U50 verification. The two synthetic example images were
created within the project.

The [clock adapter](docs/clock_adapter.md) was created within the
project on 2026-08-26 as a transparent replacement implementation. Original
rights in it are assigned to CERN-OHL-S-2.0. It contains no encrypted AMD
original code. No specific AMD confirmation of this optional workaround is
available; this absence is not treated as an established prohibition on
publication or use.

AMD tools, platform packages, XRT, license files and MNIST must be obtained
separately and are not included. PyTorch, Torchvision and NumPy are external
dependencies; their library sources are not included. Their respective terms
apply. The project licenses do not grant third-party rights. Tool reports and
license texts retain their original notices; the project licenses cover only
original project contributions within them.
