# Optional clock adapter for the Alveo U50

## Purpose and tested combination

Platform `xilinx_u50_gen3x16_xdma_5_202210_1` contains an encrypted
`clk_metadata_adapter` that failed with `Synth 8-5809` in the tested
Vivado/Vitis 2026.1 build. The replacement transparently connects `clk_in`
to `clk_out`. All three reference models were built with it and tested on
the U50 on 2026-09-23. Neither its necessity nor its operation has been
confirmed for other combinations. The [timing findings](timing_review.md)
remain unresolved.

## Enabling the option

Replacement is disabled by default. Enable it for the combination above:

```bash
LEGACY_U50_CLOCK_ADAPTER_COMPAT=1 python tools/build_reference.py lif \
  --mode xclbin --platform "$PLATFORM"
```

`PLATFORM` must point to the locally installed `.xpfm`. For the
[unattended reference run](unattended_u50.md), export the same variable before
starting. The runner records this choice in its run identity. Accepted values
are `0` (off, default) and `1` (on); other values are rejected.

After `vpl.generate_target`, `deploy/u50/build.sh` replaces only matching files
in the generated build directory, backs up the originals as
`.encrypted-original`, and resumes at `vpl.config_hw_runs`. The installed
platform is unchanged. The replacement includes no decryptor and changes no
license keys.

## Provenance and licensing

`deploy/u50/compat/clk_metadata_adapter_v1_0_vl_rfs.v` was created by Codex as
a new project file on 2026-08-26. The archived creation record contains the
current nine-line source. Its hardware function is `assign clk_out = clk_in`.
The replacement is supplied under **CERN-OHL-S-2.0** to the extent that original
rights exist; see the [license assignments](../LICENSES/licensing.json).
This does not relicense the AMD IP block.

No specific AMD confirmation of this workaround is available. This absence
is not treated as an established prohibition on publication or use. Separately
obtained AMD tools and platform materials remain subject to their own terms.
AMD originals, backup copies, platform packages and bitstreams are not part
of this source publication.
