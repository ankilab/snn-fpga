"""Generate the six synthesis ROMs used by hodgkin_huxley_neuron.sv."""

from __future__ import annotations

import argparse
from pathlib import Path

from snn_reference.hodgkin_huxley_fixed import (
    HodgkinHuxleyFixedConfig,
    build_rate_tables,
)


TABLE_NAMES = (
    "alpha_m",
    "beta_m",
    "alpha_h",
    "beta_h",
    "alpha_n",
    "beta_n",
)


def write_tables(output: Path, config: HodgkinHuxleyFixedConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    digits = (config.rate_width + 3) // 4
    mask = (1 << config.rate_width) - 1
    for name, table in zip(TABLE_NAMES, build_rate_tables(config), strict=True):
        text = "\n".join(f"{value & mask:0{digits}x}" for value in table) + "\n"
        (output / f"hh_{name}.mem").write_text(text, encoding="ascii", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "hh_luts",
    )
    args = parser.parse_args()
    config = HodgkinHuxleyFixedConfig()
    write_tables(args.output, config)
    print(f"wrote {len(TABLE_NAMES)} x {config.lut_size} entries to {args.output}")


if __name__ == "__main__":
    main()
