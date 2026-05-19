"""Convert ThermogenicSignatures.xlsx → pyscalop/signatures_data/thermogenic.gmt.

Reads the raw xlsx (one column per signature, NaN-padded), drops the three
excluded signature sets, writes a Broad GMT (name<TAB>desc<TAB>genes...).

Run from the repo root:
    python analysis/prep_thermogenic_gmt.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "thermogenic_raw.xlsx"
OUT = REPO / "pyscalop" / "signatures_data" / "thermogenic.gmt"

EXCLUDE = {
    "REACTOME_White_adipocyte_differentiation_R-HSA-381340",
    "C5.GOBP_REGULATION_OF_BROWN_FAT_CELL_DIFFERENTIATION",
    "C5.GOBP_POSITIVE_REGULATION_OF_BROWN_FAT_CELL_DIFFERENTIATION",
}

DESCRIPTION = "thermogenic signature curated by J. Laffy"


def main() -> None:
    df = pd.read_excel(SRC, sheet_name="Signature Genes")
    sigs: dict[str, list[str]] = {}
    for col in df.columns:
        if col in EXCLUDE:
            continue
        genes = df[col].dropna().astype(str).str.strip()
        genes = [g for g in genes if g]
        if genes:
            sigs[col] = genes

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for name, genes in sigs.items():
            f.write(f"{name}\t{DESCRIPTION}\t" + "\t".join(genes) + "\n")

    print(f"Wrote {len(sigs)} signatures to {OUT.relative_to(REPO)}")
    for name, genes in sigs.items():
        print(f"  {name}\t{len(genes)} genes")


if __name__ == "__main__":
    main()
