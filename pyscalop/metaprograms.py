"""Aggregate per-sample programs into cross-sample metaprograms.

Port of scalop::metaprograms. For each metacluster (a grouping of per-sample
programs), pool the constituent programs' top genes and their LFC profiles,
keep genes that recur across enough samples, and order by recurrence × LFC.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def metaprograms(
    programs: Mapping[str, Sequence[str]],
    profiles: Mapping[str, pd.Series],
    metaclusters: Mapping[str, Sequence[str]],
    *,
    samples: Sequence[str] | None = None,
    freq_cutoff: int = 3,
    order_priority: str = "freq",
    return_data: bool = False,
) -> dict[str, list[str]] | tuple[dict[str, list[str]], pd.DataFrame]:
    """Compute metaprograms from per-sample programs and their LFC profiles.

    Parameters
    ----------
    programs : dict[program_name -> gene list]
        Per-sample programs (e.g. output of ``programs(...)['programs']`` for
        each sample, then merged).
    profiles : dict[program_name -> Series(gene -> lfc)]
        Full LFC profile of each program.
    metaclusters : dict[metaprogram_name -> list of program_names]
        Which programs go together (e.g. from clustering program-program
        similarity).
    samples : list[str], optional
        Sample IDs to extract from program names. If None, inferred from
        ``programs`` keys.
    freq_cutoff : int
        Minimum number of distinct samples a gene must appear in to be kept.
    order_priority : {'freq', 'lfc'}
        Whether to sort genes within a metaprogram by recurrence first or LFC
        first.

    Returns
    -------
    dict[metaprogram_name -> gene list]   (and optionally the underlying df)
    """
    if samples is None:
        samples = list(programs.keys())

    # allowed (gene, program) pairs — the gene must be in that program's top set
    allowed = pd.DataFrame(
        [(g, name) for name, genes in programs.items() for g in genes],
        columns=["gene", "program"],
    )

    # mp -> programs
    mp_df = pd.DataFrame(
        [(mp, prog) for mp, progs in metaclusters.items() for prog in progs],
        columns=["mp", "program"],
    )

    # full LFC table
    lfc_long = pd.concat(
        [s.rename("lfc").to_frame().assign(program=name).reset_index()
         for name, s in profiles.items()],
        ignore_index=True,
    ).rename(columns={"index": "gene"})

    df = mp_df.merge(lfc_long, on="program", how="outer").merge(allowed, on=["program", "gene"], how="inner")

    # extract sample id from program name
    sample_re = re.compile("|".join(re.escape(s) for s in samples))
    df["sample"] = df["program"].apply(lambda x: (m := sample_re.search(x)) and m.group(0))

    # mean lfc per (mp, sample, gene)
    df = (df.groupby(["mp", "sample", "gene"], as_index=False)
            .agg(tum_lfc=("lfc", "mean")))

    df["freq"] = df.groupby(["mp", "gene"])["sample"].transform("nunique")
    df = df.groupby(["mp", "gene"], as_index=False).agg(
        mp_lfc=("tum_lfc", "mean"), freq=("freq", "first"),
    )
    df = df.loc[df["freq"] >= freq_cutoff].copy()

    if order_priority == "freq":
        df = df.sort_values(["mp", "freq", "mp_lfc"], ascending=[True, False, False])
    else:
        df = df.sort_values(["mp", "mp_lfc"], ascending=[True, False])

    mp = {name: g["gene"].tolist() for name, g in df.groupby("mp", sort=False)}
    if return_data:
        return mp, df
    return mp
