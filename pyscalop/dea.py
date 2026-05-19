"""Differential expression analysis (port of scalop::dea).

Per-gene t-test of group cells vs the rest (or vs an explicit second group),
returning log2 fold-changes and BH-adjusted p-values.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .utils import as_dataframe, rowcenter


def dea(
    m,
    group: Sequence[str] | Mapping[str, Sequence[str]],
    group2: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    *,
    lfc: float | None = np.log2(2),
    p: float | None = 1e-2,
    pmethod: str = "fdr_bh",
    alternative: str = "greater",
    arrange_by: str = "lfc",
    return_val: str = "df",
    center_rows: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Per-gene differential expression for one or more cell groups.

    Parameters
    ----------
    m : DataFrame | ndarray | AnnData
        Expression matrix, genes x cells (log-normalised).
    group, group2
        Either a single list of cell IDs (compares vs the rest of m's columns,
        or vs group2 if provided) or a dict of named groups.
    lfc, p
        Filtering cutoffs; pass None to skip.
    pmethod
        Adjustment method (statsmodels naming, e.g. 'fdr_bh', 'bonferroni').
    alternative
        't-test alternative: 'greater' is the typical "this group up vs rest".
    arrange_by : {'lfc', 'p', 'none'}
    return_val : {'df', 'lfc', 'p', 'gene'}
    """
    m = as_dataframe(m)
    groups = _as_named_groups(group, default="group")
    groups2 = _as_named_groups(group2, default="group2") if group2 is not None else None

    out = {}
    for name, cells in groups.items():
        if groups2 is None:
            out[name] = _dea_one(
                m, cells, None,
                lfc=lfc, p=p, pmethod=pmethod,
                alternative=alternative, arrange_by=arrange_by,
                return_val=return_val, center_rows=center_rows,
            )
        else:
            for name2, cells2 in groups2.items():
                key = f"{name}__vs__{name2}"
                out[key] = _dea_one(
                    m, cells, cells2,
                    lfc=lfc, p=p, pmethod=pmethod,
                    alternative=alternative, arrange_by=arrange_by,
                    return_val=return_val, center_rows=center_rows,
                )

    if len(out) == 1:
        return next(iter(out.values()))
    return out


def _as_named_groups(g, default: str) -> dict[str, list[str]]:
    if isinstance(g, Mapping):
        return {k: list(v) for k, v in g.items()}
    return {default: list(g)}


def _dea_one(
    m: pd.DataFrame, cells, cells2,
    *, lfc, p, pmethod, alternative, arrange_by, return_val, center_rows,
) -> pd.DataFrame:
    cells = list(cells)
    if cells2 is None:
        cells2 = [c for c in m.columns if c not in set(cells)]
    cells2 = list(cells2)

    sub = m.loc[:, cells + cells2]
    if center_rows:
        sub = rowcenter(sub)

    a = sub.loc[:, cells].values
    b = sub.loc[:, cells2].values

    # log2 fold-change on the (possibly centered) values: mean(a) - mean(b).
    # Since rowcenter subtracts the same row mean from both groups, this is
    # equivalent to the raw difference of group means in log-space.
    lfc_vec = a.mean(axis=1) - b.mean(axis=1)

    # Welch t-test, vectorised
    tstat, pval = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="propagate")
    if alternative == "greater":
        pval = np.where(tstat > 0, pval / 2, 1 - pval / 2)
    elif alternative == "less":
        pval = np.where(tstat < 0, pval / 2, 1 - pval / 2)
    # else two-sided already

    # Adjust
    try:
        from statsmodels.stats.multitest import multipletests
        _, padj, *_ = multipletests(np.nan_to_num(pval, nan=1.0), method=pmethod)
    except ImportError:
        from scipy.stats import false_discovery_control
        padj = false_discovery_control(np.nan_to_num(pval, nan=1.0), method="bh")

    df = pd.DataFrame({
        "gene": m.index,
        "foldchange": lfc_vec,
        "t": tstat,
        "p": pval,
        "p_adj": padj,
    })

    if lfc is not None:
        df = df.loc[df["foldchange"] >= lfc]
    if p is not None:
        df = df.loc[df["p_adj"] <= p]
    if arrange_by == "lfc":
        df = df.sort_values("foldchange", ascending=False)
    elif arrange_by == "p":
        df = df.sort_values("p_adj")

    if return_val == "df":
        return df.reset_index(drop=True)
    if return_val == "gene":
        return df["gene"].tolist()
    if return_val == "lfc":
        return df.set_index("gene")["foldchange"]
    if return_val == "p":
        return df.set_index("gene")["p_adj"]
    raise ValueError(f"Unknown return_val: {return_val}")
