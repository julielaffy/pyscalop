"""Identify intra-sample expression programs (port of scalop::programs).

Hierarchical clustering of cells → DEA per cluster → filter clusters by number
of significant genes → jaccard-filter overlapping clusters. Returns clusters,
top-N program gene lists, and full LFC profiles.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .dea import dea
from .hca import hca_groups
from .utils import as_dataframe, jaccard_filter, rowcenter


def programs(
    m,
    groups: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None = None,
    *,
    nsig1: int = 50,
    nsig2: int = 10,
    jaccard: float = 0.7,
    p: float = 0.01,
    lfc: float = math.log2(2),
    pmethod: str = "fdr_bh",
) -> dict:
    """Find cell clusters with robust differential expression.

    Returns
    -------
    dict with keys:
        programs : dict[str, list[str]]  top-N genes per cluster
        profiles : dict[str, Series]     full per-gene LFC per cluster
        groups   : dict[str, list[str]]  retained cell clusters
        deas     : dict[str, DataFrame]  full DEA tables per cluster
    """
    m = as_dataframe(m)
    if groups is None:
        clusters = hca_groups(rowcenter(m))
        groups_dict = {f"c{i+1}": g for i, g in enumerate(clusters)}
    elif isinstance(groups, Mapping):
        groups_dict = {k: list(v) for k, v in groups.items()}
    else:
        groups_dict = {f"c{i+1}": list(g) for i, g in enumerate(groups)}

    deas = dea(m, group=groups_dict, p=p, lfc=lfc, pmethod=pmethod,
               arrange_by="lfc", return_val="df")
    # ``dea`` returns a single DataFrame when only one group; normalise.
    if isinstance(deas, pd.DataFrame):
        deas = {next(iter(groups_dict)): deas}

    sig1 = {n: int((df["p_adj"] <= 0.01).sum()) for n, df in deas.items()}
    sig2 = {n: int((df["p_adj"] <= 0.001).sum()) for n, df in deas.items()}
    sig3 = {n: int((df["p_adj"] <= 0.0001).sum()) for n, df in deas.items()}

    keep = [n for n in deas if sig1[n] >= nsig1 and sig2[n] >= nsig2]
    keep.sort(key=lambda n: (-sig1[n], -sig2[n], -sig3[n]))

    # Jaccard-filter on the top program genes (mirrors scalop)
    top = {n: deas[n]["gene"].head(nsig1).tolist() for n in keep}
    kept = jaccard_filter(top, threshold=jaccard)

    final_groups = {n: groups_dict[n] for n in kept}
    profiles = dea(m, group=final_groups, p=None, lfc=None, pmethod=pmethod,
                   arrange_by="none", return_val="lfc")
    if isinstance(profiles, pd.Series):
        profiles = {next(iter(final_groups)): profiles}

    return {
        "programs": {n: top[n] for n in kept},
        "profiles": profiles,
        "groups": final_groups,
        "deas": {n: deas[n] for n in kept},
    }
