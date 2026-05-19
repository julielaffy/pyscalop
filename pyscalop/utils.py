"""Shared helpers: matrix centering, jaccard, type coercion."""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_dataframe(m, gene_axis: str = "rows") -> pd.DataFrame:
    """Coerce m to a genes-by-cells DataFrame.

    Accepts DataFrame, ndarray, or AnnData. AnnData is transposed since its
    convention is cells-by-genes.
    """
    try:
        import anndata as ad
    except ImportError:
        ad = None

    if ad is not None and isinstance(m, ad.AnnData):
        x = m.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        return pd.DataFrame(x.T, index=m.var_names, columns=m.obs_names)

    if isinstance(m, pd.DataFrame):
        return m if gene_axis == "rows" else m.T

    if isinstance(m, np.ndarray):
        return pd.DataFrame(m)

    raise TypeError(f"Unsupported matrix type: {type(m).__name__}")


def rowcenter(m: pd.DataFrame) -> pd.DataFrame:
    """Subtract row means (gene-wise centering)."""
    return m.sub(m.mean(axis=1), axis=0)


def colcenter(m: pd.DataFrame) -> pd.DataFrame:
    return m.sub(m.mean(axis=0), axis=1)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_filter(groups: dict[str, list], threshold: float = 0.7) -> list[str]:
    """Greedy filter: drop later groups that overlap an earlier kept group above threshold.

    Order matters — keep input ordered by priority (e.g. by significance).
    Returns the names of kept groups.
    """
    names = list(groups)
    kept: list[str] = []
    for name in names:
        if all(jaccard(groups[name], groups[k]) < threshold for k in kept):
            kept.append(name)
    return kept
