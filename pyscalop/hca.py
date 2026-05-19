"""Hierarchical clustering (port of scalop::hca and friends).

The R package exposes a family of functions: ``hca``, ``hca_groups``,
``hca_tree``, ``hca_order``, ``hca_reorder``. Here we provide the same set
backed by ``scipy.cluster.hierarchy``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist, squareform

from .utils import as_dataframe


def _correlation_distance(m: pd.DataFrame, method: str = "pearson") -> np.ndarray:
    """Return condensed distance vector = 1 - correlation between columns."""
    cor = m.corr(method=method)
    d = 1 - cor.values
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2  # symmetrise tiny float asymmetries
    return squareform(d, checks=False)


def hca(
    x,
    *,
    cor_method: str | None = "pearson",
    dist_method: str = "euclidean",
    cluster_method: str = "average",
    h: float | None = None,
    k: int | None = None,
    min_size: int | float = 5,
    max_size: int | float = 0.5,
) -> dict:
    """Hierarchical clustering analysis.

    Returns a dict with keys ``tree`` (linkage matrix), ``order``, ``groups``
    (list of cluster-membership lists) and ``cr`` (correlation matrix if used).
    """
    x = as_dataframe(x)
    n = x.shape[1]

    cr = None
    if cor_method and cor_method != "none":
        cr = x.corr(method=cor_method)
        if dist_method == "none":
            condensed = squareform(1 - cr.values, checks=False)
        else:
            condensed = _correlation_distance(x, method=cor_method)
    else:
        condensed = pdist(x.T.values, metric=dist_method)

    Z = sch.linkage(condensed, method=cluster_method)
    order = sch.leaves_list(Z)
    leaf_labels = [x.columns[i] for i in order]

    if k is not None:
        labels = sch.fcluster(Z, t=k, criterion="maxclust")
    elif h is not None:
        labels = sch.fcluster(Z, t=h, criterion="distance")
    else:
        labels = sch.fcluster(Z, t=2, criterion="maxclust")

    # group cells by cluster label
    groups: dict[int, list[str]] = {}
    for lab, cell in zip(labels, x.columns):
        groups.setdefault(int(lab), []).append(cell)

    # filter by size
    min_n = int(np.ceil(min_size * n)) if 0 < min_size < 1 else int(min_size)
    max_n = int(np.floor(max_size * n)) if 0 < max_size <= 1 else int(max_size)
    groups = {k_: v for k_, v in groups.items() if min_n <= len(v) <= max_n}

    return {
        "tree": Z,
        "order": leaf_labels,
        "groups": list(groups.values()),
        "labels": labels,
        "cr": cr,
    }


def hca_groups(x, **kwargs) -> list[list[str]]:
    return hca(x, **kwargs)["groups"]


def hca_order(x, **kwargs) -> list[str]:
    return hca(x, **kwargs)["order"]
