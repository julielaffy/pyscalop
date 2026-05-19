"""Smoke tests: synthetic data exercising each module end-to-end."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyscalop as ps


def _toy_matrix(n_genes: int = 200, n_cells: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = [f"g{i:04d}" for i in range(n_genes)]
    cells = [f"c{i:03d}" for i in range(n_cells)]
    M = rng.normal(loc=2.0, scale=1.0, size=(n_genes, n_cells))
    # plant a real signal: first 20 genes elevated in first 30 cells
    M[:20, :30] += 3.0
    return pd.DataFrame(M, index=genes, columns=cells)


def test_sig_scores_raw():
    """Without bin-matched controls, the planted signal should be obvious."""
    m = _toy_matrix()
    sigs = {"planted": [f"g{i:04d}" for i in range(20)]}
    scores = ps.sig_scores(m, sigs, expr_center=False)
    assert scores.shape == (80, 1)
    high = scores.loc[m.columns[:30], "planted"].mean()
    low = scores.loc[m.columns[30:], "planted"].mean()
    assert high > low


def test_sig_scores_with_controls():
    """With bin-matched controls and a realistic gene count, planted still wins."""
    m = _toy_matrix(n_genes=2000, n_cells=80)
    sigs = {"planted": [f"g{i:04d}" for i in range(20)],
            "random":  [f"g{i:04d}" for i in range(500, 520)]}
    scores = ps.sig_scores(m, sigs, expr_nbin=30, expr_binsize=50, random_state=0)
    assert scores.shape == (80, 2)
    high = scores.loc[m.columns[:30], "planted"].mean()
    low = scores.loc[m.columns[30:], "planted"].mean()
    assert high > low


def test_sig_scores_permute():
    m = _toy_matrix(n_genes=120, n_cells=60)
    sigs = {"planted": [f"g{i:04d}" for i in range(20)]}
    out = ps.sig_scores(m, sigs, permute=True, n_perm=10, random_state=0, perm_batch=5)
    assert set(out.columns) == {"id", "sig", "score", "fdr"}
    assert len(out) == 60  # cells x sigs


def test_dea_runs():
    m = _toy_matrix()
    cells_a = list(m.columns[:30])
    df = ps.dea(m, group=cells_a, lfc=None, p=None)
    assert {"gene", "foldchange", "p", "p_adj"}.issubset(df.columns)
    # planted genes should have positive fold changes
    planted = df.set_index("gene").loc[[f"g{i:04d}" for i in range(20)], "foldchange"]
    assert (planted > 0).all()


def test_hca_groups_returns_cells():
    m = _toy_matrix()
    groups = ps.hca_groups(m, k=3, min_size=2, max_size=0.9)
    flat = [c for g in groups for c in g]
    assert set(flat).issubset(set(m.columns))


def test_thermogenic_signatures_load():
    via_attr = ps.signatures.thermogenic         # PEP 562 attribute access
    via_load = ps.signatures.load("thermogenic")  # function call by name
    assert via_attr is via_load                   # both go through the cache
    assert "lit.thermogenic" in via_attr
    excluded = {
        "REACTOME_White_adipocyte_differentiation_R-HSA-381340",
        "C5.GOBP_REGULATION_OF_BROWN_FAT_CELL_DIFFERENTIATION",
        "C5.GOBP_POSITIVE_REGULATION_OF_BROWN_FAT_CELL_DIFFERENTIATION",
    }
    assert not (excluded & via_attr.keys())
    assert "thermogenic" in ps.signatures.list_available()


def test_programs_finds_planted_cluster():
    m = _toy_matrix()
    # give it the true partition so we don't rely on clustering recovering it
    groups = {"planted": list(m.columns[:30]), "other": list(m.columns[30:])}
    res = ps.programs(m, groups=groups, nsig1=10, nsig2=3)
    assert "planted" in res["programs"]
