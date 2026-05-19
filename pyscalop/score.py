"""Signature scoring (port of scalop::sigScores) with optional permutation FDR.

Combines ``sigScores`` and ``permuteSigScores`` from the R package into one
function: pass ``permute=True`` to run permutation testing and get per-cell
per-signature FDR.

The permutation logic faithfully mirrors R's ``permuteSigScores``:
per iteration we permute expression values across cells (independently per
gene), compute scores on the permuted matrix, derive a within-iteration
across-cells threshold ``mean + 2*sd`` per signature, and tally per cell
whether the observed score beats that threshold. After ``n_perm`` iterations
we run a per-cell binomial test against the global success rate and BH-adjust
within each signature.

Bin assignments are computed once (row means are invariant under per-row
shuffling, so they don't change) but bin-matched control gene sets are
re-sampled per iteration to match R's behaviour. The shuffling and scoring
matmuls are batched / vectorised — that's where most of the speedup comes from.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import as_dataframe, rowcenter


SigDict = Mapping[str, Sequence[str]]


# ---------- Public API ----------

def sig_scores(
    m,
    sigs,
    *,
    groups: Mapping[str, Sequence[str]] | None = None,
    center_rows: bool = True,
    expr_center: bool = True,
    expr_bin_m=None,
    expr_nbin: int = 30,
    expr_binsize: int = 100,
    conserved_genes: float = 0.7,
    replace: bool = False,
    permute: bool = False,
    n_perm: int = 50,
    alternative: str = "greater",
    random_state: int | None = None,
    perm_batch: int = 5,
) -> pd.DataFrame:
    """Score cells against gene signatures.

    Parameters
    ----------
    m : DataFrame | ndarray | AnnData
        Expression matrix of genes x cells. Not row-centered (will be centered
        internally if ``center_rows=True``).
    sigs : list[str] or dict[str, list[str]]
        A single signature (list of gene names) or a dict of named signatures.
    groups : dict[str, list[str]], optional
        Cell IDs grouped by sample for intra-sample scoring. If provided, the
        matrix is split and each sample is scored independently.
    expr_center : bool
        Subtract bin-matched control signature scores (the Tirosh-lab method).
    permute : bool
        If True, also compute permutation FDR by shuffling expression values
        across cells independently per gene; returns a long DataFrame.
    n_perm : int
        Number of permutations when ``permute=True``.
    alternative : {'greater', 'less', 'two.sided'}
        Direction of test against the permutation null.

    Returns
    -------
    DataFrame
        If ``permute=False``: cells x signatures of scores.
        If ``permute=True``: long DataFrame with columns ``[id, sig, score, fdr]``.
    """
    m = as_dataframe(m)
    if isinstance(sigs, (list, tuple, np.ndarray)) and not isinstance(sigs[0], (list, tuple)):
        sigs = {"sig1": list(sigs)}
    sigs = dict(sigs)
    sigs = filter_sigs(sigs, ref=m.index, conserved=conserved_genes)
    if not sigs:
        raise ValueError("No signatures left after filtering against matrix rownames.")

    if groups is not None:
        per_sample = []
        for name, cells in groups.items():
            sub = m.loc[:, list(cells)]
            s = _sig_scores_one(
                sub, sigs,
                center_rows=center_rows, expr_center=expr_center,
                expr_bin_m=sub if expr_bin_m is None else expr_bin_m,
                expr_nbin=expr_nbin, expr_binsize=expr_binsize, replace=replace,
                random_state=random_state,
            )
            per_sample.append(s)
        scores = pd.concat(per_sample, axis=0)
    else:
        scores = _sig_scores_one(
            m, sigs,
            center_rows=center_rows, expr_center=expr_center,
            expr_bin_m=expr_bin_m, expr_nbin=expr_nbin,
            expr_binsize=expr_binsize, replace=replace,
            random_state=random_state,
        )

    if not permute:
        return scores

    fdr = _permute_fdr(
        m=m, sigs=sigs, observed=scores,
        center_rows=center_rows, expr_center=expr_center,
        expr_bin_m=expr_bin_m, expr_nbin=expr_nbin,
        expr_binsize=expr_binsize, replace=replace,
        n_perm=n_perm, alternative=alternative,
        random_state=random_state, batch=perm_batch,
    )
    return _melt_scores_fdr(scores, fdr)


# ---------- Building blocks ----------

def filter_sigs(sigs: SigDict, ref, conserved: float = 0.7) -> dict[str, list[str]]:
    """Drop genes not in ``ref``; drop signatures whose retained fraction < conserved."""
    ref = set(ref)
    out = {}
    for name, genes in sigs.items():
        kept = [g for g in genes if g in ref]
        if not genes:
            continue
        if len(kept) / len(genes) >= conserved and len(kept) >= 1:
            out[name] = kept
    return out


def base_scores(m: pd.DataFrame, sigs: SigDict) -> pd.DataFrame:
    """Mean expression of signature genes per cell, vectorised across signatures."""
    S = _sig_indicator(sigs, m.index)              # (n_sigs, n_genes)
    sizes = S.sum(axis=1)                          # (n_sigs,)
    raw = S @ m.values                             # (n_sigs, n_cells)
    raw = raw / sizes[:, None]
    return pd.DataFrame(raw.T, index=m.columns, columns=list(sigs))


def bin_genes(m: pd.DataFrame, nbin: int = 30) -> pd.Series:
    """Assign each gene to an equally-sized bin based on its row mean."""
    means = m.mean(axis=1).sort_values()
    edges = np.linspace(0, len(means), nbin + 1).astype(int)
    bins = np.zeros(len(means), dtype=int)
    for i in range(nbin):
        bins[edges[i]:edges[i + 1]] = i
    return pd.Series(bins, index=means.index)


def binmatch(
    genes: Sequence[str],
    bins: pd.Series,
    n: int = 100,
    replace: bool = False,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Sample n bin-matched control genes per gene in ``genes``."""
    rng = rng or np.random.default_rng()
    bin_ids = bins.loc[list(genes)].values
    bin_to_genes: dict[int, np.ndarray] = {}
    for b in np.unique(bin_ids):
        bin_to_genes[b] = bins.index[bins.values == b].to_numpy()
    out: list[str] = []
    for b in bin_ids:
        pool = bin_to_genes[b]
        size = n if (replace or n <= len(pool)) else len(pool)
        out.extend(rng.choice(pool, size=size, replace=replace).tolist())
    return out


# ---------- Internals ----------

def _sig_indicator(sigs: SigDict, gene_index) -> np.ndarray:
    pos = {g: i for i, g in enumerate(gene_index)}
    S = np.zeros((len(sigs), len(gene_index)), dtype=np.float64)
    for r, genes in enumerate(sigs.values()):
        for g in genes:
            S[r, pos[g]] = 1.0
    return S


def _sig_scores_one(
    m, sigs, *, center_rows, expr_center, expr_bin_m,
    expr_nbin, expr_binsize, replace, random_state,
) -> pd.DataFrame:
    mc = rowcenter(m) if center_rows else m
    scores = base_scores(mc, sigs)
    if not expr_center:
        return scores

    bin_ref = m if expr_bin_m is None else as_dataframe(expr_bin_m)
    bins = bin_genes(bin_ref, nbin=expr_nbin)
    rng = np.random.default_rng(random_state)
    ctrl_sigs = {name: binmatch(genes, bins, n=expr_binsize, replace=replace, rng=rng)
                 for name, genes in sigs.items()}
    ctrl_scores = base_scores(mc, ctrl_sigs)
    return scores - ctrl_scores


def _permute_fdr(
    *, m, sigs, observed, center_rows, expr_center, expr_bin_m,
    expr_nbin, expr_binsize, replace, n_perm, alternative,
    random_state, batch,
) -> pd.DataFrame:
    """Faithful port of scalop::permuteSigScores's permutation testing.

    Per iteration: permute genes across cells, score, derive a within-iteration
    across-cells threshold (mean + 2*sd) per signature, tally per cell whether
    the observed score beats that threshold. Then run a per-cell binomial test
    against the global success rate and BH-adjust within each signature.
    """
    rng = np.random.default_rng(random_state)
    bin_ref = m if expr_bin_m is None else as_dataframe(expr_bin_m)
    bins = bin_genes(bin_ref, nbin=expr_nbin)
    sig_names = list(sigs)
    sig_genes_list = [list(sigs[n]) for n in sig_names]

    S = _sig_indicator(sigs, m.index)
    sig_sizes = S.sum(axis=1)[:, None]

    M = m.values.astype(np.float64, copy=False)
    G, C = M.shape
    n_sigs = len(sigs)
    observed_vals = observed.values.T  # (n_sigs, n_cells)

    passed_counts = np.zeros((n_sigs, C), dtype=np.int64)
    total_passed = np.zeros(n_sigs, dtype=np.int64)

    for start in range(0, n_perm, batch):
        b = min(batch, n_perm - start)
        idx = np.argsort(rng.random((b, G, C)), axis=2)
        Mp = np.take_along_axis(M[None, :, :], idx, axis=2)
        if center_rows:
            Mp = Mp - Mp.mean(axis=2, keepdims=True)

        for k in range(b):
            scores_k = (S @ Mp[k]) / sig_sizes  # (n_sigs, C)
            if expr_center:
                ctrl_sigs_k = {n: binmatch(g, bins, n=expr_binsize, replace=replace, rng=rng)
                               for n, g in zip(sig_names, sig_genes_list)}
                Sc = _sig_indicator(ctrl_sigs_k, m.index)
                ctrl_sizes = Sc.sum(axis=1)[:, None]
                ctrl_k = (Sc @ Mp[k]) / ctrl_sizes
                scores_k = scores_k - ctrl_k

            mean_k = scores_k.mean(axis=1, keepdims=True)
            sd_k = scores_k.std(axis=1, ddof=1, keepdims=True)
            if alternative == "greater":
                passed_k = observed_vals > (mean_k + 2 * sd_k)
            elif alternative == "less":
                passed_k = observed_vals < (mean_k - 2 * sd_k)
            elif alternative in ("two.sided", "two-sided", "two_sided"):
                passed_k = (observed_vals > (mean_k + 2 * sd_k)) | (observed_vals < (mean_k - 2 * sd_k))
            else:
                raise ValueError(f"Unknown alternative: {alternative}")
            passed_counts += passed_k.astype(np.int64)
            total_passed += passed_k.sum(axis=1)

    from scipy.stats import binom, false_discovery_control

    fdr = np.empty((n_sigs, C))
    for s in range(n_sigs):
        p_s = total_passed[s] / (n_perm * C)
        if p_s <= 0:
            fdr[s, :] = 1.0
            continue
        pvals = binom.sf(passed_counts[s, :] - 1, n_perm, p_s)
        fdr[s, :] = false_discovery_control(pvals, method="bh")

    return pd.DataFrame(fdr.T, index=observed.index, columns=sig_names)


def _melt_scores_fdr(scores: pd.DataFrame, fdr: pd.DataFrame) -> pd.DataFrame:
    s = scores.reset_index().melt(id_vars=scores.index.name or "index",
                                   var_name="sig", value_name="score")
    f = fdr.reset_index().melt(id_vars=fdr.index.name or "index",
                                var_name="sig", value_name="fdr")
    id_col = scores.index.name or "index"
    out = s.merge(f, on=[id_col, "sig"], how="outer")
    return out.rename(columns={id_col: "id"})
