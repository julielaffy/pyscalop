"""Side-by-side validation: R scalop vs Python pyscalop signature scoring.

Generates a synthetic matrix with a planted signal, runs scalop::sigScores
and scalop::permuteSigScores via Rscript, runs the equivalent in pyscalop,
and reports agreement and timing.

Run from the repo root:
    python analysis/validate_perm.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pyscalop as ps


REPO = Path(__file__).resolve().parents[1]
IO = REPO / "data" / "_validate_perm_io"
R_SCRIPT = REPO / "analysis" / "validate_perm.R"

# --- knobs ---
N_GENES = 3000          # large enough that each bin has >> binsize genes
N_CELLS = 80
N_PLANTED_GENES = 30
N_PLANTED_CELLS = 25
SIGNAL = 2.5
N_PERM = 30
EXPR_NBIN = 30
EXPR_BINSIZE = 50
SEED = 42
# -------------


def make_synthetic() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(SEED)
    M = rng.normal(loc=2.0, scale=1.0, size=(N_GENES, N_CELLS))
    M[:N_PLANTED_GENES, :N_PLANTED_CELLS] += SIGNAL
    genes = [f"g{i:04d}" for i in range(N_GENES)]
    cells = [f"c{i:03d}" for i in range(N_CELLS)]
    m = pd.DataFrame(M, index=genes, columns=cells)
    sigs = {
        "planted": [f"g{i:04d}" for i in range(N_PLANTED_GENES)],
        "random":  [f"g{i:04d}" for i in range(150, 150 + N_PLANTED_GENES)],
    }
    return m, sigs


def run_r() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    env = {**os.environ,
           "PYSCALOP_VALIDATE_DIR": str(IO),
           "PYSCALOP_VALIDATE_NPERM": str(N_PERM),
           "PYSCALOP_VALIDATE_NBIN": str(EXPR_NBIN),
           "PYSCALOP_VALIDATE_BINSIZE": str(EXPR_BINSIZE)}
    subprocess.run(["Rscript", str(R_SCRIPT)], check=True, env=env)
    r_scores = pd.read_csv(IO / "r_scores.tsv", sep="\t", index_col=0)
    r_perm = pd.read_csv(IO / "r_perm.tsv", sep="\t")
    timing = pd.read_csv(IO / "r_timing.tsv", sep="\t")
    return r_scores, r_perm, dict(zip(timing["step"], timing["seconds"]))


def run_py(m, sigs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    t0 = time.perf_counter()
    py_scores = ps.sig_scores(m, sigs, random_state=SEED,
                              expr_nbin=EXPR_NBIN, expr_binsize=EXPR_BINSIZE)
    t_obs = time.perf_counter() - t0

    t0 = time.perf_counter()
    py_perm = ps.sig_scores(m, sigs, permute=True, n_perm=N_PERM,
                            alternative="greater", random_state=SEED,
                            expr_nbin=EXPR_NBIN, expr_binsize=EXPR_BINSIZE)
    t_perm = time.perf_counter() - t0
    return py_scores, py_perm, {"sigScores": t_obs, "permuteSigScores": t_perm}


def compare_observed(r_scores: pd.DataFrame, py_scores: pd.DataFrame) -> None:
    print("\n=== Observed scores (sigScores) ===")
    sigs = sorted(set(r_scores.columns) & set(py_scores.columns))
    for sig in sigs:
        r = r_scores[sig]
        p = py_scores[sig].reindex(r.index)
        cor = float(r.corr(p))
        diff = (r - p).abs()
        print(f"  {sig:<10s}  pearson r = {cor:6.4f}   "
              f"max|Δ| = {diff.max():.4f}   mean|Δ| = {diff.mean():.4f}")


def compare_permutation(r_perm: pd.DataFrame, py_perm: pd.DataFrame,
                        planted_cells: set[str], fdr_cut: float = 0.05) -> None:
    print(f"\n=== Permutation FDR agreement (cutoff FDR <= {fdr_cut}) ===")
    sigs = sorted(set(r_perm["sig"]) & set(py_perm["sig"]))
    for sig in sigs:
        r_sub = r_perm[r_perm["sig"] == sig]
        p_sub = py_perm[py_perm["sig"] == sig]
        r_sig = set(r_sub.loc[r_sub["fdr"] <= fdr_cut, "id"])
        p_sig = set(p_sub.loc[p_sub["fdr"] <= fdr_cut, "id"])
        union = r_sig | p_sig
        jacc = len(r_sig & p_sig) / len(union) if union else 1.0

        r_tp = len(r_sig & planted_cells); r_fp = len(r_sig - planted_cells)
        p_tp = len(p_sig & planted_cells); p_fp = len(p_sig - planted_cells)
        print(f"  {sig:<10s}  R-sig={len(r_sig):3d}  Py-sig={len(p_sig):3d}  "
              f"Jaccard(R,Py)={jacc:.3f}  "
              f"R TP/FP={r_tp:>2d}/{r_fp:<2d}  Py TP/FP={p_tp:>2d}/{p_fp:<2d}")

    # Correlation of FDR values themselves
    print("\n  Pearson r of FDR values per signature:")
    for sig in sigs:
        merged = (r_perm[r_perm["sig"] == sig].set_index("id")
                  .join(py_perm[py_perm["sig"] == sig].set_index("id"),
                        lsuffix="_r", rsuffix="_py", how="inner"))
        cor = float(merged["fdr_r"].corr(merged["fdr_py"]))
        print(f"    {sig:<10s} fdr cor = {cor:6.4f}")


def main() -> None:
    IO.mkdir(parents=True, exist_ok=True)
    m, sigs = make_synthetic()
    m.to_csv(IO / "matrix.tsv", sep="\t")
    with open(IO / "sigs.tsv", "w") as f:
        for name, gs in sigs.items():
            f.write(name + "\t" + "\t".join(gs) + "\n")
    print(f"Wrote {m.shape[0]}x{m.shape[1]} matrix and {len(sigs)} sigs to {IO}")

    print("\n--- Running R ---")
    r_scores, r_perm, r_t = run_r()
    print("\n--- Running Python ---")
    py_scores, py_perm, py_t = run_py(m, sigs)

    compare_observed(r_scores, py_scores)

    planted_cells = {f"c{i:03d}" for i in range(N_PLANTED_CELLS)}
    compare_permutation(r_perm, py_perm, planted_cells)

    print("\n=== Timing ===")
    for step in ["sigScores", "permuteSigScores"]:
        rs = r_t[step]
        ps_ = py_t[step]
        print(f"  {step:<18s}  R: {rs:7.3f}s   Py: {ps_:7.3f}s   "
              f"speedup: {rs/ps_:5.1f}x")


if __name__ == "__main__":
    sys.exit(main())
