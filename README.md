# pyscalop

Single-cell analysis operations — Python port of [scalop](https://github.com/jlaffy/scalop).

A small toolbox for scRNA-seq analysis with a focus on intra-tumour expression
programs (as in Gavish et al. 2023). Built on `numpy`, `pandas`, `scipy`,
`anndata`.

## Install

```bash
pip install -e .
```

## Quick start

```python
import pyscalop as ps

# m: pandas DataFrame, genes x cells, log-normalised, not row-centered
# sigs: dict of {signature_name: [gene1, gene2, ...]}
scores = ps.sig_scores(m, sigs)

# Differential expression for one or more cell groups vs the rest
deas = ps.dea(m, groups={"clusterA": cells_A, "clusterB": cells_B})

# Find intra-tumour programs (clustering + DEA + jaccard filtering)
res = ps.programs(m)

# Aggregate programs across samples into metaprograms
mp = ps.metaprograms(res["programs"], res["profiles"], metaclusters)
```

## Layout

- `pyscalop/` — importable package
- `analysis/` — analysis scripts (source functions, consume data, emit results/plots)
- `data/` — important intermediate data
- `tests/`

## Status

v0.1 — core modules ported: `score`, `dea`, `programs`, `metaprograms`, `utils`, `plot`.
