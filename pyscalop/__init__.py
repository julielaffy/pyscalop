"""pyscalop: single-cell analysis operations (Python port of scalop)."""

from .score import sig_scores, base_scores, filter_sigs, bin_genes, binmatch
from .dea import dea
from .hca import hca, hca_groups, hca_order
from .programs import programs
from .metaprograms import metaprograms
from .utils import rowcenter, jaccard, jaccard_filter, as_dataframe
from .plot import set_style, savefig
from . import signatures

__version__ = "0.1.0"

__all__ = [
    "sig_scores",
    "base_scores",
    "filter_sigs",
    "bin_genes",
    "binmatch",
    "dea",
    "hca",
    "hca_groups",
    "hca_order",
    "programs",
    "metaprograms",
    "rowcenter",
    "jaccard",
    "jaccard_filter",
    "as_dataframe",
    "set_style",
    "savefig",
    "signatures",
]
