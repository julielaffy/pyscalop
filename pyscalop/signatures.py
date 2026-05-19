"""Bundled and user-supplied gene signatures.

Built-in signature sets ship inside the package as GMT files
(``pyscalop/signatures_data/<name>.gmt``). They load lazily and cache.

Usage
-----
Attribute-style (R-flavoured, no parens; loads on first access, cached):

    sigs = ps.signatures.thermogenic

Function call by name (handy when the name is dynamic):

    sigs = ps.signatures.load("thermogenic")

User files outside the package:

    sigs = ps.signatures.from_gmt("path/to/my_sigs.gmt")
"""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path


# ---------- I/O ----------

def from_gmt(path: str | Path) -> dict[str, list[str]]:
    """Parse a Broad GMT file: name<TAB>description<TAB>gene1<TAB>gene2..."""
    out: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *genes = parts
            out[name] = [g for g in genes if g]
    return out


def to_gmt(sigs: dict[str, list[str]], path: str | Path, description: str = "") -> None:
    """Write signatures as a GMT file."""
    with open(path, "w") as f:
        for name, genes in sigs.items():
            f.write(f"{name}\t{description}\t" + "\t".join(genes) + "\n")


# ---------- Built-in loaders ----------

@lru_cache(maxsize=None)
def _load_bundled(name: str) -> dict[str, list[str]]:
    fname = f"{name}.gmt"
    with resources.files("pyscalop.signatures_data").joinpath(fname).open() as f:
        out: dict[str, list[str]] = {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sname, _desc, *genes = parts
            out[sname] = [g for g in genes if g]
    return out


def list_available() -> list[str]:
    """Return the names of bundled signature sets."""
    files = resources.files("pyscalop.signatures_data").iterdir()
    return sorted(p.stem for p in files if p.name.endswith(".gmt"))


def load(name: str) -> dict[str, list[str]]:
    """Load a bundled signature set by name."""
    if name not in list_available():
        raise ValueError(f"No bundled signature set named {name!r}. "
                         f"Available: {list_available()}")
    return _load_bundled(name)


# ---------- R-flavoured attribute access (PEP 562) ----------

def __getattr__(name: str):
    """Allow `ps.signatures.thermogenic` (no parens) — same cached dict."""
    if name in list_available():
        return _load_bundled(name)
    raise AttributeError(f"module 'pyscalop.signatures' has no attribute {name!r}")


def __dir__() -> list[str]:
    base = ["from_gmt", "to_gmt", "list_available", "load"]
    return sorted(set(base) | set(list_available()))
