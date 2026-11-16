#!/usr/bin/env python
"""Download the public inputs of the benchmark (about 2 MB) into ./data.

    python examples/wol/fetch_data.py

Files come from a pinned commit of the Web of Life repository
(https://github.com/biocore/wol, Zhu et al. 2019; BSD 3-Clause) and are checked
against SHA-256 digests. Nothing is redistributed with this repository.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
WOL = "https://raw.githubusercontent.com/biocore/wol/ff92d2251c0ea72299a2ead1aecfd748a9acd698/data"
FILES = {
    "astral.cons.nid.nwk": (
        f"{WOL}/trees/astral/branch_length/cons/astral.cons.nid.nwk",
        "3d600bc365c26ae1492f72d384f21c0c22afeb7a2e5c8b9eb5611d6efe9cdf8d",
    ),
    "metadata.tsv.xz": (
        f"{WOL}/genomes/metadata.tsv.xz",
        "b811ff3c69b6a6f9a94d55cc9a177bced4dddd8b1132603c47669bf132c84e9b",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DATA.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = DATA / name
        if not (target.exists() and sha256(target) == expected):
            print(f"fetching {name}")
            urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URLs
        if sha256(target) != expected:
            print(f"checksum mismatch for {name}", file=sys.stderr)
            return 1
        print(f"ok  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
