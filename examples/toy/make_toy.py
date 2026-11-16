#!/usr/bin/env python
"""Generate the toy data set used in the README and the tests (seeded).

    python examples/toy/make_toy.py

A 200-tip random tree. Thirty tips get a long terminal branch and a low
quality score, mimicking the fast-evolving, incomplete genomes that plain PD
maximisation is drawn to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pdpick import parse_newick

HERE = Path(__file__).parent
N_TIPS = 200
SEED = 20261113


def main() -> None:
    rng = np.random.default_rng(SEED)
    nodes = [f"g{i + 1:03d}" for i in range(N_TIPS)]
    names = list(nodes)
    while len(nodes) > 1:
        i, j = sorted(rng.choice(len(nodes), size=2, replace=False), reverse=True)
        a, b = nodes.pop(i), nodes.pop(j)
        nodes.append(f"({a}:{rng.exponential(0.1):.6f},{b}:{rng.exponential(0.1):.6f})")
    tree = parse_newick(nodes[0] + ";")
    index = tree.tip_index()
    poor = set(rng.choice(names, size=30, replace=False))
    rows = ["genome\tquality\tclade"]
    for name in names:
        if name in poor:
            tree.length[index[name]] += rng.uniform(0.3, 0.6)
            quality = rng.uniform(0.2, 0.6)
        else:
            quality = rng.uniform(0.85, 1.0)
        rows.append(f"{name}\t{quality:.3f}\t{'poor' if name in poor else 'good'}")
    (HERE / "tree.nwk").write_text(tree.to_newick(digits=8, internal_names=False) + "\n")
    (HERE / "quality.tsv").write_text("\n".join(rows) + "\n")
    (HERE / "required.txt").write_text("# model organisms that must be kept\ng001\ng002\n")
    print(f"wrote tree.nwk, quality.tsv and required.txt to {HERE}")


if __name__ == "__main__":
    main()
