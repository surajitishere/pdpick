#!/usr/bin/env python
"""Benchmark on the 10,575-genome Web of Life phylogeny (Zhu et al. 2019).

    python examples/wol/benchmark.py

Compares, for several selection sizes k:

* ``pd``            exact PD maximisation
* ``pd_quality``    PD maximisation with a quality weight, giving up at most 5% of PD
* ``filter_pd``     drop genomes below a quality threshold, then maximise PD
* ``stratified``    phylum-stratified: one per phylum, the rest in proportion to
                    phylum size, best-quality genomes within each phylum
* ``random``        uniform random sampling (mean of 50 draws)

Quality is ``max(0, completeness - 5 x contamination) / 100`` from the CheckM
estimates in the Web of Life metadata. Writes ``results/benchmark.tsv`` and
``results/tradeoff.tsv`` (the PD-quality frontier at k = 500).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdpick import evaluate, rank_tips, read_newick, select
from pdpick.baselines import random_sample, stratified_sample

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SIZES = (50, 100, 200, 500, 1000, 2000)
FILTER = 0.9
N_RANDOM = 50
MAX_LOSS = 0.05


def load():
    tree = read_newick(DATA / "astral.cons.nid.nwk")
    meta = pd.read_csv(DATA / "metadata.tsv.xz", sep="\t", dtype=str)
    meta = meta.rename(columns={"#genome": "genome"}).set_index("genome").loc[tree.tip_names]
    quality = (
        pd.to_numeric(meta.completeness) - 5 * pd.to_numeric(meta.contamination)
    ).clip(lower=0) / 100
    return tree, meta, quality


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    tree, meta, quality = load()
    weights = quality.to_dict()
    groups = {"phylum": meta.phylum.dropna().to_dict(), "genus": meta.genus.dropna().to_dict()}
    strata = meta.phylum.fillna("unclassified").to_dict()
    tips = tree.tip_names
    rng = np.random.default_rng(1)

    plain = rank_tips(tree)
    filtered = rank_tips(tree, exclude=quality.index[quality < FILTER].tolist())
    rows = []
    for k in SIZES:
        picks = {
            "pd": plain.top(k),
            "pd_quality": select(tree, k, weights, max_pd_loss=MAX_LOSS).top(k),
            "filter_pd": filtered.top(k),
            "stratified": stratified_sample(strata, k, weights),
        }
        for method, chosen in picks.items():
            rows.append({"method": method, **evaluate(tree, chosen, weights, groups)})
        draws = pd.DataFrame(
            [evaluate(tree, random_sample(tips, k, rng), weights, groups) for _ in range(N_RANDOM)]
        )
        rows.append({"method": "random", **draws.mean().to_dict()})
        print(f"k = {k} done")
    table = pd.DataFrame(rows)
    optimum = table[table.method == "pd"].set_index("k").pd
    table["pd_vs_optimum"] = table.pd / table.k.map(optimum)
    table.to_csv(RESULTS / "benchmark.tsv", sep="\t", index=False, float_format="%.5g")

    # The frontier at k = 500: weighted selection against hard quality thresholds.
    k = 500
    frontier = []
    for lam in [0.0, *np.logspace(-2, 1.3, 30)]:
        r = plain if lam == 0 else rank_tips(tree, weights, lam)
        frontier.append({"approach": "weighted", "parameter": lam,
                         **evaluate(tree, r.top(k), weights, groups)})  # fmt: skip
    for threshold in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98):
        r = rank_tips(tree, exclude=quality.index[quality < threshold].tolist())
        frontier.append({"approach": "threshold", "parameter": threshold,
                         **evaluate(tree, r.top(k), weights, groups)})  # fmt: skip
    frontier = pd.DataFrame(frontier)
    frontier["pd_vs_optimum"] = frontier.pd / plain.pd_at(k)
    frontier.to_csv(RESULTS / "tradeoff.tsv", sep="\t", index=False, float_format="%.5g")

    # Who gets picked by plain PD maximisation?
    top = plain.top(k)
    size = pd.to_numeric(meta.total_length) / 1e6
    lines = [
        f"genomes in tree: {len(tips)}",
        f"quality, all genomes: mean {quality.mean():.3f}, median {quality.median():.3f}",
        f"quality, PD-optimal {k}: mean {quality[top].mean():.3f}, median {quality[top].median():.3f}",
        f"genome size (Mb), all genomes: median {size.median():.2f}",
        f"genome size (Mb), PD-optimal {k}: median {size[top].median():.2f}",
        f"quality < {FILTER}: {(quality < FILTER).mean():.1%} of all genomes, "
        f"{(quality[top] < FILTER).mean():.1%} of the PD-optimal {k}",
    ]
    (RESULTS / "who_gets_picked.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    cols = ["method", "k", "pd_vs_optimum", "pd_fraction", "adcl", "mean_weight", "n_phylum", "n_genus"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
