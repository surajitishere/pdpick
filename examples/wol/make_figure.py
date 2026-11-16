#!/usr/bin/env python
"""Benchmark figure (reads results/benchmark.tsv and results/tradeoff.tsv).

    python examples/wol/make_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Series colours are fixed per method and checked for colour-blind separation.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}  # fmt: skip
LABELS = {
    "pd_quality": "PD + quality weight",
    "filter_pd": "quality ≥ 0.9, then PD",
    "stratified": "phylum-stratified, best quality",
    "random": "random",
}


def style(ax, t) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(which="both", colors=t["muted"], labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def main(theme: str) -> None:
    t = THEMES[theme]
    bench = pd.read_csv(RESULTS / "benchmark.tsv", sep="\t")
    trade = pd.read_csv(RESULTS / "tradeoff.tsv", sep="\t")
    colour = {
        "pd_quality": t["series"][0],
        "filter_pd": t["series"][1],
        "stratified": t["series"][2],
        "random": t["muted"],
    }
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=160, facecolor=t["surface"], layout="constrained",
        gridspec_kw={"wspace": 0.12},
    )  # fmt: skip

    # ---- left: PD retained against selection size, labelled at the line ends
    style(ax1, t)
    ends = {}
    for method, label in LABELS.items():
        rows = bench[bench.method == method].sort_values("k")
        ax1.plot(rows.k, rows.pd_vs_optimum, color=colour[method], linewidth=2, marker="o",
                 markersize=5, markeredgecolor=t["surface"], markeredgewidth=1)  # fmt: skip
        ends[label] = (rows.pd_vs_optimum.iloc[-1], colour[method])
    ax1.axhline(1.0, color=t["axis"], linewidth=0.8)
    ax1.annotate("exact PD optimum", xy=(0.01, 1.0), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points", fontsize=8, color=t["muted"])  # fmt: skip
    ax1.set_xscale("log")
    ax1.set_xticks(sorted(bench.k.unique()))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.minorticks_off()
    ax1.set_ylim(0.4, 1.06)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for label, (y, c) in ends.items():
        ax1.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(8, 0),
                     textcoords="offset points", va="center", fontsize=9, color=t["ink"],
                     annotation_clip=False)  # fmt: skip
        ax1.plot([1.008], [y], marker="s", markersize=5, color=c, clip_on=False,
                 transform=ax1.get_yaxis_transform())  # fmt: skip
    ax1.set_xlabel("genomes selected (of 10,575)")
    ax1.set_ylabel("phylogenetic diversity, % of optimum")
    ax1.set_title("Diversity kept by each sampling scheme", fontsize=11, loc="left")

    # ---- right: the diversity-quality frontier at k = 500
    style(ax2, t)
    weighted = trade[trade.approach == "weighted"].sort_values("mean_weight")
    hard = trade[trade.approach == "threshold"].sort_values("mean_weight")
    ax2.plot(weighted.mean_weight, weighted.pd_vs_optimum, color=t["series"][0], linewidth=2,
             label="quality as a weight (varying λ)")  # fmt: skip
    ax2.plot(hard.mean_weight, hard.pd_vs_optimum, color=t["series"][1], linewidth=2, marker="o",
             markersize=5, markeredgecolor=t["surface"], markeredgewidth=1,
             label="quality as a threshold (varying cut-off)")  # fmt: skip
    at500 = bench[bench.k == 500].set_index("method")
    for method, marker in (("stratified", "D"), ("random", "s")):
        ax2.scatter(at500.loc[method, "mean_weight"], at500.loc[method, "pd_vs_optimum"], s=60,
                    marker=marker, color=colour[method], edgecolors=t["surface"], linewidths=1.5,
                    zorder=5, label=LABELS[method])  # fmt: skip
    ax2.set_ylim(0.4, 1.06)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("mean quality of the 500 selected genomes")
    ax2.set_ylabel("phylogenetic diversity, % of optimum")
    ax2.set_title("Trading diversity for genome quality (k = 500)", fontsize=11, loc="left")
    legend = ax2.legend(frameon=False, fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(t["ink2"])

    out = HERE / f"benchmark_{theme}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main("light")
    main("dark")
