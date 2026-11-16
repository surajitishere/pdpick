"""Scoring a selection with measures that PD maximisation does not optimise."""

from __future__ import annotations

import numpy as np

from .core import pd_of, total_pd
from .tree import Tree

__all__ = ["nearest_selected_distance", "evaluate"]


def nearest_selected_distance(tree: Tree, selected) -> np.ndarray:
    """Path length from every node to the closest selected tip.

    Two sweeps: the closest selected tip *below* each node, then the closest
    one reached by going *up* through the parent.
    """
    index = tree.tip_index()
    chosen = [index[t] if isinstance(t, str) else int(t) for t in selected]
    if not chosen:
        raise ValueError("selection is empty")
    n = tree.n_nodes
    parent, length = tree.parent, tree.length
    down = np.full(n, np.inf)
    down[chosen] = 0.0
    for i in range(n - 1, 0, -1):
        p = parent[i]
        reach = down[i] + length[i]
        if reach < down[p]:
            down[p] = reach
    best = down.copy()
    for i in range(1, n):
        reach = best[parent[i]] + length[i]
        if reach < best[i]:
            best[i] = reach
    return best


def evaluate(
    tree: Tree,
    selected,
    weights: dict[str, float] | None = None,
    groups: dict[str, dict[str, str]] | None = None,
) -> dict[str, float]:
    """Summary statistics of a selection.

    ``pd_fraction``     rooted PD of the selection / total tree length
    ``adcl``            mean distance from every tip of the tree to its closest
                        selected tip (Matsen et al. 2013); lower is better
    ``max_distance``    the largest such distance (covering radius)
    ``mean_weight``     mean weight of the selected tips, if weights are given
    ``n_<group>``       number of distinct labels covered, for each grouping in
                        ``groups`` (e.g. ``{"phylum": {tip: label}}``)
    """
    selected = list(selected)
    dist = nearest_selected_distance(tree, selected)[tree.tips]
    pd_value = pd_of(tree, selected, rooted=True)
    out = {
        "k": len(selected),
        "pd": pd_value,
        "pd_fraction": pd_value / total_pd(tree),
        "adcl": float(dist.mean()),
        "max_distance": float(dist.max()),
    }
    if weights is not None:
        out["mean_weight"] = float(np.mean([weights.get(t, 0.0) for t in selected]))
    for name, mapping in (groups or {}).items():
        labels = {mapping.get(t) for t in selected}
        labels.discard(None)
        labels.discard("")
        out[f"n_{name}"] = len(labels)
    return out
