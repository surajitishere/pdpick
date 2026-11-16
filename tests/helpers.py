"""Shared helpers for the test-suite."""

from __future__ import annotations

import itertools

import numpy as np

from pdpick.tree import Tree, parse_newick


def random_newick(n_tips: int, rng: np.random.Generator, polytomies: bool = False, scale: float = 0.3) -> str:
    """A random rooted tree built by repeatedly joining random lineages."""
    nodes = [f"t{i}" for i in range(n_tips)]
    while len(nodes) > 1:
        m = 2
        if polytomies and len(nodes) >= 3:
            m = int(rng.integers(2, min(4, len(nodes)) + 1))
        picked = sorted(rng.choice(len(nodes), size=m, replace=False), reverse=True)
        parts = [nodes.pop(i) for i in picked]
        nodes.append("(" + ",".join(f"{p}:{rng.exponential(scale):.6f}" for p in parts) + ")")
    return nodes[0] + ";"


def random_tree(n_tips: int, rng: np.random.Generator, **kwargs) -> Tree:
    return parse_newick(random_newick(n_tips, rng, **kwargs))


def pairwise_tip_distances(tree: Tree) -> dict[tuple[str, str], float]:
    depth = tree.depths()
    out = {}
    tips = tree.tips
    for i, a in enumerate(tips):
        for b in tips[i + 1 :]:
            m = tree.mrca([a, b])
            out[tuple(sorted((tree.names[a], tree.names[b])))] = depth[a] + depth[b] - 2 * depth[m]
    return out


def brute_force_pd(tree: Tree, tips: list[str], rooted: bool) -> float:
    """PD from first principles: sum the branches whose two sides both matter.

    A branch counts for rooted PD if a selected tip lies below it, and for
    unrooted PD if selected tips lie on both sides of it.
    """
    chosen = set(tips)
    total = 0.0
    for v in range(1, tree.n_nodes):
        below = {tree.names[t] for t in tree.descendants_tips(v)}
        inside = len(chosen & below)
        if inside and (rooted or inside < len(chosen)):
            total += tree.length[v]
    return total


def best_by_enumeration(tree, k, rooted, weights, lam, required=(), exclude=()):
    allowed = [t for t in tree.tip_names if t not in set(exclude)]
    best = -np.inf
    for subset in itertools.combinations(allowed, k):
        if not set(required) <= set(subset):
            continue
        value = brute_force_pd(tree, list(subset), rooted)
        value += lam * sum((weights or {}).get(t, 0.0) for t in subset)
        best = max(best, value)
    return best
