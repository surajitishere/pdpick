"""Exact maximisation of (quality-weighted) phylogenetic diversity.

Phylogenetic diversity (PD; Faith 1992) of a set of tips is the total length
of the branches that connect them -- to each other (*unrooted*) or to each
other and the root (*rooted*). Choosing the ``k`` tips with the largest PD is
one of the rare subset-selection problems where a greedy algorithm is exactly
optimal (Steel 2005; Pardi & Goldman 2005), and the optimal sets for
successive ``k`` are nested.

The implementation uses the *long-path decomposition* of the tree. Give every
branch to the deepest tip below it. Each tip then owns one contiguous chain of
branches running up from itself, the chains partition the tree, and a chain's
length is exactly the PD its tip adds when the tips are taken in order of
decreasing chain length. Selecting ``k`` tips is therefore a sort.

Two extensions keep that exactness:

* **Weights.** Maximising ``PD(S) + lam * sum(w_i for i in S)`` is PD
  maximisation on a tree whose terminal branches are lengthened by
  ``lam * w_i``. Any per-tip preference -- assembly quality, type-strain
  status, availability of phenotype data -- can be traded against diversity
  this way, and each ``lam`` yields an exactly optimal set for its objective.
* **Required and excluded tips.** Branches already covered by required tips
  contribute nothing further, so they are set to zero before the
  decomposition; excluded tips are removed from the tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tree import Tree

__all__ = ["Ranking", "rank_tips", "select", "chain_lengths", "pd_of", "total_pd", "farthest_tip"]


def chain_lengths(tree: Tree, length: np.ndarray | None = None) -> np.ndarray:
    """Length of the long-path chain owned by every tip (0 for internal nodes).

    ``length`` overrides the tree's branch lengths (same indexing). Ties between
    children are broken towards the child visited first in pre-order, which
    makes the result deterministic.
    """
    length = tree.length if length is None else np.asarray(length, dtype=float)
    n = tree.n_nodes
    parent = tree.parent.tolist()
    ln = length.tolist()
    depth_below = [0.0] * n  # longest path from the node down to a tip
    best_child = [-1] * n
    for i in range(n - 1, 0, -1):
        p = parent[i]
        reach = depth_below[i] + ln[i]
        # ">=" so that, scanning in reverse pre-order, the earliest child wins ties
        if best_child[p] == -1 or reach >= depth_below[p]:
            depth_below[p] = reach
            best_child[p] = i
    chain = np.zeros(n)
    is_tip = tree.is_tip
    acc = [0.0] * n  # chain length accumulated from the top of the chain down to here
    for i in range(1, n):
        p = parent[i]
        acc[i] = (acc[p] if best_child[p] == i else 0.0) + ln[i]
        if is_tip[i]:
            chain[i] = acc[i]
    return chain


def total_pd(tree: Tree) -> float:
    """Total branch length of the tree."""
    return float(tree.length[1:].sum())


def pd_of(tree: Tree, tips, rooted: bool = True) -> float:
    """Phylogenetic diversity of a set of tips, computed directly.

    Rooted PD is the length of the union of root-to-tip paths. Unrooted PD is
    the length of the smallest subtree connecting the tips (zero for one tip).
    This function is independent of the selection code and is what the tests
    use to check it.
    """
    index = tree.tip_index()
    nodes = [index[t] if isinstance(t, str) else int(t) for t in tips]
    if not nodes:
        return 0.0
    below = np.zeros(tree.n_nodes, dtype=np.int64)
    for v in nodes:
        below[v] += 1
    for i in range(tree.n_nodes - 1, 0, -1):
        below[tree.parent[i]] += below[i]
    covered = below > 0
    covered[0] = False
    if not rooted:
        # drop branches above the MRCA: those carry every selected tip
        covered &= below < len(set(nodes))
    return float(tree.length[covered].sum())


def farthest_tip(tree: Tree, node: int, extra: np.ndarray | None = None) -> tuple[int, float]:
    """The tip farthest from ``node`` and its distance.

    ``extra`` adds a per-node bonus to tips (used for weighted diameters).
    """
    hung = tree.reroot(node)
    depth = hung.depths()
    # map back: reroot keeps names but not indices, so carry indices through names
    order = _reroot_order(tree, node)
    score = depth.copy()
    if extra is not None:
        score = score + extra[order]
    tips = np.flatnonzero(hung.is_tip & (np.arange(hung.n_nodes) != 0))
    best = tips[np.argmax(score[tips])]
    return int(order[best]), float(score[best])


def _reroot_order(tree: Tree, node: int) -> np.ndarray:
    """Original node index of every node of ``tree.reroot(node)``, in its pre-order."""
    order: list[int] = []
    stack: list[tuple[int, int]] = [(int(node), -1)]
    while stack:
        v, came_from = stack.pop()
        order.append(v)
        onward = [c for c in tree.children[v] if c != came_from]
        up = int(tree.parent[v])
        if up >= 0 and up != came_from:
            onward.append(up)
        for u in reversed(onward):
            stack.append((u, v))
    return np.array(order, dtype=np.int64)


@dataclass
class Ranking:
    """Tips in the order an exact greedy PD maximiser takes them.

    ``table`` has one row per tip that can be selected, best first:

    ``rank``        1-based position
    ``tip``         tip name
    ``required``    forced into the selection
    ``gain``        PD added by this tip given all earlier ones (branch length only)
    ``weight``      the tip's weight (0 if none was given)
    ``objective_gain``  ``gain + lam * weight``; non-increasing after the required block
    ``pd``          cumulative PD of the first ``rank`` tips
    ``pd_fraction`` ``pd`` as a fraction of the PD of all selectable tips
    """

    table: pd.DataFrame
    rooted: bool
    lam: float
    n_required: int
    pd_all: float

    def top(self, k: int) -> list[str]:
        """Names of the optimal ``k``-tip selection."""
        if k < self.n_required:
            raise ValueError(f"k={k} is smaller than the number of required tips ({self.n_required})")
        if k > len(self.table):
            raise ValueError(f"k={k} exceeds the {len(self.table)} selectable tips")
        return self.table.tip.iloc[:k].tolist()

    def pd_at(self, k: int) -> float:
        return float(self.table.pd.iloc[k - 1]) if k > 0 else 0.0


def rank_tips(
    tree: Tree,
    weights: dict[str, float] | None = None,
    lam: float = 0.0,
    required: list[str] | None = None,
    exclude: list[str] | None = None,
    rooted: bool = True,
) -> Ranking:
    """Order all tips so that every prefix is an optimal selection.

    Parameters
    ----------
    tree : Tree
        Rooted tree with non-negative branch lengths.
    weights : dict, optional
        Tip name -> non-negative weight. Missing tips get weight 0.
    lam : float
        Branch length one unit of weight is worth. ``0`` ignores the weights.
    required : list of str, optional
        Tips that must be selected; they occupy the first ranks.
    exclude : list of str, optional
        Tips that may not be selected. They are removed before PD is measured.
    rooted : bool
        Rooted PD (default) counts the path to the root; unrooted PD counts
        only the subtree spanned by the selection.

    Notes
    -----
    For every ``k`` from ``max(len(required), 2 if unrooted else 1)`` up, the
    first ``k`` tips maximise ``PD + lam * (sum of weights)`` among all
    ``k``-tip selections that contain the required tips. In unrooted mode
    without required tips the first tip is one end of the (weighted) diameter;
    a single tip has zero unrooted PD, so rank 1 is only meaningful there as
    part of a larger selection.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    required = list(dict.fromkeys(required or []))
    excluded = set(exclude or [])
    clash = excluded.intersection(required)
    if clash:
        raise ValueError(f"tips both required and excluded: {sorted(clash)[:3]}")
    names = set(tree.tip_names)
    for label, group in (("required", required), ("excluded", excluded)):
        unknown = [t for t in group if t not in names]
        if unknown:
            raise KeyError(f"{len(unknown)} {label} tips are not in the tree, e.g. {unknown[:3]}")
    root_stem = 0.0  # path from the original root to the root left after exclusion
    if excluded:
        kept = [t for t in tree.tip_names if t not in excluded]
        full_index = tree.tip_index()
        root_stem = float(tree.depths()[tree.mrca([full_index[t] for t in kept])])
        tree = tree.prune(kept)
    else:
        tree = tree.suppress_unary()
    if np.any(tree.length < 0):
        raise ValueError("branch lengths must be non-negative")

    n = tree.n_nodes
    index = tree.tip_index()
    w = np.zeros(n)
    if weights:
        for name, value in weights.items():
            if name in index:
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"weight of {name!r} must be finite and non-negative")
                w[index[name]] = value
    bonus = lam * w
    is_tip = tree.is_tip

    # Work on a copy rooted where PD should be measured from. ``origin`` maps the
    # working tree's nodes back to ``tree``.
    origin = np.arange(n)
    work = tree
    forced = [index[t] for t in required]
    if not rooted:
        if forced:
            anchor = forced[0]
        else:
            start = int(tree.tips[0])
            anchor, _ = farthest_tip(tree, start, bonus)
            forced = [anchor]
        origin = _reroot_order(tree, anchor)
        work = tree.reroot(anchor)
    position = np.empty(n, dtype=np.int64)
    position[origin] = np.arange(n)

    # Weights become extra terminal branch length. (In unrooted mode the anchor
    # is the root of ``work``; its own bonus is a constant and can be ignored.)
    length = work.length.copy()
    tip_mask = is_tip[origin]
    length[tip_mask] += bonus[origin][tip_mask]

    # Branches covered by required tips add nothing more: zero them.
    covered = np.zeros(n, dtype=bool)
    for v in forced:
        u = int(position[v])
        while u > 0 and not covered[u]:
            covered[u] = True
            u = int(work.parent[u])
    reduced = length.copy()
    reduced[covered] = 0.0
    chain = chain_lengths(work, reduced)  # objective gain of each tip

    forced_set = set(forced)
    candidates = [int(v) for v in tree.tips if int(v) not in forced_set]
    # Sort by objective gain, then weight, then name: deterministic under ties.
    candidates.sort(key=lambda v: (-chain[position[v]], -w[v], tree.names[v]))
    order = forced + candidates

    # Report PD in branch-length units, measured directly on the unweighted tree.
    gains = _marginal_pd(work, position, order)
    if rooted and len(gains):
        gains[0] += root_stem  # any selection also spans the stem above the pruned root
    pd_cum = np.cumsum(gains)
    objective_gain = np.array([chain[position[v]] for v in order])
    objective_gain[: len(forced)] = np.nan
    pd_all = pd_cum[-1] if len(order) else 0.0
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "tip": [tree.names[v] for v in order],
            "required": [i < len(required) for i in range(len(order))],
            "gain": gains,
            "weight": w[order],
            "objective_gain": objective_gain,
            "pd": pd_cum,
            "pd_fraction": pd_cum / pd_all if pd_all > 0 else np.zeros(len(order)),
        }
    )
    return Ranking(table=table, rooted=rooted, lam=float(lam), n_required=len(required), pd_all=float(pd_all))


def _marginal_pd(work: Tree, position: np.ndarray, order: list[int]) -> np.ndarray:
    """PD added by each tip of ``order`` in turn, in plain branch lengths.

    ``work`` is rooted at the tree's root (rooted PD) or at the first tip of
    ``order`` (unrooted PD), so in both cases the gain is the path from the tip
    up to the nearest branch that is already covered.
    """
    covered = np.zeros(work.n_nodes, dtype=bool)
    covered[0] = True
    parent = work.parent
    length = work.length
    gains = np.zeros(len(order))
    for i, v in enumerate(order):
        u = int(position[v])
        total = 0.0
        while not covered[u]:
            covered[u] = True
            total += length[u]
            u = int(parent[u])
        gains[i] = total
    return gains


def select(
    tree: Tree,
    k: int,
    weights: dict[str, float] | None = None,
    lam: float | None = None,
    max_pd_loss: float | None = None,
    required: list[str] | None = None,
    exclude: list[str] | None = None,
    rooted: bool = True,
) -> Ranking:
    """Ranking tuned for a selection of size ``k``.

    Give either ``lam`` directly, or ``max_pd_loss``: the largest fraction of
    the optimal PD at size ``k`` you are willing to trade for higher total
    weight. The latter finds, by bisection, the largest ``lam`` whose optimal
    ``k``-selection keeps at least ``(1 - max_pd_loss)`` of the unweighted
    optimum. With neither, or without weights, the weights are ignored.
    """
    kwargs = {"required": required, "exclude": exclude, "rooted": rooted}
    if lam is not None and max_pd_loss is not None:
        raise ValueError("give lam or max_pd_loss, not both")
    if not weights or (lam is None and max_pd_loss is None):
        return rank_tips(tree, None, 0.0, **kwargs)
    if lam is not None:
        return rank_tips(tree, weights, lam, **kwargs)
    if not 0 <= max_pd_loss < 1:
        raise ValueError("max_pd_loss must be in [0, 1)")
    base = rank_tips(tree, weights, 0.0, **kwargs)
    if k > len(base.table):
        raise ValueError(f"k={k} exceeds the {len(base.table)} selectable tips")
    target = (1.0 - max_pd_loss) * base.pd_at(k)
    spread = max(float(np.ptp(base.table.weight.values)), 1e-300)
    low, high = 0.0, base.pd_all / spread  # at ``high`` weight differences dominate everything
    best = base
    trial = rank_tips(tree, weights, high, **kwargs)
    if trial.pd_at(k) >= target:
        return trial
    for _ in range(40):
        mid = np.sqrt(low * high) if low > 0 else high * 1e-6
        trial = rank_tips(tree, weights, mid, **kwargs)
        if trial.pd_at(k) >= target:
            low, best = mid, trial
        else:
            high = mid
        if low > 0 and high / low < 1.02:
            break
    return best
