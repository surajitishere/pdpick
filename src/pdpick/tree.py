"""Array-backed rooted trees and a small Newick reader/writer.

Nodes are stored in **pre-order**: the root is node 0 and ``parent[i] < i`` for
every other node. That single invariant keeps tree algorithms simple and fast
-- a post-order sweep is "walk the arrays backwards", a pre-order sweep is
"walk them forwards", a subtree is a contiguous block, and no pointer chasing
or recursion is needed.

Deliberately small: parsing and writing Newick, pruning to a tip set, resolving
polytomies, re-rooting, MRCA lookup and a few per-node summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Tree", "NewickError", "parse_newick", "read_newick"]


class NewickError(ValueError):
    """Raised when a Newick string cannot be parsed."""


_STRUCTURAL = set("(),:;[")


@dataclass
class Tree:
    """A rooted tree held in flat, pre-ordered arrays.

    Attributes
    ----------
    parent : (n,) int array
        Index of each node's parent; ``-1`` for the root (node 0).
    length : (n,) float array
        Length of the branch above each node (``0.0`` for the root).
    names : list of str
        Node labels; ``""`` where the Newick string had none.
    virtual : (n,) bool array
        ``True`` for nodes that were inserted by :meth:`binarize` to resolve a
        polytomy. They sit at zero distance from the real node they came from
        and are excluded from reports.
    """

    parent: np.ndarray
    length: np.ndarray
    names: list[str]
    virtual: np.ndarray | None = None
    children: list[list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.parent = np.asarray(self.parent, dtype=np.int64)
        self.length = np.asarray(self.length, dtype=np.float64)
        n = self.parent.size
        if self.length.size != n or len(self.names) != n:
            raise ValueError("parent, length and names must have the same size")
        if n == 0 or self.parent[0] != -1:
            raise ValueError("node 0 must be the root (parent == -1)")
        if n > 1 and not np.all(self.parent[1:] < np.arange(1, n)):
            raise ValueError("nodes must be in pre-order (parent[i] < i)")
        if n > 1 and np.any(self.parent[1:] < 0):
            raise ValueError("only node 0 may be a root")
        if self.virtual is None:
            self.virtual = np.zeros(n, dtype=bool)
        self.children = [[] for _ in range(n)]
        for i in range(1, n):
            self.children[self.parent[i]].append(i)

    # ------------------------------------------------------------------ basics
    @property
    def n_nodes(self) -> int:
        return int(self.parent.size)

    @property
    def is_tip(self) -> np.ndarray:
        return np.fromiter((len(c) == 0 for c in self.children), bool, self.n_nodes)

    @property
    def tips(self) -> np.ndarray:
        """Indices of tip nodes, in pre-order."""
        return np.flatnonzero(self.is_tip)

    @property
    def n_tips(self) -> int:
        return int(self.is_tip.sum())

    @property
    def tip_names(self) -> list[str]:
        return [self.names[i] for i in self.tips]

    def tip_index(self) -> dict[str, int]:
        """Map tip name -> node index. Raises if tip names are missing or repeated."""
        out: dict[str, int] = {}
        for i in self.tips:
            name = self.names[i]
            if name == "":
                raise ValueError("tree has an unnamed tip")
            if name in out:
                raise ValueError(f"duplicate tip name: {name!r}")
            out[name] = int(i)
        return out

    def is_binary(self) -> bool:
        return all(len(c) in (0, 2) for c in self.children)

    # --------------------------------------------------------------- summaries
    def depths(self) -> np.ndarray:
        """Root-to-node path length for every node."""
        d = np.zeros(self.n_nodes)
        for i in range(1, self.n_nodes):
            d[i] = d[self.parent[i]] + self.length[i]
        return d

    def tip_counts(self) -> np.ndarray:
        """Number of tips descending from every node (tips count themselves)."""
        c = self.is_tip.astype(np.int64)
        for i in range(self.n_nodes - 1, 0, -1):
            c[self.parent[i]] += c[i]
        return c

    def mean_tip_distance(self) -> np.ndarray:
        """Mean path length from every node to the tips below it.

        On an ultrametric tree this is the node's age. On a tree in
        substitutions per site it is the natural stand-in for age: how much
        branch length separates the node from the data, on average.
        """
        count = self.is_tip.astype(np.float64)
        total = np.zeros(self.n_nodes)
        for i in range(self.n_nodes - 1, 0, -1):
            p = self.parent[i]
            total[p] += total[i] + count[i] * self.length[i]
            count[p] += count[i]
        return total / count

    def edge_heights(self) -> np.ndarray:
        """Number of edges on the longest path from each node down to a tip."""
        h = np.zeros(self.n_nodes, dtype=np.int64)
        for i in range(self.n_nodes - 1, 0, -1):
            p = self.parent[i]
            if h[i] + 1 > h[p]:
                h[p] = h[i] + 1
        return h

    def edge_depths(self) -> np.ndarray:
        """Number of edges between the root and each node."""
        d = np.zeros(self.n_nodes, dtype=np.int64)
        for i in range(1, self.n_nodes):
            d[i] = d[self.parent[i]] + 1
        return d

    # -------------------------------------------------------------------- MRCA
    def mrca(self, nodes) -> int:
        """Most recent common ancestor of a collection of node indices."""
        nodes = [int(x) for x in nodes]
        if not nodes:
            raise ValueError("mrca() needs at least one node")
        a = nodes[0]
        for b in nodes[1:]:
            # Pre-order means an ancestor always has a smaller index, so walking
            # the larger index upwards converges on the MRCA.
            while a != b:
                if a > b:
                    a = int(self.parent[a])
                else:
                    b = int(self.parent[b])
        return a

    def descendants_tips(self, node: int) -> np.ndarray:
        """Tip indices below ``node`` (a contiguous pre-order block)."""
        end = self.subtree_end(node)
        block = np.arange(node, end)
        return block[self.is_tip[node:end]]

    def subtree_end(self, node: int) -> int:
        """One past the last pre-order index in the subtree rooted at ``node``."""
        end = node + 1
        n = self.n_nodes
        while end < n and self.parent[end] >= node:
            end += 1
        return end

    # ------------------------------------------------------------ restructuring
    def prune(self, keep_tips) -> Tree:
        """Return the tree induced by a subset of tips.

        Nodes left with a single child are removed and their branch lengths are
        merged into the child, so path lengths between the kept tips are
        unchanged. If the old root ends up with one child the new root is the
        MRCA of the kept tips.
        """
        keep_tips = set(keep_tips)
        index = self.tip_index()
        unknown = keep_tips - index.keys()
        if unknown:
            example = sorted(unknown)[:3]
            raise KeyError(f"{len(unknown)} tips not in tree, e.g. {example}")
        n = self.n_nodes
        kept_below = np.zeros(n, dtype=np.int64)
        for name in keep_tips:
            kept_below[index[name]] = 1
        if kept_below.sum() < 2:
            raise ValueError("need at least two tips to keep")
        n_kept_children = np.zeros(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            if kept_below[i]:
                p = self.parent[i]
                kept_below[p] += kept_below[i]
                n_kept_children[p] += 1

        is_tip = self.is_tip
        emit = (is_tip & (kept_below > 0)) | (n_kept_children >= 2)
        new_index = np.full(n, -1, dtype=np.int64)
        anchor = np.full(n, -1, dtype=np.int64)  # nearest emitted proper ancestor (new index)
        acc = np.zeros(n)  # branch length accumulated since that ancestor
        new_parent: list[int] = []
        new_length: list[float] = []
        new_names: list[str] = []
        new_virtual: list[bool] = []
        for i in range(n):
            if kept_below[i] == 0:
                continue
            p = self.parent[i]
            if p >= 0:
                if emit[p]:
                    anchor[i] = new_index[p]
                    acc[i] = self.length[i]
                else:
                    anchor[i] = anchor[p]
                    acc[i] = acc[p] + self.length[i]
            if emit[i]:
                new_index[i] = len(new_parent)
                new_parent.append(int(anchor[i]))
                new_length.append(float(acc[i]) if anchor[i] >= 0 else 0.0)
                new_names.append(self.names[i])
                new_virtual.append(bool(self.virtual[i]))
        return Tree(
            np.array(new_parent), np.array(new_length), new_names, np.array(new_virtual, dtype=bool)
        )

    def suppress_unary(self) -> Tree:
        """Remove degree-two nodes (merging their branches)."""
        if all(len(c) != 1 for c in self.children):
            return self
        return self.prune(self.tip_names)

    def binarize(self) -> Tree:
        """Resolve polytomies with zero-length branches.

        A node with m > 2 children becomes a ladder of m - 1 nodes joined by
        zero-length branches. Likelihoods and the marginal posteriors of the
        original nodes are unchanged; the inserted nodes are flagged in
        ``virtual``. Unary nodes are removed first.
        """
        tree = self.suppress_unary()
        if tree.is_binary():
            return tree
        return _binarize_explicit(tree)

    def reroot(self, node: int) -> Tree:
        """The same tree hung from an existing node (a tip or an internal node).

        Branch lengths and node names travel with the nodes. Re-rooting at a tip
        makes that tip the root with a single child, which is what is needed to
        measure everything relative to it. If the old root had two children it
        is left in place as a node with one child; call :meth:`suppress_unary`
        to remove it.
        """
        node = int(node)
        parent: list[int] = []
        length: list[float] = []
        names: list[str] = []
        virtual: list[bool] = []
        # (old node, old node we arrived from, new parent index, branch length)
        stack: list[tuple[int, int, int, float]] = [(node, -1, -1, 0.0)]
        while stack:
            v, came_from, new_par, ln = stack.pop()
            new = len(parent)
            parent.append(new_par)
            length.append(ln)
            names.append(self.names[v])
            virtual.append(bool(self.virtual[v]))
            onward = [(c, float(self.length[c])) for c in self.children[v] if c != came_from]
            up = int(self.parent[v])
            if up >= 0 and up != came_from:
                onward.append((up, float(self.length[v])))
            for u, edge in reversed(onward):
                stack.append((u, v, new, edge))
        return Tree(np.array(parent), np.array(length), names, np.array(virtual, dtype=bool))

    # ---------------------------------------------------------------- output
    def to_newick(self, digits: int = 10, internal_names: bool = True) -> str:
        """Serialise to Newick (iteratively, so deep trees are fine)."""
        out = [""] * self.n_nodes
        fmt = f"{{:.{digits}g}}"
        for i in range(self.n_nodes - 1, -1, -1):
            kids = self.children[i]
            label = _quote(self.names[i]) if (not kids or internal_names) else ""
            body = "(" + ",".join(out[k] for k in kids) + ")" if kids else ""
            for k in kids:
                out[k] = ""  # free memory as we go
            if i == 0:
                out[i] = body + label + ";"
            else:
                out[i] = body + label + ":" + fmt.format(self.length[i])
        return out[0]


def _binarize_explicit(tree: Tree) -> Tree:
    """Build a binary copy of ``tree`` (see :meth:`Tree.binarize`).

    A node with children k0..k_{m-1}, m > 2, becomes
    ``node -> (k0, v1)``, ``v1 -> (k1, v2)``, ..., ``v_{m-2} -> (k_{m-2}, k_{m-1})``
    with every ``v`` a zero-length virtual node. Items are pushed so that they
    pop in pre-order.
    """
    parent: list[int] = []
    length: list[float] = []
    names: list[str] = []
    virtual: list[bool] = []
    # Work items: ("node", old, new_parent, length) or ("ladder", old, k, new_parent)
    # where a ladder item creates the virtual node holding children k.. of ``old``.
    stack: list[tuple] = [("node", 0, -1, 0.0)]
    while stack:
        item = stack.pop()
        if item[0] == "node":
            _, old, par, ln = item
            new = len(parent)
            parent.append(par)
            length.append(ln)
            names.append(tree.names[old])
            virtual.append(bool(tree.virtual[old]))
            kids = tree.children[old]
            if len(kids) <= 2:
                for k in reversed(kids):
                    stack.append(("node", k, new, float(tree.length[k])))
            else:
                stack.append(("ladder", old, 1, new))
                stack.append(("node", kids[0], new, float(tree.length[kids[0]])))
        else:
            _, old, k, par = item
            kids = tree.children[old]
            new = len(parent)
            parent.append(par)
            length.append(0.0)
            names.append("")
            virtual.append(True)
            if k == len(kids) - 2:
                stack.append(("node", kids[k + 1], new, float(tree.length[kids[k + 1]])))
            else:
                stack.append(("ladder", old, k + 1, new))
            stack.append(("node", kids[k], new, float(tree.length[kids[k]])))
    return Tree(np.array(parent), np.array(length), names, np.array(virtual, dtype=bool))


def _quote(name: str) -> str:
    if name == "":
        return ""
    if any(ch in _STRUCTURAL or ch.isspace() or ch in "']" for ch in name):
        return "'" + name.replace("'", "''") + "'"
    return name


def parse_newick(text: str) -> Tree:
    """Parse a Newick string into a :class:`Tree`.

    Handles quoted labels, ``[comments]``, internal labels, scientific-notation
    branch lengths and polytomies. Missing branch lengths become ``0.0``.
    Underscores in labels are kept as they are.
    """
    parent: list[int] = [-1]
    length: list[float] = [0.0]
    names: list[str] = [""]
    cur = 0
    i = 0
    n = len(text)
    depth = 0
    finished = False
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch == "(":
            parent.append(cur)
            length.append(0.0)
            names.append("")
            cur = len(parent) - 1
            depth += 1
            i += 1
        elif ch == ",":
            if depth == 0:
                raise NewickError(f"unexpected ',' at position {i}")
            parent.append(parent[cur])
            length.append(0.0)
            names.append("")
            cur = len(parent) - 1
            i += 1
        elif ch == ")":
            if depth == 0:
                raise NewickError(f"unbalanced ')' at position {i}")
            cur = parent[cur]
            depth -= 1
            i += 1
        elif ch == "[":
            j = text.find("]", i)
            if j < 0:
                raise NewickError("unterminated [comment]")
            i = j + 1
        elif ch == ":":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            k = j
            while k < n and text[k] not in _STRUCTURAL and text[k] != ")" and not text[k].isspace():
                k += 1
            try:
                length[cur] = float(text[j:k])
            except ValueError as err:
                raise NewickError(f"bad branch length {text[j:k]!r} at position {j}") from err
            i = k
        elif ch == ";":
            finished = True
            break
        elif ch == "'":
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise NewickError("unterminated quoted label")
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(text[j])
                j += 1
            names[cur] = "".join(buf)
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in _STRUCTURAL and text[j] != ")" and not text[j].isspace():
                j += 1
            names[cur] = text[i:j]
            i = j
    if depth != 0:
        raise NewickError("unbalanced parentheses")
    if not finished and len(parent) == 1 and names[0] == "":
        raise NewickError("empty Newick string")
    if len(parent) < 2:
        raise NewickError("tree has a single node")
    length[0] = 0.0
    return Tree(np.array(parent), np.array(length), names)


def read_newick(path: str | Path) -> Tree:
    """Read the first tree in a Newick file."""
    return parse_newick(Path(path).read_text())
