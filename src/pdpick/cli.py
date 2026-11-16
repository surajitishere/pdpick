"""Command-line interface: ``pdpick select``, ``pdpick tradeoff``, ``pdpick evaluate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import rank_tips, select
from .evaluate import evaluate
from .tree import read_newick


def read_list(path: str | None) -> list[str]:
    """Tip names from a file: one per line, or the ``tip`` column of a pdpick table.

    Blank lines and ``#`` comments are skipped.
    """
    if not path:
        return []
    rows = [
        line.rstrip("\n").split("\t")
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if rows and "tip" in rows[0]:
        column = rows[0].index("tip")
        return [row[column].strip() for row in rows[1:]]
    return [row[0].strip() for row in rows]


def read_weights(path: str, column: str, id_col: str | None = None) -> dict[str, float]:
    """Tip name -> weight from a TSV/CSV table. Missing values are skipped."""
    table = pd.read_csv(path, sep=None, engine="python", dtype=str)
    id_col = id_col or table.columns[0]
    for col in (id_col, column):
        if col not in table.columns:
            raise KeyError(f"column {col!r} not found; available: {list(table.columns)}")
    values = pd.to_numeric(table[column], errors="coerce")
    keep = values.notna()
    return dict(zip(table.loc[keep, id_col], values[keep].astype(float), strict=True))


def _weights(args) -> dict[str, float] | None:
    if not args.weights:
        return None
    if not args.weight_col:
        raise ValueError("--weights needs --weight-col")
    weights = read_weights(args.weights, args.weight_col, args.id_col)
    if args.min_weight is not None:
        args._below = [t for t, v in weights.items() if v < args.min_weight]
    return weights


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tree", required=True, help="rooted Newick tree with branch lengths")
    p.add_argument("-k", type=int, required=True, help="number of tips to select")
    p.add_argument("--weights", help="table with a per-tip weight (TSV or CSV with header)")
    p.add_argument("--weight-col", help="column of --weights holding the weight")
    p.add_argument("--id-col", help="column of --weights holding tip names (default: first)")
    p.add_argument("--min-weight", type=float, help="exclude tips whose weight is below this")
    p.add_argument("--require", help="file listing tips that must be selected")
    p.add_argument("--exclude", help="file listing tips that may not be selected")
    p.add_argument("--unrooted", action="store_true", help="maximise unrooted PD")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdpick", description="Exact, quality-aware phylogenetic-diversity subsampling."
    )
    parser.add_argument("--version", action="version", version=f"pdpick {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="choose k tips")
    _add_common(sel)
    group = sel.add_mutually_exclusive_group()
    group.add_argument("--lam", type=float, help="branch length one unit of weight is worth")
    group.add_argument(
        "--max-pd-loss", type=float,
        help="largest fraction of optimal PD to give up for higher weight (e.g. 0.02)",
    )  # fmt: skip
    sel.add_argument("--out", required=True, help="output TSV (selected tips, in rank order)")
    sel.add_argument("--full-ranking", action="store_true", help="write every tip, not only the top k")

    trade = sub.add_parser("tradeoff", help="tabulate PD against mean weight over a range of lam")
    _add_common(trade)
    trade.add_argument("--points", type=int, default=25, help="number of lam values")
    trade.add_argument("--out", help="output TSV (default: stdout)")

    ev = sub.add_parser("evaluate", help="score an existing selection")
    ev.add_argument("--tree", required=True)
    ev.add_argument("--selection", required=True, help="file listing the selected tips")
    ev.add_argument("--weights")
    ev.add_argument("--weight-col")
    ev.add_argument("--id-col")
    return parser


def _constraints(args, weights):
    exclude = read_list(args.exclude)
    exclude += [t for t in getattr(args, "_below", []) if t not in set(exclude)]
    tree = read_newick(args.tree)
    tips = set(tree.tip_names)
    return tree, read_list(args.require), [t for t in exclude if t in tips]


def tradeoff_table(tree, k, weights, required, exclude, rooted, points: int = 25) -> pd.DataFrame:
    """PD and mean weight of the optimal ``k``-selection for a log-spaced range of ``lam``."""
    base = rank_tips(tree, weights, 0.0, required, exclude, rooted)
    pd_opt = base.pd_at(k)
    spread = max(float(np.ptp(base.table.weight.values)), 1e-300)
    typical = float(base.table.gain.iloc[k - 1]) / spread  # weight worth one marginal pick
    lams = [0.0, *np.logspace(-2, 3, points - 1) * max(typical, 1e-12)]
    rows = []
    for lam in lams:
        r = base if lam == 0 else rank_tips(tree, weights, lam, required, exclude, rooted)
        top = r.table.iloc[:k]
        rows.append(
            {
                "lam": lam,
                "pd": r.pd_at(k),
                "pd_retained": r.pd_at(k) / pd_opt if pd_opt > 0 else 1.0,
                "mean_weight": float(top.weight.mean()),
                "min_weight": float(top.weight.min()),
                "shared_with_pd_optimum": len(set(top.tip) & set(base.table.tip.iloc[:k])) / k,
            }
        )
    return pd.DataFrame(rows)


def _run(args) -> int:
    if args.command == "evaluate":
        tree = read_newick(args.tree)
        weights = read_weights(args.weights, args.weight_col, args.id_col) if args.weights else None
        print(json.dumps(evaluate(tree, read_list(args.selection), weights), indent=2))
        return 0

    weights = _weights(args)
    tree, required, exclude = _constraints(args, weights)
    rooted = not args.unrooted
    if args.command == "tradeoff":
        if weights is None:
            raise ValueError("tradeoff needs --weights and --weight-col")
        table = tradeoff_table(tree, args.k, weights, required, exclude, rooted, args.points)
        text = table.to_csv(sep="\t", index=False, float_format="%.6g")
        if args.out:
            Path(args.out).write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    ranking = select(
        tree, args.k, weights, lam=args.lam, max_pd_loss=args.max_pd_loss,
        required=required, exclude=exclude, rooted=rooted,
    )  # fmt: skip
    chosen = ranking.top(args.k)
    table = ranking.table if args.full_ranking else ranking.table.iloc[: args.k]
    table.to_csv(args.out, sep="\t", index=False, float_format="%.6g")
    base = ranking if ranking.lam == 0 else rank_tips(tree, None, 0.0, required, exclude, rooted)
    kind = "rooted" if rooted else "unrooted"
    print(f"selected {args.k} of {len(ranking.table)} selectable tips ({len(required)} required)")
    print(
        f"{kind} PD {ranking.pd_at(args.k):.6g} = {ranking.pd_at(args.k) / ranking.pd_all:.1%} of all "
        f"selectable tips; {ranking.pd_at(args.k) / base.pd_at(args.k):.2%} of the unweighted optimum"
    )
    if weights is not None:
        mean_all = float(np.mean([weights.get(t, 0.0) for t in ranking.table.tip]))
        mean_sel = float(np.mean([weights.get(t, 0.0) for t in chosen]))
        mean_base = float(np.mean([weights.get(t, 0.0) for t in base.top(args.k)]))
        print(
            f"mean weight {mean_sel:.4g} (unweighted optimum {mean_base:.4g}, all tips {mean_all:.4g}); "
            f"lam = {ranking.lam:.4g}"
        )
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, KeyError, FileNotFoundError) as err:
        message = err.args[0] if err.args else str(err)
        print(f"pdpick: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
