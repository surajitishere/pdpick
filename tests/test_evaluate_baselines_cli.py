import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers import pairwise_tip_distances, random_tree
from pdpick import evaluate, nearest_selected_distance, rank_tips, read_newick
from pdpick.baselines import allocate, random_sample, stratified_sample
from pdpick.cli import main, read_list, read_weights, tradeoff_table

TOY = Path(__file__).parents[1] / "examples" / "toy"


def test_nearest_selected_distance_matches_pairwise_distances():
    rng = np.random.default_rng(1)
    tree = random_tree(25, rng, polytomies=True)
    selected = list(rng.choice(tree.tip_names, size=5, replace=False))
    dist = nearest_selected_distance(tree, selected)
    pairs = pairwise_tip_distances(tree)
    for tip, node in tree.tip_index().items():
        expected = 0.0 if tip in selected else min(pairs[tuple(sorted((tip, s)))] for s in selected)
        assert dist[node] == pytest.approx(expected)
    with pytest.raises(ValueError):
        nearest_selected_distance(tree, [])


def test_evaluate_reports_coverage_and_groups():
    tree = read_newick(TOY / "tree.nwk")
    every = evaluate(tree, tree.tip_names)
    assert every["pd_fraction"] == pytest.approx(1.0) and every["adcl"] == 0.0
    quality = read_weights(TOY / "quality.tsv", "quality")
    clade = pd.read_csv(TOY / "quality.tsv", sep="\t").set_index("genome").clade.to_dict()
    top = rank_tips(tree).top(40)
    result = evaluate(tree, top, quality, {"clade": clade})
    assert result["k"] == 40 and 0 < result["pd_fraction"] < 1
    assert result["max_distance"] >= result["adcl"] > 0
    assert result["n_clade"] == 2
    assert result["mean_weight"] == pytest.approx(np.mean([quality[t] for t in top]))


def test_pd_maximisation_prefers_the_long_branched_poor_genomes_and_weights_fix_it():
    tree = read_newick(TOY / "tree.nwk")
    table = pd.read_csv(TOY / "quality.tsv", sep="\t").set_index("genome")
    quality = table.quality.to_dict()
    poor = set(table.index[table.clade == "poor"])
    plain = rank_tips(tree).top(40)
    weighted = rank_tips(tree, quality, lam=1.0).top(40)
    assert len(poor & set(plain)) / 40 > 2 * len(poor) / len(table)  # over-represented
    assert len(poor & set(weighted)) < len(poor & set(plain)) / 3


def test_allocate():
    sizes = {"a": 50, "b": 30, "c": 15, "d": 5}
    assert allocate(sizes, 2) == {"a": 1, "b": 1, "c": 0, "d": 0}
    assert allocate(sizes, 4) == dict.fromkeys(sizes, 1)
    quota = allocate(sizes, 20)
    assert sum(quota.values()) == 20 and min(quota.values()) >= 1
    assert quota["a"] > quota["b"] > quota["c"] >= quota["d"]
    assert allocate(sizes, 100) == sizes
    assert allocate({"x": 1, "y": 99}, 50) == {"y": 49, "x": 1}
    with pytest.raises(ValueError):
        allocate(sizes, 101)


def test_stratified_and_random_samples():
    strata = {f"t{i}": "abc"[i % 3] for i in range(30)}
    score = {f"t{i}": float(i) for i in range(30)}
    picks = stratified_sample(strata, 6, score)
    assert sorted(picks) == ["t24", "t25", "t26", "t27", "t28", "t29"]
    rng = np.random.default_rng(2)
    random_within = stratified_sample(strata, 9, rng=rng)
    assert len(set(random_within)) == 9
    assert {strata[t] for t in random_within} == {"a", "b", "c"}
    with pytest.raises(ValueError):
        stratified_sample(strata, 3)
    sample = random_sample(sorted(strata), 10, rng)
    assert len(set(sample)) == 10 and set(sample) <= set(strata)


def test_tradeoff_table_is_monotone():
    tree = read_newick(TOY / "tree.nwk")
    quality = read_weights(TOY / "quality.tsv", "quality")
    table = tradeoff_table(tree, 40, quality, [], [], True, points=12)
    assert table.lam.iloc[0] == 0 and table.pd_retained.iloc[0] == 1.0
    assert np.all(np.diff(table.pd.values) <= 1e-9)
    assert np.all(np.diff(table.mean_weight.values) >= -1e-9)
    assert table.mean_weight.iloc[-1] > table.mean_weight.iloc[0] + 0.2


def test_readers(tmp_path):
    plain = tmp_path / "list.txt"
    plain.write_text("# comment\nA\n\nB\textra\n")
    assert read_list(str(plain)) == ["A", "B"]
    assert read_list(None) == []
    table = tmp_path / "w.csv"
    table.write_text("id,score\nA,0.5\nB,\nC,x\nD,2\n")
    assert read_weights(str(table), "score") == {"A": 0.5, "D": 2.0}
    with pytest.raises(KeyError):
        read_weights(str(table), "nope")


def test_cli_select_tradeoff_evaluate(tmp_path, capsys):
    out = tmp_path / "picks.tsv"
    common = ["--tree", str(TOY / "tree.nwk"), "-k", "40"]
    weights = ["--weights", str(TOY / "quality.tsv"), "--weight-col", "quality"]
    assert main(["select", *common, *weights, "--max-pd-loss", "0.05",
                 "--require", str(TOY / "required.txt"), "--out", str(out)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert "2 required" in printed and "of the unweighted optimum" in printed
    picks = pd.read_csv(out, sep="\t")
    assert len(picks) == 40 and picks.tip.iloc[:2].tolist() == ["g001", "g002"]
    assert read_list(str(out)) == picks.tip.tolist()

    assert main(["select", *common, "--unrooted", "--full-ranking", "--out", str(out)]) == 0
    assert len(pd.read_csv(out, sep="\t")) == 200
    capsys.readouterr()

    assert main(["select", *common, *weights, "--min-weight", "0.8", "--out", str(out)]) == 0
    assert "of 170 selectable tips" in capsys.readouterr().out
    quality = read_weights(str(TOY / "quality.tsv"), "quality")
    assert min(quality[t] for t in pd.read_csv(out, sep="\t").tip) >= 0.8

    assert main(["tradeoff", *common, *weights, "--points", "6"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("lam\tpd\tpd_retained")

    assert main(["evaluate", "--tree", str(TOY / "tree.nwk"), "--selection", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["k"] == 40


def test_cli_errors(tmp_path, capsys):
    common = ["--tree", str(TOY / "tree.nwk"), "-k", "40", "--out", str(tmp_path / "x.tsv")]
    assert main(["select", *common, "--weights", str(TOY / "quality.tsv")]) == 2
    assert "--weight-col" in capsys.readouterr().err
    assert main(["select", "--tree", str(TOY / "tree.nwk"), "-k", "999", "--out", "x"]) == 2
    assert main(["tradeoff", "--tree", str(TOY / "tree.nwk"), "-k", "10"]) == 2
