import numpy as np
import pytest

from helpers import best_by_enumeration, brute_force_pd, random_tree
from pdpick.core import chain_lengths, farthest_tip, pd_of, rank_tips, select, total_pd
from pdpick.tree import parse_newick


def test_pd_of_on_a_known_tree():
    tree = parse_newick("((A:1,B:2):3,(C:4,D:5):6);")
    assert pd_of(tree, ["A"]) == 4.0
    assert pd_of(tree, ["A", "B"]) == 6.0
    assert pd_of(tree, ["A", "C"]) == 14.0
    assert pd_of(tree, ["A", "B"], rooted=False) == 3.0
    assert pd_of(tree, ["A", "C"], rooted=False) == 14.0
    assert pd_of(tree, ["A"], rooted=False) == 0.0
    assert pd_of(tree, []) == 0.0
    assert pd_of(tree, tree.tip_names) == total_pd(tree) == 21.0


@pytest.mark.parametrize("rooted", [True, False])
def test_pd_of_matches_the_definition(rooted):
    rng = np.random.default_rng(1)
    for _ in range(30):
        tree = random_tree(int(rng.integers(3, 12)), rng, polytomies=True)
        k = int(rng.integers(1, tree.n_tips + 1))
        subset = list(rng.choice(tree.tip_names, size=k, replace=False))
        assert pd_of(tree, subset, rooted) == pytest.approx(brute_force_pd(tree, subset, rooted))


def test_chain_lengths_partition_the_tree():
    rng = np.random.default_rng(2)
    tree = random_tree(60, rng, polytomies=True)
    chain = chain_lengths(tree)
    assert chain.sum() == pytest.approx(total_pd(tree))
    assert np.all(chain[~tree.is_tip] == 0.0)
    deepest = tree.tips[np.argmax(tree.depths()[tree.tips])]
    assert chain[deepest] == pytest.approx(tree.depths()[deepest])
    assert chain.max() == pytest.approx(chain[deepest])


def test_known_greedy_order():
    tree = parse_newick("((A:1,B:2):3,(C:4,D:5):6);")
    ranking = rank_tips(tree)
    assert ranking.table.tip.tolist() == ["D", "B", "C", "A"]
    assert ranking.table.gain.tolist() == [11.0, 5.0, 4.0, 1.0]
    assert ranking.table.pd.tolist() == [11.0, 16.0, 20.0, 21.0]
    assert ranking.top(2) == ["D", "B"]
    assert ranking.table.pd_fraction.iloc[-1] == 1.0


@pytest.mark.parametrize("trial", range(60))
def test_every_prefix_is_optimal(trial):
    """The central claim, checked against exhaustive enumeration."""
    rng = np.random.default_rng(1000 + trial)
    tree = random_tree(int(rng.integers(4, 10)), rng, polytomies=trial % 2 == 0)
    if trial % 5 == 0:  # zero-length branches create ties
        tree.length[int(rng.integers(1, tree.n_nodes))] = 0.0
    names = tree.tip_names
    rooted = trial % 3 != 0
    weights = {t: float(rng.random()) for t in names} if trial % 2 else None
    lam = float(rng.exponential(0.5)) if weights else 0.0
    shuffled = list(rng.permutation(names))
    required = shuffled[: int(rng.integers(1, 3))] if trial % 4 == 0 else []
    exclude = shuffled[3:4] if trial % 7 == 0 else []

    ranking = rank_tips(tree, weights, lam, required, exclude, rooted)
    assert len(ranking.table) == len(names) - len(exclude)
    assert ranking.table.tip.iloc[: len(required)].tolist() == required
    for k in range(max(len(required), 1 if rooted else 2), len(ranking.table) + 1):
        chosen = ranking.top(k)
        value = brute_force_pd(tree, chosen, rooted) + lam * sum((weights or {}).get(t, 0) for t in chosen)
        assert value == pytest.approx(
            best_by_enumeration(tree, k, rooted, weights, lam, required, exclude), abs=1e-9
        )
        assert ranking.pd_at(k) == pytest.approx(brute_force_pd(tree, chosen, rooted), abs=1e-9)


def test_objective_gains_never_increase():
    rng = np.random.default_rng(3)
    tree = random_tree(300, rng)
    weights = {t: float(rng.random()) for t in tree.tip_names}
    for rooted in (True, False):
        table = rank_tips(tree, weights, 0.2, rooted=rooted).table
        gains = table.objective_gain.dropna().values
        assert np.all(np.diff(gains) <= 1e-12)
        assert np.all(np.diff(table.pd.values) >= -1e-12)


def test_unrooted_ranking_starts_with_the_diameter():
    rng = np.random.default_rng(4)
    tree = random_tree(40, rng)
    depth = tree.depths()
    best = max(
        depth[a] + depth[b] - 2 * depth[tree.mrca([a, b])] for a in tree.tips for b in tree.tips if a < b
    )
    ranking = rank_tips(tree, rooted=False)
    assert ranking.pd_at(2) == pytest.approx(best)
    assert ranking.pd_at(1) == 0.0
    assert ranking.table.pd.iloc[-1] == pytest.approx(total_pd(tree.suppress_unary()))


def test_farthest_tip():
    tree = parse_newick("((A:1,B:2):3,(C:4,D:5):6);")
    index = tree.tip_index()
    tip, dist = farthest_tip(tree, index["A"])
    assert tree.names[tip] == "D" and dist == 15.0
    bonus = np.zeros(tree.n_nodes)
    bonus[index["B"]] = 100.0
    tip, dist = farthest_tip(tree, index["A"], bonus)
    assert tree.names[tip] == "B" and dist == 103.0


def test_weights_only_break_ties_when_lam_is_tiny_and_dominate_when_huge():
    rng = np.random.default_rng(5)
    tree = random_tree(80, rng)
    weights = {t: float(rng.random()) for t in tree.tip_names}
    base = rank_tips(tree)
    tiny = rank_tips(tree, weights, 1e-9)
    assert tiny.top(20) == base.top(20)
    huge = rank_tips(tree, weights, 1e9)
    by_weight = sorted(weights, key=lambda t: -weights[t])
    assert set(huge.top(20)) == set(by_weight[:20])


def test_excluded_tips_do_not_change_what_pd_means():
    tree = parse_newick("((A:1,B:2):3,(C:4,D:5):6);")
    ranking = rank_tips(tree, exclude=["C", "D"])
    # the stem above the remaining clade still counts towards rooted PD
    assert ranking.table.tip.tolist() == ["B", "A"]
    assert ranking.table.pd.tolist() == [5.0, 6.0]
    assert ranking.pd_at(2) == pd_of(tree, ["A", "B"])
    assert rank_tips(tree, exclude=["C", "D"], rooted=False).pd_at(2) == 3.0


def test_input_validation():
    tree = parse_newick("((A:1,B:2):3,(C:4,D:5):6);")
    with pytest.raises(KeyError):
        rank_tips(tree, required=["Z"])
    with pytest.raises(KeyError):
        rank_tips(tree, exclude=["Z"])
    with pytest.raises(ValueError, match="both required and excluded"):
        rank_tips(tree, required=["A"], exclude=["A"])
    with pytest.raises(ValueError):
        rank_tips(tree, {"A": 1.0}, lam=-1.0)
    with pytest.raises(ValueError):
        rank_tips(tree, {"A": -1.0}, lam=1.0)
    ranking = rank_tips(tree, required=["A", "C"])
    with pytest.raises(ValueError):
        ranking.top(1)
    with pytest.raises(ValueError):
        ranking.top(5)
    negative = parse_newick("((A:1,B:-2):3,C:4);")
    with pytest.raises(ValueError, match="non-negative"):
        rank_tips(negative)


def test_select_with_a_pd_budget():
    rng = np.random.default_rng(6)
    tree = random_tree(400, rng)
    index = tree.tip_index()
    weights = {}
    for name in tree.tip_names:  # long terminal branches go with low weight
        poor = rng.random() < 0.2
        if poor:
            tree.length[index[name]] += 0.8
        weights[name] = 0.3 if poor else 1.0
    k = 60
    base = select(tree, k)
    assert base.lam == 0.0
    mean = lambda r: float(np.mean([weights[t] for t in r.top(k)]))  # noqa: E731
    previous = mean(base)
    for loss in (0.01, 0.05, 0.2):
        tuned = select(tree, k, weights, max_pd_loss=loss)
        assert tuned.pd_at(k) >= (1 - loss) * base.pd_at(k) - 1e-9
        assert mean(tuned) >= previous - 1e-9
        previous = mean(tuned)
    assert previous > mean(base) + 0.2
    assert select(tree, k, weights, lam=0.5).lam == 0.5
    with pytest.raises(ValueError):
        select(tree, k, weights, lam=0.1, max_pd_loss=0.1)
    with pytest.raises(ValueError):
        select(tree, k, weights, max_pd_loss=1.5)


def test_large_tree_is_fast_and_consistent():
    rng = np.random.default_rng(7)
    tree = random_tree(5000, rng)
    ranking = rank_tips(tree)
    assert ranking.pd_all == pytest.approx(total_pd(tree))
    for k in (10, 500, 2500):
        assert ranking.pd_at(k) == pytest.approx(pd_of(tree, ranking.top(k)))
