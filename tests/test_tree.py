import numpy as np
import pytest

from helpers import pairwise_tip_distances, random_tree
from pdpick.tree import NewickError, parse_newick


def test_parse_basic_structure():
    t = parse_newick("((A:1,B:2)AB:0.5,C:3)root;")
    assert t.names == ["root", "AB", "A", "B", "C"]
    assert t.parent.tolist() == [-1, 0, 1, 1, 0]
    assert t.length.tolist() == [0.0, 0.5, 1.0, 2.0, 3.0]
    assert t.tip_names == ["A", "B", "C"]
    assert t.is_binary()


def test_parse_quotes_comments_and_scientific_notation():
    t = parse_newick("('it''s a tip':1e-3,[comment]'B (x)':2.5E+1 , C_c : 0.1)[&x=1];")
    assert t.tip_names == ["it's a tip", "B (x)", "C_c"]
    assert np.allclose(t.length[1:], [1e-3, 25.0, 0.1])
    assert not t.is_binary()


def test_missing_lengths_default_to_zero_and_root_length_is_dropped():
    t = parse_newick("((A,B),C):7;")
    assert np.all(t.length == 0.0)


@pytest.mark.parametrize("bad", ["", "((A,B);", "(A,B));", "(A:x,B:1);", "A;", "('A,B);"])
def test_parse_errors(bad):
    with pytest.raises(NewickError):
        parse_newick(bad)


def test_duplicate_and_unnamed_tips_are_rejected_on_lookup():
    with pytest.raises(ValueError, match="duplicate"):
        parse_newick("(A:1,A:1);").tip_index()
    with pytest.raises(ValueError, match="unnamed"):
        parse_newick("(A:1,:1);").tip_index()


def test_newick_round_trip():
    rng = np.random.default_rng(0)
    for poly in (False, True):
        t = random_tree(30, rng, polytomies=poly)
        t.names[0] = "the root"
        again = parse_newick(t.to_newick(digits=17))
        assert again.names == t.names
        assert again.parent.tolist() == t.parent.tolist()
        assert np.allclose(again.length, t.length)


def test_mrca_and_descendants():
    t = parse_newick("(((A:1,B:1)x:1,C:1)y:1,(D:1,E:1)z:1)r;")
    idx = t.tip_index()
    assert t.names[t.mrca([idx["A"], idx["B"]])] == "x"
    assert t.names[t.mrca([idx["A"], idx["C"]])] == "y"
    assert t.names[t.mrca([idx["B"], idx["E"]])] == "r"
    assert t.names[t.mrca([idx["D"], idx["E"], idx["D"]])] == "z"
    y = t.names.index("y")
    assert [t.names[i] for i in t.descendants_tips(y)] == ["A", "B", "C"]


def test_summaries_on_a_known_tree():
    t = parse_newick("((A:1,B:3)x:2,C:4)r;")
    assert t.depths().tolist() == [0.0, 2.0, 3.0, 5.0, 4.0]
    assert t.tip_counts().tolist() == [3, 2, 1, 1, 1]
    # x: mean(1, 3) = 2 ; r: mean(3, 5, 4) = 4
    assert np.allclose(t.mean_tip_distance(), [4.0, 2.0, 0.0, 0.0, 0.0])
    assert t.edge_heights().tolist() == [2, 1, 0, 0, 0]


def test_prune_preserves_distances_between_kept_tips():
    rng = np.random.default_rng(1)
    t = random_tree(40, rng, polytomies=True)
    keep = sorted(rng.choice(t.tip_names, size=15, replace=False))
    sub = t.prune(keep)
    assert sorted(sub.tip_names) == keep
    assert all(len(c) != 1 for c in sub.children)
    before = pairwise_tip_distances(t)
    after = pairwise_tip_distances(sub)
    assert all(np.isclose(after[key], before[key]) for key in after)


def test_prune_errors():
    t = parse_newick("((A:1,B:1):1,C:1);")
    with pytest.raises(KeyError):
        t.prune(["A", "Z"])
    with pytest.raises(ValueError):
        t.prune(["A"])


def test_binarize_resolves_polytomies_without_changing_distances():
    rng = np.random.default_rng(2)
    t = random_tree(35, rng, polytomies=True)
    b = t.binarize()
    assert b.is_binary()
    assert b.tip_names == t.tip_names
    assert int((~b.virtual).sum()) == t.n_nodes
    assert np.all(b.length[b.virtual] == 0.0)
    before = pairwise_tip_distances(t)
    after = pairwise_tip_distances(b)
    assert all(np.isclose(after[key], before[key]) for key in before)
    # real nodes keep their pre-order, so labels line up one to one
    assert [n for n, v in zip(b.names, b.virtual, strict=True) if not v] == t.names


def test_binarize_removes_unary_nodes():
    t = parse_newick("(((A:1,B:1):1):1,C:1);")
    b = t.binarize()
    assert b.is_binary() and b.n_nodes == 5
    idx = b.tip_index()
    assert b.depths()[idx["A"]] == 3.0


def test_reroot_preserves_every_distance():
    rng = np.random.default_rng(3)
    t = random_tree(25, rng, polytomies=True)
    before = pairwise_tip_distances(t)
    internal = int(np.flatnonzero(~t.is_tip)[5])
    hung = t.reroot(internal)
    assert hung.names[0] == t.names[internal]
    assert sorted(hung.tip_names) == sorted(t.tip_names)
    after = pairwise_tip_distances(hung)
    assert all(np.isclose(after[key], before[key]) for key in before)


def test_reroot_at_a_tip_measures_distances_from_it():
    t = parse_newick("((A:1,B:2)x:3,(C:4,D:5)y:6)r;")
    hung = t.reroot(t.tip_index()["A"])
    assert hung.names[0] == "A" and len(hung.children[0]) == 1
    depth = dict(zip(hung.names, hung.depths(), strict=True))
    assert depth == {"A": 0.0, "x": 1.0, "B": 3.0, "r": 4.0, "y": 10.0, "C": 14.0, "D": 15.0}
    # the old root now has one child; removing it leaves distances untouched
    tidy = hung.reroot(1).suppress_unary()
    assert "r" not in tidy.names
