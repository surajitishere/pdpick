"""Reference sampling schemes to compare PD maximisation against."""

from __future__ import annotations

import numpy as np

__all__ = ["random_sample", "stratified_sample", "allocate"]


def random_sample(tips: list[str], k: int, rng: np.random.Generator) -> list[str]:
    """``k`` tips drawn uniformly without replacement."""
    return [tips[i] for i in rng.choice(len(tips), size=k, replace=False)]


def allocate(sizes: dict[str, int], k: int) -> dict[str, int]:
    """Split ``k`` picks across strata: one each first, the rest in proportion to size.

    If ``k`` is smaller than the number of strata, the ``k`` largest strata get
    one pick each. The proportional part uses largest remainders and never
    exceeds a stratum's size. Ties are broken by stratum name.
    """
    if k > sum(sizes.values()):
        raise ValueError("k exceeds the number of tips")
    names = sorted(sizes, key=lambda s: (-sizes[s], s))
    if k <= len(names):
        return {s: (1 if i < k else 0) for i, s in enumerate(names)}
    quota = dict.fromkeys(names, 1)
    remaining = k - len(names)
    while remaining > 0:
        room = {s: sizes[s] - quota[s] for s in names if sizes[s] > quota[s]}
        total = sum(room.values())
        share = {s: remaining * room[s] / total for s in room}
        gave = 0
        for s in room:
            add = min(int(share[s]), room[s])
            quota[s] += add
            gave += add
        if gave == 0:  # hand out the leftovers by largest remainder
            for s in sorted(room, key=lambda s: (-(share[s] - int(share[s])), s))[:remaining]:
                quota[s] += 1
                gave += 1
        remaining -= gave
    return quota


def stratified_sample(
    strata: dict[str, str],
    k: int,
    score: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """Taxonomy-stratified sampling.

    ``strata`` maps every tip to a stratum label (for example its phylum).
    Picks are allocated with :func:`allocate`; within a stratum the tips with
    the highest ``score`` are taken (ties by name), or random ones when
    ``score`` is ``None`` and ``rng`` is given.
    """
    members: dict[str, list[str]] = {}
    for tip, label in strata.items():
        members.setdefault(label, []).append(tip)
    quota = allocate({s: len(m) for s, m in members.items()}, k)
    out: list[str] = []
    for label in sorted(members):
        tips = sorted(members[label])
        n = quota[label]
        if n == 0:
            continue
        if score is not None:
            tips.sort(key=lambda t: (-score.get(t, 0.0), t))
            out.extend(tips[:n])
        elif rng is not None:
            out.extend(tips[i] for i in rng.choice(len(tips), size=n, replace=False))
        else:
            raise ValueError("give a score or a random generator")
    return out
