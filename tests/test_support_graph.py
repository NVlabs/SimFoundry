# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cycle-safe support levels: exact DAG equivalence with the old BFS, and
termination + well-defined levels on the cyclic graphs that used to hang."""

from collections import defaultdict, deque

import numpy as np
import pytest

from simfoundry.pipeline.support_graph import support_levels


def d(**kw):
    return defaultdict(set, {k: set(v) for k, v in kw.items()})


def old_bfs_capped(phrases, on_top_of, cap):
    """The pre-fix algorithm, verbatim, with a cap standing in for the hang."""
    supports = defaultdict(set)
    for b, unders in on_top_of.items():
        for a in unders:
            supports[a].add(b)
    support_level = {}
    queue = deque(p for p in phrases if len(supports.get(p, set())) == 0)
    for leaf in queue:
        support_level[leaf] = 0
    steps = 0
    while queue:
        steps += 1
        assert steps <= cap, "old algorithm would hang"
        current = queue.popleft()
        for below in on_top_of.get(current, set()):
            new_level = support_level[current] + 1
            if below not in support_level or support_level[below] < new_level:
                support_level[below] = new_level
                queue.append(below)
    for p in phrases:
        support_level.setdefault(p, 0)
    return support_level


@pytest.mark.parametrize("desc,on_top_of,expected,n_groups", [
    ("reported hang: leaf on a mutual pair", d(C={"A", "B"}, A={"B"}, B={"A"}),
     {"C": 0, "A": 1, "B": 1}, 1),
    ("leafless mutual pair", d(A={"B"}, B={"A"}), {"A": 0, "B": 0}, 1),
    ("chain", d(a={"b"}, b={"c"}), {"a": 0, "b": 1, "c": 2}, 0),
    ("diamond takes the longest path", d(c={"a", "b"}, b={"a"}),
     {"c": 0, "b": 1, "a": 2}, 0),
    ("object under a cycle", d(A={"B", "T"}, B={"A", "T"}),
     {"A": 0, "B": 0, "T": 1}, 1),
    ("3-cycle with a leaf", d(L={"x"}, x={"y"}, y={"z"}, z={"x"}),
     {"L": 0, "x": 1, "y": 1, "z": 1}, 1),
    ("self-loop", d(A={"A", "B"}), {"A": 0, "B": 1}, 1),
    ("edge to unknown phrase ignored", d(A={"ghost"}), {"A": 0}, 0),
])
def test_structured_cases(desc, on_top_of, expected, n_groups):
    levels, groups = support_levels(sorted(expected), on_top_of)
    assert levels == expected, desc
    assert len(groups) == n_groups, desc


def _random_graph(rng, n, p_edge, force_dag):
    phrases = [f"o{i}" for i in range(n)]
    rank = {p: r for p, r in zip(phrases, rng.permutation(n))}
    on_top_of = defaultdict(set)
    for i in phrases:
        for j in phrases:
            if i == j or (force_dag and rank[i] >= rank[j]):
                continue
            if rng.random() < p_edge:
                on_top_of[i].add(j)
    return phrases, on_top_of


def _reachable(on_top_of, src):
    seen, stack = {src}, [src]
    while stack:
        for w in on_top_of.get(stack.pop(), ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def test_random_dags_match_old_algorithm_exactly():
    rng = np.random.default_rng(7)
    for _ in range(400):
        n = int(rng.integers(1, 22))
        phrases, on_top_of = _random_graph(rng, n, float(rng.uniform(0.05, 0.5)), True)
        levels, groups = support_levels(phrases, on_top_of)
        assert levels == old_bfs_capped(phrases, on_top_of, cap=10 * n * n + 100)
        assert groups == []


def test_random_cyclic_graphs_terminate_with_consistent_levels():
    rng = np.random.default_rng(11)
    for _ in range(400):
        n = int(rng.integers(2, 25))
        phrases, on_top_of = _random_graph(rng, n, float(rng.uniform(0.05, 0.6)), False)
        levels, groups = support_levels(phrases, on_top_of)
        assert set(levels) == set(phrases)
        assert all(0 <= lv < n for lv in levels.values())
        for g in groups:
            assert len({levels[m] for m in g}) == 1
        reach = {p: _reachable(on_top_of, p) for p in phrases}
        for b, unders in on_top_of.items():
            for a in unders:
                if a == b:
                    continue
                if b in reach[a]:  # same SCC: shared level
                    assert levels[a] == levels[b]
                else:  # cross edge: a is strictly deeper than b
                    assert levels[a] >= levels[b] + 1


def test_deep_chain_and_giant_ring():
    n = 5000
    chain = defaultdict(set, {f"n{i}": {f"n{i + 1}"} for i in range(n - 1)})
    levels, groups = support_levels([f"n{i}" for i in range(n)], chain)
    assert levels["n0"] == 0 and levels[f"n{n - 1}"] == n - 1 and not groups

    ring = defaultdict(set, {f"n{i}": {f"n{(i + 1) % n}"} for i in range(n)})
    levels, groups = support_levels([f"n{i}" for i in range(n)], ring)
    assert set(levels.values()) == {0}
    assert len(groups) == 1 and len(groups[0]) == n
