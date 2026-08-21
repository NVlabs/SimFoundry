# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support-level computation for the stage-5 object removal order.

Overlapping or ambiguous masks can make the pairwise support tests infer
mutual edges (A on top of B and B on top of A), and a longest-path relaxation
over such a graph never terminates. Strongly connected components are
condensed first — the condensation is acyclic by construction — and levels
are the longest path from the topmost objects through it. On an acyclic graph
this matches the plain longest-path levels exactly; members of a cycle share
one level and the removal order falls through to the later tie-breakers.
"""

from __future__ import annotations

from itertools import count


def _tarjan_scc(nodes, succ):
    """Iterative Tarjan. Returns components in reverse topological order
    (every edge points from a later component to an earlier one)."""
    index_of = {}
    lowlink = {}
    on_stack = set()
    stack = []
    comps = []
    counter = count()
    for root in nodes:
        if root in index_of:
            continue
        index_of[root] = lowlink[root] = next(counter)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ.get(root, ())))]
        while work:
            node, it = work[-1]
            descended = False
            for w in it:
                if w not in index_of:
                    index_of[w] = lowlink[w] = next(counter)
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    descended = True
                    break
                if w in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[w])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index_of[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                comps.append(comp)
    return comps


def support_levels(phrases, on_top_of):
    """Cycle-safe support levels. Returns (levels, mutual_groups).

    ``on_top_of[b]`` is the set of objects b sits on top of. Level 0 = nothing
    on top (removed first); an object's level is the longest chain of objects
    stacked above it. ``mutual_groups`` lists any mutually-supporting groups
    (cycles) that were condensed; their members share a level.
    """
    phrases = list(phrases)
    known = set(phrases)
    succ = {p: [w for w in sorted(on_top_of.get(p, ())) if w in known and w != p]
            for p in phrases}
    self_loops = {p for p in phrases if p in on_top_of.get(p, ())}

    comps = _tarjan_scc(phrases, succ)
    comp_of = {n: i for i, comp in enumerate(comps) for n in comp}

    # reversed() is topological order, so each component's level is final
    # before it relaxes its successors.
    comp_level = [0] * len(comps)
    for ci in reversed(range(len(comps))):
        for node in comps[ci]:
            for w in succ[node]:
                cj = comp_of[w]
                if cj != ci and comp_level[cj] < comp_level[ci] + 1:
                    comp_level[cj] = comp_level[ci] + 1

    levels = {n: comp_level[comp_of[n]] for n in phrases}
    mutual_groups = [sorted(comp) for comp in comps
                     if len(comp) > 1 or comp[0] in self_loops]
    return levels, mutual_groups
