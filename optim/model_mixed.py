#!/usr/bin/env python3
"""
Mixed-distance tile packing MILP -- threshold-count objective.

At each site we may place a tile of any available code distance (or nothing).
A larger-distance tile has a lower logical error rate where the chip is quiet but
a bigger footprint, so it competes against fitting more small tiles. We maximize
the number of valid, mutually non-overlapping tiles placed, choosing the distance
per site. The set of distances defaults to every one found in the data directory
(see packing_data.available_distances), so d3/d5/d7/... are handled uniformly:

    max   sum_i y_i                       (i ranges over candidates)
    s.t.  y_i allowed only if ler_i <= tau      (baked into the candidate set)
          sum_{d} y_{(p,d)} <= 1          for each origin p  (<= 1 tile per site)
          sum_{i in C} y_i <= 1           for each maximal clique C
          y_i in {0, 1}

A candidate is a (origin, distance) pair with its own footprint span
s = 2*d + 1 (see packing_data.Candidate / mixed_candidates). Because the two
distances have different footprints, the conflict rule is ASYMMETRIC:

    a ~ b  <=>  -s_a <= r_a - r_b <= s_b  and  -s_a <= c_a - c_b <= s_b

which reduces to the symmetric |dr| <= s, |dc| <= s of the single-distance
models when s_a == s_b. Footprints are still axis-aligned boxes, so Helly's
theorem still applies and the clique (site-cover) tightening carries over; the
clique membership test just uses each candidate's own span. See
`mixed_distance_model.tex`.

Usage:
    python3 model_mixed.py <tau> [--distances 3,5,7] [--data-dir DIR]
"""

import argparse
from collections import Counter, defaultdict
from typing import FrozenSet, List

import gurobipy as gp
from gurobipy import GRB

from packing_data import Candidate, available_distances, mixed_candidates


def conflict(a: Candidate, b: Candidate) -> bool:
    """
    True if two placements overlap (share a physical qubit).

    Footprint of a is [r_a, r_a + s_a] x [c_a, c_a + s_a]; likewise for b. The
    row windows overlap iff r_a <= r_b + s_b and r_b <= r_a + s_a, i.e.
    -s_a <= (r_a - r_b) <= s_b; same for columns. Overlap needs both axes.
    Note the asymmetry: the upper bound uses b's span, the lower bound a's.
    """
    dr = a.origin[0] - b.origin[0]
    dc = a.origin[1] - b.origin[1]
    return (-a.span <= dr <= b.span) and (-a.span <= dc <= b.span)


def maximal_cliques_mixed(cands: List[Candidate]) -> List[FrozenSet[int]]:
    """
    Enumerate maximal conflict cliques as covered-site memberships.

    For axis-aligned boxes (any sizes), a set of pairwise-overlapping tiles has a
    common covered site (Helly), and its corner (max origin row, max origin col)
    is an origin coordinate of one of its members. So every maximal clique is
    produced by some witness (a, b) whose members are exactly the candidates
    covering it. Cliques are sets of candidate INDICES (two candidates can share
    an origin). Non-maximal cliques (proper subsets of another) are dropped.

    Spatial indexing keeps this tractable on large chips. A candidate covers
    witness row ``a`` only if its origin row lies in ``[a - max_span, a]`` (span
    is small: 2d+1), so instead of rescanning all candidates per witness we bucket
    by origin row/col and touch only the local band. This turns the naive
    O(rows * cols * |cands|) scan -- ~24 billion iterations on a 200x200 chip --
    into work proportional to the (small) local footprint density. The witness
    column is drawn only from the origin columns present in the row band, since a
    clique's corner column is always a member's origin column.
    """
    if not cands:
        return []
    max_span = max(c.span for c in cands)

    by_row = defaultdict(list)
    for i, c in enumerate(cands):
        by_row[c.origin[0]].append(i)
    rows = sorted(by_row)

    cliques = set()
    for a in rows:
        # Candidates whose footprint covers row a (origin row in [a-max_span, a],
        # then the exact per-candidate span test a <= origin_row + span).
        row_members = [
            i
            for r in range(a - max_span, a + 1)
            for i in by_row.get(r, ())
            if a <= cands[i].origin[0] + cands[i].span
        ]
        if len(row_members) < 2:
            continue
        # Bucket that band by origin column; the witness column ranges only over
        # columns actually present among these members.
        by_col = defaultdict(list)
        for i in row_members:
            by_col[cands[i].origin[1]].append(i)
        for b in by_col:
            members = frozenset(
                i
                for cc in range(b - max_span, b + 1)
                for i in by_col.get(cc, ())
                if b <= cands[i].origin[1] + cands[i].span
            )
            if len(members) >= 2:
                cliques.add(members)

    return _drop_non_maximal(cliques)


def _drop_non_maximal(cliques) -> List[FrozenSet[int]]:
    """
    Keep only maximal cliques (drop any that is a proper subset of another).

    A proper superset of clique ``C`` must contain every member of ``C``, so it is
    found among the cliques containing ``C``'s rarest member. Pivoting on that
    member bounds each check to a small candidate set instead of the whole
    collection, avoiding the O(|cliques|^2) all-pairs comparison.
    """
    clique_list = list(cliques)
    containing = defaultdict(list)
    for k, c in enumerate(clique_list):
        for i in c:
            containing[i].append(k)

    maximal = []
    for k, c in enumerate(clique_list):
        pivot = min(c, key=lambda i: len(containing[i]))
        if not any(clique_list[k2] > c for k2 in containing[pivot] if k2 != k):
            maximal.append(c)
    return maximal


def build_and_solve(tau, distances=None, data_dir=None, verbose: bool = True):
    if distances is None:
        distances = available_distances(data_dir)
    cands = mixed_candidates(tau, distances=distances, data_dir=data_dir)
    cliques = maximal_cliques_mixed(cands)

    model = gp.Model("tile_packing_mixed")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables: y_i = 1 if candidate i (a distance-d tile at its
    # origin) is placed. Keyed by index so co-located D3/D5 never collide.
    y = model.addVars(len(cands), vtype=GRB.BINARY, name="y")

    # Objective: maximize the number of placed tiles.
    model.setObjective(y.sum(), GRB.MAXIMIZE)

    # Non-overlap: at most one tile per maximal clique.
    for k, clique in enumerate(cliques):
        model.addConstr(gp.quicksum(y[i] for i in clique) <= 1, name=f"clique_{k}")

    # At most one tile per origin (choose D3 or D5, not both). Redundant under the
    # clique constraints -- co-located candidates both cover the origin site, so
    # they already share a clique -- but kept explicit for clarity and to guard
    # the model if clique enumeration is ever bypassed.
    by_origin = {}
    for i, c in enumerate(cands):
        by_origin.setdefault(c.origin, []).append(i)
    for origin, idxs in by_origin.items():
        if len(idxs) > 1:
            model.addConstr(gp.quicksum(y[i] for i in idxs) <= 1, name=f"origin_{origin}")

    # LP relaxation bound (illustrates formulation tightness).
    model.update()  # flush pending vars/constraints before copying
    lp = model.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    lp_bound = lp.ObjVal

    model.optimize()

    chosen = sorted(cands[i] for i in range(len(cands)) if y[i].X > 0.5)
    _report(tau, distances, cands, cliques, lp_bound, model, chosen)
    return chosen


def _report(tau, distances, cands, cliques, lp_bound, model, chosen) -> None:
    # Sanity check: no two chosen tiles conflict, and none share an origin.
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            assert not conflict(a, b), f"overlap in solution: {a} vs {b}"
    origins = [c.origin for c in chosen]
    assert len(origins) == len(set(origins)), "two placed tiles share an origin"

    n_by_d = Counter(c.distance for c in chosen)
    cand_by_d = Counter(c.distance for c in cands)
    print("\n=== mixed-distance (threshold-count) formulation ===")
    print(f"tau = {tau}, distances = {tuple(distances)}")
    counts = ", ".join(
        f"D{d}: {cand_by_d.get(d, 0)}" for d in sorted(cand_by_d)
    )
    print(f"|candidates| = {len(cands)} ({counts}), {len(cliques)} clique constraints")
    print(f"LP relaxation bound : {lp_bound:.4f}")
    placed = ", ".join(f"D{d}: {n_by_d.get(d, 0)}" for d in sorted(distances))
    print(f"integer optimum     : {int(round(model.ObjVal))} tiles placed ({placed})")
    print(f"solve time          : {model.Runtime:.3f} s")
    print(f"optimality gap      : {model.MIPGap * 100:.4f}%")
    print("placed (origin, distance, ler):")
    for c in chosen:
        print(f"  {c.origin}  D{c.distance}  ler = {c.ler:.4g}")


def _parse_distances(text: str) -> List[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mixed-distance tile packing MILP (threshold-count)."
    )
    ap.add_argument("tau", type=float, help="LER validity threshold")
    ap.add_argument(
        "--distances",
        type=_parse_distances,
        default=None,
        help="comma-separated code distances to consider "
        "(default: all distances found in the data dir)",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="experiment folder to read results flakes from "
        "(default: $QSNOW_DATA_DIR or the demo path)",
    )
    args = ap.parse_args()
    build_and_solve(args.tau, distances=args.distances, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
