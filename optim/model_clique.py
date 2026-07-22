#!/usr/bin/env python3
"""
Tile packing MILP -- TIGHTER formulation (clique / site-cover constraints).

Same objective and feasible integer set as `model_pairwise.py`, but the
non-overlap requirement is expressed with clique constraints instead of edges:

    max   sum_{p in V} x_p
    s.t.  sum_{p in C_g} x_p <= 1   for each maximal clique C_g
          x_p in {0, 1}

Because tile footprints are equal-size axis-aligned boxes, any set of pairwise
overlapping tiles shares a common point (Helly's theorem), so every maximal
clique corresponds to a chip window covered by all its tiles. A window with
top-left corner (a, b) covers exactly the origins in [a, a+s] x [b, b+s]; sliding
the corner over the distinct origin rows/cols enumerates all maximal cliques and
covers every conflicting pair. Each clique constraint dominates the pairwise edges
it spans, giving a tighter LP relaxation.

Usage:
    python3 model_clique.py <tau> [--distance D]
"""

import argparse
from typing import FrozenSet, List

import gurobipy as gp
from gurobipy import GRB

from packing_data import Coord, footprint_span, valid_placements


def maximal_cliques(placements: List[Coord], span: int) -> List[FrozenSet[Coord]]:
    """
    Enumerate maximal conflict cliques as covered-window memberships.

    For each candidate window corner (a, b) drawn from the distinct origin rows
    and columns, collect the origins inside [a, a+s] x [b, b+s]; these mutually
    overlap, so they form a clique. Deduplicate and drop any clique contained in
    another to keep only the maximal ones.
    """
    rows = sorted({r for r, _ in placements})
    cols = sorted({c for _, c in placements})

    cliques = set()
    for a in rows:
        for b in cols:
            members = frozenset(
                p for p in placements if a <= p[0] <= a + span and b <= p[1] <= b + span
            )
            if len(members) >= 2:
                cliques.add(members)

    # Keep only maximal cliques (drop any proper subset of another).
    return [c for c in cliques if not any(c < other for other in cliques)]


def build_and_solve(tau: float, distance: int = 3, data_dir=None, verbose: bool = True):
    V = valid_placements(tau, distance=distance, data_dir=data_dir)
    span = footprint_span(distance)
    cliques = maximal_cliques(V, span)

    model = gp.Model("tile_packing_clique")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables: x_p = 1 if a tile is placed at origin p.
    x = model.addVars(V, vtype=GRB.BINARY, name="x")

    # Objective: maximize the number of placed tiles.
    model.setObjective(x.sum(), GRB.MAXIMIZE)

    # Non-overlap: at most one tile per maximal clique.
    for i, clique in enumerate(cliques):
        model.addConstr(gp.quicksum(x[p] for p in clique) <= 1, name=f"clique_{i}")

    # LP relaxation bound (illustrates formulation tightness).
    model.update()  # flush pending vars/constraints before copying
    lp = model.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    lp_bound = lp.ObjVal

    model.optimize()

    chosen = sorted(p for p in V if x[p].X > 0.5)
    _report(tau, distance, span, V, cliques, lp_bound, model, chosen)
    return chosen


def _report(tau, distance, span, V, cliques, lp_bound, model, chosen) -> None:
    # Sanity check: no two chosen tiles may conflict.
    for i, p in enumerate(chosen):
        for q in chosen[i + 1 :]:
            assert not (abs(p[0] - q[0]) <= span and abs(p[1] - q[1]) <= span), (
                f"overlap in solution: {p} vs {q}"
            )
    print("\n=== clique (tighter) formulation ===")
    print(f"tau = {tau:g}, distance = {distance}, span s = {span}")
    print(f"|V| = {len(V)} variables, {len(cliques)} clique constraints")
    print(f"LP relaxation bound : {lp_bound:.4f}")
    print(f"integer optimum     : {int(round(model.ObjVal))} tiles placed")
    print(f"solve time          : {model.Runtime:.3f} s")
    print(f"optimality gap      : {model.MIPGap * 100:.4f}%")
    print("placed origins:", chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile packing MILP (clique formulation).")
    ap.add_argument("tau", type=float, help="LER validity threshold")
    ap.add_argument("--distance", type=int, default=3, help="code distance (default 3)")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="experiment folder to read results flakes from "
        "(default: $QSNOW_DATA_DIR or the demo path)",
    )
    args = ap.parse_args()
    build_and_solve(args.tau, args.distance, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
