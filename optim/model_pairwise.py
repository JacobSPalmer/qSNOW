#!/usr/bin/env python3
"""
Tile packing MILP -- WEAKER formulation (pairwise non-overlap constraints).

Maximizes the number of valid, mutually non-overlapping distance-`d` tiles placed
on the checkerboard chip. This file uses the pairwise edge constraints:

    max   sum_{p in V} x_p
    s.t.  x_p + x_q <= 1   for every conflicting pair {p, q}
          x_p in {0, 1}

Two placements conflict iff their footprints overlap: |dr| <= s and |dc| <= s,
with s = 2*d + 1 (see packing_data.footprint_span). This edge formulation is
correct but has a comparatively weak LP relaxation; see `model_clique.py` for the
tighter clique version.

Usage:
    python3 model_pairwise.py <tau> [--distance D]
"""

import argparse
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB
from packing_data import Coord, footprint_span, valid_placements


def conflict_pairs(placements: List[Coord], span: int) -> List[Tuple[Coord, Coord]]:
    """All unordered pairs whose footprints overlap (|dr| <= span and |dc| <= span)."""
    pairs: List[Tuple[Coord, Coord]] = []
    n = len(placements)
    for i in range(n):
        ri, ci = placements[i]
        for j in range(i + 1, n):
            rj, cj = placements[j]
            if abs(ri - rj) <= span and abs(ci - cj) <= span:
                pairs.append((placements[i], placements[j]))
    return pairs


def build_and_solve(tau: float, distance: int = 3, data_dir=None, verbose: bool = True):
    V = valid_placements(tau, distance=distance, data_dir=data_dir)
    span = footprint_span(distance)
    pairs = conflict_pairs(V, span)

    model = gp.Model("tile_packing_pairwise")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables: x_p = 1 if a tile is placed at origin p.
    x = model.addVars(V, vtype=GRB.BINARY, name="x")

    # Objective: maximize the number of placed tiles.
    model.setObjective(x.sum(), GRB.MAXIMIZE)

    # Non-overlap: pairwise edge constraints.
    for p, q in pairs:
        model.addConstr(x[p] + x[q] <= 1, name=f"conflict_{p}_{q}")

    # LP relaxation bound (illustrates formulation tightness).
    model.update()  # flush pending vars/constraints before copying
    lp = model.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    lp_bound = lp.ObjVal

    model.optimize()

    chosen = sorted(p for p in V if x[p].X > 0.5)
    _report(tau, distance, span, V, pairs, lp_bound, model, chosen)
    return chosen


def _report(tau, distance, span, V, pairs, lp_bound, model, chosen) -> None:
    # Sanity check: no two chosen tiles may conflict.
    for i, p in enumerate(chosen):
        for q in chosen[i + 1 :]:
            assert not (abs(p[0] - q[0]) <= span and abs(p[1] - q[1]) <= span), (
                f"overlap in solution: {p} vs {q}"
            )
    print("\n=== pairwise (weaker) formulation ===")
    print(f"tau = {tau:g}, distance = {distance}, span s = {span}")
    print(f"|V| = {len(V)} variables, {len(pairs)} pairwise constraints")
    print(f"LP relaxation bound : {lp_bound:.4f}")
    print(f"integer optimum     : {int(round(model.ObjVal))} tiles placed")
    print(f"solve time          : {model.Runtime:.3f} s")
    print(f"optimality gap      : {model.MIPGap * 100:.4f}%")
    print("placed origins:", chosen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tile packing MILP (pairwise formulation)."
    )
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
