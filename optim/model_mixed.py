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
                           [--lp-bound-time SECONDS]

The LP-relaxation bound printed in the report is a diagnostic (it shows the clique
formulation is tight), not an input to the MIP. It is a full extra LP solve, so on
large instances it is capped at `--lp-bound-time` seconds and skipped if it does
not finish (0 disables it).
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import FrozenSet, List

import gurobipy as gp
from gurobipy import GRB

from packing_data import (
    Candidate,
    available_distances,
    default_data_dir,
    mixed_candidates,
)

# Where solved packings are cached. Each case (data_dir, tau, distances) maps to
# one JSON file here (see `solution_path`); a matching file is loaded instead of
# re-solving. Override per-run with `solution_dir` / `--solution-dir`.
DEFAULT_SOLUTION_DIR = Path(__file__).resolve().parent / "solutions"


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


def _tau_slug(tau) -> str:
    """Filesystem-safe rendering of tau (scalar, or a per-distance mapping)."""
    if isinstance(tau, dict):
        return "_".join(f"d{d}-{tau[d]:g}" for d in sorted(tau))
    return f"{tau:g}"


def solution_path(tau, distances, data_dir=None, solution_dir=None) -> Path:
    """
    Cache file for one solve case, named by (data_dir, tau, distances).

    The name carries a human-readable slug of the data directory, the tau value,
    and the distance list, plus a short hash of the resolved data-dir path so that
    two experiment folders sharing a leaf name never collide. `distances` should be
    the concrete list actually solved (resolve ``None`` to ``available_distances``
    first) so the key reflects what was computed.
    """
    dd = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else default_data_dir().resolve()
    )
    sol_dir = (
        Path(solution_dir).expanduser()
        if solution_dir is not None
        else DEFAULT_SOLUTION_DIR
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(dd.parts[-2:])).strip("_")
    digest = hashlib.sha1(str(dd).encode()).hexdigest()[:8]
    ds = "-".join(str(d) for d in sorted(distances))
    name = f"{slug}__tau_{_tau_slug(tau)}__d{ds}__{digest}.json"
    return sol_dir / name


def _save_solution(path, tau, distances, data_dir, cands, cliques, lp_bound, model, chosen):
    """Write the solved packing (plus report metadata) to `path` atomically."""
    dd = (
        str(Path(data_dir).expanduser().resolve())
        if data_dir is not None
        else str(default_data_dir().resolve())
    )
    cand_by_d = Counter(c.distance for c in cands)
    data = {
        "tau": tau,
        "distances": list(distances),
        "data_dir": dd,
        "n_candidates": len(cands),
        "candidates_by_distance": {str(d): cand_by_d.get(d, 0) for d in distances},
        "n_cliques": len(cliques),
        "lp_bound": lp_bound,
        "objective": int(round(model.ObjVal)),
        "solve_time": model.Runtime,
        "mip_gap": model.MIPGap,
        "placed": [
            [c.origin[0], c.origin[1], c.distance, c.span, c.ler] for c in chosen
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)  # atomic, so a crash mid-write never leaves a half file


def _load_solution(path):
    """Read a cached solve; returns (metadata dict, sorted list of Candidates)."""
    with open(path) as f:
        data = json.load(f)
    chosen = sorted(
        Candidate((r, c), d, span, ler) for r, c, d, span, ler in data["placed"]
    )
    return data, chosen


def build_and_solve(
    tau,
    distances=None,
    data_dir=None,
    verbose: bool = True,
    lp_bound_time: float = 5.0,
    solution_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
):
    if distances is None:
        distances = available_distances(data_dir)
    distances = sorted(distances)

    # Solution cache: a matching file for this (data_dir, tau, distances) case is
    # loaded instead of re-solving, unless `refresh` forces a fresh solve or
    # `use_cache` is off. This is what makes repeated visualizations of a large
    # instance instant.
    sol_path = solution_path(tau, distances, data_dir=data_dir, solution_dir=solution_dir)
    if use_cache and not refresh and sol_path.exists():
        data, chosen = _load_solution(sol_path)
        _report(
            tau,
            distances,
            {int(k): v for k, v in data.get("candidates_by_distance", {}).items()},
            data.get("n_candidates", len(chosen)),
            data.get("n_cliques", 0),
            data.get("lp_bound"),
            "not stored",
            data.get("objective", len(chosen)),
            data.get("solve_time", 0.0),
            data.get("mip_gap", 0.0),
            chosen,
            source=sol_path,
            verbose=verbose,
        )
        return chosen

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

    # LP relaxation bound: a DIAGNOSTIC only (it shows the clique formulation is
    # tight -- typically integral -- so the MIP barely branches). It is a full
    # extra LP solve of the whole model and is NOT used by model.optimize(), so on
    # large instances it is pure overhead. We cap it at `lp_bound_time` seconds and
    # skip it (reporting the skip) if it does not solve in that budget; pass
    # lp_bound_time=0 to disable it outright.
    model.update()  # flush pending vars/constraints before copying
    lp_bound = None
    lp_skip = None  # human-readable reason, if skipped
    if lp_bound_time > 0:
        lp = model.relax()
        lp.Params.OutputFlag = 0
        lp.Params.TimeLimit = lp_bound_time
        lp.optimize()
        if lp.Status == GRB.OPTIMAL:
            lp_bound = lp.ObjVal
        else:
            lp_skip = f"did not solve within {lp_bound_time:g}s budget"
    else:
        lp_skip = "disabled (lp_bound_time=0)"

    model.optimize()

    chosen = sorted(cands[i] for i in range(len(cands)) if y[i].X > 0.5)
    if use_cache:
        _save_solution(
            sol_path, tau, distances, data_dir, cands, cliques, lp_bound, model, chosen
        )
    _report(
        tau,
        distances,
        Counter(c.distance for c in cands),
        len(cands),
        len(cliques),
        lp_bound,
        lp_skip,
        int(round(model.ObjVal)),
        model.Runtime,
        model.MIPGap,
        chosen,
        source=sol_path if use_cache else None,
        cached=False,
    )
    return chosen


def _report(
    tau,
    distances,
    cand_by_d,
    n_candidates,
    n_cliques,
    lp_bound,
    lp_skip,
    objective,
    solve_time,
    mip_gap,
    chosen,
    source=None,
    cached=True,
    verbose: bool = True,
) -> None:
    # Sanity check: no two chosen tiles conflict, and none share an origin. This
    # also validates a loaded cache file against the geometry rules.
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            assert not conflict(a, b), f"overlap in solution: {a} vs {b}"
    origins = [c.origin for c in chosen]
    assert len(origins) == len(set(origins)), "two placed tiles share an origin"

    if not verbose:
        if source is not None:
            verb = "loaded" if cached else "saved"
            print(f"solution {verb}: {source}")
        return

    n_by_d = Counter(c.distance for c in chosen)
    print("\n=== mixed-distance (threshold-count) formulation ===")
    if source is not None:
        print(f"solution {'loaded from' if cached else 'saved to'}: {source}")
    print(f"tau = {tau}, distances = {tuple(distances)}")
    counts = ", ".join(f"D{d}: {cand_by_d.get(d, 0)}" for d in sorted(cand_by_d))
    print(f"|candidates| = {n_candidates} ({counts}), {n_cliques} clique constraints")
    if lp_bound is not None:
        print(f"LP relaxation bound : {lp_bound:.4f}")
    else:
        print(f"LP relaxation bound : skipped ({lp_skip}; diagnostic only)")
    placed = ", ".join(f"D{d}: {n_by_d.get(d, 0)}" for d in sorted(distances))
    print(f"integer optimum     : {objective} tiles placed ({placed})")
    print(f"solve time          : {solve_time:.3f} s{'  (cached)' if cached else ''}")
    print(f"optimality gap      : {mip_gap * 100:.4f}%")
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
    ap.add_argument(
        "--lp-bound-time",
        type=float,
        default=5.0,
        help="seconds to spend on the (diagnostic) LP-relaxation bound before "
        "skipping it; 0 disables it (default: 5)",
    )
    ap.add_argument(
        "--solution-dir",
        default=None,
        help="directory to read/write cached solutions "
        f"(default: {DEFAULT_SOLUTION_DIR})",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-solve and overwrite any cached solution for this case",
    )
    ap.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="do not read or write the solution cache",
    )
    args = ap.parse_args()
    build_and_solve(
        args.tau,
        distances=args.distances,
        data_dir=args.data_dir,
        lp_bound_time=args.lp_bound_time,
        solution_dir=args.solution_dir,
        use_cache=args.use_cache,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
