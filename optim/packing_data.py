#!/usr/bin/env python3
"""
Precompute the valid tile-placement set V for a given LER threshold tau.

Shared data layer for the two Gurobi packing models. A candidate origin
(row, col) read from the distance-3 square-packing results `.flake` file is
"valid" when its measured logical error rate satisfies `ler <= tau`; the set of
such origins is `V` in the MILP (see `square_packing_model.pdf`).

Usage (standalone):
    python3 packing_data.py <tau> [--distance D] [--data-dir DIR]

The experiment folder read for results flakes defaults to a demo path, and can be
changed with `--data-dir` (or the QSNOW_DATA_DIR environment variable).

Usage (as a module):
    from packing_data import valid_placements, footprint_span
    V = valid_placements(0.20, distance=3)   # list of (row, col)
    s = footprint_span(distance=3)           # 7

Both code distances share the same schema; the distance selects which results
`.flake` is read (see `flake_path`), so D3 and D5 are handled by the same code.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

Coord = Tuple[int, int]


class Candidate(NamedTuple):
    """A candidate tile placement: a distance-`distance` tile at `origin`.

    Used by the mixed-distance model, where at one site we may choose between a
    D3 and a D5 tile (or none). `span = 2*distance + 1` is the footprint span
    (see `footprint_span`) and `ler` is the measured logical error rate.
    """

    origin: Coord
    distance: int
    span: int
    ler: float

# This module lives in qSNOW/optim/; the experiment data stays in the demo tree.
# Paths are resolved relative to this file so imports and CLI runs work regardless
# of the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# Default experiment folder to read results flakes from. Override per-run with a
# `data_dir` argument (CLI: `--data-dir`) or persistently with the QSNOW_DATA_DIR
# environment variable; see `default_data_dir`.
DEFAULT_DATA_DIR = (
    SCRIPT_DIR.parent / "demo" / "flakes" / "experiments" / "15x15" / "mean_0_01"
)


def default_data_dir() -> Path:
    """Experiment folder used when no explicit `data_dir` is passed.

    Resolves to `$QSNOW_DATA_DIR` if that environment variable is set, otherwise
    the built-in demo default (`DEFAULT_DATA_DIR`).
    """
    env = os.environ.get("QSNOW_DATA_DIR")
    return Path(env).expanduser() if env else DEFAULT_DATA_DIR


def flake_path(distance: int = 3, data_dir: Union[str, Path, None] = None) -> Path:
    """
    Locate the square-packing results `.flake` for a given code distance.

    Searches `data_dir` (defaulting to `default_data_dir()`). Results flakes carry
    the distance as a `d<distance>` token but otherwise vary by naming convention,
    e.g. `results_squarepacking_rsc_memory_z_d5_<timestamp>.flake` (older) and
    `results_d5_30x30_mean_0_001_<timestamp>.flake` (newer). We glob on the
    `d<distance>` token so both are found without hard-coding the timestamp or the
    surrounding descriptor. If several runs match, the lexicographically last
    (newest timestamp) is used.
    """
    data_dir = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
    pattern = f"results_*d{distance}_*.flake"
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no results flake for distance {distance} in {data_dir} "
            f"(pattern {pattern!r})"
        )
    return matches[-1]


def available_distances(data_dir: Union[str, Path, None] = None) -> List[int]:
    """
    Code distances that have a results `.flake` in `data_dir`, sorted ascending.

    Discovers distances from the `d<N>` token in the results filenames (both
    naming conventions carry it, e.g. `..._z_d5_...` and `results_d7_30x30_...`),
    so a folder can hold any collection of distances and be picked up
    automatically as more are added.
    """
    dd = Path(data_dir).expanduser() if data_dir is not None else default_data_dir()
    dists = set()
    for p in dd.glob("results_*.flake"):
        for m in re.finditer(r"_d(\d+)_", p.name):
            dists.add(int(m.group(1)))
    return sorted(dists)


def load_lers(
    distance: int = 3,
    data_dir: Union[str, Path, None] = None,
    path: Optional[Path] = None,
) -> Dict[Coord, float]:
    """
    Load {(row, col): ler} from a square-packing results `.flake` (JSON) file.

    The file is located by `distance` within `data_dir` unless an explicit `path`
    to a specific flake is given (which takes precedence).
    """
    path = path if path is not None else flake_path(distance, data_dir)
    with open(path) as f:
        data = json.load(f)
    lers: Dict[Coord, float] = {}
    for key, entry in data["results"].items():
        r, c = (int(v) for v in key.split(","))
        lers[(r, c)] = entry["ler"]
    return lers


def valid_placements(
    tau: float,
    distance: int = 3,
    data_dir: Union[str, Path, None] = None,
    path: Optional[Path] = None,
) -> List[Coord]:
    """Return V = sorted origins whose LER is within the threshold `ler <= tau`."""
    lers = load_lers(distance=distance, data_dir=data_dir, path=path)
    return sorted(p for p, ler in lers.items() if ler <= tau)


def mixed_candidates(
    tau: Union[float, Dict[int, float]],
    distances: Optional[Iterable[int]] = None,
    data_dir: Union[str, Path, None] = None,
) -> List[Candidate]:
    """
    Union of valid candidate placements across a collection of code distances.

    For each distance `d` we load its results flake, keep the origins whose
    `ler <= tau_d`, and emit a `Candidate(origin, d, footprint_span(d), ler)`.
    `distances` defaults to every distance found in `data_dir` (see
    `available_distances`), so new distances are included automatically. `tau` may
    be a single threshold applied to every distance, or a per-distance mapping
    `{distance: tau_d}` (e.g. to demand a stricter LER of the larger tiles). The
    result is sorted by (origin, distance).

    Note: a larger tile's origin set is a subset of a smaller tile's (its bigger
    footprint cannot fit in the chip's edge band), so a site offers a choice among
    only the distances that both fit and are valid there; edge-band sites offer
    just the smaller distances. This falls out naturally from the per-distance
    flakes with no special-casing.
    """
    if distances is None:
        distances = available_distances(data_dir)
    cands: List[Candidate] = []
    for d in distances:
        tau_d = tau[d] if isinstance(tau, dict) else tau
        span = footprint_span(d)
        for origin, ler in load_lers(distance=d, data_dir=data_dir).items():
            if ler <= tau_d:
                cands.append(Candidate(origin, d, span, ler))
    return sorted(cands)


def chip_grid_dim(
    distance: int = 3,
    data_dir: Union[str, Path, None] = None,
    path: Optional[Path] = None,
) -> Tuple[int, int]:
    """
    Full checkerboard extent (n_rows, n_cols) of the chip, i.e. coords 0..n-1.

    A "15x15" chip is a 30x30 checkerboard. The dimension is read from the sibling
    experiment `.flake` referenced by the results file (its stored chip `length`
    is the unit size, half the internal grid), falling back to 30x30.
    """
    path = path if path is not None else flake_path(distance, data_dir)
    try:
        with open(path) as f:
            exp_name = json.load(f).get("experiment")
        with open(path.parent / exp_name) as f:
            chip = json.load(f)["exp"]["chip"]
        return 2 * int(chip["height"]), 2 * int(chip["length"])
    except (OSError, KeyError, TypeError, ValueError):
        return 30, 30


def footprint_span(distance: int = 3) -> int:
    """
    Footprint span s used by the conflict rule.

    A distance-`d` tile placed at (r, c) occupies the checkerboard window
    [r, r+s] x [c, c+s] with s = 2*d + 1 (= tile.length - 1). Two tiles conflict
    (share a qubit) iff |dr| <= s and |dc| <= s.
    """
    return 2 * distance + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute the valid tile-placement set V for a given tau."
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

    lers = load_lers(distance=args.distance, data_dir=args.data_dir)
    V = valid_placements(args.tau, distance=args.distance, data_dir=args.data_dir)
    print(f"reading from: {flake_path(args.distance, args.data_dir)}")
    print(
        f"d = {args.distance}, tau = {args.tau:g}: "
        f"|V| = {len(V)} valid of {len(lers)} candidate origins"
    )
    for r, c in V:
        print(f"  ({r:2d}, {c:2d})  ler = {lers[(r, c)]:.4f}")


if __name__ == "__main__":
    main()
