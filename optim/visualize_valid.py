#!/usr/bin/env python3
"""
Visualize which tile-placement locations are *valid* for a given LER threshold tau.

A placement origin (row, col) read from the distance-3 square-packing results
`.flake` file is "valid" when its measured logical error rate (LER) satisfies
`ler <= tau`. This script plots every candidate origin on the checkerboard,
distinguishing valid from invalid locations.

Usage:
    python3 visualize_valid.py <tau>

Example:
    python3 visualize_valid.py 0.20
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

# This script lives in qSNOW/optim/; the experiment data stays in the demo tree.
# Paths are resolved relative to this file so the script works regardless of the
# current working directory. Figures are written next to the script (in optim/).
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    SCRIPT_DIR.parent / "demo" / "flakes" / "experiments" / "15x15" / "mean_0_01"
)
FLAKE_PATH = DATA_DIR / "results_squarepacking_rsc_memory_z_d3_2026-07-16_17-10-14.flake"


def load_lers(flake_path: Path) -> Dict[Tuple[int, int], float]:
    """Load {(row, col): ler} from a square-packing results `.flake` (JSON) file."""
    with open(flake_path) as f:
        data = json.load(f)
    lers: Dict[Tuple[int, int], float] = {}
    for key, entry in data["results"].items():
        r, c = (int(v) for v in key.split(","))
        lers[(r, c)] = entry["ler"]
    return lers


def visualize(tau: float, flake_path: Path = FLAKE_PATH) -> Path:
    """Plot valid vs. invalid placement origins for threshold `tau`; save a PNG."""
    lers = load_lers(flake_path)

    valid = [(r, c) for (r, c), ler in lers.items() if ler <= tau]
    invalid = [(r, c) for (r, c), ler in lers.items() if ler > tau]

    fig, ax = plt.subplots(figsize=(8, 8))

    if invalid:
        ax.scatter(
            [c for _, c in invalid],
            [r for r, _ in invalid],
            marker="x",
            c="#c0392b",
            s=60,
            linewidths=1.5,
            label=f"invalid  (ler > {tau:g})   [{len(invalid)}]",
        )
    if valid:
        ax.scatter(
            [c for _, c in valid],
            [r for r, _ in valid],
            marker="o",
            c="#27ae60",
            s=70,
            edgecolors="#145a32",
            linewidths=0.8,
            label=f"valid  (ler ≤ {tau:g})   [{len(valid)}]",
        )

    ax.set_xlabel("column (STIM coord)")
    ax.set_ylabel("row (STIM coord)")
    ax.set_title(
        f"Valid D3 tile-placement locations  (τ = {tau:g})\n"
        f"{len(valid)} valid of {len(lers)} candidate origins"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, chip-layout style
    ax.grid(True, which="both", color="0.9", linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.95)

    out_path = SCRIPT_DIR / f"valid_locations_tau_{tau:g}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <tau>", file=sys.stderr)
        raise SystemExit(2)
    try:
        tau = float(sys.argv[1])
    except ValueError:
        print(f"error: tau must be a number, got {sys.argv[1]!r}", file=sys.stderr)
        raise SystemExit(2)

    lers = load_lers(FLAKE_PATH)
    n_valid = sum(1 for ler in lers.values() if ler <= tau)
    out = visualize(tau)
    print(f"tau = {tau:g}: {n_valid} valid of {len(lers)} candidate origins")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
