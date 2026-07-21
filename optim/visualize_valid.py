#!/usr/bin/env python3
"""
Visualize which tile-placement locations are *valid* for a given LER threshold tau.

A placement origin (row, col) read from the distance-3 square-packing results
`.flake` file is "valid" when its measured logical error rate (LER) satisfies
`ler <= tau`. This script plots every candidate origin on the checkerboard,
distinguishing valid from invalid locations.

Usage:
    python3 visualize_valid.py <tau> [--distance D]

Example:
    python3 visualize_valid.py 0.20 --distance 5
"""

import argparse

import matplotlib.pyplot as plt

# Data loading (and the flake-per-distance resolution) lives in packing_data so
# D3 and D5 are handled the same way; figures are written next to this script.
from packing_data import SCRIPT_DIR, load_lers


def visualize(tau: float, distance: int = 3, data_dir=None):
    """Plot valid vs. invalid placement origins for threshold `tau`; save a PNG."""
    lers = load_lers(distance=distance, data_dir=data_dir)

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
        f"Valid D{distance} tile-placement locations  (τ = {tau:g})\n"
        f"{len(valid)} valid of {len(lers)} candidate origins"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, chip-layout style
    ax.grid(True, which="both", color="0.9", linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.95)

    out_path = SCRIPT_DIR / f"valid_locations_d{distance}_tau_{tau:g}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize valid vs. invalid tile-placement locations."
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
    n_valid = sum(1 for ler in lers.values() if ler <= args.tau)
    out = visualize(args.tau, args.distance, data_dir=args.data_dir)
    print(
        f"d = {args.distance}, tau = {args.tau:g}: "
        f"{n_valid} valid of {len(lers)} candidate origins"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
