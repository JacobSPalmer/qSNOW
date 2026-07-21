#!/usr/bin/env python3
"""
Visualize the optimized tile packing on the full checkerboard chip.

Builds on the valid-locations plot (`visualize_valid.py`): it shows the full
30x30 chip lattice (coords 0..29, including regions with no candidate origin),
the valid vs. invalid candidate origins for threshold tau, and then draws a
square around the footprint of every tile chosen by the optimization.

A distance-d tile placed at origin (r, c) occupies the window [r, r+s] x [c, c+s]
with s = 2*d + 1, which is the square drawn around each placed tile.

Usage:
    python3 visualize_packing.py <tau> [--distance D] [--formulation clique|pairwise]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from packing_data import (
    chip_grid_dim,
    footprint_span,
    load_lers,
    valid_placements,
)


def solve(tau: float, distance: int, formulation: str):
    """Return the chosen placements from the requested formulation (quietly)."""
    if formulation == "pairwise":
        from model_pairwise import build_and_solve
    else:
        from model_clique import build_and_solve
    return build_and_solve(tau, distance=distance, verbose=False)


def visualize(tau: float, distance: int = 3, formulation: str = "clique") -> Path:
    lers = load_lers()
    V = set(valid_placements(tau))
    span = footprint_span(distance)
    n_rows, n_cols = chip_grid_dim()
    chosen = solve(tau, distance, formulation)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Full chip: every checkerboard qubit position (0..n-1), incl. empty corners.
    chip_pts = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if r % 2 == c % 2
    ]
    ax.scatter(
        [c for _, c in chip_pts],
        [r for r, _ in chip_pts],
        marker=".",
        c="0.82",
        s=8,
        zorder=1,
    )

    # Candidate origins: valid vs. invalid (same encoding as visualize_valid.py).
    valid = [p for p in lers if p in V]
    invalid = [p for p in lers if p not in V]
    if invalid:
        ax.scatter(
            [c for _, c in invalid], [r for r, _ in invalid],
            marker="x", c="#c0392b", s=45, linewidths=1.3, zorder=2,
        )
    if valid:
        ax.scatter(
            [c for _, c in valid], [r for r, _ in valid],
            marker="o", c="#27ae60", s=45, edgecolors="#145a32",
            linewidths=0.6, zorder=2,
        )

    # Placed tiles: a square around each footprint [r, r+s] x [c, c+s].
    pad = 0.5  # nudge so boundary qubits sit inside the drawn square
    for i, (r, c) in enumerate(chosen, start=1):
        ax.add_patch(
            Rectangle(
                (c - pad, r - pad), span + 2 * pad, span + 2 * pad,
                facecolor="#2980b9", edgecolor="#1b4f72",
                alpha=0.22, linewidth=2.2, zorder=3,
            )
        )
        ax.text(
            c + span / 2, r + span / 2, str(i),
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1b4f72", zorder=4,
        )

    # Chip boundary.
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), n_cols, n_rows,
            fill=False, edgecolor="0.4", linewidth=1.0, zorder=1,
        )
    )

    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-1, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, chip-layout style
    ax.set_xticks(range(0, n_cols, 2))
    ax.set_yticks(range(0, n_rows, 2))
    ax.set_xlabel("column (STIM coord)")
    ax.set_ylabel("row (STIM coord)")
    ax.set_title(
        f"Optimized D{distance} tile packing  (τ = {tau:g}, {formulation})\n"
        f"{len(chosen)} tiles placed  ·  {len(valid)} valid of "
        f"{len(lers)} candidate origins  ·  chip {n_rows}×{n_cols}"
    )

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#27ae60",
                   markeredgecolor="#145a32", label=f"valid origin (ler ≤ {tau:g})"),
        plt.Line2D([], [], marker="x", linestyle="", color="#c0392b",
                   label=f"invalid origin (ler > {tau:g})"),
        plt.Line2D([], [], marker=".", linestyle="", color="0.82",
                   label="chip qubit"),
        Patch(facecolor="#2980b9", edgecolor="#1b4f72", alpha=0.35,
              label=f"placed tile (footprint {span+1}×{span+1})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.95, borderaxespad=0.0)

    out_path = Path(__file__).resolve().parent / f"packing_tau_{tau:g}_{formulation}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize the optimized tile packing.")
    ap.add_argument("tau", type=float, help="LER validity threshold")
    ap.add_argument("--distance", type=int, default=3, help="code distance (default 3)")
    ap.add_argument("--formulation", choices=["clique", "pairwise"], default="clique",
                    help="which model to solve (default clique)")
    args = ap.parse_args()
    out = visualize(args.tau, args.distance, args.formulation)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
