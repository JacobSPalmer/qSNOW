#!/usr/bin/env python3
"""
Visualize the optimized MIXED-distance tile packing on the chip.

Companion to `visualize_packing.py` for the mixed model (`model_mixed.py`), where
each placed tile may be a distance-3 or a distance-5 patch.

Every candidate origin is classified by which distances are *valid* there
(ler <= tau), and drawn with a distinct marker:

    1) no valid tile      -- faint gray dot
    2) only D3 valid      -- blue circle
    3) only D5 valid      -- orange triangle
    4) both D3 and D5 valid-- green star

On top of that, a footprint square is drawn around every placed tile, colored by
its distance (D3 vs D5), so both the validity landscape and the chosen packing
are readable at once. A distance-d tile at origin (r, c) occupies
[r, r+s] x [c, c+s] with s = 2*d + 1, so D3 and D5 draw as different-size squares.

Usage:
    python3 visualize_mixed.py <tau> [--distances 3,5] [--data-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from model_mixed import _parse_distances, build_and_solve
from packing_data import chip_grid_dim, footprint_span, load_lers

# Per-distance tile colors (fill/edge). Extend if more distances are ever used.
DIST_COLORS = {
    3: {"face": "#2980b9", "edge": "#1b4f72", "name": "D3"},
    5: {"face": "#e67e22", "edge": "#9c640c", "name": "D5"},
}
FALLBACK = {"face": "#7f8c8d", "edge": "#2c3e50", "name": "D?"}

# Origin-validity category markers. `low`/`high` are filled in per run from the
# two distances; `none`/`both` are fixed.
CAT_NONE = {"marker": ".", "face": "#d5d5d5", "edge": "#d5d5d5", "size": 12}
CAT_BOTH = {"marker": "*", "face": "#2ecc71", "edge": "#145a32", "size": 90}
CAT_LOW = {"marker": "o", "size": 26}  # only the smaller distance valid
CAT_HIGH = {"marker": "^", "size": 34}  # only the larger distance valid


def _tau_for(tau, d: int) -> float:
    """Threshold for distance d (scalar tau, or a {distance: tau} mapping)."""
    return tau[d] if isinstance(tau, dict) else tau


def classify_origins(tau, low: int, high: int, data_dir=None):
    """
    Sort every candidate origin into {none, low, high, both} by validity.

    Returns a dict of category -> list of (r, c) origins, where `low`/`high` mean
    "only the smaller/larger distance is valid here" and `both` means both are.
    Origins where the larger tile does not fit (chip edge band) simply never
    qualify as high/both -- they fall into low or none.
    """
    lers_low = load_lers(distance=low, data_dir=data_dir)
    lers_high = load_lers(distance=high, data_dir=data_dir)
    tau_low, tau_high = _tau_for(tau, low), _tau_for(tau, high)

    cats = {"none": [], "low": [], "high": [], "both": []}
    for p in set(lers_low) | set(lers_high):
        lv = p in lers_low and lers_low[p] <= tau_low
        hv = p in lers_high and lers_high[p] <= tau_high
        if lv and hv:
            cats["both"].append(p)
        elif lv:
            cats["low"].append(p)
        elif hv:
            cats["high"].append(p)
        else:
            cats["none"].append(p)
    return cats


def _scatter(ax, pts, marker, face, edge, size, zorder, lw=0.6):
    if pts:
        ax.scatter(
            [c for _, c in pts],
            [r for r, _ in pts],
            marker=marker,
            s=size,
            facecolors=face,
            edgecolors=edge,
            linewidths=lw,
            zorder=zorder,
        )


def visualize(tau: float, distances=(3, 5), data_dir=None) -> Path:
    distances = tuple(distances)
    low, high = min(distances), max(distances)
    col_low = DIST_COLORS.get(low, FALLBACK)
    col_high = DIST_COLORS.get(high, FALLBACK)

    n_rows, n_cols = chip_grid_dim(distance=low, data_dir=data_dir)
    cats = classify_origins(tau, low, high, data_dir=data_dir)
    chosen = build_and_solve(tau, distances=distances, data_dir=data_dir, verbose=False)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Origin validity markers (drawn under the translucent tiles). Order matters
    # only for overlap; "none" is most numerous and least important, so first.
    _scatter(
        ax,
        cats["none"],
        CAT_NONE["marker"],
        CAT_NONE["face"],
        CAT_NONE["edge"],
        CAT_NONE["size"],
        zorder=2,
        lw=0,
    )
    _scatter(
        ax,
        cats["low"],
        CAT_LOW["marker"],
        col_low["face"],
        col_low["edge"],
        CAT_LOW["size"],
        zorder=2.3,
    )
    _scatter(
        ax,
        cats["high"],
        CAT_HIGH["marker"],
        col_high["face"],
        col_high["edge"],
        CAT_HIGH["size"],
        zorder=2.3,
    )
    _scatter(
        ax,
        cats["both"],
        CAT_BOTH["marker"],
        CAT_BOTH["face"],
        CAT_BOTH["edge"],
        CAT_BOTH["size"],
        zorder=2.6,
    )

    # Placed tiles: a footprint square [r, r+s] x [c, c+s], colored by distance.
    pad = 0.5  # nudge so boundary qubits sit inside the drawn square
    for i, cand in enumerate(chosen, start=1):
        r, c = cand.origin
        s = cand.span
        col = DIST_COLORS.get(cand.distance, FALLBACK)
        ax.add_patch(
            Rectangle(
                (c - pad, r - pad),
                s + 2 * pad,
                s + 2 * pad,
                facecolor=col["face"],
                edgecolor=col["edge"],
                alpha=0.20,
                linewidth=2.0,
                zorder=3,
            )
        )
        ax.text(
            c + s / 2,
            r + s / 2,
            str(i),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=col["edge"],
            zorder=4,
        )

    # Chip boundary.
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            n_cols,
            n_rows,
            fill=False,
            edgecolor="0.4",
            linewidth=1.0,
            zorder=1,
        )
    )

    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-1, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, chip-layout style
    step = 2 if n_cols <= 32 else 4
    ax.set_xticks(range(0, n_cols, step))
    ax.set_yticks(range(0, n_rows, step))
    ax.set_xlabel("column (STIM coord)")
    ax.set_ylabel("row (STIM coord)")

    placed_counts = {d: sum(1 for c in chosen if c.distance == d) for d in distances}
    placed_str = "  ".join(
        f"{DIST_COLORS.get(d, FALLBACK)['name']}: {placed_counts[d]}" for d in distances
    )
    ax.set_title(
        f"Mixed-distance tile packing  (τ = {tau:g})\n"
        f"{len(chosen)} tiles placed  ·  {placed_str}  ·  chip {n_rows}×{n_cols}"
    )

    nm_low, nm_high = col_low["name"], col_high["name"]
    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker=CAT_NONE["marker"],
            linestyle="",
            color=CAT_NONE["face"],
            label=f"no valid tile  [{len(cats['none'])}]",
        ),
        plt.Line2D(
            [],
            [],
            marker=CAT_LOW["marker"],
            linestyle="",
            markerfacecolor=col_low["face"],
            markeredgecolor=col_low["edge"],
            color=col_low["face"],
            label=f"only {nm_low} valid  [{len(cats['low'])}]",
        ),
        plt.Line2D(
            [],
            [],
            marker=CAT_HIGH["marker"],
            linestyle="",
            markerfacecolor=col_high["face"],
            markeredgecolor=col_high["edge"],
            color=col_high["face"],
            label=f"only {nm_high} valid  [{len(cats['high'])}]",
        ),
        plt.Line2D(
            [],
            [],
            marker=CAT_BOTH["marker"],
            linestyle="",
            markerfacecolor=CAT_BOTH["face"],
            markeredgecolor=CAT_BOTH["edge"],
            color=CAT_BOTH["face"],
            markersize=11,
            label=f"both {nm_low} & {nm_high} valid  [{len(cats['both'])}]",
        ),
    ]
    for d in distances:
        col = DIST_COLORS.get(d, FALLBACK)
        span = footprint_span(d)
        legend_handles.append(
            Patch(
                facecolor=col["face"],
                edgecolor=col["edge"],
                alpha=0.4,
                label=f"placed {col['name']} (footprint {span + 1}×{span + 1})",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.95,
        borderaxespad=0.0,
    )

    out_path = Path(__file__).resolve().parent / f"packing_mixed_tau_{tau:g}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize the mixed-distance tile packing."
    )
    ap.add_argument("tau", type=float, help="LER validity threshold")
    ap.add_argument(
        "--distances",
        type=_parse_distances,
        default=[3, 5],
        help="comma-separated code distances (default 3,5)",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="experiment folder to read results flakes from "
        "(default: $QSNOW_DATA_DIR or the demo path)",
    )
    args = ap.parse_args()
    out = visualize(args.tau, distances=args.distances, data_dir=args.data_dir)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
