#!/usr/bin/env python3
"""
Visualize the optimized MIXED-distance tile packing on the chip.

Companion to `visualize_packing.py` for the mixed model (`model_mixed.py`), where
each placed tile may be one of any number of code distances (d3, d5, d7, ...).

Every candidate origin is classified by its SMALLEST valid distance -- the
cheapest (densest-packing) code that meets `ler <= tau` there. Because a larger
distance always has a lower LER (monotonicity of the data), the valid distances
at a site form an upper set, so this single value characterizes the whole valid
set. Origins are drawn with a distinct per-distance marker, plus a faint gray dot
for sites where no distance is valid.

On top of that, a footprint square is drawn around every placed tile, colored by
its distance, so both the validity landscape and the chosen packing read at once.
A distance-d tile at origin (r, c) occupies [r, r+s] x [c, c+s] with s = 2*d + 1,
so different distances draw as different-size squares.

Usage:
    python3 visualize_mixed.py <tau> [--distances 3,5,7] [--data-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from model_mixed import _parse_distances, build_and_solve
from packing_data import (
    available_distances,
    chip_grid_dim,
    footprint_span,
    load_lers,
    mixed_candidates,
)

# Fixed (face, edge) colors for the common distances so plots stay consistent;
# any other distance falls back to a colormap (see `distance_styles`).
BASE_COLORS = {
    3: ("#2980b9", "#1b4f72"),
    5: ("#e67e22", "#9c640c"),
    7: ("#8e44ad", "#5b2c6f"),
    9: ("#16a085", "#0e6252"),
    11: ("#c0392b", "#7b241c"),
    13: ("#d4ac0d", "#7d6608"),
}
# Distinct marker shapes cycled per distance (ascending).
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "h", "p"]
NO_VALID = {"marker": ".", "color": "#d5d5d5", "size": 12}


def _darken(color, factor: float = 0.6):
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def distance_styles(distances) -> dict:
    """Map each distance -> {face, edge, name, marker, size}, stable by ordering."""
    ds = sorted(distances)
    cmap = plt.cm.tab10
    styles = {}
    for i, d in enumerate(ds):
        if d in BASE_COLORS:
            face, edge = BASE_COLORS[d]
        else:
            face = mcolors.to_hex(cmap(i % 10))
            edge = _darken(face)
        styles[d] = {
            "face": face,
            "edge": edge,
            "name": f"D{d}",
            "marker": MARKERS[i % len(MARKERS)],
            "size": 24 + 10 * i,  # larger distances a touch bigger, to stand out
        }
    return styles


def _tau_for(tau, d: int) -> float:
    """Threshold for distance d (scalar tau, or a {distance: tau} mapping)."""
    return tau[d] if isinstance(tau, dict) else tau


def classify_origins(tau, distances, data_dir=None) -> dict:
    """
    Sort every candidate origin by its smallest valid distance.

    Returns {None: [...origins with no valid distance...], d: [...origins whose
    smallest valid distance is d...]}. Larger distances that do not fit at a site
    (chip edge band) simply never qualify there.
    """
    ds = sorted(distances)
    lers = {d: load_lers(distance=d, data_dir=data_dir) for d in ds}

    cats = {None: []}
    for d in ds:
        cats[d] = []
    universe = set().union(*(set(lers[d]) for d in ds)) if ds else set()
    for p in universe:
        smallest = None
        for d in ds:  # ascending -> first valid is the smallest valid
            if p in lers[d] and lers[d][p] <= _tau_for(tau, d):
                smallest = d
                break
        cats[smallest].append(p)
    return cats


def visualize(
    tau: float,
    distances=None,
    data_dir=None,
    solution_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
) -> Path:
    if distances is None:
        distances = available_distances(data_dir)
    distances = sorted(distances)
    styles = distance_styles(distances)

    n_rows, n_cols = chip_grid_dim(distance=min(distances), data_dir=data_dir)
    cats = classify_origins(tau, distances, data_dir=data_dir)
    chosen = build_and_solve(
        tau,
        distances=distances,
        data_dir=data_dir,
        verbose=False,
        solution_dir=solution_dir,
        use_cache=use_cache,
        refresh=refresh,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scale-adaptive rendering. The demo chips (~60x60) get full-size markers,
    # per-tile index labels, and heavy tile borders; big chips (e.g. 400x400,
    # thousands of tiles) would drown in those, so markers shrink, borders thin,
    # and the numeric labels are dropped once they can no longer be read.
    n_max = max(n_rows, n_cols)
    marker_scale = min(1.0, max(0.12, 60.0 / n_max))
    tile_lw = 2.0 if len(chosen) <= 200 else max(0.3, 120.0 / len(chosen))
    label_tiles = len(chosen) <= 200

    # Origin validity markers (drawn under the translucent tiles): "no valid"
    # first (most numerous, least important), then by ascending distance so the
    # rarer larger-distance markers land on top.
    none_pts = cats[None]
    if none_pts:
        ax.scatter(
            [c for _, c in none_pts], [r for r, _ in none_pts],
            marker=NO_VALID["marker"], s=NO_VALID["size"] * marker_scale,
            color=NO_VALID["color"], zorder=2,
        )
    for d in distances:
        pts = cats[d]
        st = styles[d]
        if pts:
            ax.scatter(
                [c for _, c in pts], [r for r, _ in pts],
                marker=st["marker"], s=st["size"] * marker_scale,
                facecolors=st["face"], edgecolors=st["edge"],
                linewidths=0.6 * marker_scale, zorder=2.3 + 0.01 * d,
            )

    # Placed tiles: a footprint square [r, r+s] x [c, c+s], colored by distance.
    pad = 0.5  # nudge so boundary qubits sit inside the drawn square
    for i, cand in enumerate(chosen, start=1):
        r, c = cand.origin
        s = cand.span
        st = styles.get(cand.distance)
        face = st["face"] if st else "#7f8c8d"
        edge = st["edge"] if st else "#2c3e50"
        ax.add_patch(
            Rectangle(
                (c - pad, r - pad), s + 2 * pad, s + 2 * pad,
                facecolor=face, edgecolor=edge,
                alpha=0.20, linewidth=tile_lw, zorder=3,
            )
        )
        if label_tiles:
            ax.text(
                c + s / 2, r + s / 2, str(i),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=edge, zorder=4,
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
    # Aim for ~16 evenly spaced ticks regardless of chip size (even step, since
    # coords sit on the checkerboard); avoids the label smear on large chips.
    step = max(2, int(round(n_max / 16 / 2)) * 2)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_yticks(range(0, n_rows, step))
    ax.set_xlabel("column (STIM coord)")
    ax.set_ylabel("row (STIM coord)")

    placed_counts = {d: sum(1 for c in chosen if c.distance == d) for d in distances}
    placed_str = "  ".join(f"{styles[d]['name']}: {placed_counts[d]}" for d in distances)
    ax.set_title(
        f"Mixed-distance tile packing  (τ = {tau:g})\n"
        f"{len(chosen)} tiles placed  ·  {placed_str}  ·  chip {n_rows}×{n_cols}"
    )

    # Legend: validity categories (smallest valid distance), then placed-tile
    # footprints per distance.
    legend_handles = [
        plt.Line2D([], [], marker=NO_VALID["marker"], linestyle="",
                   color=NO_VALID["color"], label=f"no valid tile  [{len(cats[None])}]"),
    ]
    for d in distances:
        st = styles[d]
        legend_handles.append(
            plt.Line2D([], [], marker=st["marker"], linestyle="",
                       markerfacecolor=st["face"], markeredgecolor=st["edge"],
                       color=st["face"],
                       label=f"smallest valid = {st['name']}  [{len(cats[d])}]")
        )
    for d in distances:
        st = styles[d]
        span = footprint_span(d)
        legend_handles.append(
            Patch(facecolor=st["face"], edgecolor=st["edge"], alpha=0.4,
                  label=f"placed {st['name']} (footprint {span + 1}×{span + 1})")
        )
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.95, borderaxespad=0.0)

    out_path = Path(__file__).resolve().parent / f"packing_mixed_tau_{tau:g}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize the mixed-distance tile packing.")
    ap.add_argument("tau", type=float, help="LER validity threshold")
    ap.add_argument("--distances", type=_parse_distances, default=None,
                    help="comma-separated code distances "
                    "(default: all found in the data dir)")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="experiment folder to read results flakes from "
        "(default: $QSNOW_DATA_DIR or the demo path)",
    )
    ap.add_argument(
        "--solution-dir",
        default=None,
        help="directory to read/write cached solutions (default: model_mixed's)",
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
    out = visualize(
        args.tau,
        distances=args.distances,
        data_dir=args.data_dir,
        solution_dir=args.solution_dir,
        use_cache=args.use_cache,
        refresh=args.refresh,
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
