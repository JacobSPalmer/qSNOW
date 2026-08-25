"""Matplotlib figures for LER distributions read back from experiment/result flakes.

`visualize.py` and `interactive.py` draw *chips* with plotly. This module draws the
*outcome* of a profiling sweep: it loads the `experiment_*.flake` / `results_*.flake`
pairs a run left behind (see `qsnow.helpers.serialize`) and renders the distribution of
logical error rate across every tile placement, one series per code distance.

Matplotlib rather than plotly here because these are publication figures: explicit `dpi`
and `Figure.savefig` matter more than hover/zoom.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from qsnow.experiments import Experiment, ExperimentResults
    from qsnow.interface.chip import Chip

__all__ = [
    "ProfileRun",
    "load_profile_runs",
    "ler_cdf",
    "ler_histogram",
]

PathLike = Union[str, Path]

# Flake filename globs written by `Experiment.save()` / `Experiment.save_results()`.
_EXPERIMENT_PATTERN = "experiment*_d{d}*"
_RESULTS_PATTERN = "result*_d{d}*"

# Distances the DATE sweeps profile by default.
_DEFAULT_DISTANCES = (3, 5, 7, 9, 11, 13, 15, 17)

# Max histogram columns before wrapping onto another row.
_MAX_HIST_COLS = 4

# How the profiled run and its baseline are distinguished. Fading alone reads as "the
# same series, printed lighter", so each panel adds a second cue: the baseline step is
# dashed in the CDF, and the baseline box is hatched *and* half-width in the box panel.
_PRIMARY_ALPHA = 0.95
_BASELINE_STEP_ALPHA = 0.2
_BASELINE_BOX_ALPHA = 0.3
_BASELINE_LINESTYLE = "--"
_BASELINE_HATCH = "///"
# The hatch is drawn on the box's *edge*, so fading it with the fill would erase it.
# Only the fill stays at _BASELINE_BOX_ALPHA; the edge and its hatch use this instead.
_BASELINE_HATCH_ALPHA = 0.65

_PRIMARY_BOX_WIDTH = 0.40
_BASELINE_BOX_WIDTH = 0.20
_SOLO_BOX_WIDTH = 0.5  # no baseline to pair with: one full-width box per distance
# Vertical dodge off the distance's tick. (w_primary + w_baseline)/4 is the closest the
# two box centres can sit without the boxes touching; the rest is breathing room.
_BOX_DODGE = (_PRIMARY_BOX_WIDTH + _BASELINE_BOX_WIDTH) / 4 + 0.045

# Neutral grey for the profiled/baseline key: the convention it states holds for every
# distance, so tinting it with any one distance's colour would be misleading.
_LEGEND_SWATCH_COLOR = "0.25"
# Used when a chip carries no usable noise-model name to label its series with.
_FALLBACK_LABELS = ("profiled", "baseline")


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


@contextmanager
def _data_dir(directory: Optional[PathLike]) -> Iterator[None]:
    """Point `serialize`'s global data root at `directory` for the duration of the block.

    The root is process-global state, so restore it on the way out - otherwise plotting
    one directory silently redirects every later export/import in the session.
    """
    # deferred import: qsnow.helpers.serialize imports qsnow.experiments, which imports
    # qsnow.visualize - importing it at module level would close that cycle.
    from qsnow.helpers import serialize

    if directory is None:
        yield
        return

    previous = serialize.get_data_dir()
    serialize.set_data_dir(directory)
    try:
        yield
    finally:
        serialize.set_data_dir(previous)


@dataclass(frozen=True)
class ProfileRun:
    """One distance's profiling sweep: the experiment setup plus its results flake."""

    distance: int
    experiment: "Experiment"
    results: "ExperimentResults"

    @property
    def chip(self) -> "Chip":
        chip = getattr(self.experiment, "chip", None)
        if chip is None:
            raise AttributeError(
                f"{type(self.experiment).__name__} has no chip, so its results cannot "
                "be profiled."
            )
        return chip

    @property
    def lers(self) -> np.ndarray:
        """Logical error rate per tile placement."""
        return np.array(
            [
                # newer flakes store `ler` outright; older ones only have the raw counts
                entry.get("ler", entry["errors"] / entry["shots"])
                for entry in self.results.results.values()
            ],
            dtype=float,
        )

    @property
    def error_counts(self) -> np.ndarray:
        """Raw logical error count per tile placement."""
        return np.array(
            [entry["errors"] for entry in self.results.results.values()], dtype=float
        )

    @property
    def shots(self) -> Optional[int]:
        return self.results.run_config.get("shots")

    def values(self, scope: Literal["ler", "errors"] = "ler") -> np.ndarray:
        return self.lers if scope == "ler" else self.error_counts

    def stats(self) -> Dict[str, float]:
        """Spread of this run's LERs. `ratio` is `inf` when some placement saw no errors."""
        lers = self.lers
        lo, hi = float(lers.min()), float(lers.max())
        return {
            "n": float(lers.size),
            "min": lo,
            "max": hi,
            "range": hi - lo,
            "ratio": hi / lo if lo else float("inf"),
            "mean": float(lers.mean()),
            # stdev is undefined for a single sample; report 0 rather than raising
            "stdev": float(lers.std(ddof=1)) if lers.size > 1 else 0.0,
            "zeros": float(np.count_nonzero(lers == 0)),
        }


def load_profile_runs(
    distances: Sequence[int] = _DEFAULT_DISTANCES,
    directory: Optional[PathLike] = None,
) -> Dict[int, ProfileRun]:
    """Load the latest experiment/results flake pair for each distance in `distances`.

    `directory` temporarily overrides the serialize data root; omit it to read from
    whatever root is already configured. Raises `FileNotFoundError` if either flake is
    missing for a requested distance.
    """
    from qsnow.helpers import serialize

    with _data_dir(directory):
        return {
            d: ProfileRun(
                distance=d,
                experiment=serialize.import_latest(
                    _EXPERIMENT_PATTERN.format(d=d), silent=True
                ),
                results=serialize.import_latest(
                    _RESULTS_PATTERN.format(d=d), silent=True
                ),
            )
            for d in distances
        }


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical CDF as `(x, y)` step coordinates, ready for `Axes.step`.

    The leading `(-inf, 0)` point draws each curve's flat run-in at y=0 from the left
    edge of the axes, so a series visibly starts at zero rather than at its smallest
    sample. Matplotlib clips the infinite endpoint to the axis bound.
    """
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0:
        return x, x
    return np.r_[-np.inf, x], np.r_[0.0, np.arange(1, x.size + 1) / x.size]


def _style_box(
    bp: Dict[str, list], color, alpha: float, hatch: Optional[str] = None
) -> None:
    """Tint every artist of one boxplot to `color` at `alpha`.

    `hatch` fills the box with a pattern as a second cue beyond the fade. It has to be
    applied through explicit RGBA rather than `set_alpha`, which would fade the edge -
    and therefore the hatch drawn on it - along with the fill.
    """
    from matplotlib.colors import to_rgba

    for patch in bp["boxes"]:
        if hatch:
            patch.set_facecolor(to_rgba(color, alpha))
            patch.set_edgecolor(to_rgba(color, _BASELINE_HATCH_ALPHA))
            patch.set_hatch(hatch)
        else:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(alpha)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color(color)
            line.set_alpha(alpha)
    for line in bp["medians"]:
        # black, and less faded than the box: a colour-matched median disappears into
        # its own fill at the baseline's low alpha
        line.set_color("black")
        line.set_alpha(min(1.0, alpha + 0.45))
        line.set_linewidth(1.3)
    for flier in bp["fliers"]:
        # a d=3 sweep has ~1400 placements; default fliers bury the box they belong to
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_alpha(alpha * 0.55)
        flier.set_markersize(2.5)


def _draw_box(
    ax,
    values,
    position: float,
    color,
    alpha: float,
    width: float,
    hatch: Optional[str] = None,
) -> None:
    """Draw one horizontal box for `values`, centred on `position`.

    `patch_artist=True` is what makes the box a fillable patch - without it matplotlib
    draws an unfilled `Line2D` that cannot take a facecolor.
    """
    bp = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        orientation="horizontal",
        patch_artist=True,
    )
    _style_box(bp, color, alpha, hatch)


def _series_labels(
    primary: ProfileRun,
    baseline: ProfileRun,
    override: Optional[Tuple[str, str]] = None,
) -> Tuple[str, str]:
    """Name the profiled and baseline series, preferring each chip's noise-model name.

    `override` wins outright. Otherwise the names come from the flake, the same source
    `_noise_caption` reads - "derived contour" vs "uniform homogeneous" on the DATE
    sweeps. A chip with no recorded model falls back to generic names: labelling the
    figure "custom vs custom" would say less than "profiled vs baseline".
    """
    if override is not None:
        return override

    def name(run: ProfileRun, fallback: str) -> str:
        model = run.chip.summary().get("noise_model") or {}
        recorded = model.get("name")
        return recorded if recorded and recorded != "custom" else fallback

    return name(primary, _FALLBACK_LABELS[0]), name(baseline, _FALLBACK_LABELS[1])


def _add_convention_legends(ax_cdf, ax_box, labels: Tuple[str, str]) -> None:
    """Add the profiled-vs-baseline key to whichever panels were drawn.

    Each panel gets the key in its own artist type - lines for the CDF, patches for the
    boxes - so the swatch matches what the reader is looking at, and in the corner that
    panel's data leaves free.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    grey = _LEGEND_SWATCH_COLOR
    if ax_cdf is not None:
        # re-adding the existing legend as a plain artist keeps it: a second `legend()`
        # call replaces the axes' legend rather than adding alongside it
        existing = ax_cdf.get_legend()
        if existing is not None:
            ax_cdf.add_artist(existing)
        ax_cdf.legend(
            [
                Line2D([], [], color=grey, lw=2.5, alpha=_PRIMARY_ALPHA),
                Line2D(
                    [],
                    [],
                    color=grey,
                    lw=2.5,
                    alpha=_BASELINE_STEP_ALPHA,
                    linestyle=_BASELINE_LINESTYLE,
                ),
            ],
            list(labels),
            loc="lower right",
            fontsize=10,
            framealpha=0.95,
        )

    if ax_box is not None:
        ax_box.legend(
            [
                Patch(facecolor=grey, edgecolor=grey, alpha=_PRIMARY_ALPHA),
                Patch(
                    facecolor=to_rgba(grey, _BASELINE_BOX_ALPHA),
                    edgecolor=to_rgba(grey, _BASELINE_HATCH_ALPHA),
                    hatch=_BASELINE_HATCH,
                ),
            ],
            list(labels),
            loc="upper right",
            fontsize=9,
            framealpha=0.95,
        )


def _noise_caption(chip: "Chip", shots: Optional[int]) -> str:
    """The `PED(type=..., mean=..., dev=..., shots=...)` line under a figure title."""
    noise_model = chip.summary().get("noise_model") or {}
    name = noise_model.get("name", "custom")
    p = np.array([n.p for n in chip.noise_map.values()], dtype=float)
    return (
        f"PED(type={name}, mean={p.mean():.2g}, dev={p.std():.2g}, shots={shots})"
    )


def _suptitle(headline: str, run: ProfileRun, add_title: str = "") -> str:
    return "\n".join(
        line
        for line in (headline, _noise_caption(run.chip, run.shots), add_title)
        if line
    )


def _chip_headline(run: ProfileRun, subject: str) -> str:
    length, height = run.chip.unit_dims
    return f"{subject} across {length}x{height} chip"


def _print_run_stats(run: ProfileRun) -> None:
    s = run.stats()
    print(
        f"Distance {run.distance} \t:->  range= {s['range']:.5g};"
        f"\t pdif= {s['ratio']:.5g};"
        f"\t mean={s['mean']:.5f};"
        f"\t dev={s['stdev']:.5f}"
    )


def _print_chip_stats(chip: "Chip") -> None:
    p = np.array([n.p for n in chip.noise_map.values()], dtype=float)
    print(
        f"\nChip(mean={p.mean()}, dev={p.std(ddof=1)}, min={p.min()}, max={p.max()})"
    )


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------


def ler_cdf(
    distances: Sequence[int] = _DEFAULT_DISTANCES,
    directory: Optional[PathLike] = None,
    *,
    whisker: bool = True,
    baseline_dir: Optional[PathLike] = None,
    labels: Optional[Tuple[str, str]] = None,
    add_title: str = "",
    dpi: Optional[int] = None,
    verbose: bool = True,
    show: bool = True,
) -> Optional["Figure"]:
    """Cumulative distribution of LER across every tile placement, one step per distance.

    `whisker` adds a horizontal boxplot of the same data beneath the CDF.
    `baseline_dir` overlays a second sweep (typically the uniform-noise baseline) in
    each distance's color, for a like-for-like comparison: a faded dashed step in the
    CDF, and a fainter, narrower, hatched box paired beneath the profiled one in the box
    panel. Both series are then named in a secondary legend, taking their names from
    each chip's recorded noise model; pass `labels=(profiled, baseline)` to override.
    `verbose` prints the per-distance spread and the chip's noise summary.

    Shows the figure. Pass `show=False` to get the `Figure` back instead, to
    `savefig` it or tweak it further.
    """
    import matplotlib.pyplot as plt

    runs = load_profile_runs(distances, directory)
    baseline = load_profile_runs(distances, baseline_dir) if baseline_dir else None

    paired = baseline is not None
    if whisker:
        # paired boxes need roughly twice the vertical room per distance
        fig, (ax_cdf, ax_box) = plt.subplots(
            2,
            1,
            height_ratios=[3, 1.7] if paired else [3, 1],
            figsize=(14, 13.5) if paired else (14, 12),
            dpi=dpi,
        )
    else:
        fig, ax_cdf = plt.subplots(figsize=(10, 6), dpi=dpi)
        ax_box = None

    palette = plt.get_cmap("tab10")
    for i, d in enumerate(distances):
        run = runs[d]
        lers = run.lers
        x, y = _ecdf(lers)
        color = palette(i % palette.N)
        ax_cdf.step(x, y, color=color, label=f"d={d} (n={lers.size})")

        if baseline is not None:
            bx, by = _ecdf(baseline[d].lers)
            ax_cdf.step(
                bx,
                by,
                color=color,
                alpha=_BASELINE_STEP_ALPHA,
                linestyle=_BASELINE_LINESTYLE,
            )

        # both panels are drawn from the same `color`, so they cannot drift apart
        if ax_box is not None:
            position = i + 1
            if baseline is not None:
                _draw_box(
                    ax_box,
                    baseline[d].lers,
                    position - _BOX_DODGE,
                    color,
                    _BASELINE_BOX_ALPHA,
                    _BASELINE_BOX_WIDTH,
                    _BASELINE_HATCH,
                )
                _draw_box(
                    ax_box,
                    lers,
                    position + _BOX_DODGE,
                    color,
                    _PRIMARY_ALPHA,
                    _PRIMARY_BOX_WIDTH,
                )
            else:
                _draw_box(
                    ax_box, lers, position, color, _PRIMARY_ALPHA, _SOLO_BOX_WIDTH
                )

        if verbose:
            _print_run_stats(run)

    first = runs[distances[0]]
    if verbose:
        _print_chip_stats(first.chip)

    ax_cdf.set_title(
        _suptitle(
            _chip_headline(first, "Distribution of LER per distance=d tile profiling"),
            first,
            add_title,
        )
    )
    ax_cdf.set_ylabel("Cumulative Distribution Probability")
    ax_cdf.set_yticks([0.0, 0.5, 1.0])
    ax_cdf.set_yticks([0.25, 0.75], minor=True)
    ax_cdf.set_xscale("log")
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.legend()
    ax_cdf.grid(which="both", axis="y")
    ax_cdf.grid(which="major", axis="x")

    if ax_box is not None:
        ax_box.set_xscale("log")
        # the boxes are drawn one call at a time above, so `tick_labels=` is not
        # available - label each distance's row (or pair of rows) explicitly
        ax_box.set_yticks(
            range(1, len(distances) + 1), [str(d) for d in distances]
        )
        ax_box.set_ylim(0.4, len(distances) + 0.6)
        ax_box.set_ylabel("Distance")
        ax_box.set_xlabel("Logical Error Rate")
    else:
        ax_cdf.set_xlabel("Logical Error Rate")

    if baseline is not None:
        _add_convention_legends(
            ax_cdf, ax_box, _series_labels(first, baseline[distances[0]], labels)
        )

    fig.tight_layout()
    if show:
        plt.show()
        # returning the figure too would draw it a second time: the notebook renders
        # a returned Figure on top of what plt.show() already drew
        return None
    return fig


def ler_histogram(
    distances: Sequence[int] = _DEFAULT_DISTANCES,
    directory: Optional[PathLike] = None,
    *,
    scope: Literal["ler", "errors"] = "ler",
    limits: Tuple[Optional[float], Optional[float]] = (None, None),
    add_title: str = "",
    bins: int = 25,
    dpi: Optional[int] = None,
    show: bool = True,
) -> Optional["Figure"]:
    """A grid of per-distance histograms of LER (or raw logical error count) by placement.

    `scope` picks the quantity binned; `limits` is an `(low, high)` x-range applied to
    every subplot so distances stay directly comparable.

    Shows the figure. Pass `show=False` to get the `Figure` back instead, to
    `savefig` it or tweak it further.
    """
    import math

    import matplotlib.pyplot as plt

    runs = load_profile_runs(distances, directory)

    cols = min(_MAX_HIST_COLS, len(distances))
    rows = math.ceil(len(distances) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 5 * rows), dpi=dpi, squeeze=False
    )
    axes = axes.flatten()

    # drop the trailing slots the distance count doesn't fill
    for ax in axes[len(distances) :]:
        fig.delaxes(ax)

    xlabel = "ler" if scope == "ler" else "# logical errors"
    for ax, d in zip(axes, distances):
        run = runs[d]
        values = run.values(scope)
        zeros = np.count_nonzero(values == 0)

        ax.hist(values, bins=bins, color="skyblue", edgecolor="black")
        ax.set_ylabel("freq")
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"d{d} (n={values.size}, len(0)={zeros})")
        ax.set_xlim(limits[0], limits[1])

    first = runs[distances[0]]
    fig.suptitle(
        _suptitle(_chip_headline(first, f"{xlabel} by count"), first, add_title)
    )

    fig.tight_layout()
    if show:
        plt.show()
        # returning the figure too would draw it a second time: the notebook renders
        # a returned Figure on top of what plt.show() already drew
        return None
    return fig
