"""Matplotlib figures for LER distributions read back from experiment/result flakes.

`visualize.py` and `interactive.py` draw *chips* with plotly. This module draws the
*outcome* of a profiling sweep: it loads the `experiment_*.flake` / `results_*.flake`
pairs a run left behind (see `qsnow.helpers.serialize`) and renders the distribution of
logical error rate across every tile placement, one series per code distance. The
sweep's *input* - the PER landscape on a chip - is drawn by `distributions.py`.

Matplotlib rather than plotly here because these are publication figures: explicit `dpi`
and `Figure.savefig` matter more than hover/zoom.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ._captions import chip_headline, chip_suptitle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from qsnow.experiments import Experiment, ExperimentResults
    from qsnow.visualize.visualize import VisualizationStyle
    from qsnow.interface.chip import Chip

__all__ = [
    "ProfileRun",
    "load_profile_runs",
    "ler_cdf",
    "ler_histogram",
    "ler_table",
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
# The CDF's confidence band is a faint wash behind each profiled step. The baseline gets
# no band: its near-vertical steps would only smear colour over the profiled bands, and
# its uncertainty is not what the figure argues from.
_BAND_ALPHA = 0.15
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
# Gap, in axes fractions, between two legends stacked in the same corner.
_LEGEND_STACK_PAD = 0.015

# Confidence level of the per-placement LER interval (`ProfileRun.ler_intervals`). The
# interval is the Wilson score interval (Wilson 1927), the textbook binomial interval that
# behaves at small counts, so a placement held at the sampler's 30-error floor gets an
# honest, asymmetric bound, and a zero-error placement gets `low = 0, high ~= 3/shots`
# (the rule of three). `sinter.fit_binomial` was considered and passed over: it returns a
# likelihood-ratio interval keyed by a Bayes factor rather than a confidence level, so
# it cannot be quoted as "95%" in a caption.
_INTERVAL_CONFIDENCE = 0.95


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

    def ler_intervals(
        self, confidence: float = _INTERVAL_CONFIDENCE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """`(low, high)` Wilson score bounds on each placement's LER, in `lers` order.

        Computed from the stored `errors`/`shots` counts, so it needs no re-run. This is
        the measurement uncertainty on each placement; the sweep itself visits every
        valid placement, so there is no sampling uncertainty on the distribution.
        """
        from scipy.stats import binomtest

        bounds = [
            binomtest(int(entry["errors"]), int(entry["shots"])).proportion_ci(
                confidence_level=confidence, method="wilson"
            )
            for entry in self.results.results.values()
        ]
        return (
            np.array([b.low for b in bounds], dtype=float),
            np.array([b.high for b in bounds], dtype=float),
        )

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

    # ------------------------------------------------------------------
    # Interactive views of the flakes behind this run
    # ------------------------------------------------------------------

    def show(
        self,
        *,
        chip_only: bool = False,
        extra_styles: Optional["Mapping[str, VisualizationStyle]"] = None,
    ) -> None:
        """Open the interactive plotly view of the flakes behind this run.

        By default this is the experiment's own view - for a packing sweep, the LER
        heatmap over every placement. `chip_only` shows just the chip the sweep ran on
        (layout / status / noise), which is the setup rather than the outcome.
        """
        if chip_only:
            self.chip.show(interactive=True, extra_styles=extra_styles)
            return
        try:
            self.experiment.show(self.results, extra_styles=extra_styles)
        except NotImplementedError:
            # a bare Experiment, or a subclass that has not implemented show(); the
            # chip it ran on is still worth putting on screen
            self.chip.show(interactive=True, extra_styles=extra_styles)

    def export(self, path: Optional[PathLike] = None, *, chip_only: bool = False, **kwargs):
        """Write the same view to a standalone interactive HTML page, returning its path.

        Mirrors `show()`: the experiment's view by default, the chip's with `chip_only`.
        Extra keyword arguments (`styles`, `title`, `label`, `include_plotlyjs`) are
        forwarded to `qsnow.visualize.interactive.export_html`.
        """
        # deferred: interactive.py pulls in the experiment stack, which imports this
        # package - the same cycle every other qsnow import in this module dodges
        from qsnow.visualize.interactive import export_html

        if chip_only:
            return export_html(self.chip, path, **kwargs)
        return export_html(self.experiment, path, results=self.results, **kwargs)


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


def _draw_band(ax, run: ProfileRun, color, alpha: float) -> None:
    """Shade the envelope of `run`'s per-placement LER intervals behind its CDF step.

    The envelope runs from the ECDF of every placement's lower bound to the ECDF of its
    upper bound: sorting each bound independently *is* that ECDF, and both share the
    step's y grid because they have the same count. `step="pre"` matches the default of
    the `Axes.step` the CDF is drawn with, so band and line agree at every riser.
    """
    low, high = run.ler_intervals()
    _, y = _ecdf(run.lers)
    # `_ecdf` prepends the (-inf, 0) run-in point for the step; a fill cannot use it
    ax.fill_betweenx(
        y[1:], np.sort(low), np.sort(high), step="pre", color=color, alpha=alpha, linewidth=0
    )


def _finish_figure(fig: "Figure", save: Optional[PathLike], show: bool) -> Optional["Figure"]:
    """Save, show, and/or return `fig` - the one place the figure functions end.

    `save` writes the figure (parent directories created) before anything is shown, at
    the dpi the figure was made with. `show=True` then draws it and returns `None`:
    returning the figure too would draw it a second time, since a notebook renders a
    returned `Figure` on top of what `plt.show()` already drew.
    """
    import matplotlib.pyplot as plt

    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()
        return None
    return fig


def _style_box(
    bp: Dict[str, list],
    color,
    alpha: float,
    hatch: Optional[str] = None,
    linewidth: Optional[float] = None,
) -> None:
    """Tint every artist of one boxplot to `color` at `alpha`.

    `linewidth` sets the weight of the box edge, whiskers, caps and median together;
    `None` keeps matplotlib's defaults (and the median's slightly heavier 1.3).

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
        if linewidth is not None:
            patch.set_linewidth(linewidth)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color(color)
            line.set_alpha(alpha)
            if linewidth is not None:
                line.set_linewidth(linewidth)
    for line in bp["medians"]:
        # black, and less faded than the box: a colour-matched median disappears into
        # its own fill at the baseline's low alpha
        line.set_color("black")
        line.set_alpha(min(1.0, alpha + 0.45))
        line.set_linewidth(1.3 if linewidth is None else linewidth)
    for flier in bp["fliers"]:
        # a d=3 sweep has ~1400 placements; default fliers bury the box they belong to
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_alpha(alpha * 0.55)
        flier.set_markersize(2.5)


# Figure width, and the height in inches of each panel of the whisker layout before
# `box_scale` is applied. Paired boxes need roughly twice the vertical room per distance.
_FIG_WIDTH = 14
_CDF_HEIGHT = {False: 9.0, True: 8.6}
_BOX_HEIGHT = {False: 3.0, True: 4.9}


def _whisker_layout(
    paired: bool, box_scale: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """`(figsize, height_ratios)` for the CDF-over-whisker layout.

    `box_scale` stretches only the box panel's share of the height. Box widths and the
    row pitch share the same data units, so fattening the boxes in place would always
    eat the gap between rows; growing the panel instead gives every row more physical
    room, and boxes and gaps scale together. The figure's width then grows by the same
    factor as its total height, so the figure keeps its aspect ratio and simply gets
    larger rather than turning into a tall strip.
    """
    if box_scale <= 0:
        raise ValueError(f"box_scale must be > 0, got {box_scale}")
    cdf_height, box_height = _CDF_HEIGHT[paired], _BOX_HEIGHT[paired] * box_scale
    growth = (cdf_height + box_height) / (_CDF_HEIGHT[paired] + _BOX_HEIGHT[paired])
    figsize = (_FIG_WIDTH * growth, cdf_height + box_height)
    return figsize, (cdf_height, box_height)


def _draw_box(
    ax,
    values,
    position: float,
    color,
    alpha: float,
    width: float,
    hatch: Optional[str] = None,
    linewidth: Optional[float] = None,
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
    _style_box(bp, color, alpha, hatch, linewidth)


def _series_labels(
    primary: ProfileRun,
    baseline: ProfileRun,
    override: Optional[Tuple[str, str]] = None,
) -> Tuple[str, str]:
    """Name the profiled and baseline series, preferring each chip's noise-model name.

    `override` wins outright. Otherwise the names come from the flake, the same source
    `_captions.noise_caption` reads - "derived contour" vs "uniform homogeneous" on the DATE
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


def _keep_legend(ax, legend) -> None:
    """Keep `legend` on `ax` across the next `ax.legend()` call.

    `Axes.legend` *replaces* the axes' legend rather than adding alongside it; re-adding
    the old one as a plain artist is what lets two legends share a panel.
    """
    if legend is not None:
        ax.add_artist(legend)


def _axes_frac(ax, artist):
    """`artist`'s drawn bounding box in the axes' own 0-1 coordinates.

    Axes fractions rather than pixels so a measurement taken once survives a later
    resize or dpi change - `savefig(dpi=...)` rescales every pixel extent on the figure.
    """
    return artist.get_window_extent().transformed(ax.transAxes.inverted())


def _pin_legend(ax, legend, x: float, y: float, loc: str) -> None:
    """Pin `legend`'s `loc` corner to `(x, y)` in axes fractions.

    `borderaxespad` is the gap matplotlib keeps between a legend and whatever it is
    anchored to. Here the anchor is already the exact point we want the corner on, so the
    pad would push the box off it - and there is no public setter for it.
    """
    legend.set_loc(loc)
    legend.set_bbox_to_anchor((x, y), transform=ax.transAxes)
    legend.borderaxespad = 0.0


def _stack_legend(ax, anchor, handles, labels: Sequence[str], **kwargs):
    """Add a second legend to `ax`, stacked beneath `anchor` rather than in a fixed corner.

    `loc="best"` scores the candidate corners against the axes' lines and patches only -
    another legend is invisible to it - so an auto-placed legend and a corner-pinned key
    will happily land on top of each other. Measuring `anchor` once the layout is final
    and hanging the new legend off its edge sidesteps the question entirely: wherever
    `anchor` went, the key follows.

    The key always ends up *under* `anchor`, reading as a footnote to it. When `anchor`
    is already on the axes floor - the common case, since "best" likes the corner a CDF
    leaves free - there is nowhere below to put it, so `anchor` is lifted by the key's
    height instead and the pair keeps its order.

    Call this *after* `tight_layout`, or the measurement is of the pre-layout axes.
    """
    # "best" is resolved at draw time, so `anchor` has no meaningful extent until one
    ax.figure.canvas.draw()
    bb = _axes_frac(ax, anchor)

    right = bb.x1 > 0.5
    x = bb.x1 if right else bb.x0
    side = "right" if right else "left"

    _keep_legend(ax, anchor)
    key = ax.legend(handles, list(labels), **kwargs)
    _pin_legend(ax, key, x, bb.y0 - _LEGEND_STACK_PAD, f"upper {side}")
    ax.figure.canvas.draw()

    floor = _axes_frac(ax, key).y0 < 0  # the key overhangs the bottom of the panel
    if floor:
        # slide the pair up as a unit: the key keeps the corner `anchor` had chosen and
        # `anchor` sits on top of it, so the reading order is unchanged either way
        _pin_legend(ax, key, x, bb.y0, f"lower {side}")
        lift = _axes_frac(ax, key).height + _LEGEND_STACK_PAD
    else:
        lift = 0.0

    # freeze `anchor` - "best" re-resolves on every draw, and would otherwise slide out
    # from under the key on the next one
    _pin_legend(ax, anchor, x, bb.y0 + lift, f"lower {side}")
    return key


def _add_convention_legends(
    ax_cdf,
    ax_box,
    labels: Tuple[str, str],
    *,
    anchor=None,
    key_loc: Optional[str] = None,
) -> None:
    """Add the profiled-vs-baseline key to whichever panels were drawn.

    Each panel gets the key in its own artist type - lines for the CDF, patches for the
    boxes - so the swatch matches what the reader is looking at.

    On the CDF panel the key stacks against `anchor` (the distance legend) so the two
    cannot collide wherever `anchor` auto-placed itself. Pass `key_loc` to pin the key to
    a corner of its own instead, leaving `anchor` free to sit elsewhere.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    grey = _LEGEND_SWATCH_COLOR
    if ax_cdf is not None:
        handles = [
            Line2D([], [], color=grey, lw=2.5, alpha=_PRIMARY_ALPHA),
            Line2D(
                [],
                [],
                color=grey,
                lw=2.5,
                alpha=_BASELINE_STEP_ALPHA,
                linestyle=_BASELINE_LINESTYLE,
            ),
        ]
        if key_loc is None and anchor is not None:
            _stack_legend(ax_cdf, anchor, handles, labels, framealpha=0.95)
        else:
            _keep_legend(ax_cdf, anchor if anchor is not None else ax_cdf.get_legend())
            ax_cdf.legend(
                handles,
                list(labels),
                loc=key_loc or "lower right",
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
            framealpha=0.95,
        )


def _suptitle(headline: str, run: ProfileRun, add_title: str = "") -> str:
    """The shared chip caption (`_captions.chip_suptitle`), fed from a run."""
    return chip_suptitle(headline, run.chip, run.shots, add_title)


def _chip_headline(run: ProfileRun, subject: str) -> str:
    return chip_headline(run.chip, subject)


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
    band: bool = True,
    band_alpha: float = _BAND_ALPHA,
    linewidth: Optional[float] = None,
    box_scale: float = 1.0,
    box_linewidth: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    baseline_dir: Optional[PathLike] = None,
    labels: Optional[Tuple[str, str]] = None,
    legend_loc: str = "best",
    key_loc: Optional[str] = None,
    add_title: str = "",
    dpi: Optional[int] = None,
    verbose: bool = True,
    save: Optional[PathLike] = None,
    show: bool = True,
    **kwargs
) -> Optional["Figure"]:
    """Cumulative distribution of LER across every tile placement, one step per distance.

    `whisker` adds a horizontal boxplot of the same data beneath the CDF.
    `band` shades, in each distance's colour, the envelope of the profiled sweep's
    per-placement 95% Wilson intervals (`ProfileRun.ler_intervals`): the ECDF of every
    placement's lower bound out to the ECDF of its upper bound. It is the measurement
    uncertainty on each point, not sampling uncertainty on the distribution, which visits
    every placement. The band is widest where the sampler's error floor binds - the
    low-LER tail of the higher distances - and near-invisible where placements saw
    thousands of errors. The baseline overlay is drawn without a band to keep the figure
    legible. `band_alpha` sets the band's opacity. `linewidth` sets the width of the
    CDF steps, profiled and baseline alike (default: matplotlib's `lines.linewidth`).
    `box_scale` stretches the whisker panel by that factor, so every box and the gap
    between rows grow together and can never overlap; the whole figure then scales up
    in proportion, keeping its aspect ratio.
    `box_linewidth` sets the weight of the box edges, whiskers, caps and medians.
    `figsize` sets the overall figure size in inches, `(width, height)`, in place of the
    computed default; with `whisker`, `box_scale` still decides how that height is split
    between the two panels.
    `baseline_dir` overlays a second sweep (typically the uniform-noise baseline) in
    each distance's color, for a like-for-like comparison: a faded dashed step in the
    CDF, and a fainter, narrower, hatched box paired beneath the profiled one in the box
    panel. Both series are then named in a secondary key, taking their names from
    each chip's recorded noise model; pass `labels=(profiled, baseline)` to override.
    `verbose` prints the per-distance spread and the chip's noise summary.

    `legend_loc` places the distance legend (default `"best"`, matplotlib's auto
    placement). By default the profiled/baseline key then stacks directly against it,
    wherever it landed, so the two cannot overlap. Pass `key_loc` to pin the key to a
    corner of its own instead - `legend_loc="upper left", key_loc="lower right"` puts
    them in opposite corners. Aiming `key_loc` at the corner `"best"` picks is the one
    combination that can still collide.

    `save` writes the figure to that path before showing it. Shows the figure; pass
    `show=False` to get the `Figure` back instead, to tweak it further.
    """
    import matplotlib.pyplot as plt

    runs = load_profile_runs(distances, directory)
    baseline = load_profile_runs(distances, baseline_dir) if baseline_dir else None

    paired = baseline is not None
    if whisker:
        default_size, height_ratios = _whisker_layout(paired, box_scale)
        fig, (ax_cdf, ax_box) = plt.subplots(
            2,
            1,
            height_ratios=height_ratios,
            figsize=figsize or default_size,
            dpi=dpi,
        )
    else:
        fig, ax_cdf = plt.subplots(figsize=figsize or (10, 6), dpi=dpi)
        ax_box = None

    palette = plt.get_cmap("tab10")
    for i, d in enumerate(distances):
        run = runs[d]
        lers = run.lers
        x, y = _ecdf(lers)
        color = palette(i % palette.N)
        ax_cdf.step(x, y, color=color, label=f"d={d} (n={lers.size})", linewidth=linewidth)
        if band:
            _draw_band(ax_cdf, run, color, band_alpha)

        if baseline is not None:
            bx, by = _ecdf(baseline[d].lers)
            ax_cdf.step(
                bx,
                by,
                color=color,
                alpha=kwargs.get('baseline_alpha', _BASELINE_STEP_ALPHA),
                linestyle=_BASELINE_LINESTYLE,
                linewidth=linewidth,
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
                    linewidth=box_linewidth,
                )
                _draw_box(
                    ax_box,
                    lers,
                    position + _BOX_DODGE,
                    color,
                    _PRIMARY_ALPHA,
                    _PRIMARY_BOX_WIDTH,
                    linewidth=box_linewidth,
                )
            else:
                _draw_box(
                    ax_box,
                    lers,
                    position,
                    color,
                    _PRIMARY_ALPHA,
                    _SOLO_BOX_WIDTH,
                    linewidth=box_linewidth,
                )

        if verbose:
            _print_run_stats(run)

    first = runs[distances[0]]
    if verbose:
        _print_chip_stats(first.chip)

    if(kwargs.get('title', '')):
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
    distance_legend = ax_cdf.legend(loc=legend_loc, framealpha=0.5)
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

    fig.tight_layout()

    # after `tight_layout`: stacking the key measures the distance legend, and a
    # measurement taken against the pre-layout axes would leave the pair misaligned
    if baseline is not None:
        _add_convention_legends(
            ax_cdf,
            ax_box,
            _series_labels(first, baseline[distances[0]], labels),
            anchor=distance_legend,
            key_loc=key_loc,
        )

    return _finish_figure(fig, save, show)


def ler_histogram(
    distances: Sequence[int] = _DEFAULT_DISTANCES,
    directory: Optional[PathLike] = None,
    *,
    scope: Literal["ler", "errors"] = "ler",
    limits: Tuple[Optional[float], Optional[float]] = (None, None),
    add_title: str = "",
    bins: int = 25,
    dpi: Optional[int] = None,
    save: Optional[PathLike] = None,
    show: bool = True,
) -> Optional["Figure"]:
    """A grid of per-distance histograms of LER (or raw logical error count) by placement.

    `scope` picks the quantity binned; `limits` is an `(low, high)` x-range applied to
    every subplot so distances stay directly comparable.

    `save` writes the figure to that path before showing it. Shows the figure; pass
    `show=False` to get the `Figure` back instead, to tweak it further.
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
    return _finish_figure(fig, save, show)


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------

# Row order of `ler_table` and the one-line reading of each stat that `explain=True`
# prints; one dict so the table and its explanation cannot disagree. "ratio" is the
# profiled value over the baseline's unless the entry says otherwise.
_STAT_MEANINGS: Dict[str, str] = {
    "n": "placements swept on each chip.",
    "best": "lowest LER over placements (the best spot on the chip); ratio < 1 means the "
            "profiled chip offers better placements than the baseline ever does.",
    "p5": "5th percentile of LER: the good end of the chip without its single luckiest placement.",
    "median": "median LER: the typical placement.",
    "p95": "95th percentile of LER: the bad end without the single unluckiest placement.",
    "worst": "highest LER over placements (the worst spot); ratio > 1 is 'up to x times worse'.",
    "spread p95/p5": "within-chip variation, robust to one outlier; on a uniform chip this is "
                     "only Monte-Carlo noise, so read the profiled value against it.",
    "spread worst/best": "full within-chip range, best placement to worst; same reading.",
    "CV": "std / mean of LER over placements; on a uniform chip this is the sampling noise "
          "of the LER estimates themselves.",
    "yield": "fraction of placements at or under the BAD threshold; 'ratio' holds the "
             "difference in percentage points (profiled minus baseline).",
}
# The ratio rows that the across-distances block summarises with its min and max.
_SUMMARISED = ("best", "median", "worst", "spread p95/p5", "spread worst/best")


def _placement_stats(lers: np.ndarray, bad: float) -> Dict[str, float]:
    """The `_STAT_MEANINGS` numbers for one sweep's per-placement LERs."""
    lo, hi = float(lers.min()), float(lers.max())
    p5, p95 = (float(q) for q in np.quantile(lers, [0.05, 0.95]))
    mean = float(lers.mean())
    return {
        "n": float(lers.size),
        "best": lo,
        "p5": p5,
        "median": float(np.median(lers)),
        "p95": p95,
        "worst": hi,
        "spread p95/p5": p95 / p5 if p5 else float("inf"),
        "spread worst/best": hi / lo if lo else float("inf"),
        "CV": float(lers.std(ddof=1)) / mean if lers.size > 1 and mean else 0.0,
        "yield": float(np.mean(lers <= bad)),
    }


def _compare(stat: str, profiled: float, baseline: float) -> float:
    """The comparison column: percentage-point difference for yield, else the ratio."""
    if stat == "yield":
        return 100.0 * (profiled - baseline)
    return profiled / baseline if baseline else float("inf")


def ler_table(
    distances: Sequence[int] = _DEFAULT_DISTANCES,
    directory: Optional[PathLike] = None,
    *,
    baseline_dir: PathLike,
    bad: float = 0.005,
    labels: Optional[Tuple[str, str]] = None,
    verbose: bool = True,
    explain: bool = False,
) -> List[Dict[str, Any]]:
    """How LER varies across placements on the profiled chip versus its baseline.

    The tabular companion to `ler_cdf`, aimed at one kind of statement: "across
    distances, LER differs by up to x times in the worst placements and y times in the
    best between the uniform and the contoured device". Per distance it reports, for
    both sweeps, the extremes and quantiles of LER over placements, two within-chip
    spread ratios, the coefficient of variation and the yield at the `bad` threshold
    (see `_STAT_MEANINGS`), with a comparison column: profiled / baseline, or for yield
    the difference in percentage points. A final block (`distance = "all"`) gives the
    min and max over distances of the ratio for the best, median and worst placements
    and the two spreads.

    Returns tidy rows `{"distance", "stat", <profiled label>, <baseline label>,
    "ratio"}` (`"min ratio"` / `"max ratio"` in the summary block), so
    `pandas.DataFrame(rows)` is the notebook table. `verbose` prints the same as text;
    `explain=True` adds one line per stat saying what it means.
    """
    runs = load_profile_runs(distances, directory)
    baseline = load_profile_runs(distances, baseline_dir)
    first = distances[0]
    p_label, b_label = _series_labels(runs[first], baseline[first], labels)

    rows: List[Dict[str, Any]] = []
    ratios: Dict[str, List[float]] = {stat: [] for stat in _SUMMARISED}
    for d in distances:
        p_stats = _placement_stats(runs[d].lers, bad)
        b_stats = _placement_stats(baseline[d].lers, bad)
        for stat in _STAT_MEANINGS:
            ratio = _compare(stat, p_stats[stat], b_stats[stat])
            rows.append({"distance": d, "stat": stat, p_label: p_stats[stat], b_label: b_stats[stat], "ratio": ratio})
            if stat in ratios:
                ratios[stat].append(ratio)
    for stat in _SUMMARISED:
        rows.append({"distance": "all", "stat": stat, "min ratio": min(ratios[stat]), "max ratio": max(ratios[stat])})

    if verbose:
        _print_ler_table(rows, p_label, b_label, distances)
    if explain:
        print("\nStats (ratio = profiled / baseline unless stated):")
        for stat, meaning in _STAT_MEANINGS.items():
            print(f"  {stat:<18} {meaning}")
        print(f"  {'all':<18} min and max over distances of each ratio: the 'up to x times' and "
              f"'at least y times' across the sweep.")
    return rows


def _print_ler_table(rows: List[Dict[str, Any]], p_label: str, b_label: str, distances: Sequence[int]) -> None:
    w = max(len(p_label), len(b_label), 10)
    for d in distances:
        print(f"\nDistance {d}")
        print(f"  {'stat':<18} {p_label:>{w}} {b_label:>{w}} {'ratio':>10}")
        for r in rows:
            if r["distance"] == d:
                print(f"  {r['stat']:<18} {r[p_label]:>{w}.4g} {r[b_label]:>{w}.4g} {r['ratio']:>10.4g}")
    print("\nAcross distances")
    print(f"  {'stat':<18} {'min ratio':>10} {'max ratio':>10}")
    for r in rows:
        if r["distance"] == "all":
            print(f"  {r['stat']:<18} {r['min ratio']:>10.4g} {r['max ratio']:>10.4g}")
