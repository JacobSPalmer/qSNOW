"""Matplotlib figures of the marginal distributions a chip carries.

`profiling.py` draws the *outcome* of a sweep, read back from flakes. This module draws
the sweep's *input*: the physical error rates (PER) sitting on a live chip's sites and
couplers. The interactive heatmaps show *where* those rates sit; the figures here show
their shape, which is what the landscape generators (`generate_gaussian_noise`,
`generate_skewed_contour_noise`, ...) are chosen for.

Matplotlib for the same reason as `profiling.py`: these are publication figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from ._captions import chip_headline, chip_suptitle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from qsnow.interface.chip import Chip

__all__ = ["per_histogram"]

Panel = Literal["sites", "couplers"]
Which = Literal["sites", "couplers", "both"]
Marker = Literal["mean", "median"]

_PANEL_ORDER: Tuple[Panel, ...] = ("sites", "couplers")
_MARKERS: Tuple[Marker, ...] = ("mean", "median")

# Same fill as `ler_histogram`, so the input and outcome figures read as one family.
_HIST_COLOR = "skyblue"
_HIST_EDGE = "black"
# Markers are drawn in one neutral ink and told apart by linestyle: the histogram is a
# single series per panel, so a second hue would suggest a second series.
_MARKER_COLOR = "0.25"
_MARKER_LINESTYLES: Dict[Marker, str] = {"mean": "-", "median": "--"}
_MARKER_STATS = {"mean": np.mean, "median": np.median}

# Wide enough for the `PED(...)` caption line on a single-panel figure.
_MIN_FIG_WIDTH = 7.0


def _panels_for(which: Which) -> Tuple[Panel, ...]:
    if which == "both":
        return _PANEL_ORDER
    if which in _PANEL_ORDER:
        return (which,)
    raise ValueError(f"which must be 'sites', 'couplers' or 'both', given {which!r}.")


def _rates_for(chip: "Chip", panel: Panel) -> np.ndarray:
    profiles = chip.noise_map if panel == "sites" else chip.coupler_map
    return np.array([n.p for n in profiles.values()], dtype=float)


def _panel_title(chip: "Chip", panel: Panel, n: int) -> str:
    """`sites (n=…)`, or `couplers (n=…, derived: <mode>)` while the couplers still
    equal the endpoint combination they were derived from - otherwise the panel would
    present the site data back as if it were a second measurement."""
    title = f"{panel} (n={n}"
    if panel == "couplers" and not chip.has_independent_couplers:
        title += f", derived: {chip.spec.coupler_mode}"
    return title + ")"


def _draw_markers(ax: "Axes", values: np.ndarray, markers: Sequence[Marker]) -> None:
    for marker in markers:
        if marker not in _MARKERS:
            raise ValueError(f"markers may contain 'mean' or 'median', given {marker!r}.")
        stat = float(_MARKER_STATS[marker](values))
        ax.axvline(
            stat,
            color=_MARKER_COLOR,
            linestyle=_MARKER_LINESTYLES[marker],
            linewidth=1.5,
            label=f"{marker}={stat:.3g}",
        )
    if markers:
        ax.legend(frameon=False)


def per_histogram(
    chip: "Chip",
    *,
    which: Which = "both",
    markers: Sequence[Marker] = _MARKERS,
    limits: Tuple[Optional[float], Optional[float]] = (None, None),
    add_title: str = "",
    bins: int = 25,
    dpi: Optional[int] = None,
    show: bool = True,
) -> Optional["Figure"]:
    """Histogram(s) of a chip's physical error rates, per site and/or per coupler.

    `which` picks the panels: `"sites"`, `"couplers"`, or `"both"` side by side on a
    shared x-axis so the two marginals compare directly. `markers` adds a vertical line
    per named statistic (`"mean"`, `"median"`; pass `()` for none) with its value in the
    legend. `limits` is an `(low, high)` x-range applied to every panel.

    A coupler panel is labelled `derived: <mode>` while the coupler rates are still the
    endpoint combination `derive_coupler_noise` produced, since they then carry no
    information the site panel does not.

    Shows the figure. Pass `show=False` to get the `Figure` back instead, to `savefig`
    it or tweak it further.
    """
    import matplotlib.pyplot as plt

    panels = _panels_for(which)
    markers = tuple(markers)

    # One panel is narrower than the caption line, so it gets a floor width.
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(max(4.5 * len(panels), _MIN_FIG_WIDTH), 5),
        dpi=dpi,
        sharex=True,
        squeeze=False,
    )
    for ax, panel in zip(axes.flatten(), panels):
        values = _rates_for(chip, panel)
        ax.hist(values, bins=bins, color=_HIST_COLOR, edgecolor=_HIST_EDGE)
        _draw_markers(ax, values, markers)
        ax.set_ylabel("freq")
        ax.set_xlabel("PER")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(_panel_title(chip, panel, values.size))
        ax.set_xlim(limits[0], limits[1])

    fig.suptitle(chip_suptitle(chip_headline(chip, "PER by count"), chip, None, add_title))

    fig.tight_layout()
    if show:
        plt.show()
        # returning the figure too would draw it a second time: the notebook renders
        # a returned Figure on top of what plt.show() already drew
        return None
    return fig
