"""Core visualization library for qSNOW objects."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil, floor, log10
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from qsnow.interface.models import CSSType, Coupler, Qubit, Status, Tag, Coord

if TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure
    from qsnow.interface.chip import Chip
    from qsnow.interface.tile import LogicalTile

# ------------------------------------------------------------------
# Visualization style classes/types
# ------------------------------------------------------------------

ColorLike = Union[str, float]

# Horizontal strip reserved per colorbar: the bar itself plus its tick labels. A
# side-mounted title sits to the *right* of its bar, which is where the next bar would go,
# so each additional bar also buys a gap wide enough for the previous bar's title.
_COLORBAR_PX = 100
_COLORBAR_TITLE_PX = 40

# Plot-area sizing. One scale covers both axes (see `_compute_geometry`), so the cap and
# floor bound the *longer* side and the shorter one follows from the chip's aspect ratio.
_PX_PER_COORD = 60
_MIN_FIG_PX = 600
_MAX_FIG_PX = 900

# Half a cell of breathing room before the first coordinate and after the last.
# `_compute_geometry` sizes the plot area from these and `_apply_frame` installs them as
# the axis ranges; they must not drift, or plotly's scaleanchor silently distorts the
# range to reconcile the two.
_RANGE_LO = -0.75
_RANGE_HI = -0.25

_SHAPE_TYPES = {
    "s": "rect",
    "o": "circle",
}


@dataclass
class QubitStyle:
    color: ColorLike = "lightgray"
    marker: str = "s"
    size: float = (
        0.5  # diameter in data (axis) units, so qubits scale naturally on zoom/pan
    )
    alpha: float = 1.0
    edgecolor: str = "black"
    linewidths: float = 0.75
    custom_hovertext: Optional[str] = None

@dataclass
class ColorbarSpec:
    # A plotly colorscale name, or an explicit [[frac, color], ...] list (see
    # `_floored_colorscale`).
    colorscale: Union[str, List[List[Any]]]
    cmin: float
    cmax: float
    label: str = ""
    # Explicit tick positions/labels, in the same units as cmin/cmax. A log-scaled style
    # feeds log10 values as colors and uses these to label the bar in the original units.
    tickvals: Optional[List[float]] = None
    ticktext: Optional[List[str]] = None
    # 'right' keeps a lone title from eating into the bar's length; 'top' reads better
    # when several bars sit side by side and their titles would otherwise run vertically.
    title_side: str = "right"


@dataclass
class CouplerStyle:
    """How one coupler is drawn, represented as a line between the two qubits it joins."""

    color: ColorLike = "lightgray"
    width: float = 2.0
    alpha: float = 1.0
    custom_hovertext: Optional[str] = None


# TODO - add options in logical style for text or perhaps add a title style
@dataclass
class LogicalStyle:
    title: Optional[str] = None
    edgecolor: ColorLike = "red"
    facecolor: ColorLike = "none"
    alpha: float = 1.0
    linewidth: float = 2.0


@dataclass
class VisualizationStyle:
    style_fn: Callable[[Qubit], QubitStyle]
    colorbar: Optional[ColorbarSpec] = None
    logical_style: Optional[Callable[[Tag], LogicalStyle]] = None
    # One-line description of what this style shows, surfaced as a caption
    # next to the style dropdown in `visualize_interactive()`.
    desc: Optional[str] = None
    # Optional edge layer, mirroring `logical_style`. Left None by every default style:
    # while couplers sit at their derived value they are a function of the qubit rates
    # already on screen, so drawing them would restate that data as if it were a second
    # measurement (and couplers outnumber qubits ~2:1). Style factories opt in via their
    # own `couplers=` argument - see `noise_heatmap_style`.
    coupler_style: Optional[Callable[[Coupler], CouplerStyle]] = None
    # Give the coupler layer its own scale instead of sharing `colorbar`. Needed whenever
    # qubit and coupler rates occupy different ranges - measured two-qubit rates typically
    # run several times higher, and one shared scale would flatten the qubit variation.
    # Costs a second (invisible) trace, since plotly draws one colorbar per trace.
    coupler_colorbar: Optional[ColorbarSpec] = None
    # Write each qubit's index into its marker. Off by default: legible at a few hundred
    # qubits, illegible past that, and it costs one annotation per qubit.
    qubit_labels: bool = False


_STATUS_COLORS = {
    Status.INACTIVE: "lightgray",
    Status.LOGICAL: "steelblue",
    Status.ANCILLA: "orange",
}

_CSS_COLORS = {
    CSSType.X_CHECK: "firebrick",
    CSSType.Z_CHECK: "dodgerblue",
    CSSType.DATA: "dimgrey",
    CSSType.BUFFER: "pink",
    CSSType.UNASSIGNED: "lightgray",
}

_STATUS_ALPHA = {
    Status.INACTIVE: 0.5,
    Status.LOGICAL: 1.0,
    Status.ANCILLA: 0.75,
}


def _hovertext_format(base: str, **kwargs):
    return f"{base}<br>" + "<br>".join(
        [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    )


def _default_qubit_style_by_css_type(qubit: Qubit) -> QubitStyle:
    return QubitStyle(
        color=_CSS_COLORS.get(qubit.type, "lightgray"),
        alpha=_STATUS_ALPHA.get(qubit.status, 0.5),
        custom_hovertext=_hovertext_format(
            f"({qubit.loc[0]}, {qubit.loc[1]})",
            type=qubit.type.name,
            noise=f"{qubit.noise.p:.4f}",
        ),
    )


def _default_qubit_style_by_status(qubit: Qubit) -> QubitStyle:
    return QubitStyle(
        color=_STATUS_COLORS.get(qubit.status, "lightgray"),
        alpha=_STATUS_ALPHA.get(qubit.status, 0.5),
        custom_hovertext=_hovertext_format(
            f"({qubit.loc[0]}, {qubit.loc[1]})",
            status=qubit.status.name,
            noise=f"{qubit.noise.p:.4f}",
        ),
    )


def _default_logical_style(tag: Tag, **kwargs) -> LogicalStyle:
    return LogicalStyle(title=tag.name if tag.name else None, **kwargs)


# ------------------------------------------------------------------
# Importable visualization styles
# ------------------------------------------------------------------

default_style = VisualizationStyle(
    style_fn=_default_qubit_style_by_status, logical_style=_default_logical_style
)
css_style = VisualizationStyle(
    style_fn=_default_qubit_style_by_css_type, logical_style=_default_logical_style
)


def _show_couplers(chip: Chip, couplers: Union[bool, str]) -> bool:
    """Resolve a `couplers=` argument, where "auto" means "only if they'd show anything".

    A coupler layer is worth its ink only when the rates are not still a function of the
    qubit rates the same figure already draws - see `Chip.has_independent_couplers`.
    """
    return chip.has_independent_couplers if couplers == "auto" else bool(couplers)


def with_couplers(
    style: VisualizationStyle, chip: Chip, couplers: Union[bool, str] = "auto"
) -> VisualizationStyle:
    """Return `style` with the coupler edge layer enabled or disabled.

    A copy, never a mutation, so the shared module-level style singletons stay pristine.
    """
    if not _show_couplers(chip, couplers):
        return replace(style, coupler_style=None)

    # Color by rate only where the style has a scale to read it against; on a
    # non-heatmap view (status, CSS type) the edges are topology, not data, so they get
    # one neutral color rather than a float plotly has no way to interpret.
    if style.colorbar is not None:
        coupler_style = coupler_heatmap_style(chip).coupler_style
    else:

        def coupler_style(coupler: Coupler) -> CouplerStyle:
            return CouplerStyle(
                color="lightgray",
                width=1.5,
                custom_hovertext=_coupler_hovertext(coupler),
            )

    return replace(style, coupler_style=coupler_style)


def _floored_colorscale(name: str, floor: float = 0.25, steps: int = 9) -> List[List[Any]]:
    """`name` with its palest end trimmed off.

    Couplers are drawn as bare lines with no outline, so a near-white low end - which
    `Blues` and `hot_r` both have - vanishes against the figure's light background and
    takes the lattice topology with it. Qubit markers do not need this: their black edge
    keeps them visible however pale the fill.
    """
    from plotly.colors import sample_colorscale

    fracs = [floor + (1.0 - floor) * i / (steps - 1) for i in range(steps)]
    return [[i / (steps - 1), c] for i, c in enumerate(sample_colorscale(name, fracs))]


def _label_color(fill: ColorLike) -> str:
    """Black or white, whichever reads against `fill`.

    Qubit labels cannot be a fixed colour: `hot_r` is white at its low end and black at its
    high end, so any single choice disappears at one extreme. Perceived luminance picks the
    readable one, which also keeps custom colorscales working.
    """
    if not isinstance(fill, str) or not fill.startswith("rgb"):
        return "white"
    try:
        r, g, b = (float(v) for v in fill[fill.index("(") + 1 : fill.index(")")].split(",")[:3])
    except (ValueError, IndexError):
        return "white"
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"


def _coupler_hovertext(coupler: Coupler) -> str:
    (x0, y0), (x1, y1) = coupler.ends
    return _hovertext_format(
        f"({x0}, {y0}) <-> ({x1}, {y1})", noise=f"{coupler.noise.p:.4f}"
    )


def noise_heatmap_style(
    chip: Chip,
    colorscale: str = "hot_r",
    limits: Optional[Tuple[float, float]] = None,
    desc: Optional[str] = None,
    *,
    couplers: Union[bool, str] = "auto",
) -> VisualizationStyle:
    """Color each qubit by its noise value `p`, normalized across the chip.

    Couplers join the same scale when they carry rates of their own; `couplers=True`/`False`
    forces the layer on or off.
    """
    show_couplers = _show_couplers(chip, couplers)
    if limits:
        cmin, cmax = limits
    else:
        p_values = [qubit.noise.p for qubit in chip.qubits]
        # one shared scale: qubit and coupler p are the same quantity on the same bound,
        # so a single colorbar reads both (and plotly allows only one per trace anyway)
        if show_couplers:
            p_values += [c.noise.p for c in chip.couplers]
        cmin, cmax = min(p_values), max(p_values)

    def style_fn(qubit: Qubit) -> QubitStyle:
        # raw value; the colorscale mapping is applied trace-wide by `visualize()`
        return QubitStyle(
            color=qubit.noise.p,
            custom_hovertext=_hovertext_format(
                f"({qubit.loc[0]}, {qubit.loc[1]})",
                status=qubit.status.name,
                noise=f"{qubit.noise.p:.4f}",
            ),
        )

    def logical_style_fn(tag: Tag) -> LogicalStyle:
        return _default_logical_style(tag, edgecolor="blue")

    def coupler_style_fn(coupler: Coupler) -> CouplerStyle:
        return CouplerStyle(
            color=coupler.noise.p, custom_hovertext=_coupler_hovertext(coupler)
        )

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=ColorbarSpec(
            colorscale=colorscale, cmin=cmin, cmax=cmax, label="Noise (p)"
        ),
        logical_style=logical_style_fn,
        desc=desc,
        coupler_style=coupler_style_fn if show_couplers else None,
    )


def _log_ticks(lo: float, hi: float) -> Tuple[List[float], List[str]]:
    """Colorbar ticks for a log10-scaled range, at the 1/2/3/4/6 x 10^k positions.

    Returns positions in log10 space (what the colors are keyed on) paired with labels in
    the original units, so a reader sees error rates rather than their logarithms.
    """
    # Sub-decade ticks only while the range is narrow enough to need them; over a wide
    # span they collapse into an unreadable stack, so step back to decades.
    span = hi - lo
    mantissas = (1, 2, 3, 4, 6) if span <= 1.5 else (1, 3) if span <= 3 else (1,)
    ticks: List[float] = []
    k = floor(lo)
    while k <= ceil(hi):
        for m in mantissas:
            v = m * 10.0**k
            if lo <= log10(v) <= hi:
                ticks.append(v)
        k += 1
    if not ticks:
        ticks = [10.0**lo, 10.0**hi]
    return [log10(v) for v in ticks], [f"{v:.3g}" for v in ticks]


def _log_colorbar(values: List[float], colorscale: str, label: str) -> Tuple[ColorbarSpec, Callable[[float], float]]:
    """A log10 colorbar over `values`, plus the transform to apply to each raw value.

    The colour machinery stays linear; only the numbers handed to it are logarithms, which
    keeps `resolve_fill` and the colorbar code free of any scale special-casing.
    """
    positive = [v for v in values if v > 0]
    lo, hi = (log10(min(positive)), log10(max(positive))) if positive else (-3.0, -2.0)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    tickvals, ticktext = _log_ticks(lo, hi)
    spec = ColorbarSpec(
        colorscale=colorscale,
        cmin=lo,
        cmax=hi,
        label=label,
        tickvals=tickvals,
        ticktext=ticktext,
    )
    # Values at or below zero pin to the bottom of the scale rather than blowing up.
    return spec, (lambda v: log10(v) if v > 0 else lo)


# Two coherent looks for the dual-scale view. A preset flips marker, palette, titles,
# widths and scale *together*, so neither look can be half-applied into something that
# belongs to neither.
_DEVICE_PRESETS: Dict[str, Dict[str, Any]] = {
    # The package's own language - identical to `noise_heatmap_style` in every respect a
    # single-layer heatmap has an opinion about. The coupler scale is the one deviation,
    # and it is irreducible: two bars must be told apart.
    "qsnow": dict(
        marker="s",
        size=0.5,
        edgecolor="black",
        linewidths=0.75,
        qubit_colorscale="hot_r",
        coupler_colorscale=_floored_colorscale("Blues"),
        coupler_width=7.0,
        title_side="right",
        qubit_label="Qubit (p)",
        coupler_label="Coupler (p)",
        logical_edgecolor="blue",
        log=False,
    ),
    # The device site-map look: circular qubits, perceptually uniform ramps, log rates.
    "device": dict(
        marker="o",
        size=0.7,
        edgecolor="#e5ecf6",
        linewidths=1.5,
        qubit_colorscale="Viridis_r",
        coupler_colorscale=_floored_colorscale("Magma_r", floor=0.15),
        coupler_width=7.0,
        title_side="top",
        qubit_label="p<sub>qubit</sub>",
        coupler_label="p<sub>coupler</sub>",
        logical_edgecolor="red",
        log=True,
    ),
}

# TODO - allow for custom device stylings (i.e., manually override each individual attr above via a helper function that can be passed into preset such that preset is ultimately a dict)
def device_heatmap_style(
    chip: Chip,
    *,
    preset: str = "qsnow",
    log: Optional[bool] = None,
    labels: bool = False,
    qubit_colorscale: Optional[Union[str, List[List[Any]]]] = None,
    coupler_colorscale: Optional[Union[str, List[List[Any]]]] = None,
    limits: Optional[Tuple[float, float]] = None,
    coupler_limits: Optional[Tuple[float, float]] = None,
    desc: Optional[str] = None,
) -> VisualizationStyle:
    """Qubit and coupler error rates together, each on its own independently scaled colorbar.

    The view for a chip whose couplers carry measured rates. Unlike `noise_heatmap_style`,
    the two layers do *not* share a scale: two-qubit rates typically run several times
    higher than single-qubit ones, and one shared range would compress all the qubit
    variation into the bottom of the bar. Two scales cost a second (invisible) trace,
    since plotly draws at most one colorbar per trace.

    Two looks are available through `preset`. The default `"qsnow"` follows the package's
    own conventions - square markers, black edges, the `hot_r` ramp, a blue logical outline
    - so it sits beside `noise_heatmap_style` without looking like a different tool; the
    coupler's own `Blues` ramp is the single, unavoidable deviation. `"device"` is the
    denser device site-map look: circular qubits, perceptually uniform ramps, log rates.

    `log` follows the preset unless given explicitly. `labels=True` writes each qubit's
    index into its marker - readable on a few hundred qubits, not on a few thousand, which
    is why it is off by default.
    """
    try:
        look = _DEVICE_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"Unknown device heatmap preset '{preset}'. "
            f"Known presets: {sorted(_DEVICE_PRESETS)}."
        ) from None

    log = look["log"] if log is None else log
    qubit_colorscale = qubit_colorscale or look["qubit_colorscale"]
    coupler_colorscale = coupler_colorscale or look["coupler_colorscale"]
    qubit_label, coupler_label = look["qubit_label"], look["coupler_label"]

    qubit_p = [q.noise.p for q in chip.qubits]
    coupler_p = [c.noise.p for c in chip.couplers] or [0.0]

    if log:
        qubit_bar, qubit_tx = _log_colorbar(qubit_p, qubit_colorscale, qubit_label)
        coupler_bar, coupler_tx = _log_colorbar(
            coupler_p, coupler_colorscale, coupler_label
        )
        if limits:
            qubit_bar.cmin, qubit_bar.cmax = log10(limits[0]), log10(limits[1])
        if coupler_limits:
            coupler_bar.cmin, coupler_bar.cmax = (
                log10(coupler_limits[0]),
                log10(coupler_limits[1]),
            )
    else:
        qubit_tx = coupler_tx = lambda v: v
        qubit_bar = ColorbarSpec(
            qubit_colorscale, *(limits or (min(qubit_p), max(qubit_p))), qubit_label
        )
        coupler_bar = ColorbarSpec(
            coupler_colorscale,
            *(coupler_limits or (min(coupler_p), max(coupler_p))),
            coupler_label,
        )

    qubit_bar.title_side = coupler_bar.title_side = look["title_side"]

    def style_fn(qubit: Qubit) -> QubitStyle:
        return QubitStyle(
            color=qubit_tx(qubit.noise.p),
            marker=look["marker"],
            size=look["size"],
            edgecolor=look["edgecolor"],
            linewidths=look["linewidths"],
            custom_hovertext=_hovertext_format(
                f"({qubit.loc[0]}, {qubit.loc[1]})",
                status=qubit.status.name,
                # 4 significant figures, not 4 decimals: this style exists to show the
                # sub-1e-4 rates that `:.4f` would flatten to '0.0000'.
                noise=f"{qubit.noise.p:.4g}",
            ),
        )

    def coupler_style_fn(coupler: Coupler) -> CouplerStyle:
        return CouplerStyle(
            color=coupler_tx(coupler.noise.p),
            width=look["coupler_width"],
            custom_hovertext=_coupler_hovertext(coupler),
        )

    def logical_style_fn(tag: Tag) -> LogicalStyle:
        return _default_logical_style(tag, edgecolor=look["logical_edgecolor"])

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=qubit_bar,
        logical_style=logical_style_fn,
        desc=desc,
        coupler_style=coupler_style_fn,
        coupler_colorbar=coupler_bar,
        qubit_labels=labels,
    )


def coupler_heatmap_style(
    chip: Chip,
    colorscale: str = "hot_r",
    limits: Optional[Tuple[float, float]] = None,
    desc: Optional[str] = None,
) -> VisualizationStyle:
    """Color each *coupler* by its rate, with the qubits greyed out behind them.

    The mirror of `noise_heatmap_style` for the edge layer: use it when the couplers are
    the subject rather than context.
    """
    if limits:
        cmin, cmax = limits
    else:
        p_values = [c.noise.p for c in chip.couplers] or [0.0]
        cmin, cmax = min(p_values), max(p_values)

    def style_fn(qubit: Qubit) -> QubitStyle:
        return QubitStyle(
            color="lightgray",
            custom_hovertext=_hovertext_format(
                f"({qubit.loc[0]}, {qubit.loc[1]})",
                status=qubit.status.name,
                noise=f"{qubit.noise.p:.4f}",
            ),
        )

    def coupler_style_fn(coupler: Coupler) -> CouplerStyle:
        return CouplerStyle(
            color=coupler.noise.p,
            width=3.0,
            custom_hovertext=_coupler_hovertext(coupler),
        )

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=ColorbarSpec(
            colorscale=colorscale, cmin=cmin, cmax=cmax, label="Coupler noise (p)"
        ),
        logical_style=_default_logical_style,
        desc=desc,
        coupler_style=coupler_style_fn,
    )


def packing_profile_style(
    chip: Chip, profiles: Dict[Any, Dict], desc: Optional[str] = None
):
    def style_fn(qubit: Qubit) -> QubitStyle:
        profile = profiles.get(qubit.loc, {})
        valid = True if profile else False
        return QubitStyle(
            color="pink" if valid else "lightgray",
            custom_hovertext=_hovertext_format(
                base=f"{(qubit.loc[0], qubit.loc[1])} -> {profile.get('bound')}" if valid else f"{(qubit.loc[0], qubit.loc[1])}",
                valid=valid,
                ler=profile.get("ler", None),
            ),
        )

    return VisualizationStyle(style_fn=style_fn, desc=desc)


def custom_heatmap_style(
    chip: Chip,
    float_map: Dict[Coord, float],
    float_label: str,
    *,
    colorscale: str = "hot_r",
    limits: Optional[Tuple[float, float]] = None,
    colorbar_label: Optional[str] = None,
    additional_hovertext: Optional[Dict[str, Dict[Coord, Any]]] = None,
    desc: Optional[str] = None
) -> VisualizationStyle:
    if limits:
        cmin, cmax = limits
    elif float_map:
        cmin, cmax = min(float_map.values()), max(float_map.values())
    else:
        raise ValueError(
            "custom_heatmap_style requires a non-empty `float_map` or explicit `limits`."
        )

    def qubit_style_fn(qubit: Qubit) -> QubitStyle:
        # raw value; the colorscale mapping is applied trace-wide by `visualize()`
        if qubit.loc:
            heatmap_val = float_map.get(qubit.loc)
            hovertext_dict = {float_label: f"{heatmap_val:.4f}" if heatmap_val is not None else None}

            if additional_hovertext: 
                for attr_title, attr_map in additional_hovertext.items():
                    attr_val = attr_map.get(qubit.loc, None)
                    if attr_val:
                        attr_val = f"{attr_val:.4f}" if isinstance(attr_val, float) else attr_val
                    hovertext_dict |= {attr_title: attr_val}

            base = hovertext_dict.pop('base', None) or f"({qubit.loc[0]}, {qubit.loc[1]})"

            return QubitStyle(
                color=heatmap_val if heatmap_val is not None else "lightgray",
                custom_hovertext=_hovertext_format(
                    base = base,
                    **hovertext_dict
                ),
            )
        else:
            raise AttributeError('Qubit must have location in order to be visualized.')

    return VisualizationStyle(
        style_fn=qubit_style_fn,
        colorbar=ColorbarSpec(colorscale=colorscale, cmin=cmin, cmax=cmax, label=colorbar_label or float_label),
        logical_style=None,
        desc=desc,
    )


def area_selection_style(
    chip: Chip,
    selection: Dict[Any, Qubit],
    show_logicals=False,
    desc: Optional[str] = None,
):
    def style_fn(qubit: Qubit) -> QubitStyle:
        selected = selection.get(qubit.loc, None)
        return QubitStyle(
            color="red" if selected else "lightgray",
            custom_hovertext=_hovertext_format(
                f"({qubit.loc[0]}, {qubit.loc[1]})",
                status=qubit.status.name,
                type=qubit.type.name,
            ),
        )

    return VisualizationStyle(
        style_fn=style_fn,
        logical_style=_default_logical_style if show_logicals else None,
        desc=desc,
    )


def _discrete_colormap_fn(
    n_range: Tuple[int, int], palette: Optional[List[str]] = None
) -> Callable[[int], str]:
    """Map an integer index within `n_range` to a discrete color sampled from `palette`."""
    resolved_palette: List[str]
    if palette is None:
        from plotly.colors import qualitative

        resolved_palette = qualitative.Set1
    else:
        resolved_palette = palette

    lo, hi = n_range
    span = max(hi - lo, 1)

    def color_fn(i: int) -> str:
        idx = int((i - lo) / span * (len(resolved_palette) - 1)) % len(resolved_palette)
        return resolved_palette[idx]

    return color_fn


# ------------------------------------------------------------------
# Figure assembly
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _LayoutGeometry:
    """Pixel/fraction geometry of a figure, shared by every view rendered into it."""

    margin_l: int
    margin_r: int
    margin_t: int
    margin_b: int
    base_width: int
    fig_height: int
    colorbar_px: int
    domain_frac: float
    colorbar_step: float
    colorbar_y: float
    colorbar_len: float


def _axis_ranges(chip: Chip) -> Tuple[List[float], List[float]]:
    """
    The x and y ranges the frame installs - the single source of the figure's data aspect.

    The y range runs high-to-low so row 0 renders at the top, matching how a chip's
    coordinates are read.
    """
    return (
        [_RANGE_LO, chip.length + _RANGE_HI],
        [chip.height + _RANGE_HI, _RANGE_LO],
    )


def _axis_spans(chip: Chip) -> Tuple[float, float]:
    """Coordinate width and height of `_axis_ranges`, i.e. the aspect the plot must match."""
    (x0, x1), (y1, y0) = _axis_ranges(chip)
    return (x1 - x0, y1 - y0)


def _compute_geometry(
    chip: Chip, n_colorbars: int = 0, *, extra_top_margin: int = 0
) -> _LayoutGeometry:
    ## Misc. Colorbar Spacing Configuration ##

    # NOTE - Reserve a fixed pixel strip for the pesky lil colorbar explicitly and pin the plot domain so it never needs to encroach.
    # TODO - this still is not fit flush to the side of the plot like it ideally should
    base_margin = 40
    margin_l, margin_r, margin_b = base_margin, base_margin, base_margin
    margin_t = base_margin + 20 + extra_top_margin

    # One px-per-coordinate scale for both axes, sized off the longer side, so the plot
    # area's aspect matches the data's and plotly's scaleanchor has nothing to reconcile.
    # Clamping each axis independently (as this used to) collapsed the frame to a square
    # whenever both sides saturated the same bound - which on a checkerboard chip, whose
    # coordinate span is `pitch` times its unit extent, was essentially always.
    span_x, span_y = _axis_spans(chip)
    longest = max(span_x, span_y)
    fig_px = min(_MAX_FIG_PX, max(_MIN_FIG_PX, longest * _PX_PER_COORD))
    scale = (fig_px - 2 * base_margin) / longest
    plot_px_width = round(span_x * scale)
    plot_px_height = round(span_y * scale)
    base_width = plot_px_width + margin_l + margin_r
    fig_height = plot_px_height + margin_t + margin_b
    colorbar_px = _COLORBAR_PX * n_colorbars + _COLORBAR_TITLE_PX * max(0, n_colorbars - 1)
    domain_frac = plot_px_width / (plot_px_width + colorbar_px) if colorbar_px else 1.0

    # NOTE - Colorbar `y`/`len` are in *paper* fraction (the whole figure, margins included), not the
    # cartesian plot's own domain - so without this, the bar overshoots top/bottom by the margins.
    colorbar_bottom_frac = margin_b / fig_height
    colorbar_top_frac = 1 - margin_t / fig_height
    return _LayoutGeometry(
        margin_l=margin_l,
        margin_r=margin_r,
        margin_t=margin_t,
        margin_b=margin_b,
        base_width=base_width,
        fig_height=fig_height,
        colorbar_px=colorbar_px,
        domain_frac=domain_frac,
        # paper-fraction width of one colorbar strip, so bar `i` sits clear of bar `i-1`
        colorbar_step=(
            (_COLORBAR_PX + _COLORBAR_TITLE_PX) / (plot_px_width + colorbar_px)
            if colorbar_px
            else 0.0
        ),
        colorbar_y=(colorbar_bottom_frac + colorbar_top_frac) / 2,
        colorbar_len=colorbar_top_frac - colorbar_bottom_frac,
    )


@dataclass(frozen=True)
class _StyleLayer:
    """Everything one style contributes to a figure.

    Shapes and annotations are plain dicts (not plotly graph objects) so the same
    lists can seed `layout.shapes`/`layout.annotations` or ride inside an
    `updatemenus` button payload for interactive style switching.
    """

    shapes: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    # One entry per colorbar-bearing group: the qubit hover trace, plus a coupler hover
    # trace when the style gives couplers their own scale. Callers must add all of them
    # and keep them visible together.
    traces: List[Dict[str, Any]]


def _build_style_layer(
    chip: Chip,
    style: VisualizationStyle,
    geometry: _LayoutGeometry,
    *,
    logical_color_gradient: bool = False,
    name: str = "qubits",
) -> _StyleLayer:
    from plotly.colors import sample_colorscale

    def resolve_fill(color: ColorLike, bar: Optional[ColorbarSpec] = None) -> ColorLike:
        """Map a style's raw value onto a concrete color via the style's colorbar.

        Done in python rather than by plotly because the visible marks are layout shapes,
        which take a literal `fillcolor`; the colorbar legend is drawn separately off the
        invisible hover trace.
        """
        bar = bar if bar is not None else style.colorbar
        if bar is None or isinstance(color, str):
            return color
        span = bar.cmax - bar.cmin
        norm = (float(color) - bar.cmin) / span if span else 0.5
        return sample_colorscale(bar.colorscale, [max(0.0, min(1.0, norm))])[0]

    ## Coupler Style Configuration ##
    # Built first so the edge lines sit at the head of the shape list and therefore render
    # *beneath* the qubit marks - plotly draws shapes in list order. Skipped entirely when
    # the style declares no coupler layer, which is every default style.
    coupler_shapes: List[Dict[str, Any]] = []
    xs, ys, raw_colors, fillcolors, hovertext = [], [], [], [], []
    # A coupler layer with its own scale needs its own trace to carry its own colorbar,
    # so its hover points are kept apart; otherwise they ride the qubit trace.
    c_xs, c_ys, c_raw, c_fills, c_hover = [], [], [], [], []
    coupler_bar = style.coupler_colorbar
    if style.coupler_style is not None:
        for coupler in chip.couplers:
            cs = style.coupler_style(coupler)
            (x0, y0), (x1, y1) = coupler.ends
            linecolor = resolve_fill(cs.color, coupler_bar)

            coupler_shapes.append(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color=linecolor, width=cs.width),
                    opacity=cs.alpha,
                    layer="above",
                )
            )

            # Shapes cannot carry hover, so couplers ride the same invisible marker trace
            # the qubits already use - anchored at the edge midpoint. Keeping them in that
            # one trace is what preserves the one-trace-per-style contract that
            # `visualize_interactive`'s visibility array indexes on.
            mx, my = coupler.midpoint
            if coupler_bar is not None:
                c_xs.append(mx)
                c_ys.append(my)
                c_raw.append(cs.color)
                c_fills.append(linecolor)
                c_hover.append(cs.custom_hovertext)
            else:
                xs.append(mx)
                ys.append(my)
                raw_colors.append(cs.color if style.colorbar is not None else 0)
                fillcolors.append(linecolor)
                hovertext.append(cs.custom_hovertext)

    ## Qubit Style Configuration ##
    shapes: List[Dict[str, Any]] = coupler_shapes
    qubit_fill: Dict[Coord, ColorLike] = {}
    for qubit in chip.qubits:
        if not qubit.loc:
            raise AttributeError(
                "Qubit with uninitialized location cannot be visualized."
            )
        x, y = qubit.loc
        s = style.style_fn(qubit)

        fillcolor = resolve_fill(s.color)
        qubit_fill[(x, y)] = fillcolor

        r = s.size / 2
        shapes.append(
            dict(
                type=_SHAPE_TYPES.get(s.marker, "rect"),
                x0=x - r,
                y0=y - r,
                x1=x + r,
                y1=y + r,
                line=dict(color=s.edgecolor, width=s.linewidths),
                fillcolor=fillcolor,
                opacity=s.alpha,
                layer="above",
            )
        )

        xs.append(x)
        ys.append(y)
        raw_colors.append(s.color if style.colorbar is not None else 0)
        fillcolors.append(fillcolor)
        hovertext.append(s.custom_hovertext)

    def hover_marker_for(bar: Optional[ColorbarSpec], values, slot: int) -> Dict[str, object]:
        """An invisible marker spec that renders colorbar `slot` for `bar`, if any."""
        marker: Dict[str, object] = dict(size=20, opacity=0)
        if bar is None:
            return marker
        colorbar = dict(
            title=dict(text=bar.label, side=bar.title_side),
            x=geometry.domain_frac + slot * geometry.colorbar_step,
            xanchor="left",
            y=geometry.colorbar_y,
            yanchor="middle",
            len=geometry.colorbar_len,
        )
        if bar.tickvals is not None:
            colorbar.update(tickmode="array", tickvals=bar.tickvals, ticktext=bar.ticktext)
        marker.update(
            color=values,
            colorscale=bar.colorscale,
            cmin=bar.cmin,
            cmax=bar.cmax,
            colorbar=colorbar,
            showscale=True,
        )
        return marker

    hover_marker = hover_marker_for(style.colorbar, raw_colors, 0)

    ## Qubit Hover Box Configuration ##
    # In the default (non-colorbar) style, tint each hover box to match its qubit's fill color.
    hoverlabel = (
        None
        if style.colorbar is not None
        else dict(bgcolor=fillcolors, font=dict(color="black"))
    )

    traces: List[Dict[str, Any]] = [
        dict(
            x=xs,
            y=ys,
            mode="markers",
            marker=hover_marker,
            hovertext=hovertext,
            hoverinfo="text",
            name=name,
            hoverlabel=hoverlabel,
        )
    ]
    if coupler_bar is not None:
        # The coupler colorbar takes the slot after the qubit one, when that exists.
        traces.append(
            dict(
                x=c_xs,
                y=c_ys,
                mode="markers",
                marker=hover_marker_for(
                    coupler_bar, c_raw, 1 if style.colorbar is not None else 0
                ),
                hovertext=c_hover,
                hoverinfo="text",
                name=f"{name} couplers",
                hoverlabel=None,
            )
        )

    ## Qubit Label Configuration ##
    annotations: List[Dict[str, Any]] = []
    if style.qubit_labels:
        # Row-major reading order (left to right, top to bottom), so the index matches how
        # a device map is normally numbered rather than the grid's storage order.
        ordered = sorted(chip.grid, key=lambda c: (c[1], c[0]))
        for idx, coord in enumerate(ordered):
            annotations.append(
                dict(
                    x=coord[0],
                    y=coord[1],
                    text=str(idx),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(
                        color=_label_color(qubit_fill.get(coord, "")),
                        size=12,
                        family="Andale Mono, monospace, bold",
                    ),
                )
            )

    ## Logical Tiling Configuration ##
    if chip.tiles and style.logical_style is not None:
        edgecolor_fn = _discrete_colormap_fn((3, 17))
        for i, tile in enumerate(chip.tiles, start=0):
            ls = style.logical_style(tile.tag)
            if logical_color_gradient:
                ls.edgecolor = edgecolor_fn(i)
            x0, y0 = tile.origin[0] - 0.5, tile.origin[1] - 0.5
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x0 + tile.length,
                    y1=y0 + tile.height,
                    line=dict(color=ls.edgecolor, width=ls.linewidth),
                    fillcolor=(
                        ls.facecolor if ls.facecolor != "none" else "rgba(0,0,0,0)"
                    ),
                    opacity=ls.alpha,
                    layer="above",
                )
            )
            annotations.append(
                dict(
                    x=tile.origin[0] + tile.length - 0.55,
                    y=tile.origin[1] - 0.43,
                    text=f"tile {i}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(
                        color=ls.edgecolor, size=10, family="Andale Mono, monospace"
                    ),
                    bgcolor="rgba(128,128,128,0.25)",
                )
            )

    return _StyleLayer(shapes=shapes, annotations=annotations, traces=traces)


def _apply_frame(fig: Figure, chip: Chip, geometry: _LayoutGeometry) -> None:
    ## Axis and Layout Configuration ##
    x_range, y_range = _axis_ranges(chip)
    fig.update_xaxes(
        side="top",
        dtick=1,
        showgrid=True,
        zeroline=True,
        range=x_range,
        domain=[0, geometry.domain_frac],
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        dtick=1,
        showgrid=True,
        zeroline=True,
        range=y_range,
        scaleanchor="x",
        scaleratio=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
    )
    fig.update_layout(
        width=geometry.base_width + geometry.colorbar_px,
        height=geometry.fig_height,
        showlegend=False,
        margin=dict(
            l=geometry.margin_l,
            r=geometry.margin_r,
            t=geometry.margin_t,
            b=geometry.margin_b,
        ),
    )


def visualize(
    chip: Chip,
    fig: Optional[Figure] = None,
    style: VisualizationStyle = default_style,
    logical_color_gradient=False,
    show: bool = False,
) -> Figure:
    # Imported from the concrete submodules (not the `plotly.graph_objects` facade) since that
    # facade lazily resolves attributes via `__getattr__`, which defeats static type narrowing/hover.
    from plotly.graph_objs._figure import Figure
    from plotly.graph_objs._scattergl import Scattergl

    if fig is None:
        fig = Figure()

    n_colorbars = (style.colorbar is not None) + (style.coupler_colorbar is not None)
    geometry = _compute_geometry(chip, n_colorbars)
    layer = _build_style_layer(
        chip, style, geometry, logical_color_gradient=logical_color_gradient
    )

    # Bulk-assign via `update_layout` rather than `add_shape`/`add_annotation` in a
    # loop: each of those calls re-validates the whole figure, so doing it once per
    # qubit is O(n) validation passes instead of one - on a few hundred qubits that's
    # the difference between sub-second and 10+ second renders.
    fig.update_layout(
        shapes=list(fig.layout.shapes) + layer.shapes,
        annotations=list(fig.layout.annotations) + layer.annotations,
    )
    for trace_kwargs in layer.traces:
        fig.add_trace(Scattergl(**trace_kwargs))

    _apply_frame(fig, chip, geometry)

    if show:
        fig.show()

    return fig
