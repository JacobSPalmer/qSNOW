from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from qsnow.interface.models import CSSType, Qubit, Status, Tag

if TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure
    from qsnow.interface.chip import Chip
    from qsnow.interface.tile import LogicalTile

# ------------------------------------------------------------------
# Visualization style classes/types
# ------------------------------------------------------------------

ColorLike = Union[str, float]

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
    colorscale: str
    cmin: float
    cmax: float
    label: str = ""


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


def noise_heatmap_style(
    chip: Chip, colorscale: str = "hot_r", limits: Optional[Tuple[float, float]] = None
) -> VisualizationStyle:
    """Color each qubit by its noise value `p`, normalized across the chip."""
    if limits:
        cmin, cmax = limits
    else:
        p_values = [qubit.noise.p for qubit in chip.qubits]
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

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=ColorbarSpec(
            colorscale=colorscale, cmin=cmin, cmax=cmax, label="Noise (p)"
        ),
        logical_style=logical_style_fn,
    )


def packing_profile_style(chip: Chip, profiles: Dict[Any, Dict]):
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

    return VisualizationStyle(style_fn=style_fn)


def custom_heatmap_style(
    chip: Chip,
    float_map: Dict[Tuple[float, float], float],
    float_label: str,
    *,
    colorscale: str = "hot_r",
    limits: Optional[Tuple[float, float]] = None,
    colorbar_label: Optional[str] = None,
    additional_hovertext: Optional[Dict[str, Dict[Tuple[float, float], object]]] = None,
    style_desc: Optional[str] = None
) -> VisualizationStyle:
    if limits:
        cmin, cmax = limits
    else:
        cmin, cmax = min(float_map.values()), max(float_map.values())

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
    )


def area_selection_style(chip: Chip, selection: Dict[Any, Qubit], show_logicals=False):
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
    colorbar_y: float
    colorbar_len: float


def _compute_geometry(
    chip: Chip, reserve_colorbar: bool, *, extra_top_margin: int = 0
) -> _LayoutGeometry:
    ## Misc. Colorbar Spacing Configuration ##

    # NOTE - Reserve a fixed pixel strip for the pesky lil colorbar explicitly and pin the plot domain so it never needs to encroach.
    # TODO - this still is not fit flush to the side of the plot like it ideally should
    base_margin = 40
    margin_l, margin_r, margin_b = base_margin, base_margin, base_margin
    margin_t = base_margin + 20 + extra_top_margin

    plot_px_width = min(900, max(600, chip.length * 60)) - 2 * base_margin
    plot_px_height = min(900, max(600, chip.height * 60)) - 2 * base_margin
    base_width = plot_px_width + margin_l + margin_r
    fig_height = plot_px_height + margin_t + margin_b
    colorbar_px = 100 if reserve_colorbar else 0
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
    trace_kwargs: Dict[str, Any]
    has_colorbar: bool


def _build_style_layer(
    chip: Chip,
    style: VisualizationStyle,
    geometry: _LayoutGeometry,
    *,
    logical_color_gradient: bool = False,
    name: str = "qubits",
) -> _StyleLayer:
    from plotly.colors import sample_colorscale

    ## Qubit Style Configuration ##
    shapes: List[Dict[str, Any]] = []
    xs, ys, raw_colors, fillcolors, hovertext = [], [], [], [], []
    for qubit in chip.qubits:
        if not qubit.loc:
            raise AttributeError(
                "Qubit with uninitialized location cannot be visualized."
            )
        x, y = qubit.loc
        s = style.style_fn(qubit)

        if style.colorbar is not None and not isinstance(s.color, str):
            span = style.colorbar.cmax - style.colorbar.cmin
            norm = (float(s.color) - style.colorbar.cmin) / span if span else 0.5
            fillcolor = sample_colorscale(
                style.colorbar.colorscale, [max(0.0, min(1.0, norm))]
            )[0]
        else:
            fillcolor = s.color

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

    hover_marker: Dict[str, object] = dict(size=20, opacity=0)
    if style.colorbar is not None:
        hover_marker.update(
            color=raw_colors,
            colorscale=style.colorbar.colorscale,
            cmin=style.colorbar.cmin,
            cmax=style.colorbar.cmax,
            # title.side='right' (vs the default 'top') keeps the title from eating into `len`,
            # so the gradient itself - not the title - spans the full computed plot-aligned length.
            colorbar=dict(
                title=dict(text=style.colorbar.label, side="right"),
                x=geometry.domain_frac,
                xanchor="left",
                y=geometry.colorbar_y,
                yanchor="middle",
                len=geometry.colorbar_len,
            ),
            showscale=True,
        )

    ## Qubit Hover Box Configuration ##
    # In the default (non-colorbar) style, tint each hover box to match its qubit's fill color.
    hoverlabel = (
        None
        if style.colorbar is not None
        else dict(bgcolor=fillcolors, font=dict(color="black"))
    )

    trace_kwargs: Dict[str, Any] = dict(
        x=xs,
        y=ys,
        mode="markers",
        marker=hover_marker,
        hovertext=hovertext,
        hoverinfo="text",
        name=name,
        hoverlabel=hoverlabel,
    )

    ## Logical Tiling Configuration ##
    annotations: List[Dict[str, Any]] = []
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

    return _StyleLayer(
        shapes=shapes,
        annotations=annotations,
        trace_kwargs=trace_kwargs,
        has_colorbar=style.colorbar is not None,
    )


def _apply_frame(fig: Figure, chip: Chip, geometry: _LayoutGeometry) -> None:
    ## Axis and Layout Configuration ##
    fig.update_xaxes(
        side="top",
        dtick=1,
        showgrid=True,
        zeroline=True,
        range=[-0.75, chip.length - 0.25],
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
        range=[chip.height - 0.25, -0.75],
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

    geometry = _compute_geometry(chip, reserve_colorbar=style.colorbar is not None)
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
    fig.add_trace(Scattergl(**layer.trace_kwargs))

    _apply_frame(fig, chip, geometry)

    if show:
        fig.show()

    return fig
