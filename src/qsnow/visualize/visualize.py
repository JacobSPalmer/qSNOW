from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, Any

from qsnow.interface.models import Qubit, Status, TileTag, CSSType

if TYPE_CHECKING:
    from interface.chip import Chip, Grid
    from plotly.graph_objs._figure import Figure

# ------------------------------------------------------------------
# Visualization style classes/types
# ------------------------------------------------------------------

ColorLike = Union[str, float]

_SHAPE_TYPES = {
    's': 'rect',
    'o': 'circle',
}

@dataclass
class QubitStyle:
    color: ColorLike = 'lightgray'
    marker: str = 's'
    size: float = .5  # diameter in data (axis) units, so qubits scale naturally on zoom/pan
    alpha: float = 1.0
    edgecolor: str = 'black'
    linewidths: float = 0.75
    custom_hovertext: Optional[str] = None

@dataclass
class ColorbarSpec:
    colorscale: str
    cmin: float
    cmax: float
    label: str = ""

#TODO - add options in logical style for text or perhaps add a title style
@dataclass
class LogicalStyle:
    title: Optional[str] = None
    edgecolor: ColorLike = 'red'
    facecolor: ColorLike = 'none'
    alpha: float = 1.0
    linewidth: float = 2.0

@dataclass
class VisualizationStyle:
    style_fn: Callable[[Qubit], QubitStyle]
    colorbar: Optional[ColorbarSpec] = None
    logical_style: Optional[Callable[[TileTag], LogicalStyle]] = None

_STATUS_COLORS = {
    Status.INACTIVE: 'lightgray',
    Status.LOGICAL: 'steelblue',
    Status.ANCILLA: 'orange',
}

_CSS_COLORS = {
    CSSType.X_CHECK: 'firebrick',
    CSSType.Z_CHECK: 'dodgerblue',
    CSSType.DATA: 'dimgrey',
    CSSType.BUFFER: 'pink',
    CSSType.UNASSIGNED: 'lightgray'
}

_STATUS_ALPHA = {
    Status.INACTIVE: 0.5,
    Status.LOGICAL: 1.0,
    Status.ANCILLA: 0.75,
}

def _hovertext_format(base: str, **kwargs):
    return f"{base}<br>"+"<br>".join([f"{k}={v}" for k,v in kwargs.items() if v is not None])

def _default_qubit_style_by_css_type(qubit: Qubit) -> QubitStyle:
    return QubitStyle(color=_CSS_COLORS.get(qubit.type, 'lightgray'),
                       alpha=_STATUS_ALPHA.get(qubit.status, 0.5),
                       custom_hovertext=_hovertext_format(f"({qubit.loc[0]}, {qubit.loc[1]})", type=qubit.type.name, noise=f"{qubit.noise.p:.4f}"))

def _default_qubit_style_by_status(qubit: Qubit) -> QubitStyle:
    return QubitStyle(color=_STATUS_COLORS.get(qubit.status, 'lightgray'),
                       alpha=_STATUS_ALPHA.get(qubit.status, 0.5),
                       custom_hovertext=_hovertext_format(f"({qubit.loc[0]}, {qubit.loc[1]})", status=qubit.status.name, noise=f"{qubit.noise.p:.4f}"))

def _default_logical_style(tag: TileTag, **kwargs) -> LogicalStyle:
    return LogicalStyle(title = tag.name if tag.name else None, **kwargs)

# ------------------------------------------------------------------
# Importable visualization styles
# ------------------------------------------------------------------

default_style = VisualizationStyle(style_fn=_default_qubit_style_by_status, logical_style=_default_logical_style)
css_style = VisualizationStyle(style_fn=_default_qubit_style_by_css_type, logical_style=_default_logical_style)

def noise_heatmap_style(chip: "Chip",
                        colorscale: str = "hot_r",
                        limits: Optional[Tuple[float, float]] = None) -> VisualizationStyle:
    """Color each qubit by its noise value `p`, normalized across the chip."""
    if limits:
        cmin, cmax = limits
    else:
        p_values = [qubit.noise.p for qubit in chip.qubits]
        cmin, cmax = min(p_values), max(p_values)

    def style_fn(qubit: Qubit) -> QubitStyle:
        # raw value; the colorscale mapping is applied trace-wide by `visualize()`
        return QubitStyle(color=qubit.noise.p, 
                          custom_hovertext=_hovertext_format(f"({qubit.loc[0]}, {qubit.loc[1]})", status=qubit.status.name, noise=f"{qubit.noise.p:.4f}"))

    def logical_style_fn(tag: TileTag) -> LogicalStyle:
        return _default_logical_style(tag, edgecolor='blue')

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=ColorbarSpec(colorscale=colorscale, cmin=cmin, cmax=cmax, label="Noise (p)"),
        logical_style=logical_style_fn
    )

def packing_profile_style(chip: "Chip",
                          profiles: Dict[Any, Dict]):
    def style_fn(qubit: Qubit) -> QubitStyle:
        profile = profiles.get(qubit.loc, {})
        valid = True if profile else False
        return QubitStyle(color='pink' if valid else 'lightgray',
                          custom_hovertext=_hovertext_format(base=f"{(qubit.loc[0], qubit.loc[1])}", 
                                                             valid=valid,
                                                             bound=profile.get('bound', None),
                                                             ler=profile.get('ler', None)))
    return VisualizationStyle(
        style_fn=style_fn
    )

def area_selection_style(chip: "Chip",
                         selection: Dict[Any, Qubit],
                         show_logicals = False):
    def style_fn(qubit: Qubit) -> QubitStyle:
        selected = selection.get(qubit.loc, None)
        return QubitStyle(color='red' if selected else 'lightgray',
                          custom_hovertext=_hovertext_format(f"({qubit.loc[0]}, {qubit.loc[1]})", status=qubit.status.name, type=qubit.type.name))
    
    return VisualizationStyle(
        style_fn=style_fn,
        logical_style=_default_logical_style if show_logicals else None
    )

def _discrete_colormap_fn(n_range: Tuple[int, int], palette: Optional[List[str]] = None) -> Callable[[int], str]:
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

def visualize(
    chip: "Chip",
    fig: Optional["Figure"] = None,
    style: VisualizationStyle = default_style,
    logical_color_gradient = False,
    show: bool = False,
) -> "Figure":
    # Imported from the concrete submodules (not the `plotly.graph_objects` facade) since that
    # facade lazily resolves attributes via `__getattr__`, which defeats static type narrowing/hover.
    from plotly.graph_objs._figure import Figure
    from plotly.graph_objs._scattergl import Scattergl
    from plotly.colors import sample_colorscale

    if fig is None:
        fig = Figure()

    ## Qubit Style Configuration ##
    xs, ys, raw_colors, fillcolors, hovertext = [], [], [], [], []
    for qubit in chip.qubits:
        if not qubit.loc:
            raise AttributeError(f"Qubit with uninitialized location cannot be visualized.")
        x, y = qubit.loc
        s = style.style_fn(qubit)

        if style.colorbar is not None:
            span = style.colorbar.cmax - style.colorbar.cmin
            norm = (float(s.color) - style.colorbar.cmin) / span if span else 0.5
            fillcolor = sample_colorscale(style.colorbar.colorscale, [max(0.0, min(1.0, norm))])[0]
        else:
            fillcolor = s.color

        r = s.size / 2
        fig.add_shape(
            type=_SHAPE_TYPES.get(s.marker, 'rect'),
            x0=x - r, y0=y - r, x1=x + r, y1=y + r,
            line=dict(color=s.edgecolor, width=s.linewidths),
            fillcolor=fillcolor,
            opacity=s.alpha,
            layer='above',
        )

        xs.append(x)
        ys.append(y)
        raw_colors.append(s.color if style.colorbar is not None else 0)
        fillcolors.append(fillcolor)
        hovertext.append(s.custom_hovertext)

    ## Misc. Colorbar Spacing Configuration ##

    # NOTE - Reserve a fixed pixel strip for the pesky lil colorbar explicitly and pin the plot domain so it never needs to encroach.
    # TODO - this still is not fit flush to the side of the plot like it ideally should
    margin_l, margin_r, margin_t, margin_b = 40, 40, 60, 40
    base_width = min(900, max(600, chip.length * 60))
    fig_height = min(900, max(600, chip.height * 60))
    plot_px_width = base_width - margin_l - margin_r
    colorbar_px = 100 if style.colorbar is not None else 0
    domain_frac = plot_px_width / (plot_px_width + colorbar_px) if colorbar_px else 1.0

    # NOTE - Colorbar `y`/`len` are in *paper* fraction (the whole figure, margins included), not the
    # cartesian plot's own domain - so without this, the bar overshoots top/bottom by the margins.
    colorbar_bottom_frac = margin_b / fig_height
    colorbar_top_frac = 1 - margin_t / fig_height
    colorbar_len = colorbar_top_frac - colorbar_bottom_frac
    colorbar_y = (colorbar_bottom_frac + colorbar_top_frac) / 2

    hover_marker: Dict[str, object] = dict(size=20, opacity=0)
    if style.colorbar is not None:
        hover_marker.update(
            color=raw_colors,
            colorscale=style.colorbar.colorscale,
            cmin=style.colorbar.cmin,
            cmax=style.colorbar.cmax,
            # title.side='right' (vs the default 'top') keeps the title from eating into `len`,
            # so the gradient itself - not the title - spans the full computed plot-aligned length.
            colorbar=dict(title=dict(text=style.colorbar.label, side='right'),
                           x=domain_frac, xanchor='left',
                           y=colorbar_y, yanchor='middle', len=colorbar_len),
            showscale=True,
        )

    ## Qubit Hover Box Configuration ##
    # In the default (non-colorbar) style, tint each hover box to match its qubit's fill color.
    hoverlabel = None if style.colorbar is not None else dict(bgcolor=fillcolors, font=dict(color='black'))

    fig.add_trace(Scattergl(
        x=xs, y=ys, mode='markers', marker=hover_marker,
        hovertext=hovertext, hoverinfo='text', name='qubits',
        hoverlabel=hoverlabel,
    ))

    ## Logical Tiling Configuration ##
    if chip.tiles and style.logical_style is not None:
        edgecolor_fn = _discrete_colormap_fn((3, 17))
        for i, tile in enumerate(chip.tiles, start=0):
            ls = style.logical_style(tile.tag)
            if logical_color_gradient:
                ls.edgecolor = edgecolor_fn(i)
            x0, y0 = tile.origin[0] - .5, tile.origin[1] - .5
            fig.add_shape(
                type='rect',
                x0=x0, y0=y0, x1=x0 + tile.length, y1=y0 + tile.height,
                line=dict(color=ls.edgecolor, width=ls.linewidth),
                fillcolor=ls.facecolor if ls.facecolor != 'none' else 'rgba(0,0,0,0)',
                opacity=ls.alpha,
                layer='above',
            )
            fig.add_annotation(
                x=tile.origin[0] + tile.length - .55,
                y=tile.origin[1] - .43,
                text=f"tile {i}",
                showarrow=False,
                xanchor='right', yanchor='top',
                font=dict(color=ls.edgecolor, size=10, family='Andale Mono, monospace'),
                bgcolor='rgba(128,128,128,0.25)',
            )


    ## Axis and Layout Configuration ##
    fig.update_xaxes(side='top', dtick=1, showgrid=True, zeroline=True,
                      range=[-.75, chip.length - .25], domain=[0, domain_frac],
                      showline=True, linecolor='black', linewidth=1, mirror=True)
    fig.update_yaxes(dtick=1, showgrid=True, zeroline=True,
                      range=[chip.height - .25, -.75],
                      scaleanchor='x', scaleratio=1,
                      showline=True, linecolor='black', linewidth=1, mirror=True)
    fig.update_layout(
        width=base_width + colorbar_px,
        height=fig_height,
        showlegend=False,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
    )

    if show:
        fig.show()

    return fig
