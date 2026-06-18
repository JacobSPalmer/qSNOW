from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

from interface.models import Qubit, Status, TileTag

if TYPE_CHECKING:
    from interface.chip import Chip, Grid
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

ColorLike = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]

@dataclass
class QubitStyle:
    color: ColorLike = 'lightgray'
    marker: str = 's'
    size: float = 250
    alpha: float = 1.0
    edgecolor: str = 'black'
    linewidths: float = 0.5

@dataclass
class ColorbarSpec:
    cmap: "Colormap"
    norm: "Normalize"
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

_STATUS_ALPHA = {
    Status.INACTIVE: 0.5,
    Status.LOGICAL: 1.0,
    Status.ANCILLA: 0.75,
}

def _default_qubit_style(qubit: Qubit) -> QubitStyle:
    return QubitStyle(color=_STATUS_COLORS.get(qubit.status, 'lightgray'),
                       alpha=_STATUS_ALPHA.get(qubit.status, 0.5))

def _default_logical_style(tag: TileTag, **kwargs) -> LogicalStyle:
    return LogicalStyle(title = tag.name if tag.name else None, **kwargs)

default_style = VisualizationStyle(style_fn=_default_qubit_style, logical_style=_default_logical_style)

def noise_heatmap_style(chip: "Chip", 
                        cmap: str = "hot_r", 
                        limits: Optional[Tuple[float, float]] = None,
                        logical_style: Optional[LogicalStyle] = None) -> VisualizationStyle:
    """Color each qubit by its noise value `p`, normalized across the chip."""
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    if limits:
        norm = Normalize(vmin=limits[0], vmax=limits[1])
    else:
        p_values = [qubit.noise.p for qubit in chip.qubits]
        norm = Normalize(vmin=min(p_values), vmax=max(p_values))

    colormap = mpl.colormaps[cmap]

    def style_fn(qubit: Qubit) -> QubitStyle:
        return QubitStyle(color=colormap(norm(qubit.noise.p)))

    def logical_style_fn(tag: TileTag) -> LogicalStyle:
        return _default_logical_style(tag, edgecolor='blue')

    return VisualizationStyle(
        style_fn=style_fn,
        colorbar=ColorbarSpec(cmap=colormap, norm=norm, label="Noise (p)"),
        logical_style=logical_style_fn
    )

def _discrete_colormap_fn(n_range: Tuple[int, int], cmap: str = 'Set1') -> Callable[[int], ColorLike]:
    """Map an integer index within `n_range` to a discrete color sampled from `cmap`."""
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    colormap = mpl.colormaps[cmap]
    norm = Normalize(vmin=n_range[0], vmax=n_range[1])

    def color_fn(i: int) -> ColorLike:
        return colormap(norm(i))

    return color_fn

def visualize(
    chip: "Chip",
    ax: "Axes | None" = None,
    style: VisualizationStyle = default_style,
    logical_color_gradient = False,
    show: bool = False,
) -> "Axes":
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6,chip.length), max(6, .6*chip.height)))
    for qubit in chip.qubits:
        if not qubit.loc:
            raise AttributeError(f"Qubit with uninitialized location cannot be visualized.")
        x, y = qubit.loc
        s = style.style_fn(qubit)
        ax.scatter(x, y, c=[s.color], marker=s.marker, s=s.size, alpha=s.alpha,
                   edgecolors=s.edgecolor, linewidths=s.linewidths)

    if chip.tiles and style.logical_style is not None:
        from matplotlib.patches import Rectangle
        edgecolor_fn = _discrete_colormap_fn((3, 17), cmap='Set1')
        for i, tile in enumerate(chip.tiles, start=1):
            ls = style.logical_style(tile.tag)
            if logical_color_gradient:
                ls.edgecolor = edgecolor_fn(i)
            rect = Rectangle(xy=(tile.origin[0] - .5, tile.origin[1]-.5),
                             width=tile.length,
                             height=tile.height,
                             linewidth=ls.linewidth,
                             alpha=ls.alpha,
                             edgecolor=ls.edgecolor,
                             facecolor=ls.facecolor)
            ax.add_patch(rect)
            ax.text(x=tile.origin[0] + tile.length - .55,
                    y=tile.origin[1] - .43,
                    s=f"tile {i}",
                    ha='right',
                    va='top',
                    bbox=dict(
                        facecolor='gray',         # Opaque backing color
                        edgecolor='gray',          # Border for the text box
                        boxstyle='square,pad=0.05',  # Style and padding of the box
                        alpha=.25                # Fully opaque
                    ),
                    color=ls.edgecolor,
                    fontsize=10,
                    family=['Andale Mono'],
                    weight='bold')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    if style.colorbar is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sm = plt.cm.ScalarMappable(cmap=style.colorbar.cmap, norm=style.colorbar.norm)
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        plt.colorbar(sm, cax=cax, label=style.colorbar.label)

    if show:
        if plt.get_backend().lower() == 'agg':
            from IPython.display import display
            display(ax.get_figure())
        else:
            plt.show()

    return ax
