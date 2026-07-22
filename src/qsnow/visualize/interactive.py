"""Interactive figures: in-figure style switching and standalone HTML export."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from qsnow.visualize.visualize import (
    VisualizationStyle,
    _apply_frame,
    _build_style_layer,
    _compute_geometry,
    css_style,
    default_style,
    noise_heatmap_style,
)

if TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure

    from qsnow.experiments import ExperimentResults, SquarePackingExp
    from qsnow.interface.chip import Chip

__all__ = [
    "default_interactive_styles",
    "export_html",
    "visualize_interactive",
    "export_square_packing",
    "export_chip",
]

# Extra top-margin pixels reserved for the style dropdown above the top-side axis.
_DROPDOWN_MARGIN_PX = 50
# Further top-margin pixels reserved when the figure carries a title/subtitle.
_TITLE_MARGIN_PX = 45
# Further top-margin pixels reserved for the active style's description caption,
# only added when at least one bundled style actually has a `desc`.
_DESC_MARGIN_PX = 22


def _style_desc_annotation(desc: str) -> Dict[str, object]:
    """Caption for the active style, pinned just above the plot's top axis
    line/tick labels, inside the reserved `_DESC_MARGIN_PX` band.

    Annotation `yref="paper"` is relative to the *plotting area only* (unlike
    `layout.title.yref="container"`, which spans the whole figure) - it tops
    out at exactly 1.0 at the axis line, so a plain pixel `yshift` off that
    edge (matching the `pad` used for the dropdown/title below) is all that's
    needed; no margin/figure-height fraction math required.
    """
    return dict(
        x=0.0,
        xref="paper",
        xanchor="left",
        y=1.0,
        yref="paper",
        yanchor="bottom",
        # clears the top-side axis tick labels (~20px) and sits within the
        # reserved _DESC_MARGIN_PX band above them
        yshift=55 + _DESC_MARGIN_PX // 2,
        text=desc,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#6b7280"),
    )


def default_interactive_styles(chip: Chip) -> Dict[str, VisualizationStyle]:
    """The standard chip-level view bundle: status, CSS type, and noise heatmap."""
    return {
        "Status": replace(
            default_style,
            desc="Qubit assignment status (inactive / logical / ancilla).",
        ),
        "CSS Type": replace(
            css_style,
            desc="Qubit's Calderbank-Shor-Steane role (X-check / Z-check / data / tiling buffer)",
        ),
        "Noise": noise_heatmap_style(
            chip, desc="Heatmap of each qubit's physical error rate p."
        ),
    }


def _noise_model_text(noise_model: Optional[Dict]) -> Optional[str]:
    """One-line description of a chip's noise-model tag metadata."""
    if not noise_model:
        return None
    name = noise_model.get("name", "unknown")
    params = ", ".join(
        f"{k}={v}" for k, v in noise_model.items() if k not in ("name", "seed")
    )
    return f"{name} noise ({params})" if params else f"{name} noise"


def _chip_subtitle(chip: Chip) -> str:
    """One-line chip summary shown under the figure title."""
    summary = chip.summary()
    n_tiles = summary["n_tiles"]
    parts = [
        f"{summary['n_qubits']} qubits",
        f"{n_tiles} tile" + ("" if n_tiles == 1 else "s"),
    ]
    noise = _noise_model_text(summary["noise_model"])
    if noise:
        parts.append(noise)
    return " · ".join(parts)


def visualize_interactive(
    chip: Chip,
    styles: Optional[Mapping[str, VisualizationStyle]] = None,
    *,
    active: Optional[str] = None,
    logical_color_gradient: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """
    Build one figure with a dropdown that switches between named styles.

    `styles` maps dropdown labels to `VisualizationStyle`s (insertion order =
    dropdown order); `None` uses `default_interactive_styles(chip)`. `active`
    selects the initially shown style by name (default: the first). `title`
    defaults to `chip.tag.name` or "Chip LxH"; `subtitle` defaults to a
    one-line chip summary (qubits/tiles/noise model) shown beneath it. Pass
    `""` for either to suppress it.

    Switching is pure plotly (`updatemenus`), so the figure stays interactive
    in notebooks and through `fig.write_html()` — no Python callbacks needed.

    The figure owns `layout.shapes`/`layout.annotations`: each dropdown click
    replaces both wholesale, so anything added to them outside `styles` is
    wiped on the first switch.
    """
    from plotly.graph_objs._figure import Figure
    from plotly.graph_objs._scatter import Scatter

    if styles is None:
        styles = default_interactive_styles(chip)
    if not styles:
        raise ValueError("visualize_interactive requires at least one style.")

    names = list(styles)
    if active is None:
        active = names[0]
    if active not in styles:
        raise ValueError(f"Unknown active style {active!r}; available: {names}")
    active_idx = names.index(active)

    if title is None:
        title = chip.tag.name or f"Chip {chip.length // 2}x{chip.height // 2}"
    if subtitle is None:
        subtitle = _chip_subtitle(chip)

    # Only reserve room for the description caption row if some style actually
    # has one, so bundles without descriptions render exactly as before.
    has_desc = any(s.desc for s in styles.values())

    # Constant geometry across views (colorbar strip reserved if any style needs
    # it) so switching styles never resizes the plot.
    geometry = _compute_geometry(
        chip,
        reserve_colorbar=any(s.colorbar is not None for s in styles.values()),
        extra_top_margin=_DROPDOWN_MARGIN_PX
        + (_TITLE_MARGIN_PX if title else 0)
        + (_DESC_MARGIN_PX if has_desc else 0),
    )
    layers = {
        name: _build_style_layer(
            chip,
            style,
            geometry,
            logical_color_gradient=logical_color_gradient,
            name=name,
        )
        for name, style in styles.items()
    }
    # Per-style caption annotation (empty when that style has no `desc`), kept
    # separate from `layer.annotations` so it can ride along wherever those do.
    desc_annotations = {
        name: [_style_desc_annotation(style.desc)] if style.desc else []
        for name, style in styles.items()
    }

    fig = Figure()
    # SVG traces (not Scattergl): point counts are small and exported standalone
    # HTML shouldn't depend on WebGL availability. Only the active trace is
    # visible; plotly draws colorbars only for visible traces, so each view's
    # colorbar shows/hides automatically.
    for i, layer in enumerate(layers.values()):
        fig.add_trace(Scatter(**layer.trace_kwargs, visible=i == active_idx))

    active_layer = layers[active]
    fig.update_layout(
        shapes=active_layer.shapes,
        annotations=active_layer.annotations + desc_annotations[active],
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label=name,
                        method="update",
                        args=[
                            {"visible": [j == i for j in range(len(names))]},
                            {
                                "shapes": layer.shapes,
                                "annotations": layer.annotations
                                + desc_annotations[name],
                            },
                        ],
                    )
                    for i, (name, layer) in enumerate(layers.items())
                ],
                active=active_idx,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.0,
                yanchor="bottom",
                # lift the dropdown clear of the top-side axis tick labels
                pad=dict(b=32),
            )
        ],
    )

    _apply_frame(fig, chip, geometry)

    if title:
        # centered over the plot area (not the full paper, which includes the
        # reserved colorbar strip) and anchored to the figure's top edge so
        # title, dropdown, and axis stack predictably in the top margin
        title_conf: Dict[str, object] = dict(
            text=title,
            x=geometry.domain_frac / 2,
            xanchor="center",
            y=1.0,
            yanchor="top",
            yref="container",
            pad=dict(t=14),
            font=dict(size=18),
        )
        if subtitle:
            title_conf["subtitle"] = dict(
                text=subtitle, font=dict(size=12, color="#6b7280")
            )
        fig.update_layout(title=title_conf)

    if show:
        fig.show()

    return fig


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qSNOW - {title}</title>
<style>
  body {{
    font-family: system-ui, sans-serif;
    color: #1f2933;
    background: #fafafa;
    margin: 0;
    padding: 2rem 1.5rem;
    display: flex;
    justify-content: center;
  }}
  .page {{ width: min(100%, {page_width}px); }}
  /* indented by the figure's left margin so the header lines up with the plot frame */
  .header {{ margin: 20px 0 20px 0; max-width: 100%; }}
  h2 {{ font-size: 1rem; font-weight: 650; margin: 0; }}
  .desc {{ margin: 0.4rem 0 0; color: #616e7c; font-style: italic; }}
  /* row of stat boxes: boxes grow to fill leftover width and wrap onto new
     rows once they'd drop below their minimum width, so the row never trails
     off with a lone box hugging the left edge */
  .stats-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 1rem 0 0;
  }}
  .stat-box {{
    flex: 1 1 220px;
    padding: 0.7rem 1.2rem;
    border: 1px solid #e4e7eb;
    border-radius: 6px;
    background: #ffffff;
  }}
  .stat-box-title {{
    color: #000000;
    font-size: 0.70rem;
    font-style: bold;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 0.5rem;
  }}
  .stat-box dl {{
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 0.5rem 2rem;
    margin: 0;
  }}
  .stat-box dl > div {{ display: flex; flex-direction: column; }}
  .stat-box dt {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7b8794;
  }}
  .stat-box dd {{
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .figure {{ overflow-x: auto; }}
</style>
</head>
<body>
<div class="page">
<div class="header">
    <h2>qSNOW {title}</h2>
    {desc}
    <div class="stats-row">
        {stats}
    </div>
</div>
    <div class="figure" style="border: 1px solid #e4e7eb; border-radius: 8px;">{figure_div}</div>
</div>
</body>
</html>
"""


def _format_stat_box(title, stat_html):
    return (
        f'<div class="stat-box"><p class="stat-box-title">{escape(title)}</p>'
        f"<dl>{stat_html}</dl></div>"
    )


def _format_stat_rows(stats: Mapping[str, object]) -> str:
    """Render a stats mapping as `<dt>/<dd>` rows, formatting floats to 3 sig figs."""
    return "".join(
        f"<div><dt>{escape(name)}</dt><dd>"
        f"{escape(f'{value:.3g}' if isinstance(value, float) else str(value))}"
        f"</dd></div>"
        for name, value in stats.items()
    )


# TODO - revist the whole look of the exportable. fine for now and unimportant overall but it looks clunky and lame
# TODO - cleanup the noise and chip stats
def _noise_stats(summary: Dict[str, object]) -> Dict[str, str]:
    noise = summary.get("noise_model", {})
    stats = {}
    if isinstance(noise, dict):
        if noise.get("name"):
            stats["type"] = noise.pop("name")
        stats |= {k: v for k, v in noise.items() if k not in ("seed")}

        stats.pop("seed", None)
    return stats


def _chip_stats(summary: Dict[str, object]) -> Dict[str, str]:
    """Chip facts formatted for the export page's stats strip."""
    length, height = summary["unit_size"]
    grid_l, grid_h = summary["grid_size"]
    stats = {
        "Size": f"{length} × {height}",
        "Grid": f"{grid_l} × {grid_h}",
        "Qubits": str(summary["n_qubits"]),
        "Tiles": str(summary["n_tiles"]),
    }
    return stats


def _spp_stats(summary):
    stats = {"Placements": str(summary["placements"])}
    return stats


def _tile_stats(summary):
    length, height = summary["dims"]
    stats = {"Distance": str(summary["distance"]), "Size": f"{length} × {height}"}
    return stats


def export_html(
    obj,
    path: Optional[Union[str, Path]] = None,
    *,
    styles: Optional[Mapping[str, VisualizationStyle]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    include_plotlyjs: Union[bool, str] = True,
    **kwargs,
) -> Path:
    """
    Write a standalone interactive HTML page for qSNOW object and return its path.

    The layout and details are qSNOW object-specific and thus the object must be supported by this method, else this raises a `NotImplementedError`.
    """

    # deferred import: qsnow.helpers.serialize imports the experiment stack,
    # which imports this package (same pattern as Experiment.save)
    from qsnow.experiments import SquarePackingExp
    from qsnow.interface import Chip

    match obj:
        case SquarePackingExp():
            results = kwargs.pop("results", None)
            if results:
                return export_square_packing(
                    obj,
                    results,
                    path,
                    styles=styles,
                    title=title,
                    label=label,
                    include_plotlyjs=include_plotlyjs,
                    **kwargs,
                )
            else:
                raise AttributeError(
                    "The experiment result must be explicitly passed as keyword arguement `results`"
                )
        case Chip():
            return export_chip(
                obj,
                path,
                styles=styles,
                title=title,
                label=label,
                include_plotlyjs=include_plotlyjs,
                **kwargs,
            )
        case _:
            raise NotImplementedError(
                f"Exporting as stand-alone HTML not implemented for object type: {type(obj)}"
            )


def export_square_packing(
    exp: SquarePackingExp,
    results: ExperimentResults,
    path: Optional[Union[str, Path]] = None,
    *,
    styles: Optional[Mapping[str, VisualizationStyle]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    include_plotlyjs: Union[bool, str] = True,
    **kwargs,
) -> Path:
    """
    Write a standalone interactive HTML page for `SquarePackingExp` results and return its path.

    Either the experiment should have results stored in `Experiments.result` or `results` should be provided as a keyword argument.
    """
    # deferred import: qsnow.helpers.serialize imports the experiment stack,
    # which imports this package (same pattern as Experiment.save)
    from qsnow.helpers.serialize import _TIMESTAMP_FORMAT, get_data_dir

    chip_size_label = f"{exp.chip.length // 2}x{exp.chip.height // 2}"
    if title is None:
        title = exp.tag.name or "SP Experiment"
    if path is None:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        path = (
            get_data_dir()
            / "html"
            / f"experiment_sp_{label or chip_size_label}_{stamp}.html"
        )
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if styles is None:
        styles = exp._interactive_styles(results)

    # the page header carries the title and chip facts, so suppress both in-figure
    fig = visualize_interactive(exp.chip, styles, title="", subtitle="")

    figure_div = fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True, "displaylogo": False},  # TODO - logo
    )

    summary = exp.summary()
    chip_stats: Dict = _chip_stats(summary.get("chip"))
    chip_stats.pop("Tiles")
    chip_stats_html = _format_stat_box("Chip", _format_stat_rows(chip_stats))

    tile_stats: Dict = _tile_stats(summary.get("tile"))
    tile_stats_html = _format_stat_box("Tile", _format_stat_rows(tile_stats))

    exp_stats: Dict = _spp_stats(summary)
    exp_stats |= {"Shots": str(results.run_config["shots"])}
    exp_stats_html = _format_stat_box("Experiment", _format_stat_rows(exp_stats))

    if summary["chip"].get("noise_model", False):
        noise_model_html = _format_stat_box(
            "Noise Model", _format_stat_rows(_noise_stats(summary.get("chip")))
        )
    else:
        noise_model_html = ""

    desc_html = f'\n<p class="desc">{escape(exp.tag.desc)}</p>' if exp.tag.desc else ""
    path.write_text(
        _HTML_TEMPLATE.format(
            title=escape(title),
            desc=desc_html,
            stats=chip_stats_html + tile_stats_html + exp_stats_html + noise_model_html,
            page_width=int(fig.layout.width),
            figure_div=figure_div,
        ),
        encoding="utf-8",
    )
    return path


def export_chip(
    chip: Chip,
    path: Optional[Union[str, Path]] = None,
    *,
    styles: Optional[Mapping[str, VisualizationStyle]] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    include_plotlyjs: Union[bool, str] = True,
) -> Path:
    """
    Write a standalone interactive HTML page for `Chip` and return its path.
    """
    # deferred import: qsnow.helpers.serialize imports the experiment stack,
    # which imports this package (same pattern as Experiment.save)
    from qsnow.helpers.serialize import _TIMESTAMP_FORMAT, get_data_dir

    size_label = f"{chip.length // 2}x{chip.height // 2}"
    if title is None:
        title = chip.tag.name or "Chip"

    if path is None:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        path = get_data_dir() / "html" / f"chip_{label or size_label}_{stamp}.html"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # the page header carries the title and chip facts, so suppress both in-figure
    fig = visualize_interactive(chip, styles, title="", subtitle="")

    figure_div = fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True, "displaylogo": False},
    )
    summary = chip.summary()
    chip_stats_html = _format_stat_box("Chip", _format_stat_rows(_chip_stats(summary)))

    if summary.get("noise_model", False):
        noise_model_html = _format_stat_box(
            "Noise Model", _format_stat_rows(_noise_stats(summary))
        )
    else:
        noise_model_html = ""

    desc_html = (
        f'\n<p class="desc">{escape(chip.tag.desc)}</p>' if chip.tag.desc else ""
    )
    path.write_text(
        _HTML_TEMPLATE.format(
            title=escape(title),
            desc=desc_html,
            stats=chip_stats_html + noise_model_html,
            page_width=int(fig.layout.width),
            figure_div=figure_div,
        ),
        encoding="utf-8",
    )
    return path
