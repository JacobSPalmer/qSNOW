from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

import calib_histograms as ch

PANELS = ("(id, rx, sx, x) gate_error", "cz gate_error")
COLUMN_TITLES = ("Single-qubit gate error", "CZ gate error")
DATA_COLOR, CHIP_COLOR = "#4c72b0", "#dd8452"


def _hist(ax, values, edges, color, weights=None):
    ax.hist(values, bins=edges, weights=weights, histtype="stepfilled", alpha=0.35, color=color)
    ax.hist(values, bins=edges, weights=weights, histtype="step", color=color, linewidth=1.3)


def _stats_line(ax, text, color, fontsize):
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize,
            family="monospace", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


def _generator_stats_line(mean, median, log_sd, log_skew, stacked=False, kept=None):
    """The three generator statistics on one line, or one per line with `stacked`."""
    items = [("mean", f"{mean:.3e}"), ("med", f"{median:.3e}"), ("log sd", f"{log_sd:.3g}"), ("log skew", f"{log_skew:.3g}")]
    if kept is not None:
        items.append(("kept", f"{kept:.0%}"))
    if stacked:
        return "\n".join(f"{k} = {v}" for k, v in items)
    return "  ".join(f"{k} {v}" for k, v in items)


def load_panels(*, calibration_dir: Path, devices: Sequence[str] = ("ibm_phoenix",),
                panels: Sequence[str] = PANELS, snapshots: int | None = None,
                keep_dead: bool = True):
    fig0, series, specs = ch.plot_histograms(
        devices=list(devices), calibration_dir=calibration_dir, snapshots=snapshots,
        keep_dead=keep_dead, include=list(panels), verbose=False)
    plt.close(fig0)
    by_title = {p.title: p for p in specs}
    return series[0], [by_title[name] for name in panels]


def stats_table(*, calibration_dir: Path, devices: Sequence[str] = ("ibm_phoenix",),
                panels: Sequence[str] = PANELS, snapshots: int | None = None,
                keep_dead: bool = True, k: float = 1.5):
    data, specs = load_panels(calibration_dir=calibration_dir, devices=devices,
                              panels=panels, snapshots=snapshots, keep_dead=keep_dead)
    return ch.print_panel_summary(data, specs, k=k)


def calib_vs_chip(chip, *, calibration_dir: Path, devices: Sequence[str] = ("ibm_phoenix",),
                  panels: Sequence[str] = PANELS, snapshots: int | None = None,
                  keep_dead: bool = True, bins: int = 34, size: tuple[float, float] = (9, 6),
                  stats: bool = True, stats_fontsize: float = 7.5, stats_stacked: bool = False, stats_condense = False,
                  column_titles: tuple[str, str] = COLUMN_TITLES,
                  dpi: int | None = None, save: Path | str | None = None, **kwargs):
    data, specs = load_panels(calibration_dir=calibration_dir, devices=devices,
                              panels=panels, snapshots=snapshots, keep_dead=keep_dead)
    n_snap = len(data.frames)
    calib = []
    for spec in specs:
        v = data.values(spec)
        calib.append(v[v > 0])
    chip_rates = (np.array([q.noise.p for q in chip.qubits]),
                  np.array([c.noise.p for c in chip.couplers]))

    fig, axes = plt.subplots(2, 2, figsize=size, dpi=dpi, sharex="col", constrained_layout=True)
    for j, (title, cal, sim) in enumerate(zip(column_titles, calib, chip_rates)):
        lo, hi = min(cal.min(), sim.min()), max(cal.max(), sim.max())
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)

        _hist(axes[0, j], cal, edges, DATA_COLOR, weights=np.full(cal.size, 1.0 / n_snap))
        _hist(axes[1, j], sim, edges, CHIP_COLOR)
        if stats:
            st = ch.bulk_log_stats(cal)
            prefix = f"bulk ({st["kept"]:.0%} kept)" if stats_condense else f"bulk"
            prefix = f"{prefix}:\n" if stats_stacked else f"{prefix}: "
            _stats_line(axes[0, j], prefix + _generator_stats_line(st["mean"], st["median"], st["log sd"], st["log skew"],
                        stacked=stats_stacked, kept=None if stats_condense else st["kept"]), DATA_COLOR, stats_fontsize)
            lx = np.log10(sim)
            _stats_line(axes[1, j], _generator_stats_line(np.mean(sim), np.median(sim), lx.std(ddof=1), skew(lx, bias=False),
                        stacked=stats_stacked), CHIP_COLOR, stats_fontsize)

        axes[0, j].set_title(title)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("Physical Error Rate")

    length, height = chip.unit_dims
    axes[0, 0].set_ylabel(kwargs.get('ylabel_top', (f"{data.label.split(':')[0]}\ncount per snapshot ({n_snap} snapshots)")))
    axes[1, 0].set_ylabel(kwargs.get('ylabel_bottom', f"simulated {length}×{height} chip\ncount"))
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi)
    return fig
