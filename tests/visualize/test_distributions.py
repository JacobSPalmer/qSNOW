import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import skew
from matplotlib.figure import Figure

from qsnow.interface.chip import Chip
from qsnow.visualize.distributions import per_histogram


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def chip() -> Chip:
    c = Chip(5, 5)
    c.generate_gaussian_noise(0.01, 0.002, seed=7)
    return c


def _histo(chip, **kwargs):
    return per_histogram(chip, show=False, **kwargs)


def _stats_boxes(ax):
    return list(ax.texts)  # free text only; titles and axis labels live elsewhere


def _site_rates(chip):
    return np.array([n.p for n in chip.noise_map.values()])


def _bin_edges(ax):
    # both histogram passes draw one polygon; its distinct x vertices are the bin edges
    return np.unique(ax.patches[0].get_xy()[:, 0])


class TestPanels:
    @pytest.mark.parametrize("which, n", [("sites", 1), ("couplers", 1), ("both", 2)])
    def test_axes_count_follows_which(self, chip, which, n):
        assert len(_histo(chip, which=which).axes) == n

    def test_both_orders_sites_then_couplers(self, chip):
        titles = [ax.get_title() for ax in _histo(chip).axes]
        assert titles[0].startswith("sites") and titles[1].startswith("couplers")

    def test_titles_report_the_count(self, chip):
        ax_sites, ax_couplers = _histo(chip).axes
        assert f"n={len(chip.qubits)}" in ax_sites.get_title()
        assert f"n={len(chip.couplers)}" in ax_couplers.get_title()

    def test_invalid_which_raises(self, chip):
        with pytest.raises(ValueError, match="which"):
            _histo(chip, which="edges")

    def test_panels_share_x(self, chip):
        ax_sites, ax_couplers = _histo(chip).axes
        assert ax_couplers in ax_sites.get_shared_x_axes().get_siblings(ax_sites)

    def test_shared_axis_spans_both_marginals(self, chip):
        # regression: setting (None, None) limits froze the shared axis at the site
        # range, so couplers on a higher scale fell outside it and drew nothing
        chip.set_coupler_noise_map({c.ends: 0.2 for c in chip.couplers})
        ax_sites, ax_couplers = _histo(chip).axes
        lo, hi = ax_couplers.get_xlim()
        assert lo < 0.01 and hi > 0.2

    def test_sharex_false_gives_each_panel_its_own_axis(self, chip):
        ax_sites, ax_couplers = _histo(chip, sharex=False).axes
        assert ax_couplers not in ax_sites.get_shared_x_axes().get_siblings(ax_sites)

    def test_logx_sets_a_log_scale_with_geometric_bins(self, chip):
        fig = _histo(chip, logx=True)
        ax = fig.axes[0]
        assert all(a.get_xscale() == "log" for a in fig.axes)
        edges = _bin_edges(ax)
        ratios = edges[1:] / edges[:-1]
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-6)  # equal ratios, not widths

    def test_log_ticks_label_the_decades_only(self, chip):
        # regression: a panel spanning under a decade got every minor tick labelled in
        # `2x10^-3` notation and the labels collided
        chip.generate_gaussian_noise(0.01, 0.0015, seed=7)  # spans well under a decade
        fig = _histo(chip, which="sites", logx=True)
        ax = fig.axes[0]
        fig.canvas.draw()
        majors = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert majors and all(m.startswith("$\\mathdefault{10^{") for m in majors)
        assert all(t.get_text() == "" for t in ax.get_xticklabels(minor=True))
        assert len(ax.get_xticks(minor=True)) > 0  # the marks stay, only labels go

    def test_linear_axis_by_default(self, chip):
        assert _histo(chip, which="sites").axes[0].get_xscale() == "linear"

    def test_limits_apply_to_every_panel(self, chip):
        fig = _histo(chip, limits=(0.0, 0.02))
        assert all(ax.get_xlim() == (0.0, 0.02) for ax in fig.axes)

    def test_dpi_is_applied(self, chip):
        assert _histo(chip, dpi=300).dpi == 300

    def test_suptitle_carries_headline_and_noise_model(self, chip):
        text = _histo(chip, add_title="extra").get_suptitle()
        assert text.startswith("PER by count across 5x5 chip")
        assert "PED(type=gaussian" in text
        assert text.endswith("extra")

    def test_title_false_draws_no_suptitle(self, chip):
        """Matches `ler_cdf(title=False)`: every figure suppresses captions alike."""
        assert _histo(chip, title=False).get_suptitle() == ""

    def test_figsize_overrides_the_computed_default(self, chip):
        assert tuple(_histo(chip, figsize=(9.0, 4.0)).get_size_inches()) == (9.0, 4.0)


class TestDerivedCouplerLabel:
    def test_derived_couplers_are_labelled_with_their_mode(self, chip):
        assert "derived: mean" in _histo(chip, which="couplers").axes[0].get_title()

    def test_independent_couplers_are_not_labelled_derived(self, chip):
        ends = chip.couplers[0].ends
        chip.set_coupler_noise_map({ends: 0.5})
        assert "derived" not in _histo(chip, which="couplers").axes[0].get_title()


class TestStyle:
    def test_fill_and_outline_share_the_series_colour(self, chip):
        ax = _histo(chip, which="sites").axes[0]
        fill, outline = ax.patches
        assert fill.get_facecolor()[:3] == outline.get_edgecolor()[:3]
        assert fill.get_alpha() == pytest.approx(0.35)
        assert outline.get_fill() is False


class TestStatsBox:
    def _box(self, chip, **kwargs):
        (box,) = _stats_boxes(_histo(chip, which="sites", **kwargs).axes[0])
        return box

    def test_default_on_a_linear_axis_is_the_raw_moments(self, chip):
        rates = _site_rates(chip)
        assert self._box(chip).get_text().split("  ") == [
            f"mean {rates.mean():.3g}",
            f"med {np.median(rates):.3g}",
            f"sd {rates.std(ddof=1):.3g}",
            f"skew {skew(rates, bias=False):.3g}",
        ]

    def test_default_on_a_log_axis_takes_spread_and_skew_in_log_space(self, chip):
        rates, lx = _site_rates(chip), np.log10(_site_rates(chip))
        assert self._box(chip, logx=True).get_text().split("  ") == [
            f"mean {rates.mean():.3g}",
            f"med {np.median(rates):.3g}",
            f"log sd {lx.std(ddof=1):.3g}",
            f"log skew {skew(lx, bias=False):.3g}",
        ]

    def test_raw_mode_keeps_the_raw_moments_on_a_log_axis(self, chip):
        assert self._box(chip, logx=True, stats="raw").get_text() == self._box(chip, stats="raw").get_text()

    @pytest.mark.parametrize("stats", ["raw", "log"])
    def test_uniform_landscape_prints_nan_skew_without_warning(self, chip, stats):
        chip.generate_uniform_noise(0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            text = self._box(chip, logx=True, stats=stats).get_text()
        assert text.endswith("nan") and "mean 0.01" in text

    def test_no_box_without_stats(self, chip):
        assert _stats_boxes(_histo(chip, which="sites", stats=None).axes[0]) == []

    def test_invalid_mode_raises(self, chip):
        with pytest.raises(ValueError, match="stats"):
            _histo(chip, stats="mode")


class TestShow:
    def test_shows_by_default(self, chip, monkeypatch):
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))

        assert per_histogram(chip) is None
        assert shown == [1]

    def test_show_false_returns_the_figure_without_showing(self, chip, monkeypatch):
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))

        assert isinstance(per_histogram(chip, show=False), Figure)
        assert shown == []
