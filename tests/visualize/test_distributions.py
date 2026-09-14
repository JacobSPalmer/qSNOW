import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.legend import Legend

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


def _legends(ax):
    return [c for c in ax.get_children() if isinstance(c, Legend)]


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

    def test_couplers_only_has_no_site_panel(self, chip):
        assert "sites" not in _histo(chip, which="couplers").axes[0].get_title()

    def test_invalid_which_raises(self, chip):
        with pytest.raises(ValueError, match="which"):
            _histo(chip, which="edges")

    def test_panels_share_x(self, chip):
        ax_sites, ax_couplers = _histo(chip).axes
        assert ax_couplers in ax_sites.get_shared_x_axes().get_siblings(ax_sites)

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


class TestDerivedCouplerLabel:
    def test_derived_couplers_are_labelled_with_their_mode(self, chip):
        assert "derived: mean" in _histo(chip, which="couplers").axes[0].get_title()

    def test_independent_couplers_are_not_labelled_derived(self, chip):
        ends = chip.couplers[0].ends
        chip.set_coupler_noise_map({ends: 0.5})
        assert "derived" not in _histo(chip, which="couplers").axes[0].get_title()


class TestMarkers:
    @pytest.mark.parametrize(
        "markers, n", [(("mean", "median"), 2), (("mean",), 1), ((), 0)]
    )
    def test_one_line_per_marker(self, chip, markers, n):
        ax = _histo(chip, which="sites", markers=markers).axes[0]
        assert len(ax.lines) == n

    def test_legend_labels_carry_the_values(self, chip):
        ax = _histo(chip, which="sites").axes[0]
        labels = [t.get_text() for t in _legends(ax)[0].get_texts()]
        assert labels[0].startswith("mean=") and labels[1].startswith("median=")

    def test_mean_and_median_differ_by_linestyle_not_color(self, chip):
        mean_line, median_line = _histo(chip, which="sites").axes[0].lines
        assert mean_line.get_color() == median_line.get_color()
        assert mean_line.get_linestyle() != median_line.get_linestyle()

    def test_no_legend_without_markers(self, chip):
        assert _legends(_histo(chip, which="sites", markers=()).axes[0]) == []

    def test_invalid_marker_raises(self, chip):
        with pytest.raises(ValueError, match="markers"):
            _histo(chip, markers=("mode",))


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
