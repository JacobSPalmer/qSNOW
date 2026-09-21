import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # never open a window from the test suite

from qsnow.interface.noise import NormalContour, Uniform
from qsnow.experiments.squarepacking.game import SquarePackingExp  # noqa: E402
from qsnow.helpers import serialize  # noqa: E402
from qsnow.interface.chip import Chip  # noqa: E402
from qsnow.interface.codes.rsc import SCTile  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402

from qsnow.visualize import profiling  # noqa: E402
from qsnow.visualize.profiling import (  # noqa: E402
    ProfileRun,
    _ecdf,
    ler_cdf,
    ler_histogram,
    ler_table,
    load_profile_runs,
)

DISTANCES = (3, 5)


def _save_sweep(distance: int, lers, *, shots: int = 1_000, ler_key: bool = True):
    """Persist a synthetic profiling sweep as a real experiment/results flake pair."""
    exp = SquarePackingExp(chip=Chip(10, 10), tile=SCTile(distance))
    exp.config["shots"] = shots
    exp.results = {}
    for loc, ler in zip(exp.profile, lers):
        errors = round(ler * shots)
        entry = {"shots": shots, "errors": errors}
        if ler_key:
            entry["ler"] = ler
        exp.results[loc] = entry

    exp.save(label=f"profiling_d{distance}")
    exp.save_results(label=f"profiling_d{distance}")
    return exp


@pytest.fixture
def sweeps(tmp_path):
    """Two saved sweeps (d3, d5) under an isolated data root."""
    serialize.set_data_dir(tmp_path / "data")
    saved = {
        3: _save_sweep(3, [0.001, 0.002, 0.003, 0.004]),
        5: _save_sweep(5, [0.0005, 0.0010, 0.0015]),
    }
    serialize.set_data_dir()  # leave the global root at its default, as callers find it
    yield tmp_path / "data", saved
    serialize.set_data_dir()


@pytest.fixture(autouse=True)
def _close_figures():
    """Figures are pyplot-managed now, so drop them between tests."""
    yield
    plt.close("all")


@pytest.fixture
def data_root(sweeps):
    root, _ = sweeps
    return root


@pytest.fixture
def named_roots(tmp_path):
    """Two roots whose chips record a noise-model name, for label derivation."""
    roots = {}
    for key, model in [
        ("primary", NormalContour(0.01, 0.003, seed=1)),
        ("baseline", Uniform(0.01)),
    ]:
        root = tmp_path / key
        serialize.set_data_dir(root)
        try:
            for d in DISTANCES:
                exp = SquarePackingExp(chip=Chip(10, 10), tile=SCTile(d))
                exp.chip.spec.noise_model = model
                exp.config["shots"] = 1_000
                exp.results = {
                    loc: {"shots": 1_000, "errors": 2, "ler": 0.002}
                    for loc in exp.profile[:3]
                }
                exp.save(label=f"profiling_d{d}")
                exp.save_results(label=f"profiling_d{d}")
        finally:
            serialize.set_data_dir()
        roots[key] = root
    return roots["primary"], roots["baseline"]


class TestEcdf:
    def test_one_point_per_sample_plus_the_run_in(self):
        x, y = _ecdf([0.3, 0.1, 0.2])

        assert x.size == y.size == 4

    def test_starts_with_a_flat_run_in_at_zero(self):
        x, y = _ecdf([0.3, 0.1, 0.2])

        assert x[0] == -np.inf
        assert y[0] == 0.0

    def test_sorted_and_monotonic_ending_at_one(self):
        x, y = _ecdf([0.3, 0.1, 0.2])

        assert list(x[1:]) == [0.1, 0.2, 0.3]
        assert np.all(np.diff(y) > 0)
        assert y[-1] == pytest.approx(1.0)

    def test_empty_input_returns_empty(self):
        x, y = _ecdf([])

        assert x.size == 0 and y.size == 0


class TestLoadProfileRuns:
    def test_one_run_per_distance(self, data_root):
        runs = load_profile_runs(DISTANCES, data_root)

        assert set(runs) == set(DISTANCES)
        assert all(isinstance(r, ProfileRun) for r in runs.values())

    def test_pairs_the_matching_experiment_and_results(self, data_root):
        runs = load_profile_runs(DISTANCES, data_root)

        for d, run in runs.items():
            assert run.experiment.tile.spec.distance == d
            assert run.lers.size == len(run.results.results)

    def test_restores_the_previous_data_dir(self, data_root):
        before = serialize.get_data_dir()

        load_profile_runs(DISTANCES, data_root)

        assert serialize.get_data_dir() == before

    def test_restores_the_data_dir_even_when_a_flake_is_missing(self, data_root):
        before = serialize.get_data_dir()

        with pytest.raises(FileNotFoundError):
            load_profile_runs([7], data_root)

        assert serialize.get_data_dir() == before

    def test_omitting_directory_uses_the_configured_root(self, data_root):
        serialize.set_data_dir(data_root)
        try:
            assert set(load_profile_runs(DISTANCES)) == set(DISTANCES)
        finally:
            serialize.set_data_dir()


class TestProfileRun:
    def test_lers_use_the_stored_ler_key(self, data_root):
        run = load_profile_runs([3], data_root)[3]

        assert sorted(run.lers) == pytest.approx([0.001, 0.002, 0.003, 0.004])

    def test_lers_fall_back_to_errors_over_shots(self, tmp_path):
        serialize.set_data_dir(tmp_path / "legacy")
        try:
            _save_sweep(3, [0.001, 0.002], shots=1_000, ler_key=False)
            run = load_profile_runs([3], tmp_path / "legacy")[3]

            assert sorted(run.lers) == pytest.approx([0.001, 0.002])
        finally:
            serialize.set_data_dir()

    def test_shots_come_from_the_run_config(self, data_root):
        assert load_profile_runs([3], data_root)[3].shots == 1_000

    def test_error_counts_are_raw_counts(self, data_root):
        run = load_profile_runs([3], data_root)[3]

        assert sorted(run.error_counts) == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_values_selects_scope(self, data_root):
        run = load_profile_runs([3], data_root)[3]

        assert np.array_equal(run.values("ler"), run.lers)
        assert np.array_equal(run.values("errors"), run.error_counts)

    def test_ler_intervals_bracket_the_point_estimates(self, data_root):
        run = load_profile_runs([3], data_root)[3]

        low, high = run.ler_intervals()

        assert low.shape == high.shape == run.lers.shape
        assert np.all(low < run.lers)
        assert np.all(run.lers < high)

    def test_ler_intervals_narrow_with_more_confidence_given_up(self, data_root):
        run = load_profile_runs([3], data_root)[3]

        low95, high95 = run.ler_intervals()
        low68, high68 = run.ler_intervals(confidence=0.68)

        assert np.all(low68 > low95)
        assert np.all(high68 < high95)

    def test_ler_intervals_of_a_zero_error_placement(self, tmp_path):
        serialize.set_data_dir(tmp_path / "zeros")
        try:
            _save_sweep(3, [0.0, 0.002], shots=1_000)
            run = load_profile_runs([3], tmp_path / "zeros")[3]
        finally:
            serialize.set_data_dir()

        low, high = run.ler_intervals()
        zero = np.flatnonzero(run.error_counts == 0)[0]

        assert low[zero] == 0.0
        # the rule of three: 0 events in N shots bounds the rate at ~3/N (95%)
        assert high[zero] == pytest.approx(3 / 1_000, rel=0.3)

    def test_stats_spread(self, data_root):
        s = load_profile_runs([3], data_root)[3].stats()

        assert s["n"] == 4
        assert s["range"] == pytest.approx(0.003)
        assert s["ratio"] == pytest.approx(4.0)
        assert s["zeros"] == 0

    def test_stats_ratio_is_inf_when_a_placement_saw_no_errors(self, tmp_path):
        serialize.set_data_dir(tmp_path / "zeros")
        try:
            _save_sweep(3, [0.0, 0.002])
            s = load_profile_runs([3], tmp_path / "zeros")[3].stats()
        finally:
            serialize.set_data_dir()

        assert s["ratio"] == float("inf")
        assert s["zeros"] >= 1

    def test_stats_on_a_single_placement_does_not_raise(self, tmp_path):
        serialize.set_data_dir(tmp_path / "single")
        try:
            exp = SquarePackingExp(chip=Chip(10, 10), tile=SCTile(3))
            exp.config["shots"] = 100
            exp.results = {exp.profile[0]: {"shots": 100, "errors": 5, "ler": 0.05}}
            exp.save(label="profiling_d3")
            exp.save_results(label="profiling_d3")

            s = load_profile_runs([3], tmp_path / "single")[3].stats()
        finally:
            serialize.set_data_dir()

        assert s["n"] == 1
        assert s["stdev"] == 0.0

    def test_show_opens_the_experiment_view_with_this_runs_results(self, data_root):
        run = load_profile_runs([3], data_root)[3]
        seen = {}
        run.experiment.show = lambda results, **kw: seen.update(
            results=results, kwargs=kw
        )

        run.show()

        assert seen["results"] is run.results

    def test_show_threads_extra_styles_through(self, data_root):
        run = load_profile_runs([3], data_root)[3]
        seen = {}
        run.experiment.show = lambda results, **kw: seen.update(kw)
        styles = {"custom": object()}

        run.show(extra_styles=styles)

        assert seen["extra_styles"] is styles

    def test_chip_only_shows_the_chip_not_the_experiment(self, data_root):
        run = load_profile_runs([3], data_root)[3]
        called = []
        run.experiment.show = lambda *a, **k: called.append("experiment")
        run.chip.show = lambda **kw: called.append(("chip", kw.get("interactive")))

        run.show(chip_only=True)

        assert called == [("chip", True)]

    def test_show_falls_back_to_the_chip_when_the_experiment_has_no_view(
        self, data_root
    ):
        run = load_profile_runs([3], data_root)[3]
        called = []

        def unimplemented(*a, **k):
            raise NotImplementedError

        run.experiment.show = unimplemented
        run.chip.show = lambda **kw: called.append("chip")

        run.show()

        assert called == ["chip"]

    def test_export_passes_the_results_alongside_the_experiment(self, data_root, tmp_path):
        run = load_profile_runs([3], data_root)[3]
        out = tmp_path / "exp.html"

        written = run.export(out, include_plotlyjs="cdn")

        assert written == out
        assert out.stat().st_size > 0

    def test_export_chip_only_writes_the_chip_page(self, data_root, tmp_path):
        run = load_profile_runs([3], data_root)[3]
        out = tmp_path / "chip.html"

        written = run.export(out, chip_only=True, include_plotlyjs="cdn")

        assert written == out
        assert out.stat().st_size > 0

    def test_chip_raises_when_the_experiment_has_none(self):
        from qsnow.experiments.experiment import Experiment, ExperimentResults

        run = ProfileRun(
            distance=3,
            experiment=Experiment(),
            results=ExperimentResults(experiment_ref=None, run_config={}, results={}),
        )

        with pytest.raises(AttributeError, match="no chip"):
            run.chip


def _cdf(data_root, **kwargs):
    """`ler_cdf` in its quiet, figure-returning form, for asserting on the plot."""
    return ler_cdf(DISTANCES, data_root, verbose=False, show=False, **kwargs)


def _histo(data_root, **kwargs):
    return ler_histogram(DISTANCES, data_root, show=False, **kwargs)


def _legends(ax):
    """Every legend on `ax` - `ax.get_legend()` only ever returns the most recent."""
    return [c for c in ax.get_children() if isinstance(c, Legend)]


def _labels_of(legend):
    return [t.get_text() for t in legend.get_texts()]


def _cdf_legends(fig):
    """`(distance legend, profiled/baseline key)` from the CDF panel, after a draw.

    Placement is only resolved at draw time - `loc="best"` especially - so nothing can
    be measured until the figure has been rendered once.
    """
    fig.canvas.draw()
    ax = fig.axes[0]
    by_kind = {_labels_of(lg)[0].startswith("d="): lg for lg in _legends(ax)}
    return by_kind[True], by_kind[False]


def _box_extent(patch):
    """`(centre, height)` of one boxplot patch, from its path vertices."""
    ys = patch.get_path().vertices[:, 1]
    return (ys.min() + ys.max()) / 2, ys.max() - ys.min()


class TestLerCdf:
    def test_whisker_adds_a_boxplot_axes(self, data_root):
        assert len(_cdf(data_root).axes) == 2

    def test_without_whisker_is_a_single_axes(self, data_root):
        fig = _cdf(data_root, whisker=False)

        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == "Logical Error Rate"

    def test_one_step_per_distance(self, data_root):
        assert len(_cdf(data_root).axes[0].lines) == len(DISTANCES)

    def test_baseline_doubles_the_plotted_steps(self, data_root):
        fig = _cdf(data_root, baseline_dir=data_root)

        assert len(fig.axes[0].lines) == 2 * len(DISTANCES)

    def test_baseline_steps_share_their_distance_color(self, data_root):
        lines = _cdf(data_root, baseline_dir=data_root).axes[0].lines

        assert lines[0].get_color() == lines[1].get_color()
        assert lines[1].get_alpha() == pytest.approx(profiling._BASELINE_STEP_ALPHA)

    def test_band_adds_one_fill_per_distance(self, data_root):
        fills = _cdf(data_root).axes[0].collections

        assert len(fills) == len(DISTANCES)
        assert all(f.get_alpha() == pytest.approx(profiling._BAND_ALPHA) for f in fills)

    def test_linewidth_is_applied_to_every_step(self, data_root):
        lines = _cdf(data_root, baseline_dir=data_root, linewidth=2.5).axes[0].lines

        assert all(line.get_linewidth() == pytest.approx(2.5) for line in lines)

    def test_box_scale_stretches_only_the_box_panel(self, data_root):
        base = _cdf(data_root, baseline_dir=data_root)
        big = _cdf(data_root, baseline_dir=data_root, box_scale=1.5)

        def heights(fig):
            return fig.axes[1].get_subplotspec().get_gridspec().get_height_ratios()

        (cdf0, box0), (cdf1, box1) = heights(base), heights(big)
        assert cdf1 == cdf0
        assert box1 == pytest.approx(1.5 * box0)
        # the height grows by exactly the extra panel height, so the CDF is not squeezed
        w0, h0 = base.get_size_inches()
        w1, h1 = big.get_size_inches()
        assert h1 - h0 == pytest.approx(0.5 * box0)
        # and the width grows by the same factor, so the aspect ratio is preserved
        assert w1 / w0 == pytest.approx(h1 / h0)

    def test_box_scale_leaves_the_boxes_in_data_units_alone(self, data_root):
        base = _cdf(data_root, baseline_dir=data_root).axes[1]
        big = _cdf(data_root, baseline_dir=data_root, box_scale=1.5).axes[1]

        for b1, b2 in zip(base.patches, big.patches):
            assert _box_extent(b2) == pytest.approx(_box_extent(b1))

    def test_figsize_overrides_the_default_in_both_layouts(self, data_root):
        for kwargs in ({}, {"whisker": False}):
            fig = _cdf(data_root, figsize=(7, 5), **kwargs)

            assert tuple(fig.get_size_inches()) == pytest.approx((7, 5))

    def test_figsize_keeps_the_box_scale_split(self, data_root):
        fig = _cdf(data_root, figsize=(7, 5), box_scale=2.0)
        cdf_h, box_h = fig.axes[1].get_subplotspec().get_gridspec().get_height_ratios()

        assert box_h / cdf_h == pytest.approx(2.0 * 3.0 / 9.0)

    def test_box_scale_must_be_positive(self, data_root):
        with pytest.raises(ValueError, match="box_scale"):
            _cdf(data_root, box_scale=0)

    def test_box_linewidth_is_applied_to_every_box_artist(self, data_root):
        ax = _cdf(data_root, baseline_dir=data_root, box_linewidth=2.0).axes[1]

        assert all(p.get_linewidth() == pytest.approx(2.0) for p in ax.patches)
        # whiskers, caps and medians are the panel's Line2Ds (fliers are markers only)
        assert all(
            line.get_linewidth() == pytest.approx(2.0)
            for line in ax.lines
            if line.get_linestyle() != "None"
        )

    def test_band_alpha_is_applied(self, data_root):
        fills = _cdf(data_root, band_alpha=0.4).axes[0].collections

        assert all(f.get_alpha() == pytest.approx(0.4) for f in fills)

    def test_band_can_be_turned_off(self, data_root):
        assert len(_cdf(data_root, band=False).axes[0].collections) == 0

    def test_baseline_adds_no_band(self, data_root):
        fills = _cdf(data_root, baseline_dir=data_root).axes[0].collections

        assert len(fills) == len(DISTANCES)

    def test_band_matches_its_distance_color(self, data_root):
        from matplotlib.colors import to_rgb

        ax = _cdf(data_root).axes[0]

        for line, fill in zip(ax.lines, ax.collections):
            assert tuple(fill.get_facecolor()[0][:3]) == pytest.approx(
                to_rgb(line.get_color())
            )

    def test_band_spans_the_interval_bounds(self, data_root):
        run = load_profile_runs([3], data_root)[3]
        low, high = run.ler_intervals()

        fill = _cdf(data_root).axes[0].collections[0]
        xs = np.concatenate([p.vertices[:, 0] for p in fill.get_paths()])
        xs = xs[np.isfinite(xs)]

        assert xs.min() == pytest.approx(low.min())
        assert xs.max() == pytest.approx(high.max())

    def test_axes_are_log_scaled_and_bounded(self, data_root):
        ax = _cdf(data_root).axes[0]

        assert ax.get_xscale() == "log"
        assert ax.get_ylim() == (0.0, 1.0)

    def test_title_reports_unit_dims_not_raw_coordinates(self, data_root):
        assert "10x10 chip" in _cdf(data_root).axes[0].get_title()

    def test_add_title_is_appended(self, data_root):
        title = _cdf(data_root, add_title="NRS(type=SI1000)").axes[0].get_title()

        assert title.endswith("NRS(type=SI1000)")

    def test_title_false_draws_no_caption(self, data_root):
        """The print path: the caption is set in the paper's text instead."""
        assert _cdf(data_root, title=False).axes[0].get_title() == ""

    def test_add_title_is_ignored_when_the_title_is_off(self, data_root):
        fig = _cdf(data_root, title=False, add_title="NRS(type=SI1000)")

        assert fig.axes[0].get_title() == ""

    def test_dpi_is_applied(self, data_root):
        assert _cdf(data_root, dpi=200).dpi == 200


class TestLerCdfBoxPanel:
    """The box panel mirrors the CDF's baseline overlay: same colour per distance,
    with the baseline drawn fainter *and* narrower so it reads as subordinate."""

    def test_one_box_per_distance_without_a_baseline(self, data_root):
        ax_box = _cdf(data_root).axes[1]

        assert len(ax_box.patches) == len(DISTANCES)

    def test_baseline_doubles_the_boxes(self, data_root):
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        assert len(ax_box.patches) == 2 * len(DISTANCES)

    def test_boxes_match_their_distance_curve_color(self, data_root):
        fig = _cdf(data_root)
        ax_cdf, ax_box = fig.axes

        for line, patch in zip(ax_cdf.lines, ax_box.patches):
            assert patch.get_facecolor()[:3] == pytest.approx(line.get_color()[:3])

    def test_boxes_are_colored_even_without_a_baseline(self, data_root):
        # guards the "always colour-match" decision - the panel used to be monochrome
        ax_box = _cdf(data_root).axes[1]

        assert not all(
            p.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
            for p in ax_box.patches
        )

    def test_baseline_box_pairs_with_and_sits_below_its_profiled_box(self, data_root):
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        # drawn baseline-then-profiled per distance, so patches come in pairs
        for base_patch, primary_patch in zip(ax_box.patches[::2], ax_box.patches[1::2]):
            base_centre, base_height = _box_extent(base_patch)
            primary_centre, primary_height = _box_extent(primary_patch)

            assert base_centre < primary_centre
            assert base_height < primary_height
            assert base_patch.get_facecolor()[:3] == pytest.approx(
                primary_patch.get_facecolor()[:3]
            )

    def test_baseline_box_is_fainter_than_the_profiled_box(self, data_root):
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        # read the alpha off the RGBA fill: the hatched baseline sets it there rather
        # than through set_alpha, which would fade its edge and hatch too
        assert ax_box.patches[0].get_facecolor()[3] == pytest.approx(
            profiling._BASELINE_BOX_ALPHA
        )
        assert ax_box.patches[1].get_facecolor()[3] == pytest.approx(
            profiling._PRIMARY_ALPHA
        )

    def test_one_tick_label_per_distance_not_per_box(self, data_root):
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        assert [t.get_text() for t in ax_box.get_yticklabels()] == [
            str(d) for d in DISTANCES
        ]

    def test_figure_is_taller_when_a_baseline_is_paired(self, data_root):
        plain = _cdf(data_root).get_size_inches()[1]
        paired = _cdf(data_root, baseline_dir=data_root).get_size_inches()[1]

        assert paired > plain

    def test_baseline_box_is_hatched_and_the_profiled_box_is_not(self, data_root):
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        for base_patch, primary_patch in zip(ax_box.patches[::2], ax_box.patches[1::2]):
            assert base_patch.get_hatch() == profiling._BASELINE_HATCH
            assert primary_patch.get_hatch() is None

    def test_hatched_edge_is_stronger_than_the_fill(self, data_root):
        # set_alpha() would fade the edge - and the hatch drawn on it - with the fill
        base_patch = _cdf(data_root, baseline_dir=data_root).axes[1].patches[0]

        assert base_patch.get_facecolor()[3] == pytest.approx(
            profiling._BASELINE_BOX_ALPHA
        )
        assert base_patch.get_edgecolor()[3] == pytest.approx(
            profiling._BASELINE_HATCH_ALPHA
        )

    def test_solo_boxes_are_not_hatched(self, data_root):
        assert all(p.get_hatch() is None for p in _cdf(data_root).axes[1].patches)


class TestBaselineLegend:
    """The profiled-vs-baseline key, and the dash/hatch cues it points at."""

    def test_baseline_steps_are_dashed_and_primary_solid(self, data_root):
        lines = _cdf(data_root, baseline_dir=data_root).axes[0].lines

        for primary, base in zip(lines[::2], lines[1::2]):
            assert primary.get_linestyle() == "-"
            assert base.get_linestyle() == profiling._BASELINE_LINESTYLE

    def test_steps_are_solid_without_a_baseline(self, data_root):
        assert all(
            line.get_linestyle() == "-" for line in _cdf(data_root).axes[0].lines
        )

    def test_both_panels_gain_a_key(self, data_root):
        ax_cdf, ax_box = _cdf(data_root, baseline_dir=data_root).axes

        # the distance legend survives alongside the new one
        assert len(_legends(ax_cdf)) == 2
        assert len(_legends(ax_box)) == 1

    def test_no_key_without_a_baseline(self, data_root):
        ax_cdf, ax_box = _cdf(data_root).axes

        assert len(_legends(ax_cdf)) == 1
        assert len(_legends(ax_box)) == 0

    def test_distance_legend_is_not_replaced(self, data_root):
        ax_cdf = _cdf(data_root, baseline_dir=data_root).axes[0]

        rendered = {tuple(_labels_of(lg)) for lg in _legends(ax_cdf)}
        assert any(labels[0].startswith("d=") for labels in rendered)

    def test_labels_come_from_the_chip_noise_model(self, named_roots):
        primary_root, baseline_root = named_roots
        ax_box = ler_cdf(
            DISTANCES, primary_root, baseline_dir=baseline_root,
            verbose=False, show=False,
        ).axes[1]

        assert _labels_of(_legends(ax_box)[0]) == ["normal contour", "uniform homogeneous"]

    def test_explicit_labels_override_the_noise_model(self, named_roots):
        primary_root, baseline_root = named_roots
        ax_box = ler_cdf(
            DISTANCES, primary_root, baseline_dir=baseline_root,
            labels=("contoured", "flat"), verbose=False, show=False,
        ).axes[1]

        assert _labels_of(_legends(ax_box)[0]) == ["contoured", "flat"]

    def test_falls_back_when_the_chip_has_no_noise_model(self, data_root):
        # the `sweeps` fixture never sets one
        ax_box = _cdf(data_root, baseline_dir=data_root).axes[1]

        assert _labels_of(_legends(ax_box)[0]) == list(profiling._FALLBACK_LABELS)

    def test_whisker_false_still_gets_the_cdf_key(self, data_root):
        fig = _cdf(data_root, baseline_dir=data_root, whisker=False)

        assert len(fig.axes) == 1
        assert len(_legends(fig.axes[0])) == 2

    def test_cdf_legends_do_not_overlap(self, data_root):
        # "best" cannot see another legend, so the distance legend used to land on top
        # of the key in the lower-right corner the CDF steps leave free
        fig = _cdf(data_root, baseline_dir=data_root)
        distances, key = _cdf_legends(fig)

        assert not distances.get_window_extent().overlaps(key.get_window_extent())

    def test_stacked_key_stays_inside_the_axes(self, data_root):
        fig = _cdf(data_root, baseline_dir=data_root)
        ax = fig.axes[0]
        _, key = _cdf_legends(fig)

        panel, box = ax.get_window_extent(), key.get_window_extent()
        assert panel.contains(*box.p0) and panel.contains(*box.p1)

    def test_key_stacks_beneath_the_distance_legend(self, data_root):
        # the corner the key used to be pinned to, forced: the distance legend is on the
        # axes floor, so making room underneath means lifting it rather than flipping
        fig = _cdf(data_root, baseline_dir=data_root, legend_loc="lower right")
        ax = fig.axes[0]
        distances, key = _cdf_legends(fig)
        d_box, k_box = profiling._axes_frac(ax, distances), profiling._axes_frac(ax, key)

        assert d_box.x1 > 0.5 and d_box.y0 < 0.5
        assert k_box.x1 == pytest.approx(d_box.x1, abs=0.01)  # share the right edge
        assert k_box.y1 < d_box.y0  # key underneath
        assert not distances.get_window_extent().overlaps(key.get_window_extent())

    def test_key_stays_beneath_a_high_distance_legend(self, data_root):
        # the other branch: room below, so the key drops and nothing is lifted
        fig = _cdf(data_root, baseline_dir=data_root, legend_loc="upper right")
        ax = fig.axes[0]
        distances, key = _cdf_legends(fig)
        d_box, k_box = profiling._axes_frac(ax, distances), profiling._axes_frac(ax, key)

        assert d_box.y1 == pytest.approx(1.0, abs=0.05)  # never moved
        assert k_box.y1 < d_box.y0

    def test_key_loc_decouples_the_two_legends(self, data_root):
        fig = _cdf(
            data_root,
            baseline_dir=data_root,
            legend_loc="upper left",
            key_loc="lower right",
        )
        ax = fig.axes[0]
        distances, key = _cdf_legends(fig)
        d_box, k_box = profiling._axes_frac(ax, distances), profiling._axes_frac(ax, key)

        assert d_box.x0 < 0.5 and d_box.y1 > 0.5  # upper left
        assert k_box.x1 > 0.5 and k_box.y0 < 0.5  # lower right, not stacked

    def test_legend_placement_survives_a_redraw(self, data_root):
        # the auto-placed legend is frozen once the key hangs off it; left live, "best"
        # would re-resolve on the next draw and slide out from under the key
        fig = _cdf(data_root, baseline_dir=data_root)
        before = [lg.get_window_extent().bounds for lg in _cdf_legends(fig)]

        fig.canvas.draw()

        after = [lg.get_window_extent().bounds for lg in _cdf_legends(fig)]
        assert after == pytest.approx(before)

    def test_save_writes_the_figure(self, data_root, tmp_path):
        out = tmp_path / "figs" / "cdf.png"  # parent does not exist yet

        fig = _cdf(data_root, save=out)

        assert out.is_file()
        assert Image.open(out).size[0] > 0
        assert isinstance(fig, Figure)  # show=False still hands the figure back

    def test_save_uses_the_figure_dpi(self, data_root, tmp_path):
        out = tmp_path / "cdf.png"

        fig = _cdf(data_root, dpi=150, save=out)

        width, _ = Image.open(out).size
        assert width == pytest.approx(150 * fig.get_size_inches()[0], abs=2)

    def test_dpi_survives_to_savefig(self, data_root, tmp_path):
        fig = _cdf(data_root, dpi=200)
        out = tmp_path / "cdf.png"

        fig.savefig(out)

        width, _ = Image.open(out).size
        assert width == pytest.approx(200 * fig.get_size_inches()[0], abs=2)

    def test_verbose_prints_per_distance_and_chip_stats(self, data_root, capsys):
        ler_cdf(DISTANCES, data_root, verbose=True, show=False)

        out = capsys.readouterr().out
        assert "Distance 3" in out and "Distance 5" in out
        assert "Chip(mean=" in out

    def test_quiet_when_verbose_off(self, data_root, capsys):
        _cdf(data_root)

        assert capsys.readouterr().out == ""


class TestLerHistogram:
    def test_save_writes_the_figure(self, data_root, tmp_path):
        out = tmp_path / "histo.png"

        _histo(data_root, save=out)

        assert out.is_file()

    def test_one_axes_per_distance(self, data_root):
        assert len(_histo(data_root).axes) == len(DISTANCES)

    def test_unused_grid_slots_are_deleted(self, data_root):
        # 2 distances in a <=4-column grid leaves 2 empty slots that must not survive
        assert len(_histo(data_root).axes) == 2

    def test_scope_switches_the_x_label(self, data_root):
        assert _histo(data_root).axes[0].get_xlabel() == "ler"
        assert _histo(data_root, scope="errors").axes[0].get_xlabel() == "# logical errors"

    def test_subplot_titles_report_count_and_zeros(self, data_root):
        assert _histo(data_root).axes[0].get_title() == "d3 (n=4, len(0)=0)"

    def test_limits_apply_to_every_subplot(self, data_root):
        fig = _histo(data_root, limits=(0.0, 0.01))

        assert all(ax.get_xlim() == (0.0, 0.01) for ax in fig.axes)

    def test_dpi_is_applied(self, data_root):
        assert _histo(data_root, dpi=300).dpi == 300

    def test_suptitle_reports_the_chip(self, data_root):
        assert "10x10 chip" in _histo(data_root)._suptitle.get_text()

    def test_title_false_draws_no_suptitle(self, data_root):
        """Matches `ler_cdf(title=False)`: the two figures suppress captions alike."""
        fig = _histo(data_root, title=False)

        assert fig._suptitle is None or fig._suptitle.get_text() == ""


class TestShow:
    @pytest.fixture(params=["ler_cdf", "ler_histogram"])
    def plot(self, request, data_root):
        fn = {"ler_cdf": ler_cdf, "ler_histogram": ler_histogram}[request.param]
        kwargs = {"verbose": False} if fn is ler_cdf else {}
        return lambda **kw: fn(DISTANCES, data_root, **kwargs, **kw)

    def test_shows_by_default(self, plot, monkeypatch):
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))

        result = plot()

        assert shown == [1]
        # returning the figure as well would draw it a second time in a notebook
        assert result is None

    def test_show_false_returns_the_figure_without_showing(self, plot, monkeypatch):
        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))

        fig = plot(show=False)

        assert shown == []
        assert isinstance(fig, Figure)


class TestLerTable:
    def _rows(self, root, baseline, **kw):
        return ler_table(DISTANCES, root, baseline_dir=baseline, verbose=False, **kw)

    def test_a_sweep_against_itself_reads_as_no_difference(self, data_root):
        rows = self._rows(data_root, data_root)
        for r in rows:
            if r["distance"] == "all":
                assert r["min ratio"] == r["max ratio"] == pytest.approx(1.0)
            elif r["stat"] == "yield":
                assert r["ratio"] == pytest.approx(0.0)
            else:
                assert r["ratio"] == pytest.approx(1.0)

    def test_spread_out_profiled_against_a_constant_baseline(self, tmp_path):
        """The uniform case: one LER for every placement on the baseline, so its spreads
        are 1 while the profiled chip's exceed 1, with a better best and a worse worst."""
        roots = {}
        for key, lers in [("profiled", [0.001, 0.002, 0.004, 0.008]), ("baseline", [0.003] * 4)]:
            serialize.set_data_dir(tmp_path / key)
            try:
                for d in DISTANCES:
                    _save_sweep(d, lers)
            finally:
                serialize.set_data_dir()
            roots[key] = tmp_path / key
        rows = {(r["distance"], r["stat"]): r for r in self._rows(roots["profiled"], roots["baseline"])}
        for d in DISTANCES:
            assert rows[(d, "spread worst/best")]["baseline"] == pytest.approx(1.0)
            assert rows[(d, "spread worst/best")]["profiled"] == pytest.approx(8.0)
            assert rows[(d, "worst")]["ratio"] > 1 > rows[(d, "best")]["ratio"]
        assert rows[("all", "worst")]["max ratio"] == pytest.approx(8 / 3)

    def test_one_row_per_stat_per_distance_plus_summary_and_named_columns(self, named_roots, capsys):
        primary, baseline = named_roots
        rows = ler_table(DISTANCES, primary, baseline_dir=baseline, verbose=False, explain=True)
        per_distance = [r for r in rows if r["distance"] != "all"]
        assert len(per_distance) == len(DISTANCES) * len(profiling._STAT_MEANINGS)
        assert {r["stat"] for r in rows if r["distance"] == "all"} == set(profiling._SUMMARISED)
        assert {"normal contour", "uniform homogeneous"} <= set(per_distance[0])
        out = capsys.readouterr().out
        assert all(stat in out for stat in profiling._STAT_MEANINGS)
