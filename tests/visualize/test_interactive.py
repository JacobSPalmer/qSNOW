import pytest

from qsnow.experiments.experiment import ExperimentResults
from qsnow.experiments.squarepacking.game import SquarePackingExp
from qsnow.helpers.serialize import set_data_dir
from qsnow.visualize.interactive import (
    _format_stat_rows,
    default_interactive_styles,
    export_html,
    export_square_packing,
    visualize_interactive,
)
from qsnow.visualize.visualize import (
    VisualizationStyle,
    _default_qubit_style_by_status,
    css_style,
    default_style,
)

DEFAULT_NAMES = ["Status", "CSS Type", "Noise"]


def visible_flags(fig):
    return [trace.visible for trace in fig.data]


class TestDefaultInteractiveStylesDesc:
    def test_bundled_styles_all_have_desc(self, chip):
        styles = default_interactive_styles(chip)
        assert all(style.desc for style in styles.values())

    def test_shared_style_instances_untouched(self, chip):
        # `default_interactive_styles` must not mutate the shared module-level
        # `default_style`/`css_style` instances when attaching a description.
        default_interactive_styles(chip)
        assert default_style.desc is None
        assert css_style.desc is None


class TestStyleDescCaption:
    def test_caption_present_for_active_style_with_desc(self, chip):
        fig = visualize_interactive(chip, active="Noise")
        texts = [a.text for a in fig.layout.annotations]
        desc = default_interactive_styles(chip)["Noise"].desc
        assert desc in texts

    def test_caption_absent_for_style_without_desc(self, chip):
        no_desc = VisualizationStyle(style_fn=_default_qubit_style_by_status)
        fig = visualize_interactive(chip, {"Bare": no_desc})
        assert fig.layout.annotations == ()

    def test_caption_swaps_per_button(self, chip):
        fig = visualize_interactive(chip)
        styles = default_interactive_styles(chip)
        menu = fig.layout.updatemenus[0]
        for button in menu.buttons:
            _, relayout = button.args
            texts = [a["text"] for a in relayout["annotations"]]
            expected = styles[button.label].desc
            assert expected in texts

    def test_no_desc_bundle_matches_prior_geometry(self, chip):
        # regression: reserving room for the caption row must be opt-in - a
        # bundle where no style has a `desc` should render with exactly the
        # same figure height as before this feature existed.
        described = visualize_interactive(chip)
        no_desc = VisualizationStyle(style_fn=_default_qubit_style_by_status)
        undescribed = visualize_interactive(chip, {"Bare": no_desc})
        assert undescribed.layout.height < described.layout.height


class TestVisualizeInteractive:
    def test_one_trace_per_style_only_active_visible(self, chip):
        fig = visualize_interactive(chip)
        assert [trace.name for trace in fig.data] == DEFAULT_NAMES
        assert visible_flags(fig) == [True, False, False]

    def test_updatemenus_wiring(self, chip):
        fig = visualize_interactive(chip)
        assert len(fig.layout.updatemenus) == 1
        menu = fig.layout.updatemenus[0]
        assert [button.label for button in menu.buttons] == DEFAULT_NAMES
        assert menu.active == 0
        for i, button in enumerate(menu.buttons):
            assert button.method == "update"
            restyle, relayout = button.args
            assert restyle["visible"] == [j == i for j in range(len(DEFAULT_NAMES))]
            assert len(relayout["shapes"]) == len(chip.qubits)
            assert "annotations" in relayout

    def test_initial_state_matches_active_button(self, chip):
        fig = visualize_interactive(chip)
        button_shapes = fig.layout.updatemenus[0].buttons[0].args[1]["shapes"]
        assert len(fig.layout.shapes) == len(button_shapes)

    def test_constant_geometry_across_views(self, chip):
        # colorbar strip reserved for the whole figure; buttons touch only
        # shapes/annotations, never the sizing keys
        fig = visualize_interactive(chip)
        assert fig.layout.xaxis.domain[1] < 1.0
        for button in fig.layout.updatemenus[0].buttons:
            assert set(button.args[1]) == {"shapes", "annotations"}

    def test_active_by_name(self, chip):
        fig = visualize_interactive(chip, active="Noise")
        assert visible_flags(fig) == [False, False, True]
        assert fig.layout.updatemenus[0].active == 2

    def test_unknown_active_raises(self, chip):
        with pytest.raises(ValueError, match="Unknown active style"):
            visualize_interactive(chip, active="nope")

    def test_empty_styles_raises(self, chip):
        with pytest.raises(ValueError, match="at least one style"):
            visualize_interactive(chip, {})

    def test_tile_annotations_swap_with_style(self, chip, logical_tile):
        chip.add_tile(logical_tile, (0, 0))
        no_logical = VisualizationStyle(style_fn=_default_qubit_style_by_status)
        fig = visualize_interactive(chip, {"Status": default_style, "Bare": no_logical})
        status_button, bare_button = fig.layout.updatemenus[0].buttons
        assert len(status_button.args[1]["annotations"]) == 1
        assert len(bare_button.args[1]["annotations"]) == 0
        # the tile outline shape rides along with the qubit shapes
        assert len(status_button.args[1]["shapes"]) == len(chip.qubits) + 1
        assert len(bare_button.args[1]["shapes"]) == len(chip.qubits)

    def test_default_title_and_subtitle(self, chip):
        fig = visualize_interactive(chip)
        assert fig.layout.title.text == "Chip 5x5"
        assert "50 qubits" in fig.layout.title.subtitle.text

    def test_title_and_subtitle_suppressed(self, chip):
        fig = visualize_interactive(chip, title="", subtitle="")
        assert fig.layout.title.text is None

    def test_write_html_keeps_dropdown(self, tmp_path, chip):
        path = tmp_path / "fig.html"
        visualize_interactive(chip).write_html(path, include_plotlyjs="cdn")
        assert "updatemenus" in path.read_text()


class TestExportHtml:
    def test_explicit_path(self, tmp_path, chip):
        path = export_html(chip, tmp_path / "my_chip.html", include_plotlyjs="cdn")
        assert path == tmp_path / "my_chip.html"
        text = path.read_text()
        assert "<title>qSNOW - Chip</title>" in text
        assert "updatemenus" in text

    def test_default_path_uses_data_dir(self, tmp_path, chip):
        set_data_dir(tmp_path)
        try:
            path = export_html(chip, include_plotlyjs="cdn")
        finally:
            set_data_dir()
        assert path.parent == tmp_path / "html"
        assert path.name.startswith("chip_5x5_")
        assert path.suffix == ".html"

    def test_page_header_has_stats_and_desc(self, tmp_path, chip):
        chip.tag.desc = "conftest test chip"
        path = export_html(chip, tmp_path / "chip.html", include_plotlyjs="cdn")
        text = path.read_text()
        assert "<dt>Qubits</dt><dd>50</dd>" in text
        assert "<dt>Size</dt>" in text
        assert '<p class="desc">conftest test chip</p>' in text

    def test_export_leaves_the_chip_noise_model_intact(self, tmp_path, chip):
        # regression: the stats strip popped `name` out of the chip's live metadata
        chip.generate_gaussian_noise(0.01, 0.002, seed=1)
        before = chip.summary()["noise_model"]

        path = export_html(chip, tmp_path / "chip.html", include_plotlyjs="cdn")

        assert chip.summary()["noise_model"] == before
        assert "<dt>type</dt><dd>gaussian</dd>" in path.read_text()

    def test_default_styles_are_bundled(self, tmp_path, chip):
        path = export_html(chip, tmp_path / "chip.html", include_plotlyjs="cdn")
        text = path.read_text()
        for name in default_interactive_styles(chip):
            assert name in text


@pytest.fixture
def sp_exp(chip, logical_tile) -> SquarePackingExp:
    return SquarePackingExp(chip=chip, tile=logical_tile)


@pytest.fixture
def sp_results(sp_exp) -> ExperimentResults:
    return ExperimentResults(
        experiment_ref=None,
        run_config={"shots": 1000, "max_errors": 100, "decoder": "pymatching"},
        results={loc: {"ler": 0.01} for loc in sp_exp.profile},
    )


class TestExportSquarePacking:
    def test_explicit_path(self, tmp_path, sp_exp, sp_results):
        path = export_square_packing(
            sp_exp, sp_results, tmp_path / "sp.html", include_plotlyjs="cdn"
        )
        assert path == tmp_path / "sp.html"
        text = path.read_text()
        assert "updatemenus" in text
        assert "<dt>Placements</dt>" in text
        assert "<dt>Shots</dt><dd>1000</dd>" in text

    def test_default_path_uses_data_dir(self, tmp_path, sp_exp, sp_results):
        set_data_dir(tmp_path)
        try:
            path = export_square_packing(sp_exp, sp_results, include_plotlyjs="cdn")
        finally:
            set_data_dir()
        assert path.parent == tmp_path / "html"
        assert path.name.startswith("experiment_sp_")
        assert path.suffix == ".html"

    def test_via_export_html_dispatch(self, tmp_path, sp_exp, sp_results):
        path = export_html(
            sp_exp,
            tmp_path / "sp.html",
            results=sp_results,
            include_plotlyjs="cdn",
        )
        assert path == tmp_path / "sp.html"

    def test_export_html_requires_results_kwarg(self, tmp_path, sp_exp):
        with pytest.raises(AttributeError, match="results"):
            export_html(sp_exp, tmp_path / "sp.html")

    def test_stats_include_chip_tile_and_experiment(self, tmp_path, sp_exp, sp_results):
        path = export_square_packing(
            sp_exp, sp_results, tmp_path / "sp.html", include_plotlyjs="cdn"
        )
        text = path.read_text()
        assert "<dt>Size</dt>" in text
        assert "<dt>Distance</dt>" in text


class TestFormatStatRows:
    def test_strings_pass_through(self):
        html = _format_stat_rows({"Size": "5 x 5"})
        assert "<dt>Size</dt><dd>5 x 5</dd>" in html

    def test_floats_formatted_to_three_sig_figs(self):
        html = _format_stat_rows({"p": 0.0012345})
        assert "<dd>0.00123</dd>" in html

    def test_escapes_html_special_characters(self):
        html = _format_stat_rows({"<tag>": "<script>"})
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestCouplerView:
    def test_derived_couplers_add_no_view(self, chip):
        """While couplers restate the noise heatmap, the dropdown stays as it was."""
        chip.generate_gaussian_noise(0.01, 0.002)

        styles = default_interactive_styles(chip)

        assert list(styles) == DEFAULT_NAMES

    def test_independent_couplers_add_a_view(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        styles = default_interactive_styles(chip)

        assert list(styles) == DEFAULT_NAMES + ["Coupler Noise", "Qubit + Coupler"]
        assert all(styles[n].desc for n in ("Coupler Noise", "Qubit + Coupler"))

    def test_single_bar_views_still_map_one_to_one_onto_traces(self, chip):
        """A coupler layer sharing the qubit colorbar rides the existing trace, so those
        views keep their 1:1 style-to-trace correspondence."""
        chip.generate_gaussian_noise(0.01, 0.002)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        fig = visualize_interactive(chip)
        single_bar = list(default_interactive_styles(chip))[:4]

        assert [t.name for t in fig.data][:4] == single_bar
        assert visible_flags(fig)[0] is True
        for i in range(len(single_bar)):
            assert sum(fig.layout.updatemenus[0].buttons[i].args[0]["visible"]) == 1

    def test_coupler_view_shapes_include_the_edges(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        fig = visualize_interactive(chip)
        button = next(
            b for b in fig.layout.updatemenus[0].buttons if b.label == "Coupler Noise"
        )

        assert len(button.args[1]["shapes"]) == len(chip.qubits) + len(chip.couplers)
        # buttons still only ever swap these two keys, never geometry
        assert set(button.args[1]) == {"shapes", "annotations"}


class TestCombinedView:
    def measured(self, chip):
        chip.generate_uniform_noise(0.004)
        chip.set_coupler_noise_map({c.ends: 0.02 for c in chip.couplers})
        return chip

    def test_combined_view_joins_the_bundle_when_couplers_are_independent(self, chip):
        names = list(default_interactive_styles(self.measured(chip)))

        assert names == DEFAULT_NAMES + ["Coupler Noise", "Qubit + Coupler"]

    def test_a_dual_colorbar_view_contributes_two_traces(self, chip):
        fig = visualize_interactive(self.measured(chip))

        # four single-bar views plus the combined view's pair
        assert len(fig.data) == 6

    def test_each_button_shows_exactly_its_own_traces(self, chip):
        """Visibility is tracked per style *span*, so the combined view lights both of
        its traces while every single-trace view lights exactly one."""
        fig = visualize_interactive(self.measured(chip))
        flags = [b.args[0]["visible"] for b in fig.layout.updatemenus[0].buttons]

        assert [sum(f) for f in flags] == [1, 1, 1, 1, 2]
        assert all(len(f) == len(fig.data) for f in flags)
        # every trace is claimed by exactly one view
        assert [sum(col) for col in zip(*flags)] == [1] * len(fig.data)

    def test_buttons_still_only_swap_shapes_and_annotations(self, chip):
        fig = visualize_interactive(self.measured(chip))

        for button in fig.layout.updatemenus[0].buttons:
            assert set(button.args[1]) == {"shapes", "annotations"}

    def test_geometry_is_sized_for_the_widest_view(self, chip):
        """Two colorbars on one view must not resize the plot when others are shown."""
        from qsnow.interface.chip import Chip

        fig = visualize_interactive(self.measured(chip))
        plain = visualize_interactive(Chip(5, 5))

        from qsnow.visualize.visualize import _colorbar_strip_px, _font_px, device_heatmap_style

        # the plain view already carries the qubit bar's strip; the measured chip adds
        # the coupler bar's
        device = device_heatmap_style(self.measured(Chip(5, 5)))
        assert fig.layout.width - plain.layout.width == _colorbar_strip_px(device.coupler_colorbar, _font_px())
