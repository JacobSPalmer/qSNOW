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
    default_style,
)

DEFAULT_NAMES = ["Status", "CSS Type", "Noise"]


def visible_flags(fig):
    return [trace.visible for trace in fig.data]


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
