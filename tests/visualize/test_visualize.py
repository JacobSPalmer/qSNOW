from qsnow.interface.models import Qubit, Status
from qsnow.visualize.visualize import (
    QubitStyle,
    _default_qubit_style_by_status,
    _discrete_colormap_fn,
    area_selection_style,
    custom_heatmap_style,
    default_style,
    noise_heatmap_style,
    packing_profile_style,
    visualize,
)


class TestDefaultQubitStyle:
    def test_returns_qubit_style(self):
        style = _default_qubit_style_by_status(Qubit(loc=(0, 0), status=Status.LOGICAL))
        assert isinstance(style, QubitStyle)

    def test_color_varies_by_status(self):
        inactive_color = _default_qubit_style_by_status(
            Qubit(loc=(0, 0), status=Status.INACTIVE)
        ).color
        logical_color = _default_qubit_style_by_status(
            Qubit(loc=(0, 0), status=Status.LOGICAL)
        ).color
        assert inactive_color != logical_color


def test_default_style_uses_status_based_styling():
    assert default_style.style_fn is _default_qubit_style_by_status


class TestVisualizeFigure:
    def test_one_shape_per_qubit_single_trace(self, chip):
        fig = visualize(chip)
        assert len(fig.layout.shapes) == len(chip.qubits) == 50
        assert len(fig.data) == 1
        assert fig.layout.updatemenus == ()

    def test_default_style_uses_full_domain(self, chip):
        fig = visualize(chip)
        assert fig.layout.xaxis.domain[1] == 1.0

    def test_colorbar_style_reserves_domain_strip(self, chip):
        fig = visualize(chip, style=noise_heatmap_style(chip))
        assert fig.layout.xaxis.domain[1] < 1.0
        assert fig.data[0].marker.showscale is True

    def test_logical_color_gradient_overrides_default_edgecolor(
        self, lg_chip, logical_tile
    ):
        other_tile = logical_tile.copy()
        lg_chip.add_tile(logical_tile, (0, 0))
        lg_chip.add_tile(other_tile, (8, 0))
        fig = visualize(lg_chip, logical_color_gradient=True)
        # last two shapes are the tile outlines (qubit rects come first); the
        # default logical style's fixed "red" edgecolor should be overridden
        for shape in fig.layout.shapes[-2:]:
            assert shape.line.color != "red"


class TestCustomHeatmapStyle:
    def test_colors_by_provided_map(self, chip):
        coord_map = {q.loc: float(i) for i, q in enumerate(chip.qubits)}
        style = custom_heatmap_style(chip, coord_map, label="LER")
        assert style.colorbar is not None
        assert style.colorbar.label == "LER"
        qubit = chip.qubits[0]
        result = style.style_fn(qubit)
        assert result.color == coord_map[qubit.loc]

    def test_missing_coord_falls_back_to_lightgray(self, chip):
        style = custom_heatmap_style(chip, {}, limits=(0.0, 1.0))
        result = style.style_fn(chip.qubits[0])
        assert result.color == "lightgray"

    def test_limits_override_computed_bounds(self, chip):
        coord_map = {q.loc: 0.5 for q in chip.qubits}
        style = custom_heatmap_style(chip, coord_map, limits=(-1.0, 1.0))
        assert style.colorbar.cmin == -1.0
        assert style.colorbar.cmax == 1.0


class TestPackingProfileStyle:
    def test_valid_placement_colored_pink(self, chip):
        loc = chip.qubits[0].loc
        style = packing_profile_style(chip, {loc: {"bound": (1, 1), "ler": 0.01}})
        assert style.style_fn(chip.qubits[0]).color == "pink"

    def test_missing_placement_colored_lightgray(self, chip):
        style = packing_profile_style(chip, {})
        assert style.style_fn(chip.qubits[0]).color == "lightgray"


class TestAreaSelectionStyle:
    def test_selected_qubit_colored_red(self, chip):
        qubit = chip.qubits[0]
        style = area_selection_style(chip, {qubit.loc: qubit})
        assert style.style_fn(qubit).color == "red"

    def test_unselected_qubit_colored_lightgray(self, chip):
        style = area_selection_style(chip, {})
        assert style.style_fn(chip.qubits[0]).color == "lightgray"

    def test_show_logicals_toggles_logical_style(self, chip):
        assert area_selection_style(chip, {}).logical_style is None
        assert (
            area_selection_style(chip, {}, show_logicals=True).logical_style is not None
        )


class TestDiscreteColormapFn:
    def test_maps_index_within_palette(self):
        color_fn = _discrete_colormap_fn((0, 3), palette=["a", "b", "c", "d"])
        assert color_fn(0) == "a"
        assert color_fn(3) == "d"

    def test_different_indices_can_differ(self):
        color_fn = _discrete_colormap_fn((0, 10), palette=["a", "b", "c"])
        assert color_fn(0) != color_fn(10)
