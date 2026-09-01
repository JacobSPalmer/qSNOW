import pytest

from qsnow.interface.models import Qubit, Status
from qsnow.visualize.visualize import (
    CouplerStyle,
    QubitStyle,
    VisualizationStyle,
    _default_qubit_style_by_status,
    _discrete_colormap_fn,
    area_selection_style,
    custom_heatmap_style,
    default_style,
    _log_ticks,
    _COLORBAR_PX,
    _COLORBAR_TITLE_PX,
    _floored_colorscale,
    _label_color,
    coupler_heatmap_style,
    device_heatmap_style,
    noise_heatmap_style,
    packing_profile_style,
    visualize,
    with_couplers,
)


def test_visualization_style_desc_defaults_to_none():
    style = VisualizationStyle(style_fn=_default_qubit_style_by_status)
    assert style.desc is None


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


class TestNoiseHeatmapStyleDesc:
    def test_defaults_to_none(self, chip):
        assert noise_heatmap_style(chip).desc is None

    def test_threads_through(self, chip):
        style = noise_heatmap_style(chip, desc="what this shows")
        assert style.desc == "what this shows"


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

    def test_x_axis_range_not_widened_by_scaleanchor(self, lg_chip):
        # regression: on a square chip, the (fixed) extra top margin reserved for
        # the top-side axis labels used to shrink the plot area's height relative
        # to its width; since the y-axis is scaleanchor-locked to x, plotly
        # silently widened the *x* range to compensate (growing with chip size)
        # while the y range stayed exactly as requested. Both should now resolve
        # to exactly what was requested.
        fig = visualize(lg_chip)
        requested_x = fig.layout.xaxis.range
        requested_y = fig.layout.yaxis.range
        full = fig.full_figure_for_development(warn=False)
        assert full.layout.xaxis.range == requested_x
        assert full.layout.yaxis.range == requested_y

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
        style = custom_heatmap_style(chip, coord_map, "LER")
        assert style.colorbar is not None
        assert style.colorbar.label == "LER"
        qubit = chip.qubits[0]
        result = style.style_fn(qubit)
        assert result.color == coord_map[qubit.loc]

    def test_missing_coord_falls_back_to_lightgray(self, chip):
        style = custom_heatmap_style(chip, {}, "LER", limits=(0.0, 1.0))
        result = style.style_fn(chip.qubits[0])
        assert result.color == "lightgray"

    def test_limits_override_computed_bounds(self, chip):
        coord_map = {q.loc: 0.5 for q in chip.qubits}
        style = custom_heatmap_style(chip, coord_map, "LER", limits=(-1.0, 1.0))
        assert style.colorbar.cmin == -1.0
        assert style.colorbar.cmax == 1.0

    def test_colorbar_label_defaults_to_float_label(self, chip):
        style = custom_heatmap_style(chip, {}, "LER", limits=(0.0, 1.0))
        assert style.colorbar.label == "LER"

    def test_colorbar_label_override(self, chip):
        style = custom_heatmap_style(
            chip, {}, "LER", limits=(0.0, 1.0), colorbar_label="Logical error rate"
        )
        assert style.colorbar.label == "Logical error rate"

    def test_style_desc_threads_into_visualization_style(self, chip):
        style = custom_heatmap_style(
            chip, {}, "LER", limits=(0.0, 1.0), desc="what this shows"
        )
        assert style.desc == "what this shows"


class TestPackingProfileStyle:
    def test_valid_placement_colored_pink(self, chip):
        loc = chip.qubits[0].loc
        style = packing_profile_style(chip, {loc: {"bound": (1, 1), "ler": 0.01}})
        assert style.style_fn(chip.qubits[0]).color == "pink"

    def test_missing_placement_colored_lightgray(self, chip):
        style = packing_profile_style(chip, {})
        assert style.style_fn(chip.qubits[0]).color == "lightgray"

    def test_desc_threads_through(self, chip):
        style = packing_profile_style(chip, {}, desc="what this shows")
        assert style.desc == "what this shows"


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

    def test_desc_threads_through(self, chip):
        style = area_selection_style(chip, {}, desc="what this shows")
        assert style.desc == "what this shows"


class TestDiscreteColormapFn:
    def test_maps_index_within_palette(self):
        color_fn = _discrete_colormap_fn((0, 3), palette=["a", "b", "c", "d"])
        assert color_fn(0) == "a"
        assert color_fn(3) == "d"

    def test_different_indices_can_differ(self):
        color_fn = _discrete_colormap_fn((0, 10), palette=["a", "b", "c"])
        assert color_fn(0) != color_fn(10)


def line_shapes(fig):
    return [s for s in fig.layout.shapes if s.type == "line"]


class TestCouplerLayer:
    def test_styles_without_a_coupler_layer_draw_no_edges(self, chip):
        """The default seam: coupler_style is None, so output is byte-identical to
        before couplers existed."""
        fig = visualize(chip, style=default_style)

        assert default_style.coupler_style is None
        assert line_shapes(fig) == []

    def test_enabled_layer_draws_one_line_per_coupler(self, chip):
        chip.generate_uniform_noise(0.01)
        style = with_couplers(default_style, chip, couplers=True)

        fig = visualize(chip, style=style)

        assert len(line_shapes(fig)) == len(chip.couplers) == 81

    def test_edges_are_prepended_so_qubits_draw_on_top(self, chip):
        chip.generate_uniform_noise(0.01)

        fig = visualize(chip, style=with_couplers(default_style, chip, couplers=True))
        shapes = list(fig.layout.shapes)

        assert all(s.type == "line" for s in shapes[: len(chip.couplers)])
        assert all(s.type != "line" for s in shapes[len(chip.couplers) :])

    def test_coupler_layer_does_not_add_a_trace(self, chip):
        """Couplers ride the existing invisible hover trace, so the one-trace contract
        that `visualize_interactive`'s visibility array indexes on still holds."""
        chip.generate_uniform_noise(0.01)

        fig = visualize(chip, style=with_couplers(default_style, chip, couplers=True))

        assert len(fig.data) == 1
        assert len(fig.data[0].x) == len(chip.qubits) + len(chip.couplers)

    def test_coupler_hovertext_reaches_the_trace(self, chip):
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        fig = visualize(chip, style=coupler_heatmap_style(chip))
        hovers = [h for h in fig.data[0].hovertext if h and "<->" in h]

        assert len(hovers) == len(chip.couplers)
        assert any("0.2000" in h for h in hovers)

    def test_hover_points_are_anchored_at_edge_midpoints(self, chip):
        chip.generate_uniform_noise(0.01)

        fig = visualize(chip, style=with_couplers(default_style, chip, couplers=True))
        midpoints = {c.midpoint for c in chip.couplers}

        # couplers come first in the parallel arrays, matching the shape order
        assert set(zip(fig.data[0].x, fig.data[0].y)) >= midpoints


class TestCouplerAutoDetection:
    def test_auto_is_off_while_couplers_are_derived(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)

        fig = visualize(chip, style=noise_heatmap_style(chip))

        assert not chip.has_independent_couplers
        assert line_shapes(fig) == []

    def test_auto_turns_on_once_a_coupler_is_overridden(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        fig = visualize(chip, style=noise_heatmap_style(chip))

        assert len(line_shapes(fig)) == len(chip.couplers)

    def test_explicit_true_overrides_auto_off(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)

        fig = visualize(chip, style=noise_heatmap_style(chip, couplers=True))

        assert len(line_shapes(fig)) == len(chip.couplers)

    def test_explicit_false_overrides_auto_on(self, chip):
        chip.generate_gaussian_noise(0.01, 0.002)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        fig = visualize(chip, style=noise_heatmap_style(chip, couplers=False))

        assert line_shapes(fig) == []

    def test_shared_singletons_are_never_mutated(self, chip):
        chip.generate_uniform_noise(0.01)

        with_couplers(default_style, chip, couplers=True)

        assert default_style.coupler_style is None

    def test_enabled_layer_widens_the_shared_colorbar(self, chip):
        """Qubit and coupler p are the same quantity, so one colorbar spans both."""
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        assert noise_heatmap_style(chip).colorbar.cmax == 0.2
        assert noise_heatmap_style(chip, couplers=False).colorbar.cmax == 0.01


def measured(chip, qubit_p=0.004, coupler_p=0.02):
    """A chip whose couplers carry rates of their own, an order above the qubits."""
    chip.generate_uniform_noise(qubit_p)
    chip.set_coupler_noise_map({c.ends: coupler_p for c in chip.couplers})
    return chip


class TestLogTicks:
    def test_narrow_range_gets_sub_decade_ticks(self):
        import math

        _, text = _log_ticks(math.log10(2e-3), math.log10(1e-2))

        assert text == ["0.002", "0.003", "0.004", "0.006", "0.01"]

    def test_wide_range_falls_back_to_decades(self):
        import math

        _, text = _log_ticks(math.log10(1e-4), math.log10(2e-2))

        assert text == ["0.0001", "0.0003", "0.001", "0.003", "0.01"]

    def test_tick_positions_are_log10_of_their_labels(self):
        import math

        vals, text = _log_ticks(math.log10(2e-3), math.log10(1e-2))

        assert vals == [pytest.approx(math.log10(float(t))) for t in text]


class TestDeviceHeatmapStyle:
    def test_declares_two_independent_colorbars(self, chip):
        style = device_heatmap_style(measured(chip))

        assert style.colorbar is not None
        assert style.coupler_colorbar is not None
        assert style.colorbar.colorscale != style.coupler_colorbar.colorscale

    def test_the_two_scales_do_not_share_a_range(self, chip):
        """The whole point: coupler rates an order above qubit rates would otherwise
        flatten every qubit into the bottom of one shared bar."""
        style = device_heatmap_style(measured(chip, qubit_p=0.004, coupler_p=0.02))

        assert style.colorbar.cmax != style.coupler_colorbar.cmax

    def test_emits_one_trace_per_colorbar(self, chip):
        fig = visualize(chip, style=device_heatmap_style(measured(chip)))

        assert len(fig.data) == 2
        assert [t.marker.showscale for t in fig.data] == [True, True]

    def test_the_colorbars_do_not_overlap(self, chip):
        fig = visualize(chip, style=device_heatmap_style(measured(chip)))

        xs = [t.marker.colorbar.x for t in fig.data]
        assert xs[0] < xs[1]

    def test_reserves_a_strip_per_colorbar(self, chip):
        """The second bar buys its own strip plus a gap for the first bar's side title,
        which would otherwise be drawn straight over it."""
        one = visualize(chip, style=noise_heatmap_style(measured(chip)))
        two = visualize(chip, style=device_heatmap_style(chip))

        assert two.layout.width - one.layout.width == _COLORBAR_PX + _COLORBAR_TITLE_PX

    def test_log_scale_feeds_log10_values_against_log10_bounds(self, chip):
        style = device_heatmap_style(measured(chip, qubit_p=0.004), log=True)

        assert style.colorbar.cmin < 0  # log10(0.004) is negative
        assert style.colorbar.ticktext is not None

    def test_linear_scale_keeps_raw_units(self, chip):
        style = device_heatmap_style(measured(chip, qubit_p=0.004), log=False)

        assert style.colorbar.cmin == pytest.approx(0.004)
        assert style.colorbar.ticktext is None

    def test_labels_are_off_by_default(self, chip):
        fig = visualize(chip, style=device_heatmap_style(measured(chip)))

        assert fig.layout.annotations == ()

    def test_labels_number_every_qubit_in_reading_order(self, chip):
        fig = visualize(chip, style=device_heatmap_style(measured(chip), labels=True))
        notes = list(fig.layout.annotations)

        assert len(notes) == len(chip.qubits)
        assert [n.text for n in notes] == [str(i) for i in range(len(chip.qubits))]
        # index 0 is the top-left qubit; reading order is left-to-right, top-to-bottom
        assert (notes[0].x, notes[0].y) == min(chip.grid, key=lambda c: (c[1], c[0]))

    def test_zero_rates_pin_to_the_bottom_of_the_log_scale(self, chip):
        """A p of 0 has no logarithm; it must clamp rather than raise."""
        chip.generate_uniform_noise(0.004)
        chip.set_coupler_noise_map({c.ends: 0.0 for c in chip.couplers})

        fig = visualize(chip, style=device_heatmap_style(chip))

        assert len(fig.data) == 2


class TestDeviceHeatmapPresets:
    def test_default_preset_matches_the_house_marker_conventions(self, chip):
        """Asserted against QubitStyle's own defaults, not literals, so this tracks the
        house style if it ever moves."""
        style = device_heatmap_style(measured(chip))
        house, actual = QubitStyle(), style.style_fn(chip.qubits[0])

        assert (actual.marker, actual.size, actual.edgecolor, actual.linewidths) == (
            house.marker,
            house.size,
            house.edgecolor,
            house.linewidths,
        )

    def test_default_preset_uses_the_house_qubit_ramp(self, chip):
        style = device_heatmap_style(measured(chip))

        assert style.colorbar.colorscale == "hot_r"
        assert style.colorbar.colorscale != style.coupler_colorbar.colorscale

    def test_default_preset_matches_the_house_logical_outline(self, chip):
        """noise_heatmap_style outlines tiles in blue; the device view must agree."""
        device = device_heatmap_style(measured(chip)).logical_style(chip.tag)
        house = noise_heatmap_style(chip).logical_style(chip.tag)

        assert device.edgecolor == house.edgecolor == "blue"

    def test_default_preset_mounts_bar_titles_at_the_side(self, chip):
        style = device_heatmap_style(measured(chip))

        assert style.colorbar.title_side == "right"
        assert style.coupler_colorbar.title_side == "right"

    def test_device_preset_flips_the_whole_set_together(self, chip):
        style = device_heatmap_style(measured(chip), preset="device")
        marker = style.style_fn(chip.qubits[0])

        assert marker.marker == "o"
        assert style.colorbar.colorscale == "Viridis_r"
        assert style.colorbar.title_side == "top"
        assert style.colorbar.ticktext is not None  # device preset is log
        assert style.logical_style(chip.tag).edgecolor == "red"

    def test_log_follows_the_preset_unless_given(self, chip):
        measured(chip)

        assert device_heatmap_style(chip).colorbar.ticktext is None
        assert device_heatmap_style(chip, preset="device").colorbar.ticktext is not None

    @pytest.mark.parametrize("preset, log", [("qsnow", True), ("device", False)])
    def test_explicit_log_overrides_either_preset(self, chip, preset, log):
        style = device_heatmap_style(measured(chip), preset=preset, log=log)

        assert (style.colorbar.ticktext is not None) is log

    def test_unknown_preset_names_the_valid_ones(self, chip):
        with pytest.raises(ValueError, match="Known presets"):
            device_heatmap_style(chip, preset="nope")

    def test_explicit_colorscales_override_the_preset(self, chip):
        style = device_heatmap_style(measured(chip), qubit_colorscale="Greys")

        assert style.colorbar.colorscale == "Greys"


class TestLabelContrast:
    @pytest.mark.parametrize(
        "fill, expected",
        [
            ("rgb(255, 255, 255)", "black"),  # hot_r's low end
            ("rgb(0, 0, 0)", "white"),        # hot_r's high end
            ("rgb(8, 48, 107)", "white"),     # Blues' high end
            ("lightgray", "white"),           # unparseable named color
            ("", "white"),
        ],
    )
    def test_picks_the_readable_color(self, fill, expected):
        assert _label_color(fill) == expected

    def test_labels_stay_legible_across_the_house_ramp(self, chip):
        """hot_r is white at the low end, so a fixed white label would vanish on every
        good qubit - both colors must appear on a chip that spans the ramp."""
        chip.generate_uniform_noise(0.001)
        for q in chip.qubits[:10]:
            q.noise.p = 0.05
        chip.set_coupler_noise_map({c.ends: 0.02 for c in chip.couplers})

        fig = visualize(chip, style=device_heatmap_style(chip, labels=True))
        colors = {n.font.color for n in fig.layout.annotations if n.text.isdigit()}

        assert colors == {"black", "white"}


class TestFlooredColorscale:
    def test_trims_the_palest_end(self, chip):
        """A bare line with no outline vanishes at Blues' near-white low end."""
        full = _floored_colorscale("Blues", floor=0.0)[0][1]
        floored = _floored_colorscale("Blues", floor=0.25)[0][1]

        assert full != floored
        assert _label_color(full) == "black"    # i.e. the raw low end is very pale
        assert _floored_colorscale("Blues")[-1][1] == _floored_colorscale("Blues", 0.0)[-1][1]

    def test_spans_the_full_zero_to_one_domain(self):
        scale = _floored_colorscale("Blues")

        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0
