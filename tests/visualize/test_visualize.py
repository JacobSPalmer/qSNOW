from qsnow.interface.models import Qubit, Status
from qsnow.visualize.visualize import (
    QubitStyle,
    _default_qubit_style_by_status,
    default_style,
    noise_heatmap_style,
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
