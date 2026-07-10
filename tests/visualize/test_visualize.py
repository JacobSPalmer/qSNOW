from qsnow.interface.models import Qubit, Status
from qsnow.visualize.visualize import (
    QubitStyle,
    _default_qubit_style_by_status,
    default_style,
)


class TestDefaultQubitStyle:
    def test_returns_qubit_style(self):
        style = _default_qubit_style_by_status(Qubit(loc=(0, 0), status=Status.LOGICAL))
        assert isinstance(style, QubitStyle)

    def test_color_varies_by_status(self):
        inactive_color = _default_qubit_style_by_status(Qubit(loc=(0, 0), status=Status.INACTIVE)).color
        logical_color = _default_qubit_style_by_status(Qubit(loc=(0, 0), status=Status.LOGICAL)).color
        assert inactive_color != logical_color


def test_default_style_uses_status_based_styling():
    assert default_style.style_fn is _default_qubit_style_by_status
