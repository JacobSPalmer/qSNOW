import pytest

from qsnow.interface.grid import Grid
from qsnow.interface.models import Qubit


class TestGridConstruction:
    @pytest.mark.parametrize("length,height", [(0, 5), (5, 0), (-1, 5)])
    def test_rejects_non_positive_dimensions(self, length, height):
        with pytest.raises(ValueError):
            Grid(length, height)

    def test_bound_reflects_origin_length_height(self):
        grid = Grid(4, 6, origin=(1, 1))
        assert grid.bound == (4, 6)


class TestGridQubitAccess:
    def test_loc_raises_for_missing_coordinate(self):
        grid = Grid(3, 3)
        with pytest.raises(KeyError):
            grid.loc((0, 0))

    def test_loc_returns_stored_qubit(self):
        grid = Grid(3, 3)
        qubit = Qubit(loc=(0, 0))
        grid.grid[(0, 0)] = qubit
        assert grid.loc((0, 0)) is qubit


class TestGridRegionQueries:
    def test_is_empty_region_true_when_all_qubits_inactive(self):
        grid = Grid(5, 5)
        for x, y in [(0, 0), (2, 0), (0, 2), (2, 2)]:
            grid.grid[(x, y)] = Qubit(loc=(x, y))

        assert grid.is_empty_region((0, 0), (2, 2)) is True

    def test_is_empty_region_false_when_a_qubit_is_active(self):
        from qsnow.interface.models import Status

        grid = Grid(5, 5)
        for x, y in [(0, 0), (2, 0), (0, 2), (2, 2)]:
            grid.grid[(x, y)] = Qubit(loc=(x, y))
        grid.loc((2, 2)).status = Status.LOGICAL

        assert grid.is_empty_region((0, 0), (2, 2)) is False
