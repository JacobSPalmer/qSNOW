import pytest

from qsnow.interface.chip import Chip
from qsnow.interface.models import Status


class TestChipConstruction:
    def test_dimensions_are_doubled_for_checkerboard(self, chip):
        assert chip.length == 10
        assert chip.height == 10

    def test_only_checkerboard_positions_are_populated(self, chip):
        for (x, y) in chip.grid.keys():
            assert x % 2 == y % 2

    def test_all_qubits_start_inactive(self, chip):
        assert all(q.status == Status.INACTIVE for q in chip.qubits)


class TestChipTilePlacement:
    def test_add_tile_at_default_origin_succeeds(self, chip, logical_tile):
        assert chip.add_tile(logical_tile) is True
        assert logical_tile in chip.tiles

    def test_add_tile_activates_underlying_qubits(self, chip, logical_tile):
        chip.add_tile(logical_tile)
        assert any(q.status != Status.INACTIVE for q in chip.qubits)

    def test_add_tile_rejects_overlapping_placement(self, chip, logical_tile, surface_code_circuit):
        from qsnow.interface.chip import LogicalTile

        chip.add_tile(logical_tile)
        overlapping_tile = LogicalTile(surface_code_circuit)

        with pytest.warns(UserWarning):
            result = chip.add_tile(overlapping_tile, loc=(0, 0))

        assert result is False

    def test_add_tile_rejects_off_checkerboard_loc(self, chip, logical_tile):
        with pytest.raises(ValueError):
            chip.add_tile(logical_tile, loc=(1, 0))
