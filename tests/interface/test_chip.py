import pytest

from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.models import Status


class TestChipConstruction:
    def test_dimensions_are_doubled_for_checkerboard(self, chip: Chip):
        assert chip.length == 10
        assert chip.height == 10

    def test_only_checkerboard_positions_are_populated(self, chip: Chip):
        for x, y in chip.grid.keys():
            assert x % 2 == y % 2

    def test_all_qubits_start_inactive(self, chip: Chip):
        assert all(q.status == Status.INACTIVE for q in chip.qubits)

    def test_construct_chip_from_tile(self, logical_tile: LogicalTile):
        chip = Chip.from_tile(logical_tile)
        assert chip.length == logical_tile.length
        assert chip.height == logical_tile.height
        assert chip.origin == logical_tile.origin == (0, 0)


class TestChipClassProperties:
    def test_get_noise_map(self, chip: Chip):
        for q in chip.qubits:
            if q.loc[0] % 2:
                q.noise.p = 0.05

        assert all(n.p == 0.05 for c, n in chip.noise_map.items() if c[0] % 2)
        assert all(n.p == 0.0 for c, n in chip.noise_map.items() if not c[0] % 2)

    def test_get_tile_map(self, logical_tile: LogicalTile):
        chip = Chip(11, 11)

        tile1 = logical_tile.copy()
        tile2 = logical_tile.copy()
        chip.add_tile(tile1, (0, 0))
        chip.add_tile(tile2, (8, 8))

        map = chip.tile_map
        assert map.get((0, 0)) == tile1
        assert map.get((8, 8)) == tile2

    def test_set_noise_map(self, chip: Chip):
        chip.generate_random_noise()

        chip2 = Chip(chip.length // 2, chip.height // 2)
        chip2.set_noise_map(chip.noise_map)

        assert all(
            n1.p == n2.p
            for n1, n2 in zip(chip.noise_map.values(), chip2.noise_map.values())
        )


class TestNoiseGeneration:
    def test_random_noise_values_land_within_range(self, chip: Chip):
        low, high = 0.01, 0.05
        chip.generate_random_noise((low, high), seed=1)
        # regression: `round()` with no ndigits rounded every sampled p to the
        # nearest int (0), collapsing all noise values to 0.0
        assert all(low <= q.noise.p for q in chip.qubits)
        assert any(q.noise.p != 0.0 for q in chip.qubits)

    def test_random_noise_values_vary_across_qubits(self, chip: Chip):
        chip.generate_random_noise((0.01, 0.05), seed=1)
        assert len({q.noise.p for q in chip.qubits}) > 1

    def test_gaussian_noise_values_are_nonzero(self, chip: Chip):
        chip.generate_gaussian_noise(mean=0.01, deviation=0.005, seed=1)
        assert any(q.noise.p != 0.0 for q in chip.qubits)


class TestTileClassProperties:
    def test_chip_origin_and_bound(self, chip: Chip, logical_tile: LogicalTile):
        chip.add_tile(logical_tile, (2, 2))
        assert logical_tile.origin == (2, 2)
        assert logical_tile.circuit_origin == (2, 2)

        chip.tiles[0].shift_to((0, 0))
        assert logical_tile.origin == (0, 0)
        assert logical_tile.circuit_origin == (0, 0)


class TestChipTilePlacement:
    def test_add_tile_at_default_origin_succeeds(
        self, chip: Chip, logical_tile: LogicalTile
    ):
        assert chip.add_tile(logical_tile) is True
        assert logical_tile in chip.tiles

    def test_add_tile_activates_underlying_qubits(
        self, chip: Chip, logical_tile: LogicalTile
    ):
        chip.add_tile(logical_tile)
        assert any(q.status != Status.INACTIVE for q in chip.qubits)

    def test_add_tile_rejects_overlapping_placement(
        self, chip: Chip, logical_tile: LogicalTile, surface_code_circuit
    ):
        chip.add_tile(logical_tile)
        overlapping_tile = LogicalTile(surface_code_circuit)

        with pytest.warns(UserWarning):
            result = chip.add_tile(overlapping_tile, loc=(0, 0))

        assert result is False

    def test_add_tile_rejects_off_checkerboard_loc(
        self, chip: Chip, logical_tile: LogicalTile
    ):
        with pytest.raises(ValueError):
            chip.add_tile(logical_tile, loc=(1, 0))

    def test_add_tile_rejects_out_of_bounds_loc(
        self, chip: Chip, logical_tile: LogicalTile
    ):
        with pytest.warns(UserWarning):
            chip.add_tile(logical_tile, loc=(10, 10))

    def test_shift_tiles_across_chip(self, logical_tile: LogicalTile):
        chip = Chip(10, 10)
        assert chip.add_tile(logical_tile)

        logical_tile.shift_by(3, 3)
        assert chip.tiles[0].origin == (3, 3)

        logical_tile.shift_by(2, 0)
        assert chip.tiles[0].origin == (5, 3)

        logical_tile.shift_by(0, -2)
        assert chip.tiles[0].origin == (5, 1)

        logical_tile.shift_by(-5, -1)
        assert chip.tiles[0].origin == (0, 0)

    def test_removing_tiles_from_chip(self, lg_chip: Chip, logical_tile: LogicalTile):
        tile1, tile2, tile3 = (logical_tile.copy() for i in range(3))

        lg_chip.add_tile(tile1, (0, 0))
        lg_chip.add_tile(tile2, (8, 0))
        lg_chip.add_tile(tile3, (12, 8))

        t2_region = lg_chip.select_rect(*tile2.origin, *tile2.bound)
        assert all(q.is_active() for q in t2_region.values())
        assert tile2 == lg_chip.pop_tile(1)
        assert all(not q.is_active() for q in t2_region.values())


class TestChipSummary:
    def test_summary_facts(self, chip: Chip):
        summary = chip.summary()
        assert summary["unit_size"] == (5, 5)
        assert summary["grid_size"] == (10, 10)
        assert summary["n_qubits"] == 50
        assert summary["n_tiles"] == 0
        assert summary["noise_model"] is None

    def test_summary_tracks_tiles_and_noise(self, chip: Chip, logical_tile):
        chip.add_tile(logical_tile, (0, 0))
        chip.generate_gaussian_noise(0.01, 0.005, seed=7)
        summary = chip.summary()
        assert summary["n_tiles"] == 1
        assert summary["noise_model"]["name"] == "gaussian"
