from statistics import mean

import numpy as np
import pytest
from scipy import stats

from qsnow.interface.chip import Chip, LogicalTile
from qsnow.interface.lattice import CHECKERBOARD, SQUARE
from qsnow.interface.models import NoiseProfile, Status
from qsnow.interface.noise import RandomGaussian, SkewContour, p_bounds


class TestSquareLatticeChip:
    """A dense integer lattice: Chip(L, H) spans L x H coordinates, all occupied."""

    def test_dimensions_are_not_doubled(self, square_chip: Chip):
        assert (square_chip.length, square_chip.height) == (5, 5)
        assert square_chip.unit_dims == (5, 5)

    def test_every_integer_coordinate_holds_a_qubit(self, square_chip: Chip):
        assert len(square_chip.qubits) == 25
        assert set(square_chip.grid) == {(x, y) for x in range(5) for y in range(5)}

    def test_summary_reports_the_lattice(self, square_chip: Chip):
        assert square_chip.summary()["lattice"] == "square"
        assert square_chip.summary()["n_qubits"] == 25

    def test_copy_preserves_the_lattice(self, square_chip: Chip):
        # SquarePackingExp copies its chip, so losing the lattice here would
        # silently turn a square-lattice experiment into a checkerboard one
        copied = square_chip.copy()

        assert copied.lattice == SQUARE
        assert len(copied.qubits) == 25

    def test_accepts_a_mixed_parity_origin(self, square_chip: Chip, dense_circuit):
        tile = LogicalTile(dense_circuit, lattice=SQUARE)

        assert square_chip.add_tile(tile, (1, 0))

    def test_single_coordinate_shift_is_legal(self, square_chip: Chip, dense_circuit):
        tile = LogicalTile(dense_circuit, lattice=SQUARE)
        assert square_chip.add_tile(tile, (0, 0))

        tile.shift_by(1, 0)

        assert tile.origin == (1, 0)


class TestCrossLatticePlacement:
    """Lattices nest - every checkerboard site is a square site, not the reverse."""

    def test_checkerboard_tile_places_on_a_square_chip(self, logical_tile):
        chip = Chip(10, 10, lattice=SQUARE)

        assert chip.add_tile(logical_tile, (0, 0))

    def test_dense_tile_is_rejected_on_a_checkerboard_chip(self, chip: Chip, dense_circuit):
        tile = LogicalTile(dense_circuit, lattice=SQUARE)

        with pytest.raises(ValueError, match="off-lattice"):
            chip.add_tile(tile, (0, 0))

    def test_rejection_message_names_both_lattices(self, chip: Chip, dense_circuit):
        tile = LogicalTile(dense_circuit, lattice=SQUARE)

        with pytest.raises(ValueError, match="'square'.*'checkerboard'"):
            chip.add_tile(tile, (0, 0))


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

    def test_random_noise_upper_bound_is_the_range_end(self, lg_chip: Chip):
        # regression: `range[1]` was passed as scipy's `scale` (the width), so
        # (low, high) actually sampled [low, low + high]
        low, high = 0.01, 0.02
        lg_chip.generate_random_noise((low, high), seed=1)
        assert all(low <= q.noise.p <= high for q in lg_chip.qubits)
        assert max(q.noise.p for q in lg_chip.qubits) > (low + high) / 2

    def test_random_noise_values_vary_across_qubits(self, chip: Chip):
        chip.generate_random_noise((0.01, 0.05), seed=1)
        assert len({q.noise.p for q in chip.qubits}) > 1

    def test_gaussian_noise_values_are_nonzero(self, chip: Chip):
        chip.generate_gaussian_noise(mean=0.01, deviation=0.005, seed=1)
        assert any(q.noise.p != 0.0 for q in chip.qubits)

    def test_skewed_contour_assigns_every_qubit_within_bounds(self, chip: Chip):
        chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        lo, hi = p_bounds()
        assert all(lo <= q.noise.p <= hi for q in chip.qubits)
        assert all(q.noise.p > 0.0 for q in chip.qubits)

    def test_skewed_contour_is_seed_reproducible(self, chip: Chip):
        chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        first = [q.noise.p for q in chip.qubits]
        chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        assert [q.noise.p for q in chip.qubits] == first

    def test_skewed_contour_is_right_skewed(self, lg_chip: Chip):
        lg_chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        assert stats.skew([q.noise.p for q in lg_chip.qubits]) > 0.5

    def test_skewed_contour_shares_the_derived_contour_landscape(self, lg_chip: Chip):
        # Same seed and slope must give the same peaks and valleys: only the marginal
        # distribution differs, so the rank order of qubits is identical.
        lg_chip.generate_derived_contour_noise(0.01, 0.003, seed=3)
        gaussian = [q.noise.p for q in lg_chip.qubits]
        lg_chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3)
        skewed = [q.noise.p for q in lg_chip.qubits]
        assert stats.spearmanr(gaussian, skewed).statistic == pytest.approx(1.0)

    def test_skewed_contour_median_center_pins_the_median(self, lg_chip: Chip):
        lg_chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3, center="median")
        assert np.median([q.noise.p for q in lg_chip.qubits]) == pytest.approx(0.01, rel=0.1)


class TestCouplerNoiseGeneration:
    """Chip-level behaviour of `generate_coupler_noise`; the distribution semantics
    (correlation, marginals, baselines) live in tests/interface/noise/."""

    @pytest.fixture
    def landscape(self) -> Chip:
        c = Chip(6, 6)
        c.generate_noise(SkewContour(0.01, 0.003, 1.5, seed=3))
        return c

    def test_assigns_every_coupler_within_bounds(self, landscape: Chip):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.5)
        lo, hi = p_bounds()
        assert all(lo <= c.noise.p <= hi for c in landscape.couplers)
        assert landscape.has_independent_couplers

    def test_records_the_distribution_and_its_correlation(self, landscape: Chip):
        dist = SkewContour(0.05, 0.01, 1.0, seed=4)
        landscape.generate_coupler_noise(dist, correlation=0.6)
        assert landscape.spec.coupler_model == dist
        assert landscape.spec.coupler_correlation == 0.6
        summary = landscape.summary()
        assert summary["coupler_model"]["name"] == "skewed contour"
        assert summary["coupler_correlation"] == 0.6

    def test_a_site_generator_afterwards_re_derives_and_drops_the_record(self, landscape: Chip):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.5)
        landscape.generate_noise(RandomGaussian(0.01, 0.002, seed=1))
        assert landscape.spec.coupler_model is None
        assert landscape.spec.coupler_correlation is None
        assert not landscape.has_independent_couplers

    def test_a_hand_override_drops_the_record(self, landscape: Chip):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.5)
        landscape.set_coupler_noise_map({landscape.couplers[0].ends: 0.2})
        assert landscape.spec.coupler_model is None

    def test_copy_keeps_the_record(self, landscape: Chip):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.5)
        clone = landscape.copy()
        assert clone.spec.coupler_model == landscape.spec.coupler_model
        assert clone.spec.coupler_correlation == 0.5

    def test_setters_copy_whole_profiles(self, chip: Chip):
        # regression guard for the granular-profile future: a profile handed to the chip
        # is copied as a profile, not rebuilt from its `p`
        source = NoiseProfile(0.02)
        chip.set_noise_map({(0, 0): source})
        assert chip.loc((0, 0)).noise == source and chip.loc((0, 0)).noise is not source


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

    def test_abutting_placement_is_order_dependent(
        self, lg_chip: Chip, logical_tile: LogicalTile
    ):
        """Characterization, NOT desired behavior.

        A tile's keep-out region extends one coordinate past its footprint on the
        +x/+y sides only. Two tiles that abut at x=8 therefore place fine in
        left-to-right order, but the same pair is rejected right-to-left because
        the second tile's keep-out reaches into the first tile's qubits.
        Pinned so the footprint refactor is provably inert; the asymmetry itself
        is a known wart tracked separately.
        """
        left, right = logical_tile.copy(), logical_tile.copy()

        assert lg_chip.add_tile(right, (8, 0))
        with pytest.warns(UserWarning):
            assert lg_chip.add_tile(left, (0, 0)) is False

        fresh = Chip(10, 10)
        assert fresh.add_tile(logical_tile.copy(), (0, 0))
        assert fresh.add_tile(logical_tile.copy(), (8, 0))

    def test_removing_tiles_from_chip(self, lg_chip: Chip, logical_tile: LogicalTile):
        tile1, tile2, tile3 = (logical_tile.copy() for i in range(3))

        lg_chip.add_tile(tile1, (0, 0))
        lg_chip.add_tile(tile2, (8, 0))
        lg_chip.add_tile(tile3, (12, 8))

        t2_region = lg_chip.select_rect(*tile2.origin, *tile2.bound)
        assert all(q.is_active() for q in t2_region.values())
        assert tile2 == lg_chip.pop_tile(1)
        assert all(not q.is_active() for q in t2_region.values())


    def test_remove_tile_takes_that_tile_off_by_identity(
        self, lg_chip: Chip, logical_tile: LogicalTile
    ):
        first, second = logical_tile.copy(), logical_tile.copy()
        lg_chip.add_tile(first, (0, 0))
        lg_chip.add_tile(second, (10, 10))

        assert lg_chip.remove_tile(second) is second
        assert lg_chip.tiles == [first]
        assert not second.initialized()
        with pytest.raises(ValueError):
            lg_chip.remove_tile(second)

    def test_clear_tiles_removes_every_tile(self, lg_chip: Chip, logical_tile: LogicalTile):
        # regression: popping by index while enumerating skipped every other tile
        for loc in [(0, 0), (8, 0), (12, 8)]:
            lg_chip.add_tile(logical_tile.copy(), loc)

        lg_chip.clear_tiles()

        assert lg_chip.tiles == []
        assert all(not q.is_active() for q in lg_chip.qubits)


class TestChipSpec:
    def test_generators_record_a_typed_noise_model(self, chip: Chip):
        chip.generate_gaussian_noise(mean=0.01, deviation=0.002, seed=7)

        assert chip.spec.noise_model == RandomGaussian(mean=0.01, deviation=0.002, seed=7)
        assert chip.summary()["noise_model"] == {
            "name": "gaussian", "mean": 0.01, "deviation": 0.002, "seed": 7
        }

    def test_skewed_contour_records_its_shape_parameters(self, chip: Chip):
        chip.generate_skewed_contour_noise(0.01, 0.003, 1.5, seed=3, center="median")

        assert chip.spec.noise_model == SkewContour(0.01, 0.003, 1.5, "median", seed=3)
        assert chip.summary()["noise_model"] == {
            "name": "skewed contour", "location": 0.01, "deviation": 0.003, "skew": 1.5,
            "center": "median", "slope": 5, "seed": 3,
        }
        assert chip.summary()["noise_model"]["name"] == "skewed contour"

    def test_tool_state_stays_out_of_tag_metadata(self, chip: Chip):
        chip.generate_gaussian_noise(mean=0.01, deviation=0.002, seed=7)
        chip.derive_coupler_noise("max")

        assert chip.tag.metadata == {}

    def test_copy_does_not_share_tag_or_spec(self, chip: Chip):
        # regression: copies shared the original's Tag, so deriving couplers on a
        # copy rewrote the original's recorded mode
        chip.generate_uniform_noise(0.01)
        clone = chip.copy()

        clone.derive_coupler_noise("max")
        clone.tag.metadata["note"] = "only on the clone"

        assert chip.spec.coupler_mode == "mean"
        assert clone.spec.coupler_mode == "max"
        assert "note" not in chip.tag.metadata
        assert clone.spec.noise_model == chip.spec.noise_model


class TestPlacementRejection:
    def test_valid_placement_has_no_rejection(self, chip: Chip, logical_tile):
        assert chip.placement_rejection(logical_tile, (0, 0)) is None

    def test_off_lattice_origin_is_fatal(self, chip: Chip, logical_tile):
        rejection = chip.placement_rejection(logical_tile, (1, 0))
        assert rejection is not None and rejection.fatal

    def test_occupied_site_is_not_fatal(self, chip: Chip, logical_tile):
        chip.add_tile(logical_tile, (0, 0))
        rejection = chip.placement_rejection(logical_tile.copy(), (0, 0))
        assert rejection is not None and not rejection.fatal

    def test_overflow_is_not_fatal(self, chip: Chip, logical_tile):
        rejection = chip.placement_rejection(logical_tile, (10, 10))
        assert rejection is not None and not rejection.fatal

    def test_is_valid_agrees_with_rejection(self, chip: Chip, logical_tile):
        chip.add_tile(logical_tile.copy(), (0, 0))
        for loc in [(0, 0), (1, 0), (2, 2), (6, 0), (10, 10)]:
            assert chip.is_valid_tile_placement(logical_tile, loc) == (
                chip.placement_rejection(logical_tile, loc) is None
            )

    def test_ignoring_discounts_a_tiles_own_footprint(self, chip: Chip, logical_tile):
        chip.add_tile(logical_tile, (0, 0))
        own = (logical_tile.origin, logical_tile.bound)

        assert chip.placement_rejection(logical_tile, (2, 2)) is not None
        assert chip.placement_rejection(logical_tile, (2, 2), ignoring=own) is None


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


class TestChipCouplers:
    def test_couplers_are_built_from_the_lattice(self, chip: Chip):
        # Chip(5,5) on the checkerboard spans 10x10 coords -> a 9x9 grid of diagonal links
        assert len(chip.couplers) == 81

    def test_square_lattice_couples_cardinal_neighbours(self, square_chip: Chip):
        assert len(square_chip.couplers) == 40
        assert square_chip.coupler((0, 0), (0, 1)) is not None

    def test_coupler_lookup_is_order_independent(self, chip: Chip):
        assert chip.coupler((0, 0), (1, 1)) is chip.coupler((1, 1), (0, 0))

    def test_coupler_raises_for_an_unlinked_pair(self, chip: Chip):
        with pytest.raises(KeyError, match="No coupler between"):
            chip.coupler((0, 0), (2, 2))

    def test_find_coupler_returns_none_instead_of_raising(self, chip: Chip):
        assert chip.find_coupler((0, 0), (2, 2)) is None
        assert chip.find_coupler((0, 0), (1, 1)) is not None

    def test_every_coupler_joins_two_real_qubits(self, chip: Chip):
        for coupler in chip.couplers:
            assert all(e in chip.grid for e in coupler.ends)


class TestCouplerNoise:
    def test_generators_derive_couplers_from_their_endpoints(self, chip: Chip):
        chip.generate_uniform_noise(0.01)

        assert all(c.noise.p == 0.01 for c in chip.couplers)

    def test_derived_mean_reproduces_the_endpoint_average(self, chip: Chip):
        chip.generate_random_noise()

        assert all(
            c.noise.p == pytest.approx(mean([chip.loc(e).noise.p for e in c.ends]))
            for c in chip.couplers
        )

    @pytest.mark.parametrize(
        "mode, expected", [("mean", 0.03), ("max", 0.05), ("min", 0.01)]
    )
    def test_derive_modes_combine_the_endpoints(self, chip: Chip, mode, expected):
        chip.loc((0, 0)).noise.p = 0.01
        chip.loc((1, 1)).noise.p = 0.05

        chip.derive_coupler_noise(mode)

        assert chip.coupler((0, 0), (1, 1)).noise.p == pytest.approx(expected)

    def test_derive_records_the_model_in_metadata(self, chip: Chip):
        chip.derive_coupler_noise("max")

        assert chip.spec.coupler_mode == "max"

    def test_set_coupler_noise_map_accepts_floats_and_profiles(self, chip: Chip):
        chip.set_coupler_noise_map({((0, 0), (1, 1)): 0.2})
        chip.set_coupler_noise_map({((2, 2), (3, 3)): NoiseProfile(0.3)})

        assert chip.coupler((0, 0), (1, 1)).noise.p == 0.2
        assert chip.coupler((2, 2), (3, 3)).noise.p == 0.3

    def test_set_noise_map_leaves_coupler_overrides_alone(self, chip: Chip):
        """A targeted qubit edit must not silently discard a hand-set coupler rate;
        `derive_coupler_noise()` is the explicit way to re-derive."""
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        chip.set_noise_map({(0, 0): 0.05})

        assert chip.coupler((0, 0), (1, 1)).noise.p == 0.2

    def test_copy_preserves_couplers(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        clone = chip.copy()

        assert {e: n.p for e, n in clone.coupler_map.items()} == {
            e: n.p for e, n in chip.coupler_map.items()
        }

    def test_copy_couplers_are_independent_objects(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        clone = chip.copy()

        clone.coupler((0, 0), (1, 1)).noise.p = 0.5

        assert chip.coupler((0, 0), (1, 1)).noise.p == 0.01


class TestIndependentCouplerDetection:
    def test_generators_leave_couplers_derived(self, chip: Chip):
        chip.generate_gaussian_noise(0.01, 0.002)

        assert chip.has_independent_couplers is False

    def test_a_fresh_chip_has_no_independent_couplers(self, chip: Chip):
        assert chip.has_independent_couplers is False

    def test_a_direct_write_makes_a_coupler_independent(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        assert chip.has_independent_couplers is True

    def test_set_coupler_noise_map_makes_couplers_independent(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        chip.set_coupler_noise_map({((0, 0), (1, 1)): 0.2})

        assert chip.has_independent_couplers is True

    def test_regenerating_noise_keeps_the_derivation_mode(self, chip: Chip):
        # regression: generators re-derived with the default `mean`, silently
        # discarding a `max`/`min` choice recorded on the chip
        chip.generate_uniform_noise(0.01)
        chip.derive_coupler_noise("max")

        chip.generate_gaussian_noise(0.01, 0.002, seed=1)

        assert chip.spec.coupler_mode == "max"
        assert all(
            c.noise.p == max(chip.loc(e).noise.p for e in c.ends) for c in chip.couplers
        )
        assert chip.has_independent_couplers is False

    def test_re_deriving_clears_independence(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        chip.derive_coupler_noise()

        assert chip.has_independent_couplers is False

    def test_detection_respects_the_recorded_derivation_mode(self, chip: Chip):
        """A chip derived with `max` is not 'independent' just because its rates differ
        from the mean."""
        chip.loc((0, 0)).noise.p = 0.01
        chip.loc((1, 1)).noise.p = 0.05
        chip.derive_coupler_noise("max")

        assert chip.has_independent_couplers is False

    def test_summary_reports_couplers(self, chip: Chip):
        summary = chip.summary()

        assert summary["n_couplers"] == 81
        assert summary["independent_couplers"] is False
