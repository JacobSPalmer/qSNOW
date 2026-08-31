import pytest
import stim

from qsnow.interface.lattice import (
    CHECKERBOARD,
    SQUARE,
    Lattice,
    known_lattices,
    lattice_by_name,
    register_lattice,
)


class TestSites:
    def test_checkerboard_occupies_matching_parity_only(self):
        assert CHECKERBOARD.is_site((0, 0))
        assert CHECKERBOARD.is_site((1, 1))
        assert not CHECKERBOARD.is_site((1, 0))
        assert not CHECKERBOARD.is_site((0, 1))

    def test_square_occupies_every_integer_coordinate(self):
        assert all(SQUARE.is_site(c) for c in [(0, 0), (1, 0), (0, 1), (3, 7)])

    def test_is_site_accepts_the_floats_stim_emits(self):
        assert CHECKERBOARD.is_site((2.0, 4.0))
        assert not CHECKERBOARD.is_site((3.0, 4.0))

    def test_positions_counts_match_the_pitch(self):
        # a 10x10 coordinate extent: half the points on a checkerboard, all of them dense
        assert len(list(CHECKERBOARD.positions(10, 10))) == 50
        assert len(list(SQUARE.positions(10, 10))) == 100

    def test_positions_are_all_sites(self):
        assert all(CHECKERBOARD.is_site(c) for c in CHECKERBOARD.positions(7, 7))


class TestUnitCells:
    @pytest.mark.parametrize("lattice", [CHECKERBOARD, SQUARE])
    def test_span_and_units_round_trip(self, lattice: Lattice):
        assert lattice.units(*lattice.span(5, 8)) == (5, 8)

    def test_checkerboard_doubles_unit_cells(self):
        assert CHECKERBOARD.span(5, 5) == (10, 10)

    def test_square_spans_one_coordinate_per_cell(self):
        assert SQUARE.span(5, 5) == (5, 5)

    def test_keepout_margin_is_half_a_cell(self):
        assert CHECKERBOARD.keepout_margin == 1.0
        assert SQUARE.keepout_margin == 0.5


class TestConstruction:
    def test_rejects_nonpositive_pitch(self):
        with pytest.raises(ValueError):
            Lattice(name="degenerate", pitch=0)

    def test_lattices_compare_by_value_and_hash(self):
        assert Lattice(name="checkerboard", pitch=2) == CHECKERBOARD
        assert len({CHECKERBOARD, Lattice(name="checkerboard", pitch=2)}) == 1

    def test_subclass_may_override_is_site(self):
        class EvenOnly(Lattice):
            def is_site(self, coord):
                return coord[0] % 2 == 0 and coord[1] % 2 == 0

        lattice = EvenOnly(name="even", pitch=2)
        assert lattice.is_site((2, 4))
        assert not lattice.is_site((1, 1))


class TestRegistry:
    def test_builtin_lattices_are_registered(self):
        assert lattice_by_name("checkerboard") is CHECKERBOARD
        assert lattice_by_name("square") is SQUARE
        assert {"checkerboard", "square"} <= set(known_lattices())

    def test_unknown_name_raises_a_directed_error(self):
        with pytest.raises(KeyError, match="register_lattice"):
            lattice_by_name("hexagonal")

    def test_registered_lattice_is_retrievable(self):
        custom = Lattice(name="triple", pitch=3)
        register_lattice(custom)
        try:
            assert lattice_by_name("triple") is custom
        finally:
            from qsnow.interface.lattice import _LATTICES

            _LATTICES.pop("triple")


class TestInference:
    def test_infers_checkerboard_from_a_rotated_surface_code(self, surface_code_circuit):
        coords = surface_code_circuit.get_final_qubit_coordinates().values()

        assert Lattice.infer(coords) is CHECKERBOARD

    def test_infers_square_when_parity_is_violated(self):
        assert Lattice.infer([(0, 0), (1, 0), (0, 1), (1, 1)]) is SQUARE

    def test_raises_on_too_few_coordinates_to_discriminate(self):
        with pytest.raises(ValueError, match="too few"):
            Lattice.infer([(0, 0), (2, 2)])

    def test_raises_on_a_single_row(self):
        # a repetition-code chain satisfies every lattice, so it proves nothing
        with pytest.raises(ValueError, match="single row or column"):
            Lattice.infer([(0, 0), (2, 0), (4, 0), (6, 0)])

    def test_raises_on_non_integer_coordinates(self):
        circuit = stim.Circuit("""
            QUBIT_COORDS(0.5, 0.5) 0
            QUBIT_COORDS(1.5, 0.5) 1
            QUBIT_COORDS(0.5, 1.5) 2
            QUBIT_COORDS(1.5, 1.5) 3
        """)
        coords = circuit.get_final_qubit_coordinates().values()

        with pytest.raises(ValueError, match="initial_shift"):
            Lattice.infer(coords)
