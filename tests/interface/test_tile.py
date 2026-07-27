import pytest
import stim

from qsnow.interface.chip import Chip
from qsnow.interface.codes.rsc import SCTile
from qsnow.interface.models import CSSType
from qsnow.interface.rules import ChannelRule, InjectionRule, Ruleset
from qsnow.interface.tile import LogicalTile


@pytest.fixture
def four_qubit_circuit() -> stim.Circuit:
    return stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(2, 0) 1
        QUBIT_COORDS(0, 2) 2
        QUBIT_COORDS(2, 2) 3
        R 0 1 2 3
        CX 0 1
        M 0 1 2 3
    """)


@pytest.fixture
def two_qubit_circuit() -> stim.Circuit:
    return stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(2, 0) 1
        H 0
    """)


@pytest.fixture
def one_qubit_circuit() -> stim.Circuit:
    return stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        H 0
    """)


def placed(circuit, chip, ruleset=None, *, noise=None):
    """Build a LogicalTile from `circuit`, place it on `chip`, and assign noise."""
    tile = LogicalTile(circuit, x_buffer=1, y_buffer=1, ruleset=ruleset)
    assert chip.add_tile(tile)
    for coord, p in (noise or {}).items():
        chip.loc(coord).noise.p = p
    return tile


def names(circuit) -> list[str]:
    return [instr.name for instr in circuit]


class TestNoiseInjectionOrdering:
    """`_inject_circuit_noise` is the only place noise ever enters a tile's
    circuit; these pin down the `before`/`after` behavior directly so a
    regression like a dropped `before` join is caught immediately."""

    def test_before_channel_is_emitted_ahead_of_the_triggering_instruction(
        self, four_qubit_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "R", "any", before=[ChannelRule("X_ERROR", "all_qubits")]
                )
            ]
        )
        tile = placed(
            four_qubit_circuit,
            chip,
            ruleset,
            noise={(0, 0): 0.01, (2, 0): 0.02, (0, 2): 0.03, (2, 2): 0.04},
        )

        out = list(tile.circuit)
        r_idx = names(tile.circuit).index("R")

        assert [i.name for i in out[r_idx - 4 : r_idx]] == ["X_ERROR"] * 4
        assert [i.gate_args_copy() for i in out[r_idx - 4 : r_idx]] == [
            [0.01],
            [0.02],
            [0.03],
            [0.04],
        ]

    def test_after_channel_is_emitted_following_the_triggering_instruction(
        self, four_qubit_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "CX",
                    "any",
                    after=[
                        ChannelRule("DEPOLARIZE2", "active", scalar=1.5),
                        ChannelRule("DEPOLARIZE1", "idle"),
                    ],
                )
            ]
        )
        tile = placed(
            four_qubit_circuit,
            chip,
            ruleset,
            noise={(0, 0): 0.01, (2, 0): 0.02, (0, 2): 0.03, (2, 2): 0.04},
        )

        out = list(tile.circuit)
        cx_idx = names(tile.circuit).index("CX")

        assert out[cx_idx + 1].name == "DEPOLARIZE2"
        assert out[cx_idx + 1].gate_args_copy() == [pytest.approx(0.0225)]
        assert [t.value for t in out[cx_idx + 1].targets_copy()] == [0, 1]

        assert [i.name for i in out[cx_idx + 2 : cx_idx + 4]] == ["DEPOLARIZE1"] * 2
        assert out[cx_idx + 2].gate_args_copy() == [0.03]
        assert out[cx_idx + 3].gate_args_copy() == [0.04]

    def test_rule_with_failing_trigger_contributes_no_channels(
        self, two_qubit_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "H", "data", before=[ChannelRule("X_ERROR", "all_qubits")]
                )
            ]
        )
        tile = placed(two_qubit_circuit, chip, ruleset, noise={(0, 0): 0.02})

        assert names(tile.circuit) == ["QUBIT_COORDS", "QUBIT_COORDS", "H"]

    def test_exclusive_rule_stops_lower_priority_rules_from_firing(
        self, one_qubit_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "H",
                    "any",
                    after=[ChannelRule("DEPOLARIZE1", "all_qubits")],
                    exclusive=True,
                ),
                InjectionRule("H", "any", after=[ChannelRule("Z_ERROR", "all_qubits")]),
            ]
        )
        tile = placed(one_qubit_circuit, chip, ruleset, noise={(0, 0): 0.02})

        assert names(tile.circuit) == ["QUBIT_COORDS", "H", "DEPOLARIZE1"]

    def test_multiple_channel_rules_in_one_before_list_all_appear(
        self, one_qubit_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "H",
                    "any",
                    before=[
                        ChannelRule("X_ERROR", "all_qubits"),
                        ChannelRule("DEPOLARIZE1", "all_qubits"),
                    ],
                )
            ]
        )
        tile = placed(one_qubit_circuit, chip, ruleset, noise={(0, 0): 0.02})

        assert names(tile.circuit) == [
            "QUBIT_COORDS",
            "X_ERROR",
            "DEPOLARIZE1",
            "H",
        ]

    def test_circuit_end_to_end_via_default_sc_tile_ruleset(self, chip: Chip):
        """Integration check with the real ruleset used in production (rsc.py):
        the final data-qubit `M` has a `before` X_ERROR channel and nothing
        `after`, so an X_ERROR immediately preceding `M` is a direct signal
        that `before`-channel injection actually ran."""
        tile = SCTile(3, rounds=1, task="memory_z")
        assert chip.add_tile(tile)
        chip.generate_uniform_noise(0.02)

        out = list(tile.circuit)
        m_idx = names(tile.circuit).index("M")

        assert out[m_idx - 1].name == "X_ERROR"
        assert out[m_idx - 1].gate_args_copy() == [0.02]
        m_targets = {t.value for t in out[m_idx].targets_copy()}
        pre_targets = {t.value for t in out[m_idx - 1].targets_copy()}
        assert m_targets <= pre_targets


class TestCircuitPropertyAccess:
    def test_circuit_raises_before_tile_is_placed(self, logical_tile: LogicalTile):
        with pytest.raises(AttributeError):
            logical_tile.circuit

    def test_base_circuit_is_available_before_placement(
        self, logical_tile: LogicalTile
    ):
        assert logical_tile.base_circuit is not None


class TestShiftRewritesCircuitAndMetadata:
    def test_shift_by_rewrites_qubit_coords_in_circuit(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)

        tile.shift_by(2, 2)

        coords = tile._circuit.get_final_qubit_coordinates()
        assert coords == {0: [2.0, 2.0], 1: [4.0, 2.0]}

    def test_shift_by_transfers_qubit_status_and_type_to_new_coords(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)
        chip.loc((0, 0)).type = CSSType.DATA

        tile.shift_by(2, 2)

        assert chip.loc((0, 0)).is_active() is False
        assert chip.loc((0, 0)).type == CSSType.UNASSIGNED
        assert chip.loc((2, 2)).is_active() is True
        assert chip.loc((2, 2)).type == CSSType.DATA

    def test_shift_by_c2i_matches_recomputed_map(
        self, two_qubit_circuit, chip: Chip
    ):
        # shift_by builds _c2i by translating the existing map instead of
        # re-walking the circuit; it must equal the fully re-derived map.
        tile = placed(two_qubit_circuit, chip)

        tile.shift_by(2, 2)

        assert tile._c2i == tile._extract_c2i_map()

    def test_shift_to_no_op_when_already_at_target(self, two_qubit_circuit, chip: Chip):
        tile = placed(two_qubit_circuit, chip)

        tile.shift_to(tile.origin)

        assert tile.origin == (0, 0)

    def test_shift_by_rejects_off_checkerboard_shift(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)

        with pytest.raises(ValueError):
            tile.shift_by(1, 0)

    def test_shift_by_rejects_out_of_bounds_shift(self, two_qubit_circuit, chip: Chip):
        tile = placed(two_qubit_circuit, chip)

        with pytest.raises(ValueError):
            tile.shift_by(20, 20)

    def test_shift_by_rejects_overlap_with_another_tile(
        self, two_qubit_circuit, chip: Chip
    ):
        placed(two_qubit_circuit, chip)
        other = LogicalTile(two_qubit_circuit.copy(), x_buffer=1, y_buffer=1)
        assert chip.add_tile(other, (4, 0))

        with pytest.raises(ValueError):
            other.shift_by(-4, 0)


class TestTileGeometry:
    def test_circuit_origin_and_bound_reflect_qubit_extents(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)

        assert tile.circuit_origin == (0.0, 0.0)
        assert tile.circuit_bound == (2.0, 0.0)

    def test_qubit_at_index_returns_matching_qubit(self, two_qubit_circuit, chip: Chip):
        tile = placed(two_qubit_circuit, chip)

        q = tile.qubit_at_index(1)

        assert q is chip.loc((2, 0))

    def test_qubit_at_index_returns_none_for_unknown_index(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)

        assert tile.qubit_at_index(99) is None


class TestUnplacedTileAccessRaises:
    def test_chip_raises_before_placement(self, logical_tile: LogicalTile):
        with pytest.raises(AttributeError):
            logical_tile.chip

    def test_qubits_raises_before_placement(self, logical_tile: LogicalTile):
        with pytest.raises(AttributeError):
            logical_tile.qubits

    def test_initialized_false_before_placement_true_after(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = LogicalTile(two_qubit_circuit)
        assert tile.initialized() is False

        chip.add_tile(tile)
        assert tile.initialized() is True


class TestResetAndConstruction:
    def test_reset_detaches_from_chip_and_scrubs_qubits(
        self, two_qubit_circuit, chip: Chip
    ):
        tile = placed(two_qubit_circuit, chip)
        qubits = list(tile.qubits)

        tile.reset()

        assert tile.initialized() is False
        assert all(not q.is_active() for q in qubits)

    @pytest.mark.parametrize("x_buffer,y_buffer", [(-1, 1), (1, -1)])
    def test_negative_buffer_rejected(self, two_qubit_circuit, x_buffer, y_buffer):
        with pytest.raises(ValueError):
            LogicalTile(two_qubit_circuit, x_buffer=x_buffer, y_buffer=y_buffer)
