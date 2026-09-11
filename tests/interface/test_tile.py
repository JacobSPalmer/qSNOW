import pytest
import sinter
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


NOISE_CHANNELS = {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}


def noise_before(instructions, idx) -> dict:
    """The contiguous run of noise channels immediately preceding `idx`, by name.

    A rule's `before` list can hold several channels, so the triggering operation
    is not necessarily adjacent to any one of them; this collects the whole
    injected block without depending on the order within it.
    """
    block = {}
    i = idx - 1
    while i >= 0 and instructions[i].name in NOISE_CHANNELS:
        block[instructions[i].name] = instructions[i]
        i -= 1
    return block


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

    def test_exclusive_rule_whose_trigger_fails_does_not_block_lower_rules(
        self, one_qubit_circuit, chip: Chip
    ):
        # regression: exclusivity broke the scan on an operation-name match even
        # when the trigger failed, so lower-priority rules never fired
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "H",
                    "data",  # the bare tile's qubit is untyped, so this fails
                    after=[ChannelRule("DEPOLARIZE1", "all_qubits")],
                    exclusive=True,
                ),
                InjectionRule("H", "any", after=[ChannelRule("Z_ERROR", "all_qubits")]),
            ]
        )
        tile = placed(one_qubit_circuit, chip, ruleset, noise={(0, 0): 0.02})

        assert names(tile.circuit) == ["QUBIT_COORDS", "H", "Z_ERROR"]

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
        """Integration check with the real SI1000 ruleset used in production (rsc.py):
        the final data-qubit `M` carries `before` channels and nothing `after`, so the
        noise block directly preceding `M` is a direct signal that `before`-channel
        injection ran -- and that SI1000's scalars were applied to the qubit's `p`."""
        p = 0.02
        tile = SCTile(3, rounds=1, task="memory_z")
        assert chip.add_tile(tile)
        chip.generate_uniform_noise(p)

        out = list(tile.circuit)
        m_idx = names(tile.circuit).index("M")
        before = noise_before(out, m_idx)

        # SI1000: Measure(p) -> 5p, ResonatorIdle(p) -> 2p
        assert set(before) == {"X_ERROR", "DEPOLARIZE1"}
        assert before["X_ERROR"].gate_args_copy() == [pytest.approx(5 * p)]
        assert before["DEPOLARIZE1"].gate_args_copy() == [pytest.approx(2 * p)]

        m_targets = {t.value for t in out[m_idx].targets_copy()}
        measured = {t.value for t in before["X_ERROR"].targets_copy()}
        idle = {t.value for t in before["DEPOLARIZE1"].targets_copy()}

        assert m_targets <= measured
        # the resonator-idle channel is for qubits that are *not* being measured
        assert not (m_targets & idle)


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

    def test_shift_rejects_single_coordinate_overhang(self, chip: Chip):
        """Regression: shift_by used to feed an inclusive bound to an exclusive-form
        bounds check, letting a one-coordinate overhang slip through validation and
        die later with a KeyError inside _transfer_qubit_metadata."""
        tile = SCTile(3)
        assert chip.add_tile(tile, (0, 0))

        # (3,3) puts the tile's bound at (10,10) on a chip bounded at (9,9);
        # add_tile already rejects it, so shift_to must too.
        with pytest.raises(ValueError):
            tile.shift_to((3, 3))

    def test_region_queries_follow_the_tile_after_a_shift(self, chip: Chip):
        # regression: the tile's spatial index was built over its pre-shift qubits
        # and never invalidated, so a query after shifting raised KeyError
        tile = SCTile(3)
        assert chip.add_tile(tile, (0, 0))
        before = set(tile.select_rect(0, 0, 20, 20))

        tile.shift_by(2, 2)

        after = set(tile.select_rect(0, 0, 20, 20))
        assert after == {(x + 2, y + 2) for x, y in before}
        assert all(c in tile.grid for c in after)

    def test_popped_tile_can_be_placed_again_somewhere_else(self, chip: Chip):
        # regression: reset() restored the base circuit but kept the origin of the
        # last placement, so the next add_tile shifted the circuit by a stale offset
        tile = SCTile(3)
        assert chip.add_tile(tile, (2, 2))
        chip.pop_tile(0)

        assert tile.origin == tile.circuit_origin
        assert chip.add_tile(tile, (0, 0))
        assert tile.origin == (0, 0)
        assert all(c in chip.grid for c in tile.grid)

    def test_reset_tile_holds_no_chip_qubits(self, chip: Chip):
        tile = SCTile(3)
        chip.add_tile(tile, (0, 0))

        chip.pop_tile(0)

        assert tile.grid == {}
        assert tile.select_rect(0, 0, 20, 20) == {}

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


class TestInjectionMatchesFlattened:
    """Noise injection now walks the circuit *without* flattening it, recursing
    into ``REPEAT`` blocks (see ``_process_circuit``) to save compute on codes
    with many rounds. These tests pin down that the repeat-preserving injection
    produces a circuit that is *functionally identical* to injecting noise on a
    fully flattened circuit -- the old behavior -- so sampling is unchanged."""

    def _placed_sctile(self, chip: Chip) -> SCTile:
        tile = SCTile(distance=3, rounds=3)
        chip.add_tile(tile, (0, 0))
        # Non-uniform noise so channels are actually injected (and so a mistake
        # in per-round injection would surface as differing channel arguments).
        chip.generate_random_noise()
        return tile

    def test_repeat_block_is_preserved_in_injected_circuit(self, chip: Chip):
        """Guards that the code path under test is exercised: injection must keep
        the ``REPEAT`` block rather than unrolling it."""
        tile = self._placed_sctile(chip)
        assert "REPEAT" in str(tile.circuit)

    def test_injected_circuit_samples_identically_to_flattened(self, chip: Chip):
        """``strong_id`` is a deterministic hash of the (flattened) circuit, so
        equal ids guarantee identical sampling under sinter. The reference is the
        same tile injected on an already-flattened circuit, which never enters
        the ``REPEAT`` branch -- the trusted per-instruction path."""
        tile = self._placed_sctile(chip)

        # Repeat-preserving injection (the new behavior), unrolled by STIM.
        repeat_injected = tile.circuit.flattened()

        # Flatten-then-inject reference (the old behavior), built from the same
        # tile state so both share identical per-qubit noise.
        flat_injected = stim.Circuit(
            "\n".join(
                tile._process_circuit(
                    tile._circuit.flattened(), tile._extract_i2q_map()
                )
            )
        )

        # STIM structural equality -- clear, fast, and human-readable.
        assert repeat_injected == flat_injected
        # ... and the sampling-level invariant the sampler actually consumes:
        # strong_id hashes the circuit, its detector error model, the decoder,
        # and metadata, so equal ids guarantee sinter samples them identically.
        assert self._strong_id(repeat_injected) == self._strong_id(flat_injected)

    @staticmethod
    def _strong_id(circuit: stim.Circuit) -> str:
        return sinter.Task(
            circuit=circuit,
            detector_error_model=circuit.detector_error_model(),
            decoder="pymatching",
            json_metadata={},
        ).strong_id()


@pytest.fixture
def coupled_pair_circuit() -> stim.Circuit:
    """A CX across two checkerboard-adjacent sites, so the pair has a real coupler."""
    return stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 1) 1
        R 0 1
        CX 0 1
        M 0 1
    """)


class TestCouplerSourcedInjection:
    def test_coupler_source_prices_the_gate_off_the_edge(
        self, coupled_pair_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "CX",
                    "any",
                    after=[ChannelRule("DEPOLARIZE2", "active", source="coupler")],
                )
            ]
        )
        tile = placed(
            coupled_pair_circuit, chip, ruleset, noise={(0, 0): 0.01, (1, 1): 0.03}
        )
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        out = list(tile.circuit)
        cx_idx = names(tile.circuit).index("CX")

        # 0.2 from the coupler, not 0.02 from the endpoint mean
        assert out[cx_idx + 1].gate_args_copy() == [pytest.approx(0.2)]

    def test_coupler_source_still_honours_the_scalar(
        self, coupled_pair_circuit, chip: Chip
    ):
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "CX",
                    "any",
                    after=[
                        ChannelRule(
                            "DEPOLARIZE2", "active", scalar=1.5, source="coupler"
                        )
                    ],
                )
            ]
        )
        tile = placed(coupled_pair_circuit, chip, ruleset)
        chip.coupler((0, 0), (1, 1)).noise.p = 0.2

        out = list(tile.circuit)
        cx_idx = names(tile.circuit).index("CX")

        assert out[cx_idx + 1].gate_args_copy() == [pytest.approx(0.3)]

    def test_derived_couplers_reproduce_the_endpoint_mean(
        self, coupled_pair_circuit, chip: Chip
    ):
        """The migration guarantee: with couplers left at their derived default, a rule
        on the coupler source emits exactly what the endpoint mean used to."""
        chip.generate_uniform_noise(0.01)
        chip.loc((0, 0)).noise.p = 0.01
        chip.loc((1, 1)).noise.p = 0.03
        chip.derive_coupler_noise()

        rule = lambda source: Ruleset(
            injection_rules=[
                InjectionRule(
                    "CX",
                    "any",
                    after=[ChannelRule("DEPOLARIZE2", "active", source=source)],
                )
            ]
        )
        tile = placed(coupled_pair_circuit, chip, rule("coupler"))
        coupler_out = list(tile.circuit)[names(tile.circuit).index("CX") + 1]
        chip.pop_tile(0)

        tile = placed(coupled_pair_circuit, chip, rule("qubit_mean"))
        mean_out = list(tile.circuit)[names(tile.circuit).index("CX") + 1]

        assert coupler_out.gate_args_copy() == mean_out.gate_args_copy()

    def test_gate_across_an_uncoupled_pair_raises(self, four_qubit_circuit, chip: Chip):
        """(0,0) and (2,0) are both sites but two apart, so nothing couples them."""
        ruleset = Ruleset(
            injection_rules=[
                InjectionRule(
                    "CX",
                    "any",
                    after=[ChannelRule("DEPOLARIZE2", "active", source="coupler")],
                )
            ]
        )
        tile = placed(four_qubit_circuit, chip, ruleset)

        with pytest.raises(ValueError, match="No coupler joins"):
            tile.circuit


class TestSurfaceCodeCouplerNoise:
    def test_every_sc_gate_pair_is_lattice_coupled(self, chip: Chip):
        """The SI1000 CX rule reads the coupler, so a d=3 patch only builds at all if
        every CX in stim's generated circuit spans a physically coupled pair."""
        chip.generate_uniform_noise(0.01)
        tile = SCTile(distance=3)
        assert chip.add_tile(tile, (2, 2))

        depolarize2 = [i for i in tile.circuit.flattened() if i.name == "DEPOLARIZE2"]

        assert depolarize2
        assert all(i.gate_args_copy() == [pytest.approx(0.01)] for i in depolarize2)

    def test_a_defective_coupler_changes_only_its_own_gate(self, chip: Chip):
        chip.generate_uniform_noise(0.01)
        tile = SCTile(distance=3)
        assert chip.add_tile(tile, (2, 2))

        before = [
            i.gate_args_copy()[0]
            for i in tile.circuit.flattened()
            if i.name == "DEPOLARIZE2"
        ]
        # pick an edge a CX in the patch actually spans (data<->ancilla, never data<->data)
        i2c = tile._circuit.get_final_qubit_coordinates()
        cx = next(i for i in tile._circuit.flattened() if i.name == "CX")
        a, b = ((i2c[t.value][0], i2c[t.value][1]) for t in cx.target_groups()[0])
        chip.coupler(a, b).noise.p = 0.2
        after = [
            i.gate_args_copy()[0]
            for i in tile.circuit.flattened()
            if i.name == "DEPOLARIZE2"
        ]

        assert 0.2 in after
        assert sorted(set(before)) == [pytest.approx(0.01)]
        assert sorted(set(after)) == [pytest.approx(0.01), pytest.approx(0.2)]
