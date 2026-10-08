from __future__ import annotations

from typing import Dict, List, Literal, Optional

from stim import Circuit

from ..lattice import CHECKERBOARD
from ..models import Coord, CSSType, Tag, TileSpec
from ..rules import ChannelRule, InjectionRule, Ruleset
from ..tile import LogicalTile


class SCTile(LogicalTile):
    def __init__(
        self,
        distance: int,
        rounds: Optional[int] = None,
        task: Literal["memory_x", "memory_z"] = "memory_z",
        origin: Coord = (0, 0),
        *,
        ruleset: Optional[Ruleset] = None,
        tag: Optional[Tag] = None,
    ):
        # `ruleset`/`tag` are the per-instance state the generator args cannot rebuild,
        # so `copy()` and the serialize importer hand them back in through here.
        if rounds is None:
            rounds = distance
        generator = lambda t, d, r: Circuit.generated(
            code_task=f"surface_code:rotated_{t}", distance=d, rounds=r
        )
        super().__init__(
            base_circuit=generator(task, distance, rounds),
            initial_shift=None,
            x_buffer=1,
            y_buffer=1,
            origin=origin,
            # stim's rotated surface-code generators emit checkerboard coordinates
            lattice=CHECKERBOARD,
            tag=tag if tag is not None else Tag(name=f"rsc_{task}_d{distance}"),
            ruleset=ruleset,
            spec=TileSpec(
                tile_type=type(self).__name__,
                distance=distance,
                rounds=rounds,
                generator=generator,
                generator_args={
                    "code_task": f"surface_code:rotated_{task}",
                    "distance": distance,
                    "rounds": rounds,
                    "task": task,
                },
            ),
        )

    def _first_op_coords(self, operation: str) -> Dict[Coord, int]:
        """Circuit coordinates targeted by the first `operation` instruction.

        Both the stabilizer readout (`MR`) and the X-basis change (`H`) appear in
        full before the circuit's REPEAT block, so one un-flattened pass finds them -
        which also avoids `CircuitRepeatBlock`, that has no `.name`.
        """
        i2c = self._circuit.get_final_qubit_coordinates()
        for instr in self._yield_circuit_instructions(self._circuit):
            if instr.name == operation:
                return {
                    (i2c[t.value][0], i2c[t.value][1]): t.value
                    for t in instr.targets_copy()
                }
        return {}

    def _init_tile_qubit_types(self):
        # Read the CSS roles off the circuit rather than off coordinate parity: the
        # ancillas are exactly what gets reset-and-measured, and the X ancillas are
        # exactly those conjugated by H. Holds for any distance, task, and lattice.
        all_qubits = self._c2i
        all_measures = self._first_op_coords("MR")
        if not all_measures:
            raise ValueError(
                f"Cannot type qubits for {type(self).__name__}: its circuit has no MR "
                "instruction, so the stabilizer ancillas cannot be identified."
            )
        x_measures = self._first_op_coords("H")
        z_measures = {
            k: all_measures[k] for k in all_measures.keys() - x_measures.keys()
        }

        for q in self.qubits:
            if q.loc in x_measures:
                q.type = CSSType.X_CHECK
            elif q.loc in z_measures:
                q.type = CSSType.Z_CHECK
            elif q.loc in all_qubits:
                q.type = CSSType.DATA
            else:
                q.type = CSSType.BUFFER

    # TODO shift the custom_rules into the ChannelRuleset object, adding the add_rule method within the LogicalTile super class
    def _init_ruleset(self, custom_rules: List[InjectionRule] = []) -> Ruleset:

        def on_operation(op: str, trig, before, after, name=None) -> InjectionRule:
            return InjectionRule(op, trig, before, after, name=name)

        def apply_channel(channel, filter, scalar=1.0, name=None, source="qubit_mean"):
            return ChannelRule(channel, filter, scalar=scalar, name=name, source=source)

        # Default application of SI1000 ruleset
        return Ruleset(
            [
                on_operation(
                    "R",
                    "all_qubits",
                    before=[],
                    after=[
                        apply_channel("X_ERROR", "all_qubits", 2.0, name="Init")
                    ],  # InitZ(p)           -> SI1000(2p)
                ),
                on_operation(
                    "H",
                    "x_measures",
                    before=[],
                    after=[
                        apply_channel(
                            "DEPOLARIZE1", "active", 0.1, name="Clifford1"
                        ),  # AnyClifford1(p)    -> SI1000(p/10)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 0.1, name="Idle"
                        ),  # Idle(p)            -> SI1000(p/10)
                    ],
                ),
                on_operation(
                    "CX",
                    "any",
                    before=[],
                    after=[
                        # The two-qubit error is the coupler's, not the endpoints'. Couplers
                        # derive as the mean of their two qubits by default, so this is
                        # numerically identical until a coupler is given its own rate.
                        apply_channel(
                            "DEPOLARIZE2",
                            "active",
                            1,
                            name="Clifford2",
                            source="coupler",
                        ),  # AnyClifford2(p)   -> SI1000(p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 0.1, name="Idle"
                        ),  # Idle(p)           -> SI1000(p/10)
                    ],
                ),
                on_operation(
                    "MR",
                    "all_measures",
                    before=[
                        apply_channel(
                            "X_ERROR", "all_measures", 5.0, name="Measure"
                        ),  # Measure(p)        -> SI1000(5p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 2.0, name="ResonatorIdle"
                        ),  # ResonatorIdle(p)  -> SI1000(2p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 0.1, name="Idle"
                        ),  # Idle(p)           -> SI1000(p/10)
                    ],
                    after=[
                        apply_channel(
                            "X_ERROR", "all_measures", 2.0, name="Init"
                        ),  # InitZ(p)          -> SI1000(2p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 2.0, name="ResonatorIdle"
                        ),  # ResonatorIdle(p)  -> SI1000(2p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 0.1, name="Idle"
                        ),  # Idle(p)           -> SI1000(p/10)
                    ],
                ),
                on_operation(
                    "M",
                    "data",
                    before=[
                        apply_channel(
                            "X_ERROR", "all_qubits", 5.0
                        ),  # Measure(p)        -> SI1000(5p)
                        apply_channel(
                            "DEPOLARIZE1", "idle", 2.0
                        ),  # ResonatorIdle(p)  -> SI1000(2p)
                    ],
                    after=[],
                ),
            ]
            + custom_rules
        )

    def summary(self) -> Dict[str, object]:
        s = super().summary()
        s["type"] = "RSC"
        return s

    def copy(self) -> SCTile:
        """Return a fresh uninitialized copy: same code, ruleset, and annotations."""
        return SCTile(
            distance=self.spec.generator_args["distance"],
            rounds=self.spec.generator_args["rounds"],
            task=self.spec.generator_args["task"],
            **self._carried_state(),
        )
