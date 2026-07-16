from __future__ import annotations

import logging
from copy import deepcopy
from statistics import mean
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from stim import Circuit

from .grid import Grid
from .models import (
    _DEFAULT_CSSTYPE,
    _DEFAULT_STATUS,
    Coord,
    Qubit,
    ShiftFunction,
    Status,
    Tag,
    TileSpec,
)
from .rules import Ruleset

if TYPE_CHECKING:
    from .chip import Chip


class LogicalTile(Grid):
    """
    A Stim circuit anchored to a coordinate frame.

    `c2i` maps each qubit's (x, y) position (in the circuit's own frame, optionally
    shifted by `shift_function`) to its integer index in the Stim circuit.

    To place a tile on a Chip, use chip.add_tile() with a specified origin coordinate.

    Attributes:
    _base_circuit (Circuit): Initial circuit used to define the tile
    _circuit: Circuit -> Noise injected or spatially modified circuit built from the base circuit
    _chip (Optional[Chip]) ->
    _c2i: Dict[Coord, int]
    tag (Tag) -> annotation (name/desc/metadata), same plain Tag as every Grid
    spec (TileSpec) -> reconstruction fields read by serialize importers and copy()
    """

    spec: TileSpec

    def __init__(
        self,
        base_circuit: Circuit,
        origin: Coord = (0, 0),
        *,
        initial_shift: Optional[ShiftFunction] = None,
        x_buffer: int = 1,
        y_buffer: int = 1,
        ruleset: Optional[Ruleset] = None,
        tag: Optional[Tag] = None,
        spec: Optional[TileSpec] = None,
    ):
        if x_buffer < 0 or y_buffer < 0:
            raise ValueError(
                f"Buffer values cannot be lower than 0. Given x_buffer of {x_buffer} and y_buffer of {y_buffer}"
            )

        # Resolved here (before the Grid super().__init__ below) so they are usable
        # during init; the tag is passed through to Grid, which stores this same object.
        self.tag = tag if tag is not None else Tag()
        self.spec = (
            spec if spec is not None else TileSpec(tile_type=type(self).__name__)
        )

        # NOTE - This initial update is made outside _update since the circuit must be initialized first
        self._base_circuit = base_circuit.without_noise().copy()
        self._circuit = base_circuit.without_noise().copy()
        self._c2i = self._extract_c2i_map()

        # NOTE - This is the only time that _base_circuit should ever be updated is with the initial_shift, as it is used in copying this tile as a template for another
        #        The `initial_shift` allows for fundamentally changing the qubit coordinate plane of the underlying circuit,
        #        which can be used to align the circuits qubit coordiantes to use a discrete integer-based checkerboard style.
        #        For example of why this is important, see the Surface Code	Biased Memory 5x5x3 example circuit on Crumble.
        #        The circuit uses a .5 half-step checkerboard grid so using this shift function expands the circuit into the discrete integer: lambda *yx: (yx[1]*2, yx[0]*2)
        if initial_shift:
            self._base_circuit = self._shift_circuit_level_coordinates(initial_shift)
            self._update(new_circuit=self._base_circuit.copy(), new_qubits=None)
        if origin:
            self._update(
                new_circuit=self._shift_circuit_level_coordinates(
                    lambda *yx: (yx[0] + origin[0], yx[1] + origin[1])
                ),
                new_qubits=None,
                new_origin=origin,
            )

        # Initially all tiles do not have a chip assigned. Modifying the circuit object will work
        self._chip = None

        # By default, logical tile has no noise injection rules unless passed in manually or defaulted by implemented code subclass
        # TODO add this as an accessible property
        self._ruleset = ruleset if ruleset else self._init_ruleset()

        # Set the tile to have the same length and height as the circuit, this would represent a tight packing of a tile with no buffer qubits
        circuit_coords = list(self._circuit.get_final_qubit_coordinates().values())
        dimensions = tuple(int(max(coord) + 1) for coord in zip(*circuit_coords))[:2]

        super().__init__(
            length=dimensions[0] + x_buffer,
            height=dimensions[1] + y_buffer,
            origin=self.circuit_origin,
            tag=self.tag,
        )

    def __del__(self):
        if self.initialized():
            self._scrub_qubits()
            self.chip.tiles.remove(self)

    def _init_tile_qubit_status(self):
        # 2. Change all the new qubits that are now in the scope to the tile to logical
        for c in self._qubits.keys():
            self.chip.loc(c).status = Status.ANCILLA
        # 3. Change all the new qubits that are not actively instantiated in the circuit to ancilla
        for c in self._c2i.keys():
            self.chip.loc(c).status = Status.LOGICAL

    # TODO - formalize the inheritance structure/relation between parent class (LogicalTile) and child (implementable code-specific tiles)
    # NOTE - default logicaltile does not deal with typing. typing is dependent on specific code constructions. this could change, likely to use a default noise ruleset like the `SCTile`
    def _init_tile_qubit_types(self):
        pass

    def _init_ruleset(self) -> Ruleset:
        return Ruleset()

    # ------------------------------------------------------------------
    # Tile properties
    # ------------------------------------------------------------------
    @property
    def circuit(self) -> Circuit:
        """
        The STIM circuit created from the tile.
        This circuit is modified from the inputted circuit with the underlying noise features of the tile's placement within the grid.
        """
        if not self.initialized():
            raise AttributeError(
                "Tile has not been initialized to a chip. To access underlying circuit before initialization, use `tile.base_circuit`."
            )
        return self._inject_circuit_noise()

    @property
    def base_circuit(self) -> Circuit:
        """
        The original STIM circuit used to initialize the tile.
        """
        return self._base_circuit

    @property
    def circuit_origin(self) -> Coord:
        """
        The origin (upper leftmost) coordinate of the qubit coordiantes ~within~ the circuit.
        """
        coords = list(self._circuit.get_final_qubit_coordinates().values())
        return tuple(min(coord) for coord in zip(*coords))[:2]

    @property
    def circuit_bound(self) -> Coord:
        """
        The bound (bottom rightmost) coordinate of the qubit coordiantes ~within~ the circuit.
        """
        coords = list(self._circuit.get_final_qubit_coordinates().values())
        return tuple(max(coord) for coord in zip(*coords))[:2]

    @property
    def chip(self) -> Chip:
        """
        The chip that the tile is placed on. If the tile is not initialized to a chip yet, an attribute error will be raised.
        """
        if not self._chip:
            raise AttributeError("Chip not assigned for the requested tile.")
        return self._chip

    @property
    def qubits(self) -> List[Qubit]:
        """
        A flat-list of qubits within the tile.
        """
        if not self.initialized():
            raise AttributeError(
                "Tile has no underlying qubits as the tile has not been assigned a chip"
            )
        return super().qubits

    @property
    def dims(self) -> Tuple[int, int]:
        return (self.length // 2, self.height // 2)

    def qubit_at_index(self, index: int) -> Optional[Qubit]:
        """
        Return qubit cooresponding to the given circuit-level index.
        """
        coord = self._circuit.get_final_qubit_coordinates().get(index)
        if coord:
            return self.loc((coord[0], coord[1]))
        else:
            return None

    def initialized(self) -> bool:
        return True if self._chip else False

    # ------------------------------------------------------------------
    # Circuit manipulation
    # ------------------------------------------------------------------
    def _yield_circuit_instructions(self, *, flatten: bool = False):
        for instr in self._circuit.flattened() if flatten else self._circuit:
            yield instr

    def _format_instruction_to_str(
        self,
        name: str,
        targets: List[int],
        arg: Optional[List] = None,
        *,
        tag: Optional[str] = None,
    ) -> str:
        arg_s = f"({', '.join(map(str, arg))})" if arg else ""
        targ_s = " ".join(map(lambda x: str(int(x)), targets))
        tag_s = f"[{tag}]" if tag else ""
        return f"{name}{tag_s}{arg_s} {targ_s}"

    # NOTE - All updates to the underlying _circuit and _qubits map are routed through this function
    #       This ensures that all dependent objects (e.g., _c2i) properly reflect any changes to circuit and
    #       that pointers are properly carried through and new objects are not created accidentally (e.g., _qubits)
    def _update(
        self,
        new_circuit: Optional[Circuit] = None,
        new_qubits: Optional[Dict[Coord, Qubit]] = None,
        new_origin: Optional[Coord] = None,
    ):
        # 1. Update underlying circuit and c2i
        if new_circuit:
            self._circuit = new_circuit
            self._c2i = self._extract_c2i_map()
        # 2. If qubits changed, shift qubit references and update statuses
        if new_qubits:
            self._qubits = new_qubits
        # 3. Update the new origin
        if new_origin:
            self.origin = new_origin

    # TODO - cleanup
    def _transfer_qubit_metadata(self, o2n_map: Dict[Coord, Coord]):
        o_metadata = {}
        for o in o2n_map.keys():
            o_metadata[o] = {
                "status": self.chip.loc(o).status,
                "type": self.chip.loc(o).type,
            }
            self.chip.loc(o).reset_status()
            self.chip.loc(o).reset_type()

        for o, n in o2n_map.items():
            self.chip.loc(n).status = o_metadata.get(o, {}).get(
                "status", _DEFAULT_STATUS
            )
            self.chip.loc(n).type = o_metadata.get(o, {}).get("type", _DEFAULT_CSSTYPE)

    def _scrub_qubits(self):
        for q in self.qubits:
            q.reset()

    def _extract_c2i_map(self) -> Dict[Coord, int]:
        i2c = self._circuit.get_final_qubit_coordinates()
        return {(value[0], value[1]): key for key, value in i2c.items()}

    def _extract_i2q_map(self) -> Dict[int, Qubit]:
        i2c = self._circuit.get_final_qubit_coordinates()
        return {key: self.chip.loc((value[0], value[1])) for key, value in i2c.items()}

    def _shift_map(self, shift_function: ShiftFunction) -> Dict[Coord, Coord]:
        return {coord: shift_function(*coord) for coord in list(self._qubits.keys())}

    def _shift_circuit_level_coordinates(
        self, shift_function: ShiftFunction
    ) -> Circuit:
        shifted_circuit = Circuit()
        # NOTE - Issue with STIM where the circuit iterator does not update type exclusivity to CircuitInstructions when flattening
        for instr in self._yield_circuit_instructions():
            match instr.name:
                case "QUBIT_COORDS":
                    shifted_circuit.append_from_stim_program_text(
                        self._format_instruction_to_str(
                            name=instr.name,
                            targets=[q.qubit_value for q in instr.targets_copy()],  # type: ignore
                            arg=list(shift_function(*instr.gate_args_copy())),  # type: ignore
                        )
                    )
                case _:
                    shifted_circuit.append(instr)
        return shifted_circuit

    # ------------------------------------------------------------------
    # General functions
    # ------------------------------------------------------------------

    def assign_chip(self, chip: Chip, new_origin: Coord):
        self._chip = chip

        # NOTE - duped here just so that shift_to can ignore if the shift would just move the same tile to it's current position
        x0, y0 = self.origin
        x1, y1 = new_origin
        self.shift_by(int(x1 - x0), int(y1 - y0))

        self._init_tile_qubit_status()
        self._init_tile_qubit_types()

    def copy(self) -> LogicalTile:
        """Return a fresh uninitialized copy of the circuit."""
        # deepcopy so the copy never shares mutable tag/spec state (dicts included)
        return LogicalTile(
            self.base_circuit, tag=deepcopy(self.tag), spec=deepcopy(self.spec)
        )

    def reset(self) -> None:
        """Resets the tile back to uninitialized state, removing it from any active chip and housekeeping qubit statuses."""
        self._circuit = self._base_circuit.copy()
        self._scrub_qubits()
        # TODO - remove _c2i as a property and just use extract_c2i_map when necessary. It could technically save time to not have to remake the map everytime but
        #       it's hardly being used as is except just to keep track of updating it when necessary so lil bit of a headache for no purpose as is
        self._c2i = self._extract_c2i_map()
        self._chip = None

    # ------------------------------------------------------------------
    # Tile spatial movements
    # ------------------------------------------------------------------

    # TODO - at some point, just need to start tracking the shifts while only making the shifts on the returned object (i.e., when the circuit is "compiled")
    def shift_by(self, x: int, y: int):
        """Shifts the current tile within the chip by `x` spaces left or right and y units up or down. A (x, y) shift is valid IFF x%2 == y%2."""

        new_origin = (self.origin[0] + x, self.origin[1] + y)
        new_bound = (self.bound[0] + x, self.bound[1] + y)

        # VALIDATION (differs slightly from parent chips internal _validate to exclude the consideration of qubits owned by the current tile in the empty subregion query)
        if not self.chip._validate_checkerboard_loc(new_origin):
            raise ValueError(
                f"Invalid shift that violates checkboard indexing. Both x and y must both be even or both be odd, given x = {x}, y = {y}"
            )

        if not self.chip._validate_chip_bounds(new_origin, new_bound):
            raise ValueError("Invalid shift that violates chip boundaries. ")

        if not self.chip.is_empty_region_subset(
            new_origin, new_bound, self.origin, self.bound
        ):
            logger.debug(
                f"Current (O:{self.origin}, B:{self.bound}) ->  New (O:{new_origin}, B:{new_bound})"
            )
            raise ValueError(
                "Invalid shift operation that violates tile overlap constraints. This shift results in the tile overlapping an existing tile on chip."
            )

        new_qubits = self.chip.select_rect(
            self.origin[0] + x, self.origin[1] + y, self.bound[0] + x, self.bound[1] + y
        )
        new_circuit = self._shift_circuit_level_coordinates(
            lambda *coords: (coords[0] + x, coords[1] + y)
        )

        if self._qubits:
            self._transfer_qubit_metadata(
                self._shift_map(lambda *coords: (coords[0] + x, coords[1] + y))
            )

        self._update(
            new_circuit=new_circuit, new_qubits=new_qubits, new_origin=new_origin
        )

    def shift_to(self, new_origin: Coord):
        """Shifts the current tile to the given origin point. A (x, y) point is valid IFF x%2 == y%2."""
        if not (new_origin == self.origin):
            x0, y0 = self.origin
            x1, y1 = new_origin
            self.shift_by(int(x1 - x0), int(y1 - y0))

    # ------------------------------------------------------------------
    # Noise channel injection
    # ------------------------------------------------------------------

    # TODO - make debug_tags a global configuration flag when that refactor is up
    def _inject_circuit_noise(self, debug_tags=False) -> Circuit:
        circ = Circuit()
        i2q = self._extract_i2q_map()
        for instr in self._yield_circuit_instructions(flatten=True):
            before = []
            after = []
            for rule in self._ruleset.rules:
                if instr.name == rule.operation:
                    operation_targs = [
                        [t.value for t in a] for a in instr.target_groups()
                    ]  # type: ignore
                    if self._ruleset.check_trigger(rule.trigger, operation_targs, i2q):
                        for c in rule.before:
                            channel_targs = self._ruleset.apply_filter(
                                c.filter, operation_targs, i2q
                            )
                            for l in channel_targs:
                                before.append(
                                    self._format_instruction_to_str(
                                        name=c.channel,
                                        targets=[i for i in l],
                                        # TODO - move the determination of noise value to be handled by ruleset to enable arbitrary granular control of scaling
                                        arg=[
                                            mean([i2q.get(i).noise.p for i in l])  # type: ignore
                                            * c.scalar
                                        ],
                                        tag=f"{rule.operation}:{rule.trigger} -> {c.channel}:{c.filter}"
                                        if debug_tags
                                        else None
                                        if rule.name is None
                                        else rule.name,
                                    )
                                )
                        for c in rule.after:
                            channel_targs = self._ruleset.apply_filter(
                                c.filter, operation_targs, i2q
                            )
                            for l in channel_targs:
                                after.append(
                                    self._format_instruction_to_str(
                                        name=c.channel,
                                        targets=l,
                                        arg=[
                                            mean([i2q.get(i).noise.p for i in l])  # type: ignore
                                            * c.scalar
                                        ],  # type: ignore
                                        tag=f"{rule.operation}:{rule.trigger} -> {c.channel}:{c.filter}"
                                        if debug_tags
                                        else None
                                        if rule.name is None
                                        else rule.name,
                                    )
                                )
                    if rule.exclusive:
                        break

            circ.append_from_stim_program_text("\n".join(before))
            circ.append(instr)
            circ.append_from_stim_program_text("\n".join(after))

        return circ
