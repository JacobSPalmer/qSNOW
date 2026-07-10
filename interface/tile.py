from __future__ import annotations

from warnings import warn
from typing import Callable, Dict, List, Optional, Tuple

from stim import Circuit

from .grid import Grid
from .chip import Chip
from .models import Coord, Qubit, Status, ShiftFunction, TileTag, CSSType

class LogicalTile(Grid):
    """
    A Stim circuit anchored to a coordinate frame.

    `c2i` maps each qubit's (x, y) position (in the circuit's own frame, optionally
    shifted by `shift_function`) to its integer index in the Stim circuit.

    To place a tile on a Chip, use chip.add_tile() with a specified origin coordinate.
    """
    _base_circuit: Circuit
    _circuit: Circuit
    _chip: Optional[Chip]
    _c2i: Dict[Coord, int]
    tag: TileTag
    # channels: List[Channel]

    def __init__(
        self,
        circuit: Circuit,
        initial_shift: Optional[Callable[[*Tuple[float, ...]], Tuple[float, ...]]] = None,
        x_buffer: int = 1,
        y_buffer: int = 1,
        origin: Coord = (0,0),
        tag: TileTag = TileTag()
    ):
        if x_buffer < 0 or y_buffer < 0:
            raise ValueError(f"Buffer values cannot be lower than 0. Given x_buffer of {x_buffer} and y_buffer of {y_buffer}")
        
        #TODO - expand the tagging system to be more formal
        self.tag = tag

        #This initial update is made outside _update since the circuit must be initialized first
        self._base_circuit = circuit.without_noise().flattened().copy()
        self._circuit = circuit.without_noise().flattened().copy()
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
            self._update(new_circuit=self._shift_circuit_level_coordinates(lambda *yx: (yx[0] + origin[0], yx[1] + origin[1])))
            

        #Initially all tiles do not have a chip assigned. Modifying the circuit object will work
        self._chip = None

        # initially, set the tile to have the same length and height as the circuit
        # this would represent a tight packing of a tile with no buffer qubits
        circuit_coords = list(self._circuit.get_final_qubit_coordinates().values())
        dimensions = tuple(int(max(coord)+1) for coord in zip(*circuit_coords))[:2]

        super().__init__(length=dimensions[0]+x_buffer, 
                         height=dimensions[1]+y_buffer, 
                         origin=self.circuit_origin)

    # ------------------------------------------------------------------
    # Tile properties
    # ------------------------------------------------------------------
    @property
    def circuit(self) -> Circuit:
        """
        The STIM circuit created from the tile.
        This circuit is modified from the inputted circuit with the underlying noise features of the tile's placement within the grid.
        """
        return self._circuit
    
    @property
    def base_circuit(self) -> Circuit:
        """
        The original STIM circuit used to initialize the tile.
        """
        return self._base_circuit
    
    @property
    def circuit_origin(self) -> Coord:
        coords = list(self._circuit.get_final_qubit_coordinates().values())
        return tuple(min(coord) for coord in zip(*coords))[:2]

    @property
    def circuit_bound(self) -> Coord:
        coords = list(self._circuit.get_final_qubit_coordinates().values())
        return tuple(max(coord) for coord in zip(*coords))[:2]

    @property
    def chip(self) -> Chip:
        '''The chip that '''
        if not self._chip:
            raise AttributeError("Chip not assigned for the requested tile.")
        return self._chip 
    
    @property
    def qubits(self) -> List[Qubit]:
        '''A flat-list of qubits within the tile.'''
        if not self._chip:
            raise AttributeError("Tile has no underlying qubits as the tile has not been assigned a chip")
        return super().qubits
    
    def qubit_at_index(self, index: int) -> Optional[Qubit]:
        coord = self._circuit.get_final_qubit_coordinates().get(index)
        if coord:
            return self.loc((coord[0], coord[1]))
        else:
            return None

    # ------------------------------------------------------------------
    # Circuit manipulation
    # ------------------------------------------------------------------

    def _format_instruction(self, name: str, gate_args: Optional[List], gate_targs: List) -> str:
        if gate_args:
            return f"{name}({', '.join([str(i) for i in gate_args])}) {','.join([str(i) for i in gate_targs])}"
        else:
            return f"{name} {', '.join(gate_targs)}"

    #NOTE - All updates to the underlying _circuit and _qubits map are routed through this function
    #       This ensures that all dependent objects (e.g., _c2i) properly reflect any changes to circuit and
    #       that pointers are properly carried through and new objects are not created accidentally (e.g., _qubits)
    def _update(self, 
                new_circuit: Optional[Circuit] = None, 
                new_qubits: Optional[Dict[Coord, Qubit]] = None,
                new_origin: Optional[Coord] = None):
        #1. Update underlying circuit and c2i
        if new_circuit:
            self._circuit = new_circuit
            self._c2i = self._extract_c2i_map()  
        #2. If qubits changed, shift qubit references and update statuses 
        if new_qubits:
            self._realign_qubit_status(old=self._qubits, new=new_qubits)
            self._qubits = new_qubits
        #3. Update the new origin
        if new_origin:
            self.origin = new_origin

    # def _assign_type(self, rules: Tuple[CSSType, Callable]):

    def _realign_qubit_status(self, old: Dict[Coord, Qubit], new: Dict[Coord, Qubit]):
        # 1. Change all qubits that are no longer in the scope of the logical tile to inactive
        for c in old.keys():
            self.chip.loc(c).status = Status.INACTIVE
        # 2. Change all the new qubits that are now in the scope to the tile to logical
        for c in new.keys():
            self.chip.loc(c).status = Status.ANCILLA
        # 3. Change all the new qubits that are not actively instantiated in the circuit to ancilla
        for c in self._c2i.keys():
            self.chip.loc(c).status = Status.LOGICAL

    def _extract_c2i_map(self) -> Dict[Coord, int]:
        i2c = self._circuit.get_final_qubit_coordinates()
        return {(value[0], value[1]): key for key, value in i2c.items()}

    def _shift_circuit_level_coordinates(self, shift_function: ShiftFunction) -> Circuit:
        shifted_circuit = Circuit()
        #NOTE - Issue with STIM where the circuit iterator does not update type exclusivity to CircuitInstructions when flattening
        for instr in self.circuit.flattened():
            match instr.name:
                case 'QUBIT_COORDS':
                    instr = CircuitInstruction(self._format_instruction(instr.name, list(shift_function(*instr.gate_args_copy())), [q.qubit_value for q in instr.targets_copy()])) # type: ignore
                    shifted_circuit.append(instr)
                case _:
                    shifted_circuit.append(instr)
        return shifted_circuit

    # ------------------------------------------------------------------
    # General functions
    # ------------------------------------------------------------------

    def assign_chip(self, chip: Chip, new_origin: Coord):
        self._chip = chip
        self.shift_to(new_origin)

    def copy(self) -> LogicalTile:
        return LogicalTile(self.base_circuit, tag=self.tag)

    # ------------------------------------------------------------------
    # Tile spatial movements
    # ------------------------------------------------------------------

    def shift_by(self, x: int, y: int):
        if not(x%2 == y%2):
            raise ValueError(f"Invalid shift that violates checkboard indexing. Both x and y must both be even or both be odd, given x = {x}, y = {y}")
        '''Shifts the current tile within the chip by `x` spaces left or right and y units up or down. A (x, y) shift is valid IFF x%2 == y%2.'''
        new_origin = (self.origin[0] + x,  self.origin[1] + y) #(2 - 2, 0 + 2) -> (0, 2)
        new_circuit = self._shift_circuit_level_coordinates(lambda *coords: (coords[0] + x, coords[1] + y))

        # print(f"Current Origin x Bound: {self.origin} x {self.bound}")

        new_qubits = self.chip.select_rect(self.origin[0] + x, self.origin[1] + y, self.bound[0] + x, self.bound[1] + y)
        self._update(new_circuit=new_circuit, new_qubits=new_qubits, new_origin=new_origin)

        # print(f"New Bound: {self.origin} x {self.bound}")
        
    def shift_to(self, new_origin: Coord):
        '''Shifts the current tile to the given origin point. A (x, y) point is valid IFF x%2 == y%2.'''
        x0, y0 = self.origin #(2, 0)
        x1, y1 = new_origin #(0, 2)
        self.shift_by(int(x1 - x0), int(y1 - y0)) #(0-2, 2-0) -> (-2, 2)

    # ------------------------------------------------------------------
    # Noise channel injection
    # ------------------------------------------------------------------

    #TODO - Noise dynamic rule mapper. Hard coding for now.
    def inject_noise(self):
        # for instr in self.circuit.flattened():
        return NotImplemented