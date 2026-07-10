from __future__ import annotations

from warnings import warn
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from pickle import dump
from random import uniform
from statistics import mean

from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
from stim import Circuit, CircuitInstruction

from scipy.stats import truncnorm

from .grid import Grid
from .models import Coord, NoiseProfile, Qubit, Status, ShiftFunction, TileTag, CSSType, _DEFAULT_CSSTYPE, _DEFAULT_STATUS
from .rules import Ruleset, InjectionRule, ChannelRule
from qsnow.visualize import VisualizationStyle, default_style, visualize

class Chip(Grid):
    """
    Physical qubit chip as a checkerboard integer lattice.

    Chip(L, H) creates a grid spanning (0,0)–(2L, 2H) where valid qubit positions
    satisfy x % 2 == y % 2 (both even or both odd). Unit cells are 2 coordinate
    units wide, so a 5×5 logical tile starting at (0,0) covers select_rect(0, 0, 10, 10).

    Typical workflow:
      1. Instantiate Chip(L, H) and assign noise to individual qubits.
      2. Build and place LogicalTiles within the chip by specifying origin points within the (2L x 2H) chip.
      3. Retrieve and modify noise-injected circuits by accessing `tile.circuit` or shifting tiles to modify underlying Stim circuit.
    """

    def __init__(self, length: int, height: int, rotated: bool = True):
        super().__init__(2 * length, 2 * height)
        self._fill_checkerboard()
        self.tiles: List[LogicalTile] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    
    @property
    def noise_map(self) -> Dict[Coord, NoiseProfile]:
        '''Returns a map of coords to noise profile of qubit as a dict.'''
        return {coord: q.noise for coord, q in self._qubits.items()}

    @property
    def tile_map(self) -> Dict[Coord, LogicalTile]:
        '''Returns a map of chip's logical tile by the respective tile origin as dict.'''
        return {t.origin: t for t in self.tiles}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _fill_checkerboard(self) -> None:
        """Populate all valid checkerboard positions with default Qubits."""
        for x in range(self.length):
            for y in range(self.height):
                if x % 2 == y % 2:
                    coord: Coord = (x, y)
                    self._qubits[coord] = Qubit(loc=coord)

    # ------------------------------------------------------------------
    # Noise manipulation
    # ------------------------------------------------------------------

    def generate_random_noise(self, range: Tuple[float, float] = (0.01, 0.05)):
        for q in self.qubits:
            q.noise.p = round(uniform(range[0], range[1]), 5)

    def generate_gaussian_noise(self, mean, deviation, rng = None): #base26 "argonne"
        dist = truncnorm((0.000001-mean)/deviation, (1-mean)/deviation, loc=mean, scale=deviation)
        for q in self.qubits:
            q.noise.p = round(dist.rvs(1, random_state=rng)[0], 5)

    # ------------------------------------------------------------------
    # Tile operations
    # ------------------------------------------------------------------
    def add_tile(self, tile: LogicalTile, loc: Optional[Coord] = None) -> bool:
        """
        Add a `LogicalTile` to the chip. Returns `True` if tile was placed on chip successfully. 

        A tile can only be added if the space the tile would occupy on the chip is not already occupied by another tile.
        A space is free if and only if ALL qubits within the would-be tile placement are designated as `Status.INACTIVE`.

        If the `loc` parameter is not specified, the tile will attempt to be placed according to the origin and bound of the tile (i.e., the upperleftmost coordinate in the Stim circuits).
        If the `loc` parameter is specified, the tile will attempt to be shifted to be placed a the specified origin and cooresponding bound in respect to the now modified origin.
        """
        if loc:
            region_origin = loc
            region_bound = (region_origin[0] + tile.length, region_origin[1] + tile.height)
        else:
            #try and exactly place tile using tile's origin (defaults to (0,0))
            region_origin = tile.origin
            region_bound = tile.bound

        if not self._validate_tile_placements_with_warnings(tile, region_origin):
            return False

        #tile placement is valid and now modify the tile accordingly and add it to the chip
        tile.assign_chip(self, region_origin) 
        self.tiles.append(tile)
        return True

    def is_valid_tile_placement(self, origin: Coord, bound: Coord) -> bool:
        # 1. check that the loc is valid for the checkerboard styling
        if not(self._validate_checkerboard_loc(origin)):
            return False

        # 2. check if any this tile would overlap with any other tile
        if not(self._validate_empty_region(origin, bound)):
            return False

        # 3. check that this tile is within the bounds of the chip itself 
        if not (self._validate_chip_bounds(origin, bound)):
            return False
        
        return True
    
    def _validate_tile_placements_with_warnings(self, tile: LogicalTile, loc: Coord):
        origin = loc
        bound = (loc[0] + tile.length, loc[1] + tile.height)
        # 1. check that the loc is valid for the checkerboard styling
        if not(self._validate_checkerboard_loc(origin)):
            raise ValueError(f"Invalid tile placement. Both x and y must both be even or both be odd, given loc of ({origin[0]}, {origin[1]})")

        # 2. check if any this tile would overlap with any other tile
        if not(self._validate_empty_region(origin, bound)):
            warn(f"Invalid tile placement. Qubit's within the ({origin} x {bound}) are currently active.", stacklevel=2)
            return False

        # 3. check that this tile is within the bounds of the chip itself 
        if not (self._validate_chip_bounds(origin, bound)):
            warn(f"Invalid tile placement. Placement at ({origin} x {bound}) overflows chip boundaries of ({self.origin} x {self.bound}).", stacklevel=2)
            return False
        
        return True

    def _validate_checkerboard_loc(self, origin: Coord) -> bool:
        return origin[0]%2 == origin[1]%2
    
    # borderline unnecessary but keeps styling of constraint checks
    def _validate_empty_region(self, origin: Coord, bound: Coord) -> bool:
        return self.is_empty_region(origin, bound)
    
    def _validate_chip_bounds(self, origin: Coord, bound: Coord) -> bool:
        return self.length >= bound[0] and self.height >= bound[1] and self.origin[0] <= origin[0] and self.origin[1] <= origin[1]
    
    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show(self, style: Optional[VisualizationStyle] = None) -> None:
        """Display a visualization of the chip's qubit layout."""
        visualize(self, style=style or default_style, show=True)
    
    # ------------------------------------------------------------------
    # Import/Export
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            'dims': (self.length/2, self.height/2), #the original input length/height used to initialize the chip (not the 2l x 2h checkerboard size although that is saved in the underlying grid)
            'grid': super().to_dict(),
            'tiles': {c:t.to_dict() for c,t in self.tile_map.items()},
            'noise_map': self.noise_map
        }
    
    def export_noise_model(self):
        return NotImplemented
    
    def import_noise_model(self):
        return NotImplemented

    def pickle(self):
        return NotImplemented

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

    def __init__(
        self,
        circuit: Circuit,
        origin: Coord = (0,0),
        *,
        initial_shift: Optional[ShiftFunction] = None,
        x_buffer: int = 1,
        y_buffer: int = 1,
        ruleset: Optional[Ruleset] = None,
        tag: TileTag = TileTag()
    ):
        if x_buffer < 0 or y_buffer < 0:
            raise ValueError(f"Buffer values cannot be lower than 0. Given x_buffer of {x_buffer} and y_buffer of {y_buffer}")
        
        #TODO - expand the tagging system to be more formal
        self.tag = tag

        # NOTE - This initial update is made outside _update since the circuit must be initialized first
        self._base_circuit = circuit.without_noise().copy()
        self._circuit = circuit.without_noise().copy()
        self._c2i = self._extract_c2i_map()
        
        # NOTE - This is the only time that _base_circuit should ever be updated is with the initial_shift, as it is used in copying this tile as a template for another
        #        The `initial_shift` allows for fundamentally changing the qubit coordinate plane of the underlying circuit,
        #        which can be used to align the circuits qubit coordiantes to use a discrete integer-based checkerboard style.
        #        For example of why this is important, see the Surface Code	Biased Memory 5x5x3 example circuit on Crumble. 
        #        The circuit uses a .5 half-step checkerboard grid so using this shift function expands the circuit into the discrete integer: lambda *yx: (yx[1]*2, yx[0]*2)
        if initial_shift:
            self._base_circuit = self._shift_circuit_level_coordinates(initial_shift)
            self._update(new_circuit=self._base_circuit.copy(), 
                         new_qubits=None)
        if origin:
            self._update(new_circuit=self._shift_circuit_level_coordinates(lambda *yx: (yx[0] + origin[0], yx[1] + origin[1])),
                         new_qubits=None,
                         new_origin=origin)

        #Initially all tiles do not have a chip assigned. Modifying the circuit object will work
        self._chip = None
        
        #By default, logical tile has no noise injection rules unless passed in manually or defaulted by implemented code subclass
        #TODO add this as an accessible property
        self._ruleset = ruleset if ruleset else self._init_ruleset()

        # Set the tile to have the same length and height as the circuit, this would represent a tight packing of a tile with no buffer qubits
        circuit_coords = list(self._circuit.get_final_qubit_coordinates().values())
        dimensions = tuple(int(max(coord)+1) for coord in zip(*circuit_coords))[:2]

        super().__init__(length=dimensions[0]+x_buffer,
                         height=dimensions[1]+y_buffer,
                         origin=self.circuit_origin)

    def __del__(self):
        if self._qubits:
            self._scrub_qubits()

    def _init_tile_qubit_status(self):
        # 2. Change all the new qubits that are now in the scope to the tile to logical
        for c in self._qubits.keys():
            self.chip.loc(c).status = Status.ANCILLA
        # 3. Change all the new qubits that are not actively instantiated in the circuit to ancilla
        for c in self._c2i.keys():
            self.chip.loc(c).status = Status.LOGICAL

    #NOTE - default logicaltile does not deal with typing. typing is dependent on specific code constructions. this could change, likely to use a default noise ruleset like the `SCTile`
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
            raise AttributeError(f"Tile has not been initialized to a chip. To access underlying circuit before initialization, use `tile.base_circuit`.")
        return self._inject_circuit_noise()

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
    
    def initialized(self) -> bool:
        return True if self._chip else False

    @property
    def chip(self) -> Chip:
        '''The chip that '''
        if not self._chip:
            raise AttributeError("Chip not assigned for the requested tile.")
        return self._chip

    @property
    def qubits(self) -> List[Qubit]:
        '''A flat-list of qubits within the tile.'''
        if not self.initialized():
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
    def _yield_circuit_instructions(self, *, flatten: bool = False):
        for instr in self._circuit.flattened() if flatten else self._circuit:
            yield instr

    def _format_instruction_to_str(self, name: str, targets: List[int], arg: Optional[List] = None, *, tag: Optional[str] = None) -> str:
        arg_s = f"({', '.join(map(str, arg))})" if arg else ''
        targ_s = ' '.join(map(lambda x: str(int(x)), targets))
        tag_s = f'[{tag}]' if tag else ''
        return f"{name}{tag_s}{arg_s} {targ_s}"

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
            self._qubits = new_qubits
        #3. Update the new origin
        if new_origin:
            self.origin = new_origin

    #TODO - cleanup
    def _transfer_qubit_metadata(self, o2n_map: Dict[Coord, Coord]):
        o_metadata = {}
        for o in o2n_map.keys():
            o_metadata[o] = {'status':self.chip.loc(o).status, 'type':self.chip.loc(o).type} 
            self.chip.loc(o).reset_status()
            self.chip.loc(o).reset_type()
    
        for o, n in o2n_map.items():
            self.chip.loc(n).status = o_metadata.get(o, {}).get('status', _DEFAULT_STATUS)
            self.chip.loc(n).type = o_metadata.get(o, {}).get('type', _DEFAULT_CSSTYPE)

    def _extract_c2i_map(self) -> Dict[Coord, int]:
        i2c = self._circuit.get_final_qubit_coordinates()
        return {(value[0], value[1]): key for key, value in i2c.items()}
    
    def _extract_i2q_map(self) -> Dict[int, Qubit]:
        i2c = self._circuit.get_final_qubit_coordinates()
        return {key: self.chip.loc((value[0], value[1])) for key, value in i2c.items()} 

    def _shift_map(self, shift_function: ShiftFunction) -> Dict[Coord, Coord]:
        return {coord:shift_function(*coord) for coord in list(self._qubits.keys())}

    def _shift_circuit_level_coordinates(self, shift_function: ShiftFunction) -> Circuit:
        shifted_circuit = Circuit()
        #NOTE - Issue with STIM where the circuit iterator does not update type exclusivity to CircuitInstructions when flattening
        for instr in self._yield_circuit_instructions():
            match instr.name:
                case 'QUBIT_COORDS':
                    shifted_circuit.append_from_stim_program_text(self._format_instruction_to_str(name=instr.name, targets=[q.qubit_value for q in instr.targets_copy()], arg=list(shift_function(*instr.gate_args_copy()))))# type: ignore
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
        return LogicalTile(self.base_circuit, tag=self.tag)
        
    def _scrub_qubits(self):
        for q in self.qubits:
            q.reset()

    # ------------------------------------------------------------------
    # Import/export
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            'origin': self.origin,
            'bound': self.bound,
            'tag': self.tag.to_dict(),
            'circuit': self.circuit,
        }

    # ------------------------------------------------------------------
    # Tile spatial movements
    # ------------------------------------------------------------------

    #TODO - at some point, just need to start tracking the shifts while only making the shifts on the returned object (i.e., when the circuit is "compiled")
    def shift_by(self, x: int, y: int):
        '''Shifts the current tile within the chip by `x` spaces left or right and y units up or down. A (x, y) shift is valid IFF x%2 == y%2.'''
        
        new_origin = (self.origin[0] + x,  self.origin[1] + y) 
        new_bound = (self.bound[0] + x, self.bound[1] + y)

        # VALIDATION (differs slightly from parent chips internal _validate to exclude the consideration of qubits owned by the current tile in the empty subregion query)
        if not self.chip._validate_checkerboard_loc(new_origin):
            raise ValueError(f"Invalid shift that violates checkboard indexing. Both x and y must both be even or both be odd, given x = {x}, y = {y}")
        
        if not self.chip._validate_chip_bounds(new_origin, new_bound):
            raise ValueError(f"Invalid shift that violates chip boundaries. ")

        if not self.chip.is_empty_region_subset(new_origin, new_bound, self.origin, self.bound):
            # print(f"Current (O:{self.origin}, B:{self.bound}) ->  New (O:{new_origin}, B:{new_bound})")
            raise ValueError(f"Invalid shift operation that violates tile overlap constraints. This shift results in the tile overlapping an existing tile on chip.")
        
        new_qubits = self.chip.select_rect(self.origin[0] + x, self.origin[1] + y, self.bound[0] + x, self.bound[1] + y)
        new_circuit = self._shift_circuit_level_coordinates(lambda *coords: (coords[0] + x, coords[1] + y))

        if self._qubits:
            self._transfer_qubit_metadata(self._shift_map(lambda *coords: (coords[0] + x, coords[1] + y)))

        self._update(new_circuit=new_circuit, new_qubits=new_qubits, new_origin=new_origin)

    def shift_to(self, new_origin: Coord):
        '''Shifts the current tile to the given origin point. A (x, y) point is valid IFF x%2 == y%2.'''
        if not (new_origin == self.origin):
            x0, y0 = self.origin
            x1, y1 = new_origin
            self.shift_by(int(x1 - x0), int(y1 - y0))

    # ------------------------------------------------------------------
    # Noise channel injection
    # ------------------------------------------------------------------
    
    def _inject_circuit_noise(self) -> Circuit:
        circ = Circuit()
        i2q = self._extract_i2q_map()
        for instr in self._yield_circuit_instructions(flatten=True):
            before = []
            after = []
            for rule in self._ruleset.rules:
                if instr.name == rule.operation:
                    operation_targs = [[t.value for t in a] for a in instr.target_groups()] # type: ignore
                    if self._ruleset.check_trigger(rule.trigger, operation_targs, i2q):
                        for c in rule.before:
                            channel_targs = self._ruleset.apply_filter(c.filter, operation_targs, i2q)
                            for l in channel_targs:
                                before.append(self._format_instruction_to_str(name=c.channel,
                                                                              targets=[i for i in l], 
                                                                              #TODO - move the determination of noise value to be handled by ruleset to enable arbitrary granular control of scaling
                                                                              arg=[mean([i2q.get(i).noise.p for i in l]) * c.scalar],# type: ignore
                                                                              tag=f"{rule.operation}:{rule.trigger} -> {c.channel}:{c.filter}" if rule.name is None else rule.name))
                        for c in rule.after:
                            channel_targs = self._ruleset.apply_filter(c.filter, operation_targs, i2q)
                            for l in channel_targs:
                                after.append(self._format_instruction_to_str(name=c.channel, 
                                                                             targets=l, 
                                                                             arg=[mean([i2q.get(i).noise.p for i in l]) * c.scalar],# type: ignore
                                                                             tag=f"{rule.operation}:{rule.trigger} -> {c.channel}:{c.filter}" if rule.name is None else rule.name))
                    if rule.exclusive:
                        break
            
            circ.append_from_stim_program_text('\n'.join(before))
            circ.append(instr)
            circ.append_from_stim_program_text('\n'.join(after))

        return circ

            

        
