from __future__ import annotations

from warnings import warn
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from random import uniform

from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
from stim import Circuit, CircuitInstruction

from .models import Coord, NoiseProfile, Qubit, Status, ShiftFunction, TileTag
from visualize import VisualizationStyle, default_style, visualize

if TYPE_CHECKING:
    from matplotlib.axes import Axes

class Grid:
    """
    Integer checkerboard lattice. Valid positions (x, y) satisfy x % 2 == y % 2,
    spanning x ∈ [0, length] and y ∈ [0, height] (both inclusive).

    Backed by a Shapely STRtree for efficient arbitrary-region selection.
    The index is built lazily on first query and cached until invalidated.
    """
    origin: Coord
    height: int
    length: int
    _qubits: Dict[Coord, Qubit]
    _index: Optional[Tuple[List[Coord], STRtree]]

    def __init__(self, length: int, height: int, origin: Coord = (0, 0)):
        if length < 1 or height < 1:
            raise ValueError(
                f"Grid length and height must be >= 1. Got length={length}, height={height}."
            )
        self.origin = origin
        self.length = length
        self.height = height
        self._qubits: Dict[Coord, Qubit] = {}
        self._index: Optional[Tuple[List[Coord], STRtree]] = None

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------

    def _invalidate_index(self) -> None:
        self._index = None

    def _ensure_index(self) -> Tuple[List[Coord], STRtree]:
        if self._index is None:
            coords = list(self._qubits.keys())
            tree = STRtree([Point(x, y) for (x, y) in coords])
            self._index = (coords, tree)
        return self._index

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bound(self) -> Coord:
        return (self.origin[0] + self.length - 1, self.origin[1] + self.height - 1)

    # ------------------------------------------------------------------
    # Qubit access
    # ------------------------------------------------------------------

    @property
    def grid(self) -> Dict[Coord, Qubit]:
        """The full dict of (x, y) → Qubit for all occupied positions."""
        return self._qubits
    
    @property
    def qubits(self) -> List[Qubit]:
        """Return all qubits on the chip as a flat list."""
        return list(self._qubits.values())

    def loc(self, coord: Coord) -> Qubit:
        """Return the qubit at `coord`, raising KeyError if absent."""
        try:
            return self._qubits[coord]
        except KeyError:
            raise KeyError(f"No qubit at coordinate {coord}.")

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------

    def select(self, region: Polygon) -> Dict[Coord, Qubit]:
        """Return all qubits whose coordinates fall within or on `region`."""
        coords, tree = self._ensure_index()
        # predicate='covers' → region.covers(tree_item) → includes boundary points
        indices = tree.query(region, predicate='covers')
        return {coords[i]: self._qubits[coords[i]] for i in indices}

    def select_rect(self, x0: float, y0: float, x1: float, y1: float) -> Dict[Coord, Qubit]:
        """
        Select qubits in the axis-aligned rectangle [x0, x1] × [y0, y1] (inclusive).
        Coordinates are in the checkerboard system where unit cells are 2 units wide/tall.
        Example: top-left 2×2 cell block of any Chip → select_rect(0, 0, 4, 4).
        """
        return self.select(box(x0, y0, x1, y1))
    
    # ------------------------------------------------------------------
    # Region querying
    # ------------------------------------------------------------------

    def is_empty_region(self, origin: Coord, bound: Coord) -> bool:
        """
        Check if the specified rectangular region of the chip is free, with the `origin` being the top left corner and `bound` being the bottom right corner. 
        A region is empty if all qubits in rectangle are inactive.
        """
        for q in self.select_rect(origin[0], origin[1], bound[0], bound[1]).values():
            if q.is_active():
                return False
            else:
                continue
        return True


    
class Chip(Grid):
    """
    Physical qubit chip as a checkerboard integer lattice.

    Chip(L, W) creates a grid spanning (0,0)–(2L, 2W) where valid qubit positions
    satisfy x % 2 == y % 2 (both even or both odd). Unit cells are 2 coordinate
    units wide, so a 5×5 logical tile starting at (0,0) covers select_rect(0, 0, 10, 10).

    Typical workflow:
      1. Instantiate Chip(L, W) and assign noise to individual qubits.
      2. Select regions via select_rect() or select(polygon) to build LogicalTiles.
      3. Retrieve get_noise_map() to inject per-qubit noise into Stim circuits.
    """
    tiles: List[LogicalTile]

    def __init__(self, length: int, height: int, rotated = True):
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


    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show(self, style: Optional[VisualizationStyle] = None) -> "Axes":
        """Display a visualization of the chip's qubit layout."""
        return visualize(self, style=style or default_style, show=True)
    
    # ------------------------------------------------------------------
    # Tile operations
    # ------------------------------------------------------------------
    def add_tile(self, tile: LogicalTile, loc: Optional[Coord]) -> bool:
        """
        Add a `LogicalTile` to the chip. Returns `True` if tile was placed on chip successfully. 

        A tile can only be added if the space the tile would occupy on the chip is not already occupied by another tile.
        A space is free if and only if ALL qubits within the would-be tile placement are designated as `Status.INACTIVE`.

        If the `loc` parameter is not specified, the tile will attempt to be placed according to the origin and bound of the tile.
        If the `loc` parameter is specified, the tile will attempt to be shifted to be placed a the specified origin and cooresponding bound in respect to the now modified origin.
        """
        if not (tile.length <= self.length or tile.height <= self.height):
            raise ValueError("Tile dimensions must be less than or equal to the dimensions of a given chip.",
                             f"Given tile is {tile.length} x {tile.height} but chip is {self.length} x {self.height}")
        
        #TODO - add checking to make sure that the tile region is within the chip for both cases below
        if loc:
            region_origin = loc
            region_bound = (region_origin[0] + tile.length, region_origin[1] + tile.height)
        else:
            #try and exactly place tile using tile's origin (defaults to (0,0))
            region_origin = tile.origin
            region_bound = tile.bound

        if not self.is_empty_region(region_origin, region_bound):
            warn(f"Invalid tile placement. Qubit's within the ({region_origin} x {region_bound}) are currently active.")
            return False
        
        #tile placement is valid and now modify the tile accordingly and add it to the chip
        tile.assign_chip(self, region_origin) 
        self.tiles.append(tile)
        return True


class LogicalTile(Grid):
    """
    A Stim circuit anchored to a coordinate frame.

    `c2i` maps each qubit's (x, y) position (in the circuit's own frame, optionally
    shifted by `shift_function`) to its integer index in the Stim circuit.

    To place a tile on a Chip, use chip.select_rect() or chip.select() with the
    tile's bounding region to retrieve the corresponding Qubit objects.
    """
    _base_circuit: Circuit
    _circuit: Circuit
    _chip: Optional[Chip]
    _c2i: Dict[Coord, int]
    tag: TileTag

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

            

        
