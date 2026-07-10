from __future__ import annotations

from warnings import warn
from typing import Callable, Dict, List, Optional, Tuple, Union, overload

from random import uniform

from shapely import difference
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
from stim import Circuit, CircuitInstruction

from .models import Coord, NoiseProfile, Qubit, Status, ShiftFunction, TileTag, CSSType
from visualize import VisualizationStyle, default_style, visualize

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

    def loc(self, coord: Union[Coord, Tuple[float,...]]) -> Qubit:
        """Return the qubit at `coord`, raising KeyError if absent."""
        try:
            return self._qubits[(coord[0], coord[1])]
        except KeyError:
            raise KeyError(f"No qubit at coordinate {coord}.")


    # ------------------------------------------------------------------
    # Import/export
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            'origin': self.origin,
            'length': self.length,
            'height': self.length,
            'qubits': {c:q.to_dict() for c, q in self._qubits.items()}
        }

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------
    def select(self, region: Polygon, predicate='covers') -> Dict[Coord, Qubit]:
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
    
    def select_difference(self, polygon_A, polygon_B) -> Dict[Coord, Qubit]:
        """
        Select qubits in `polygon_A` that are not in `polygon_B`.
        """
        return self.select(difference(polygon_A, polygon_B))
    
    def select_rect_difference(self, origin_A: Coord, bound_A: Coord, origin_B: Coord, bound_B: Coord):
        box_A = box(origin_A[0], origin_A[1], bound_A[0], bound_A[1])
        box_B = box(origin_B[0]-1, origin_B[1]-1, bound_B[0]+1, bound_B[1]+1) #have to expand the subtracting space slightly with wonky checkerboarding setup
        return self.select_difference(box_A, box_B)
    
    # ------------------------------------------------------------------
    # Region querying
    # ------------------------------------------------------------------

    def is_empty_region_subset(self, origin_A: Coord, bound_A: Coord, origin_B: Coord, bound_B: Coord):
        region = self.select_rect_difference(origin_A, bound_A, origin_B, bound_B).values()
        
        for q in region:
            if q.is_active():
                return False
            else:
                continue
        return True

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


    