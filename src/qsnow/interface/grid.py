from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from shapely import difference
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree

from .lattice import CHECKERBOARD, Lattice
from .models import Coord, Qubit, Tag


class Grid:
    """
    A rectangular block of qubit positions spanning x ∈ [0, length] and y ∈ [0, height]
    (both inclusive), in *coordinate* units.

    Which of those coordinates actually hold a qubit is decided by the grid's
    `Lattice` - `CHECKERBOARD` (the default, x % 2 == y % 2) or `SQUARE` (every
    integer coordinate). `unit_dims` converts the coordinate extent back to unit cells.

    Backed by a Shapely STRtree for efficient arbitrary-region selection.
    The index is built lazily on first query and cached until invalidated.

    Every grid carries a `Tag` for annotation (name/desc/metadata), which
    round-trips through serialization.
    """

    origin: Coord
    height: int
    length: int
    lattice: Lattice
    tag: Tag
    _qubits: Dict[Coord, Qubit]
    _index: Optional[Tuple[List[Coord], STRtree]]

    def __init__(
        self,
        length: int,
        height: int,
        origin: Coord = (0, 0),
        tag: Optional[Tag] = None,
        *,
        lattice: Lattice = CHECKERBOARD,
    ):
        if length < 1 or height < 1:
            raise ValueError(
                f"Grid length and height must be >= 1. Got length={length}, height={height}."
            )
        self.origin = origin
        self.length = length
        self.height = height
        self.lattice = lattice
        self.tag = tag if tag is not None else Tag()
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
        return self.footprint_for(self.origin, self.length, self.height)[1]

    @property
    def unit_dims(self) -> Tuple[int, int]:
        """The grid's size in unit cells, as opposed to coordinate units."""
        return self.lattice.units(self.length, self.height)

    @staticmethod
    def footprint_for(origin: Coord, length: int, height: int) -> Tuple[Coord, Coord]:
        """The `(origin, bound)` rectangle a `length x height` grid occupies at `origin`.

        `bound` is *inclusive* - the last coordinate actually covered - which is the
        single convention used for every region query in the package. Static so a
        placement can be evaluated before anything is moved there.
        """
        return origin, (origin[0] + length - 1, origin[1] + height - 1)

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

    def loc(self, coord: Union[Coord, Tuple[float, ...]]) -> Qubit:
        """Return the qubit at `coord`, raising KeyError if absent."""
        try:
            return self._qubits[(coord[0], coord[1])]
        except KeyError:
            raise KeyError(f"No qubit at coordinate {coord}.")

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------
    def select(self, region: Polygon, predicate="covers") -> Dict[Coord, Qubit]:
        """Return all qubits whose coordinates fall within or on `region`."""
        coords, tree = self._ensure_index()
        # predicate='covers' → region.covers(tree_item) → includes boundary points
        indices = tree.query(region, predicate="covers")
        return {coords[i]: self._qubits[coords[i]] for i in indices}

    def select_rect(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Dict[Coord, Qubit]:
        """
        Select qubits in the axis-aligned rectangle [x0, x1] × [y0, y1] (inclusive).
        Coordinates are coordinate units, not unit cells - on the checkerboard a cell is
        2 units wide/tall, so the top-left 2×2 cell block is select_rect(0, 0, 4, 4).
        """
        return self.select(box(x0, y0, x1, y1))

    def select_difference(self, polygon_A, polygon_B) -> Dict[Coord, Qubit]:
        """
        Select qubits in `polygon_A` that are not in `polygon_B`.
        """
        return self.select(difference(polygon_A, polygon_B))

    def select_rect_difference(
        self, origin_A: Coord, bound_A: Coord, origin_B: Coord, bound_B: Coord
    ):
        # B is grown by half a cell because `difference` yields a closed polygon and
        # `select` covers boundary points, so B's own outermost qubits would otherwise
        # survive the subtraction. The margin stays below one coordinate step on a dense
        # lattice, so it never reaches into a neighbour's qubits.
        margin = self.lattice.keepout_margin
        box_A = box(origin_A[0], origin_A[1], bound_A[0], bound_A[1])
        box_B = box(
            origin_B[0] - margin,
            origin_B[1] - margin,
            bound_B[0] + margin,
            bound_B[1] + margin,
        )
        return self.select_difference(box_A, box_B)

    # ------------------------------------------------------------------
    # Region querying
    # ------------------------------------------------------------------

    def is_empty_region_subset(
        self, origin_A: Coord, bound_A: Coord, origin_B: Coord, bound_B: Coord
    ):
        region = self.select_rect_difference(
            origin_A, bound_A, origin_B, bound_B
        ).values()

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
