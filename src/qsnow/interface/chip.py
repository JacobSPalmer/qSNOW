from __future__ import annotations

from random import uniform
from typing import Dict, List, Optional, Tuple
from warnings import warn

from scipy.stats import truncnorm

from qsnow.visualize import VisualizationStyle, default_style, visualize

from .grid import Grid
from .models import Coord, NoiseProfile, Qubit
from .tile import LogicalTile


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
        """Returns a map of coords to noise profile of qubit as a dict."""
        return {coord: q.noise for coord, q in self._qubits.items()}

    @property
    def tile_map(self) -> Dict[Coord, LogicalTile]:
        """Returns a map of chip's logical tile by the respective tile origin as dict."""
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

    def generate_gaussian_noise(self, mean, deviation, rng=None):  # base26 "argonne"
        dist = truncnorm(
            (0.000001 - mean) / deviation,
            (1 - mean) / deviation,
            loc=mean,
            scale=deviation,
        )
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
            region_bound = (
                region_origin[0] + tile.length,
                region_origin[1] + tile.height,
            )
        else:
            # try and exactly place tile using tile's origin (defaults to (0,0))
            region_origin = tile.origin
            region_bound = tile.bound

        if not self._validate_tile_placements_with_warnings(tile, region_origin):
            return False

        # tile placement is valid and now modify the tile accordingly and add it to the chip
        tile.assign_chip(self, region_origin)
        self.tiles.append(tile)
        return True

    def remove_tile(self, index: int) -> LogicalTile:
        if not index < len(self.tiles):
            raise ValueError(
                f"Index {index} out of range for list of tiles with len {len(self.tiles)}"
            )

        tile = self.tiles.pop(index)
        tile.reset()
        return tile

    def is_valid_tile_placement(self, origin: Coord, bound: Coord) -> bool:
        # 1. check that the loc is valid for the checkerboard styling
        if not (self._validate_checkerboard_loc(origin)):
            return False

        # 2. check if any this tile would overlap with any other tile
        if not (self._validate_empty_region(origin, bound)):
            return False

        # 3. check that this tile is within the bounds of the chip itself
        if not (self._validate_chip_bounds(origin, bound)):
            return False

        return True

    def _validate_tile_placements_with_warnings(self, tile: LogicalTile, loc: Coord):
        origin = loc
        bound = (loc[0] + tile.length, loc[1] + tile.height)
        # 1. check that the loc is valid for the checkerboard styling
        if not (self._validate_checkerboard_loc(origin)):
            raise ValueError(
                f"Invalid tile placement. Both x and y must both be even or both be odd, given loc of ({origin[0]}, {origin[1]})"
            )

        # 2. check if any this tile would overlap with any other tile
        if not (self._validate_empty_region(origin, bound)):
            warn(
                f"Invalid tile placement. Qubit's within the ({origin} x {bound}) are currently active.",
                stacklevel=2,
            )
            return False

        # 3. check that this tile is within the bounds of the chip itself
        if not (self._validate_chip_bounds(origin, bound)):
            warn(
                f"Invalid tile placement. Placement at ({origin} x {bound}) overflows chip boundaries of ({self.origin} x {self.bound}).",
                stacklevel=2,
            )
            return False

        return True

    def _validate_checkerboard_loc(self, origin: Coord) -> bool:
        return origin[0] % 2 == origin[1] % 2

    # borderline unnecessary but keeps styling of constraint checks
    def _validate_empty_region(self, origin: Coord, bound: Coord) -> bool:
        return self.is_empty_region(origin, bound)

    def _validate_chip_bounds(self, origin: Coord, bound: Coord) -> bool:
        return (
            self.length >= bound[0]
            and self.height >= bound[1]
            and self.origin[0] <= origin[0]
            and self.origin[1] <= origin[1]
        )

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
            "dims": (
                self.length / 2,
                self.height / 2,
            ),  # the original input length/height used to initialize the chip (not the 2l x 2h checkerboard size although that is saved in the underlying grid)
            "grid": super().to_dict(),
            "tiles": {c: t.to_dict() for c, t in self.tile_map.items()},
            "noise_map": self.noise_map,
        }
