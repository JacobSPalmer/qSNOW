from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple
from warnings import warn

from numpy.random import SeedSequence, default_rng
from scipy.stats import truncnorm, uniform

from qsnow.visualize import (
    VisualizationStyle,
    default_interactive_styles,
    visualize,
    visualize_interactive,
)

from .grid import Grid
from .models import Coord, NoiseProfile, Qubit, Tag
from .tile import LogicalTile


class Chip(Grid):
    """
    Physical qubit chip as a checkerboard integer lattice.

    Chip(L, H) creates a grid spanning (0,0)–(2L - 1, 2H - 1) where valid qubit positions
    satisfy x % 2 == y % 2 (both even or both odd). "Unit cells" are 2 coordinate
    units wide, so a 5×5 logical tile with origin (0,0) has bound of (9, 9)

    The grid is described using selection spaces of polygons, so the 5x5 logical tile
    occupies box(0, 0, 10, 10).

    Typical workflow:
      1. Instantiate Chip(L, H) and assign noise to individual qubits (or using the pre-defined noise samplers).
      2. Build, place, and shift LogicalTiles within the chip by specifying origin points or movement shifts within the (2L x 2H) chip.
      3. Retrieve and modify noise-injected circuits by accessing `tile.circuit`.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        length: int,
        height: int,
        *,
        noise_map: Optional[Dict[Coord, NoiseProfile]] = None,
        tiles: Optional[Dict[Coord, LogicalTile]] = None,
        tag: Optional[Tag] = None,
    ):
        super().__init__(2 * length, 2 * height, tag=tag)
        self._fill_checkerboard()
        self.tiles: List[LogicalTile] = []

        if noise_map:
            self.set_noise_map(noise_map)
        if tiles:
            self.add_tiles(tiles)

    @classmethod
    def from_tile(cls, tile: LogicalTile):
        "Chip constructor that builds a chip fit to a specific"
        l = (
            tile.length // 2
        )  # Because height and length get converted to 0 indexed 2L x 2H chip
        h = tile.height // 2
        return cls(l, h, noise_map=None, tiles={(0, 0): tile})

    def _fill_checkerboard(self) -> None:
        """Populate all valid checkerboard positions with default Qubits."""
        for x in range(self.length):
            for y in range(self.height):
                if x % 2 == y % 2:
                    coord: Coord = (x, y)
                    self._qubits[coord] = Qubit(loc=coord)

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

    def summary(self) -> Dict[str, object]:
        """Compact facts describing the chip, for display surfaces
        (visualization headers, HTML exports, reprs)."""
        return {
            "unit_size": (self.length // 2, self.height // 2),
            "grid_size": (self.length, self.height),
            "n_qubits": len(self.qubits),
            "n_tiles": len(self.tiles),
            "noise_model": self.tag.metadata.get("noise_model"),
        }

    # ------------------------------------------------------------------
    # Noise manipulation
    # ------------------------------------------------------------------
    def _generate_rng_seed(self):
        return SeedSequence().entropy

    # TODO - at some point this should just become a dataclass and can migrate flakes to always have this present
    def _set_noise_metadata(self, name, **kwargs):
        self.tag.metadata["noise_model"] = {"name": name, **kwargs}

    def set_noise_map(self, noise_map: Dict[Coord, NoiseProfile] | Dict[Coord, float]):
        for c, n in noise_map.items():
            self.loc(c).noise = (
                NoiseProfile(n.p) if isinstance(n, NoiseProfile) else NoiseProfile(n)
            )

    def generate_random_noise(
        self, range: Tuple[float, float] = (0.01, 0.05), seed: int | None = None
    ):
        """
        Sets the `NoiseProfile.p` value for each qubit to a `BoundedFloat(0, 0.75)` randomly sampled from a uniform
        distribution with an upper and lower bound of the range provided.

        The `rng` determines if the random value sampling with use a seeded generator. `rng` can be provided as
        an integer value that will become the seed for an `np.random.Generator` or `None` (default).
        """
        if isinstance(seed, int):
            rng_seed = seed
        else:
            rng_seed = self._generate_rng_seed()
        rng = default_rng(seed=rng_seed)

        self._set_noise_metadata("uniform random", range=range, seed=rng_seed)

        dist = uniform(loc=range[0], scale=range[1])
        for q in self.qubits:
            q.noise.p = round(dist.rvs(1, random_state=rng)[0], 5)

    def generate_gaussian_noise(
        self, mean, deviation, seed: int | None = None
    ):  # base26 "argonne"
        """
        Sets the `NoiseProfile.p` value for each qubit to a `BoundedFloat(0, 0.75)` randomly sampled from a truncated gaussian
        distribution with the mean and deviation provided.

        The `rng` determines if the random value sampling with use a seeded generator. `rng` can be provided as a `np.random.Generator`,
        an integer value that will become the seed for an `np.random.Generator`, or `None` (default).
        """
        if isinstance(seed, int):
            rng_seed = seed
        else:
            rng_seed = self._generate_rng_seed()
        rng = default_rng(seed=rng_seed)

        self._set_noise_metadata(
            "gaussian", mean=mean, deviation=deviation, seed=rng_seed
        )

        dist = truncnorm(
            (0.0000000001 - mean) / deviation,
            (1 - mean) / deviation,
            loc=mean,
            scale=deviation,
        )
        for q in self.qubits:
            q.noise.p = round(dist.rvs(1, random_state=rng)[0], 5)

    def generate_uniform_noise(self, p):
        """
        Sets the `NoiseProfile.p` value for each qubit to a `BoundedFloat(0, 0.75)` with a physical error rate
        of the `p` provided.
        """

        self._set_noise_metadata("uniform homogeneous", p=p)

        for q in self.qubits:
            q.noise.p = p

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

    def add_tiles(
        self, tile_map: Dict[Coord, LogicalTile], *, place_valid_subset=False
    ) -> None:
        """
        Add multiple `LogicalTiles` per the provided `Coord` -> `LogicalTile` map provided.
        If `place_valid_subset` is `True`, any tile in the map with a valid loc will be placed regardless of if the other tiles validity.
        """

        if not place_valid_subset and any(
            not self._validate_tile_placements_with_warnings(t, c)
            for c, t in tile_map.items()
        ):
            return

        for c, t in tile_map.items():
            self.add_tile(t, c)

    def clear_tiles(self) -> None:
        """
        Remove and clean all tiles currently on the chip.
        """
        for i, t in enumerate(self.tiles):
            self.pop_tile(i)

    def pop_tile(self, index: int) -> LogicalTile:
        """
        Pop and clean the tile at specified index in `Chip.tiles` list.

        Important to note, this does not delete the tile object but instead scrubs the qubits
        within the tile, removes the chip from the tile (and vice versa), and returns the clean tile.
        """
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

    def show(
        self,
        style: Optional[VisualizationStyle] = None,
        *,
        extra_styles: Optional[Mapping[str, VisualizationStyle]] = None,
        interactive: bool = False,
    ) -> None:
        """
        Display a visualization of the chip's qubit layout.

        With no arguments, shows an interactive figure with a dropdown to switch
        between the standard views (status / CSS type / noise heatmap), extended
        by any `extra_styles` (name -> style). Passing `style` renders that
        single style statically instead.
        """
        if style is not None:
            visualize(self, style=style, show=True)
            return
        elif not interactive:
            visualize(self, show=True)
            return
        else:
            styles = default_interactive_styles(self)
            styles.update(extra_styles or {})
            visualize_interactive(self, styles, show=True)

    # ------------------------------------------------------------------
    # Input/ouput
    # ------------------------------------------------------------------

    def copy(self, copy_tiles: bool = False) -> Chip:
        """
        Returns a deepcopy of the chip with all parameters.

        Optionally, the tiles placed on the chips can be deep copied as well.
        """
        new_chip = Chip(
            self.length // 2, self.height // 2, noise_map=self.noise_map, tag=self.tag
        )
        if copy_tiles:
            new_chip.add_tiles({c: t.copy() for c, t in self.tile_map.items()})
        return new_chip
