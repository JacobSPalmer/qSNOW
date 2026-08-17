from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Any
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
from .lattice import CHECKERBOARD, Lattice
from .models import Coord, NoiseProfile, Qubit, Tag
from .tile import LogicalTile


class Chip(Grid):
    """
    Physical qubit chip, laid out on a `Lattice`.

    Chip(L, H) creates L x H *unit cells*; how many coordinates that spans depends on
    the lattice. On the default `CHECKERBOARD` a cell is 2 coordinate units wide, so
    Chip(5, 5) spans (0,0)-(9,9) and only positions satisfying x % 2 == y % 2 hold a
    qubit. On `SQUARE` a cell is one coordinate, so Chip(5, 5) spans (0,0)-(4,4) with a
    qubit at every integer position.

    The grid is described using selection spaces of polygons, so a 5x5 checkerboard
    logical tile occupies box(0, 0, 10, 10).

    Typical workflow:
      1. Instantiate Chip(L, H) and assign noise to individual qubits (or using the pre-defined noise samplers).
      2. Build, place, and shift LogicalTiles within the chip by specifying origin points or movement shifts.
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
        lattice: Lattice = CHECKERBOARD,
    ):
        super().__init__(*lattice.span(length, height), tag=tag, lattice=lattice)
        self._fill_lattice()
        self.tiles: List[LogicalTile] = []

        if noise_map:
            self.set_noise_map(noise_map)
        if tiles:
            self.add_tiles(tiles)

    @classmethod
    def from_tile(cls, tile: LogicalTile):
        "Chip constructor that builds a chip fit to a specific tile"
        l, h = tile.unit_dims
        return cls(l, h, noise_map=None, tiles={(0, 0): tile}, lattice=tile.lattice)

    def _fill_lattice(self) -> None:
        """Populate every position the lattice puts a qubit on with a default Qubit."""
        for coord in self.lattice.positions(self.length, self.height):
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
            "unit_size": self.unit_dims,
            "grid_size": (self.length, self.height),
            "lattice": self.lattice.name,
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

    # TODO - this works and serves it's purpose just fine for now but needs a revist and cleanup down the line
    def generate_derived_contour_noise(
            self, mean: float, deviation: float, seed: int | None = None, *, slope: int = 5, 
        ):
        from qsnow.experiments import SquarePackingExp
        from .codes import SCTile

        def adjust_map_dimensions(noise_map: Dict[Coord, Any], dims: Tuple[float, float]) -> Dict[Coord, float]:
            for coord in list(noise_map.keys()):
                if coord[0] >= dims[0] or coord[1] >= dims[1]:
                    noise_map.pop(coord)
            return noise_map

        def scale_gaussian_contour(coord_dict, new_deviation, alpha_ratio=0.1):
            import numpy as np
            coords = list(coord_dict.keys())
            orig_values = np.array(list(coord_dict.values()), dtype=float)
            deviation_factor = new_deviation / np.std(orig_values)
            
            target_mean = np.mean(orig_values)
            scaled_linear = target_mean + deviation_factor * (orig_values - target_mean)
            
            # exponential soft-clipping to the lower tail near zero
            # alpha is our soft lower bound fence (e.g., 10% of the mean)
            # this means we get close to zero but no qubit ever gets an absolute 0 error rate
            alpha = target_mean * alpha_ratio 
            
            # smooth C1-continuous blending function
            final_values = np.where(
                scaled_linear >= alpha,
                scaled_linear,
                alpha * np.exp((scaled_linear - alpha) / alpha)
            )
            
            # Step 3: Shift slightly to correct any minor mean drift caused by the tail smoothing
            mean_drift = np.mean(final_values) - target_mean
            final_values = final_values - mean_drift
            
            # Safety double-check: if the shift pushed anything below alpha, clamp it smoothly
            final_values = np.where(
                final_values >= alpha, 
                final_values, 
                alpha * np.exp((final_values - alpha) / alpha)
            )
            
            return {coords[i]: final_values[i] for i in range(len(coords))}


        if isinstance(seed, int):
            rng_seed = seed
        else:
            rng_seed = self._generate_rng_seed()

        buffer_l, buffer_h = self.unit_dims
        buffer_chip = Chip(
            buffer_l + slope + 1, buffer_h + slope + 1, lattice=self.lattice
        )
        buffer_chip.generate_gaussian_noise(mean, deviation, rng_seed)

        buffed_map = scale_gaussian_contour(
            adjust_map_dimensions(SquarePackingExp(buffer_chip, SCTile(slope))._average_per_for_candidate_placements(), (self.length, self.height)),
            deviation)

        self._set_noise_metadata("derived contour", mean = mean, deviation = deviation, slope = slope, seed = rng_seed)
        self.set_noise_map({c:NoiseProfile(p) for c,p in buffed_map.items()})


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
        # without a `loc`, try and exactly place tile using tile's origin (defaults to (0,0))
        region_origin = loc if loc else tile.origin

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

    def candidate_placements(self, tile: LogicalTile) -> List[Coord]:
        """Every origin on this chip where `tile` could currently be placed."""
        return [
            (i, j)
            for i in range(self.length)
            for j in range(self.height)
            if self.is_valid_tile_placement(tile, (i, j))
        ]

    def is_valid_tile_placement(self, tile: LogicalTile, loc: Coord) -> bool:
        footprint = self.footprint_for(loc, tile.length, tile.height)

        # 1. check that the loc is a valid site on this chip's lattice
        if not (self._validate_lattice_origin(loc)):
            return False

        # 2. check the tile's own qubits all land on chip sites at this loc
        if not (self._validate_tile_coords_on_lattice(tile, loc)):
            return False

        # 3. check if any this tile would overlap with any other tile
        if not (self._validate_empty_region(footprint)):
            return False

        # 4. check that this tile is within the bounds of the chip itself
        if not (self._validate_chip_bounds(footprint)):
            return False

        return True

    def _validate_tile_placements_with_warnings(self, tile: LogicalTile, loc: Coord):
        footprint = self.footprint_for(loc, tile.length, tile.height)
        origin, bound = footprint

        # 1. check that the loc is a valid site on this chip's lattice
        if not (self._validate_lattice_origin(origin)):
            raise ValueError(
                f"Invalid tile placement for a '{self.lattice.name}' lattice: "
                f"{self.lattice.site_rule}, given loc of ({origin[0]}, {origin[1]})"
            )

        # 2. check the tile's own qubits land on chip sites once moved to `origin`.
        # Lattices nest (every checkerboard site is a square site, not the reverse), so
        # this is decided per-coordinate rather than by comparing the two lattices.
        if not (self._validate_tile_coords_on_lattice(tile, origin)):
            raise ValueError(
                f"Tile on a '{tile.lattice.name}' lattice cannot be placed at {origin} on a "
                f"'{self.lattice.name}' chip: some of its qubits would land off-lattice, "
                f"where the chip has no qubit. Chip rule: {self.lattice.site_rule}."
            )

        # 3. check if any this tile would overlap with any other tile
        if not (self._validate_empty_region(footprint)):
            warn(
                f"Invalid tile placement. Qubit's within the ({origin} x {bound}) are currently active.",
                stacklevel=2,
            )
            return False

        # 4. check that this tile is within the bounds of the chip itself
        if not (self._validate_chip_bounds(footprint)):
            warn(
                f"Invalid tile placement. Placement at ({origin} x {bound}) overflows chip boundaries of ({self.origin} x {self.bound}).",
                stacklevel=2,
            )
            return False

        return True

    def _validate_lattice_origin(self, origin: Coord) -> bool:
        return self.lattice.is_site(origin)

    def _validate_tile_coords_on_lattice(
        self, tile: LogicalTile, origin: Coord
    ) -> bool:
        """Whether every qubit of `tile` lands on a chip site when moved to `origin`."""
        dx, dy = origin[0] - tile.origin[0], origin[1] - tile.origin[1]
        return all(
            self.lattice.is_site((x + dx, y + dy)) for x, y in tile._c2i
        )

    def _keepout(self, footprint: Tuple[Coord, Coord]) -> Tuple[Coord, Coord]:
        """The region a footprint reserves: itself plus a `+x`/`+y` margin.

        The margin is the half-cell a footprint shares with the next lattice site.
        It is also what keeps two tiles from being placed flush against each other,
        which is why it is preserved rather than dropped - existing chips (and the
        multi-tile flakes that re-place their tiles on import) were built under it.
        """
        origin, bound = footprint
        margin = self.lattice.keepout_margin
        return origin, (bound[0] + margin, bound[1] + margin)

    # borderline unnecessary but keeps styling of constraint checks
    def _validate_empty_region(self, footprint: Tuple[Coord, Coord]) -> bool:
        return self.is_empty_region(*self._keepout(footprint))

    def _validate_chip_bounds(self, footprint: Tuple[Coord, Coord]) -> bool:
        origin, bound = footprint
        return (
            self.origin[0] <= origin[0]
            and self.origin[1] <= origin[1]
            and bound[0] <= self.bound[0]
            and bound[1] <= self.bound[1]
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
            *self.unit_dims,
            noise_map=self.noise_map,
            tag=self.tag,
            lattice=self.lattice,
        )
        if copy_tiles:
            new_chip.add_tiles({c: t.copy() for c, t in self.tile_map.items()})
        return new_chip
