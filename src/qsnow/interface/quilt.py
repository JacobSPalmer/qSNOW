from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from math import ceil
from typing import Dict, List, Optional, Tuple

from .chip import _COUPLER_DERIVATIONS, Chip, PlacementRejection, _join_coords
from .models import (
    ChipSpec,
    Coord,
    Coupler,
    CouplerKey,
    CouplerMode,
    NoiseProfile,
    Qubit,
    Tag,
    coupler_key,
)
from .noise import NoiseDistribution
from .tile import LogicalTile

__all__ = ["QuiltedChip"]


def _forwarded(name: str) -> property:
    """A property that reads and writes `name` on the instance's `_source`."""
    return property(
        lambda self: getattr(self._source, name),
        lambda self, value: setattr(self._source, name, value),
    )


class _StitchedQubit(Qubit):
    """A module qubit seen from the quilt: own location, the module qubit's state."""

    noise = _forwarded("noise")
    _status = _forwarded("_status")
    _type = _forwarded("_type")

    def __init__(self, loc: Coord, source: Qubit):
        self._source = source
        self._loc = loc


class _StitchedCoupler(Coupler):
    """A module coupler seen from the quilt: own ends, the module coupler's noise."""

    noise = _forwarded("noise")

    def __init__(self, ends: CouplerKey, source: Coupler):
        self._source = source
        self._ends = coupler_key(*ends)


class QuiltedChip(Chip):
    """
    A chip stitched together from smaller chips ("modules"), used like any `Chip`.

    `modules` maps each module's offset to the module. Offsets are coordinates, the
    same units as tile locations, and like a tile origin each must be a lattice site.
    Modules are expected to abut so that neighbouring modules link directly through
    couplers. Each module keeps its own frame and can still be edited on its own: the
    quilt only holds views of the module qubits and couplers in its frame, and every
    read or write goes through to the module. The state the quilt owns is its seam
    couplers - those joining two modules.

    Recipes run on the quilt itself (`generate_noise`, ...) write through to the module
    rates but are recorded on the quilt's `spec` alone. After editing a module's noise
    directly, `derive_seam_noise` brings the seams back in line with their endpoints.
    """

    def __init__(
        self,
        modules: Dict[Coord, Chip],
        *,
        tag: Optional[Tag] = None,
        spec: Optional[ChipSpec] = None,
    ):
        _check_modules(modules)
        #TODO - (naming convention - low) - this should probably be module_map to stikc with the convention used elsewhere (like noise_map, or tile_map, etc.)
        self.modules: Dict[Coord, Chip] = dict(modules)
        self._owners: Dict[Coord, Tuple[Chip, Coord]] = {}
        self._seams: Dict[CouplerKey, Coupler] = {}

        lattice = next(iter(modules.values())).lattice
        bounds = [_module_footprint(o, m)[1] for o, m in modules.items()]
        # Rounded up: a module offset to an odd site ends part-way through a unit cell.
        length, height = (
            ceil((max(b[axis] for b in bounds) + 1) / lattice.pitch) for axis in (0, 1)
        )
        super().__init__(length, height, tag=tag, spec=spec, lattice=lattice)
        self.derive_seam_noise(self.spec.coupler_mode)

    def _fill_lattice(self) -> None:
        for (dx, dy), module in self.modules.items():
            for (x, y), qubit in module.grid.items():
                coord = (x + dx, y + dy)
                self._qubits[coord] = _StitchedQubit(coord, qubit)
                self._owners[coord] = (module, (x, y))

    def _fill_couplers(self) -> None:
        for a, b in self.lattice.edges(self.length, self.height):
            if a not in self._owners or b not in self._owners:
                continue
            (module_a, local_a), (module_b, local_b) = self._owners[a], self._owners[b]
            key = coupler_key(a, b)
            if module_a is module_b:
                self._couplers[key] = _StitchedCoupler(
                    key, module_a.coupler(local_a, local_b)
                )
            else:
                self._couplers[key] = self._seams[key] = Coupler(key)

    # ------------------------------------------------------------------
    # Modules and seams
    # ------------------------------------------------------------------

    def locate(self, coord: Coord) -> Tuple[Chip, Coord]:
        """The module owning `coord`, and `coord` in that module's frame."""
        try:
            return self._owners[(coord[0], coord[1])]
        except KeyError:
            raise KeyError(f"No module owns coordinate {coord}.") from None

    def module_footprints(self) -> Dict[Coord, Tuple[Coord, Coord]]:
        """The `(origin, bound)` each module covers in the quilt, keyed by offset."""
        return {
            offset: _module_footprint(offset, m) for offset, m in self.modules.items()
        }

    @property
    def chiplets(self) -> List[Chip]:
        """The modules/chips, returned as an unordered/unindexed list for convenience."""
        return list(self.modules.values())

    @property
    def seam_couplers(self) -> List[Coupler]:
        """The couplers joining sites on two different modules."""
        return list(self._seams.values())

    @property
    def seam_map(self) -> Dict[CouplerKey, NoiseProfile]:
        """Returns a map of seam endpoint pairs to the seam coupler's noise profile."""
        return {ends: c.noise for ends, c in self._seams.items()}

    def seam(self, a: Coord, b: Coord) -> Coupler:
        """Return the coupler joining `a` and `b`, raising KeyError if they are unlinked."""
        try:
            return self._seams[coupler_key(a, b)]
        except KeyError:
            raise KeyError(
                f"No seam between {a} and {b}. On the '{self.lattice.name}' lattice "
                f"a coupler joins sites one of {self.lattice.neighbor_offsets} apart."
            ) from None

    def derive_seam_noise(self, mode: CouplerMode = "mean") -> None:
        """Set every seam rate from its two endpoints by `mode` (mean, max, min)."""
        combine = _COUPLER_DERIVATIONS[mode]
        for c in self.seam_couplers:
            c.noise.p = combine([self.loc(e).noise.p for e in c.ends])

    def generate_seam_noise(
        self, dist: NoiseDistribution, *, correlation: Optional[float] = None
    ) -> None:
        """
        Give the seam couplers the rates `dist` produces for them, leaving every other
        coupler as it is. `correlation` is as in `generate_coupler_noise`.
        """
        rates = dist.couplers(self, correlation=correlation)
        self._write_coupler_rates({k: n for k, n in rates.items() if k in self._seams})

    # ------------------------------------------------------------------
    # Tile placement
    # ------------------------------------------------------------------

    def placement_rejection(
        self,
        tile: LogicalTile,
        loc: Coord,
        *,
        ignoring: Optional[Tuple[Coord, Coord]] = None,
    ) -> Optional[PlacementRejection]:
        """`Chip.placement_rejection`, also rejecting qubits outside every module."""
        rejection = super().placement_rejection(tile, loc, ignoring=ignoring)
        if rejection is not None:
            return rejection
        if any(c not in self._owners for c in self._landing_coords(tile, loc)):
            return PlacementRejection(
                f"Invalid tile placement. Placement at {loc} puts qubits outside every "
                "module of the quilt.",
                fatal=False,
            )
        return None

    # ------------------------------------------------------------------
    # Input/output
    # ------------------------------------------------------------------

    def copy(self, copy_tiles: bool = False) -> QuiltedChip:
        """
        Returns a deepcopy of the quilt: every module, the seam rates, tag and spec.

        Optionally, the tiles placed on the quilt can be deep copied as well.
        """
        new_quilt = QuiltedChip(
            {offset: p.copy() for offset, p in self.modules.items()},
            tag=deepcopy(self.tag),
            spec=deepcopy(self.spec),
        )
        new_quilt._write_coupler_rates(self.seam_map)
        if copy_tiles:
            new_quilt.add_tiles({c: t.copy() for c, t in self.tile_map.items()})
        return new_quilt


def _check_modules(modules: Dict[Coord, Chip]) -> None:
    """Reject module maps the quilt cannot stitch into one consistent frame."""
    if not modules:
        raise ValueError("A QuiltedChip needs at least one module.")
    lattices = {p.lattice.name for p in modules.values()}
    if len(lattices) > 1:
        raise ValueError(f"All modules must share one lattice. Got {sorted(lattices)}.")
    if len({id(p) for p in modules.values()}) < len(modules):
        raise ValueError(
            "Each module must be a distinct Chip; copy a chip to reuse it."
        )
    if any(ox < 0 or oy < 0 for ox, oy in modules):
        raise ValueError("Module offsets must be non-negative coordinates.")
    lattice = next(iter(modules.values())).lattice
    for offset in modules:
        if not lattice.is_site(offset):
            nearby = [c for c in lattice.nearest_sites(offset) if min(c) >= 0]
            raise ValueError(
                f"Module offset {offset} is not a site of the '{lattice.name}' "
                f"lattice: {lattice.site_rule}. Nearest valid offsets: "
                f"{_join_coords(nearby)}."
            )
    footprints = {o: _module_footprint(o, m) for o, m in modules.items()}
    for (oa, (a0, a1)), (ob, (b0, b1)) in combinations(footprints.items(), 2):
        if a0[0] <= b1[0] and b0[0] <= a1[0] and a0[1] <= b1[1] and b0[1] <= a1[1]:
            raise ValueError(
                f"Modules at {oa} and {ob} overlap; modules may abut but not share "
                "sites."
            )
    if any(p.tiles for p in modules.values()):
        raise ValueError(
            "Modules must be stitched before any tiles are placed; place tiles on the "
            "quilt instead."
        )


def _module_footprint(offset: Coord, module: Chip) -> Tuple[Coord, Coord]:
    """The `(origin, bound)` a module covers in the quilt when placed at `offset`."""
    return Chip.footprint_for(offset, module.length, module.height)
