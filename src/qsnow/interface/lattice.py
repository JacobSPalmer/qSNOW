from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

from .models import Coord

__all__ = [
    "Lattice",
    "CHECKERBOARD",
    "SQUARE",
    "register_lattice",
    "lattice_by_name",
    "known_lattices",
]


@dataclass(frozen=True)
class Lattice:
    """
    Which integer coordinates hold a qubit, and how coordinates relate to unit cells.

    A lattice is defined entirely by its `pitch` - the number of coordinate units per
    unit cell - because one predicate covers every case we care about:

        is_site(x, y)  <->  x % pitch == y % pitch

    With `pitch=2` that is the checkerboard rule (`x % 2 == y % 2`, half the integer
    points occupied); with `pitch=1` it is trivially true, giving a dense integer
    lattice where every coordinate holds a qubit. Stim imposes no lattice of its own -
    only its *rotated* surface-code generators happen to emit checkerboard coordinates -
    so a `Chip` must be able to host either.

    Frozen, so instances are hashable and compare by value. Subclass and override
    `is_site` for a lattice that a single pitch cannot express (heavy-hex, triangular).
    """

    name: str
    pitch: int

    def __post_init__(self) -> None:
        if self.pitch < 1:
            raise ValueError(f"Lattice pitch must be >= 1. Got {self.pitch}.")

    # ------------------------------------------------------------------
    # Sites
    # ------------------------------------------------------------------

    def is_site(self, coord: Coord) -> bool:
        """Whether `coord` is a position this lattice puts a qubit on."""
        return coord[0] % self.pitch == coord[1] % self.pitch

    def positions(self, length: int, height: int) -> Iterator[Coord]:
        """Every site within a `length x height` *coordinate* extent."""
        for x in range(length):
            for y in range(height):
                if self.is_site((x, y)):
                    yield (x, y)

    # ------------------------------------------------------------------
    # Unit cells <-> coordinates
    # ------------------------------------------------------------------

    def span(self, length: int, height: int) -> Tuple[int, int]:
        """Coordinate extent occupied by a `length x height` block of unit cells."""
        return (self.pitch * length, self.pitch * height)

    def units(self, length: int, height: int) -> Tuple[int, int]:
        """Unit cells spanned by a `length x height` coordinate extent."""
        return (length // self.pitch, height // self.pitch)

    @property
    def keepout_margin(self) -> float:
        """
        Extra coordinate margin a footprint reserves beyond its own bound.

        Half a cell - the space a footprint shares with the next lattice site. On the
        checkerboard this is `1.0`, which is what keeps two tiles from being placed
        flush against each other; it is preserved rather than dropped because existing
        chips (and the multi-tile flakes that re-place their tiles on import) were
        built under it. On a dense lattice it is `0.5`: large enough to defeat
        Shapely's closed-set `covers` predicate on a footprint's own boundary, small
        enough to never reach the neighbouring site.
        """
        return self.pitch / 2

    @property
    def site_rule(self) -> str:
        """Human-readable statement of what makes a coordinate valid, for error text."""
        if self.pitch == 1:
            return "any integer coordinate is valid"
        if self.pitch == 2:
            return "x and y must both be even or both be odd"
        return f"x % {self.pitch} must equal y % {self.pitch}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def infer(cls, coords: Iterable[Coord]) -> Lattice:
        """
        Guess the lattice a set of circuit coordinates lives on.

        Opt-in only (`lattice="auto"`), never a default: the evidence for "checkerboard"
        is the *absence* of parity-violating coordinates, which a small or degenerate
        circuit provides by accident. Raises rather than guessing when the coordinates
        cannot discriminate, so the failure is a loud one at construction rather than a
        confusing placement rejection later.
        """
        sites: List[Coord] = [(c[0], c[1]) for c in coords]
        if any(float(x) != int(x) or float(y) != int(y) for x, y in sites):
            raise ValueError(
                "Cannot infer a lattice from non-integer qubit coordinates. Rescale the "
                "circuit onto an integer grid with the tile's `initial_shift` first."
            )

        distinct = set(sites)
        if len(distinct) < 4:
            raise ValueError(
                f"Cannot infer a lattice from {len(distinct)} distinct coordinate(s); "
                "too few to tell a checkerboard from a dense grid. Pass `lattice=` explicitly."
            )
        if len({x for x, _ in distinct}) == 1 or len({y for _, y in distinct}) == 1:
            raise ValueError(
                "Cannot infer a lattice from coordinates on a single row or column; a "
                "1D chain satisfies every lattice. Pass `lattice=` explicitly."
            )

        return CHECKERBOARD if all(CHECKERBOARD.is_site(c) for c in distinct) else SQUARE


CHECKERBOARD = Lattice(name="checkerboard", pitch=2)
SQUARE = Lattice(name="square", pitch=1)

_LATTICES: Dict[str, Lattice] = {l.name: l for l in (CHECKERBOARD, SQUARE)}


def register_lattice(lattice: Lattice) -> None:
    """Register a lattice by name so serialization can rebuild it."""
    _LATTICES[lattice.name] = lattice


def lattice_by_name(name: str) -> Lattice:
    """Look up a registered lattice, raising a directed error if it is unknown."""
    try:
        return _LATTICES[name]
    except KeyError:
        raise KeyError(
            f"Unknown lattice '{name}'. Known lattices: {sorted(_LATTICES)}. "
            "Register custom lattices with register_lattice() before importing."
        ) from None


def known_lattices() -> List[str]:
    return sorted(_LATTICES)
