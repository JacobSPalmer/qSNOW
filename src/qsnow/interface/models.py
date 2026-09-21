from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Self, Tuple, overload, TypeAlias

from functools import total_ordering
from numbers import Number

if TYPE_CHECKING:
    from .noise.distribution import NoiseDistribution

Coord: TypeAlias = Tuple[float, float]
ShiftFunction: TypeAlias = Callable[[*tuple[float, ...]], Coord]
CouplerKey: TypeAlias = Tuple[Coord, Coord]


@dataclass
class Tag:
    """
    Generic annotation attached to serializable objects (grids, tiles, experiments):
    a short `name`, a freeform `desc` for context not tracked elsewhere, and a
    `metadata` dict for anything structured that only humans read. Fields that
    code reads belong in a typed spec (see `TileSpec`), never in `metadata`.
    """

    name: Optional[str] = None
    desc: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class TileSpec:
    """
    How to rebuild a tile: read by the serialize importers and `copy()`.
    Callable fields (`generator`, `initial_shift_fn`) are not serialized — the
    generator is recreated by code-subclass constructors and the initial shift
    is already baked into the tile's base circuit.
    """

    tile_type: Optional[str] = None
    distance: Optional[int] = None
    rounds: Optional[int] = None
    generator: Optional[Callable] = None
    generator_args: Dict[str, Any] = field(default_factory=dict)
    initial_shift_fn: Optional[ShiftFunction] = None


CouplerMode: TypeAlias = Literal["mean", "max", "min"]


@dataclass
class ChipSpec:
    """
    State a chip's own code reads to rebuild its noise landscape: the distribution that
    produced the site rates, the one (if any) that produced the coupler rates and the
    cross-correlation it was applied with, and how coupler rates derive from their
    endpoints when no coupler distribution is in force.

    Typed and separate from `Tag.metadata` for the same reason `TileSpec` is: metadata
    is free-form text for humans, so anything the tool branches on must not live there
    where it can be reshaped, shared between copies, or silently dropped.
    """

    noise_model: Optional["NoiseDistribution"] = None
    coupler_mode: CouplerMode = "mean"
    # The coupler-side twin of `noise_model`, and the correlation it was applied with.
    # Both None whenever the couplers are derived from their endpoints or hand-set.
    coupler_model: Optional["NoiseDistribution"] = None
    coupler_correlation: Optional[float] = None


class Status(Enum):
    INACTIVE = 0  # True if not within a logical patch; default state
    LOGICAL = 1  # True if actively used in a loaded logical patch
    ANCILLA = 2  # True if in a logical patch but not actively used by logical circuit


class CSSType(Enum):
    UNASSIGNED = 0
    X_CHECK = 1
    Z_CHECK = 2
    DATA = 3
    BUFFER = 4


_DEFAULT_STATUS = Status.INACTIVE
_DEFAULT_CSSTYPE = CSSType.UNASSIGNED

class BoundedFloat:
    """Descriptor class enforcing min_value <= value <= max_value on assignment."""

    def __init__(self, min_value: Optional[float], max_value: Optional[float]):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    @overload
    def __get__(self, obj: None, objtype: Optional[type] = None) -> Self: ...
    @overload
    def __get__(self, obj: object, objtype: Optional[type] = None) -> float: ...
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return float(getattr(obj, self.private_name))

    def __set__(self, obj: object, value: float) -> None:
        if (self.min_value is not None and self.min_value > value) or (
            self.max_value is not None and self.max_value < value
        ):
            raise ValueError(
                f"Value for '{self.name}' must be between {self.min_value if self.min_value is not None else '-INF'} and {self.max_value if self.max_value is not None else 'INF'}. Given {value}."
            )
        setattr(obj, self.private_name, value)

@total_ordering
class NoiseProfile:
    # TODO - start with seperating all operations into 3 buckets: 2-qubit (CNOT, SWAP, etc.), 1-qubit (H, Pauli's (X, Y, Z)), Idle/Measurement (M, MX, R, RX)
    #       this could be the de facto "default" noise profile of each qubit but implement it in such a way that the noise profile can be set manually so the profile supports each operation having it's own specific value for pre- and post- operation.
    p = BoundedFloat(0.0, 0.75)

    def __init__(self, p: float = 0.0):
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
    
    def __eq__(self, value) -> bool:
        if isinstance(value, Number):
            return self.p == value
        elif isinstance(value, NoiseProfile):
            return self.p == value.p
        else:
            return NotImplemented

    def __lt__(self, value) -> bool:
        if isinstance(value, Number):
            return self.p < value
        elif isinstance(value, NoiseProfile):
            return self.p < value.p
        else:
            return NotImplemented

    # NOTE - mildly pointless right now but plan to expand noise profile so adding this now avoids work later
    def copy(self):
        return NoiseProfile(self.p)


class Qubit:
    def __init__(
        self,
        loc: Optional[Coord] = None,
        noise: Optional[NoiseProfile] = None,
        status: Status = _DEFAULT_STATUS,
        type: CSSType = _DEFAULT_CSSTYPE,
    ):
        self._loc: Optional[Coord] = loc
        self.noise = noise if noise is not None else NoiseProfile()
        self._status = status
        self._type = type

    @property
    def loc(self) -> Coord:
        if not self._loc:
            raise AttributeError('Qubit location not initialized.')
        return self._loc
    
    @loc.setter
    def loc(self, new_loc) -> None:
        self._loc = new_loc
        
    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = new_status

    @property
    def type(self) -> CSSType:
        return self._type

    @type.setter
    def type(self, new_type):
        self._type = new_type

    def reset(self) -> None:
        """Reset qubit's all relevant typing or status attributes, but leaves qubit's noise profile."""
        self.reset_status()
        self.reset_type()

    def reset_status(self) -> None:
        """Resets the qubit's status to the default, initially `INACTIVE`"""
        self._status = _DEFAULT_STATUS

    def reset_type(self) -> None:
        """Resets the qubit's type to the default typing, initially `UNASSIGNED`"""
        self._type = _DEFAULT_CSSTYPE

    def is_measure(self) -> bool:
        # print(f"is either z or x -> {self.is_z_measure() or self.is_x_measure()}")
        return self.is_z_measure() or self.is_x_measure()

    def is_z_measure(self) -> bool:
        return self.type == CSSType.Z_CHECK

    def is_x_measure(self) -> bool:
        return self.type == CSSType.X_CHECK

    def is_data(self) -> bool:
        return self.type == CSSType.DATA

    def is_active(self) -> bool:
        return True if self.status != Status.INACTIVE else False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loc={self.loc}, status={self.status}, type={self.type}, noise={self.noise})"


def coupler_key(a: Coord, b: Coord) -> CouplerKey:
    """
    Order-independent key for the edge `{a, b}`, so `(a, b)` and `(b, a)` name one coupler.

    A sorted tuple rather than a frozenset: it is deterministic, it round-trips through
    JSON, and it keeps `Coupler.ends` readable.
    """
    return (a, b) if a <= b else (b, a)


class Coupler:
    """
    The link between two adjacent qubits, owning the error rate of operations across it.

    A two-qubit gate's error on real hardware is dominated by the coupler joining the
    pair, not by the pair's endpoints - so the rate lives here rather than being derived
    from the two `Qubit.noise` values. Couplers belong to the *physical* chip, exactly as
    `Qubit.noise` does: a `LogicalTile` never owns one, and `shift_by` never carries one
    along (see `LogicalTile._transfer_qubit_metadata`).

    `ends` is canonical, so a coupler built as `((2,2), (1,1))` compares and keys the same
    as one built as `((1,1), (2,2))`. Validity is the `Chip`'s business - it builds every
    coupler off its `Lattice`, so adjacency holds by construction.
    """

    def __init__(self, ends: CouplerKey, noise: Optional[NoiseProfile] = None):
        self._ends: CouplerKey = coupler_key(*ends)
        self.noise = noise if noise is not None else NoiseProfile()

    @property
    def ends(self) -> CouplerKey:
        return self._ends

    @property
    def midpoint(self) -> Coord:
        """Halfway between the two endpoints - where a renderer would anchor the edge."""
        (x0, y0), (x1, y1) = self._ends
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def other(self, coord: Coord) -> Coord:
        """The endpoint opposite `coord`."""
        a, b = self._ends
        if coord == a:
            return b
        if coord == b:
            return a
        raise KeyError(f"Coordinate {coord} is not an endpoint of {self}.")

    def __contains__(self, coord: object) -> bool:
        return coord in self._ends

    def copy(self):
        return Coupler(self._ends, self.noise.copy())

    def __repr__(self) -> str:
        a, b = self._ends
        return f"{self.__class__.__name__}(ends={a}<->{b}, noise={self.noise})"
