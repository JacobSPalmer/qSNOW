from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Self, Tuple, overload

type Coord = Tuple[float, float]
type ShiftFunction = Callable[[*tuple[float, ...]], Coord]


@dataclass
class TileTag:
    name: Optional[str] = None
    tile_type: Optional[str] = None
    initial_shift_fn: Optional[ShiftFunction] = None
    distance: Optional[int] = None
    rounds: Optional[int] = None
    generator: Optional[Callable] = None
    generator_args: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    # TODO - work thru metadata updates (i.e., default named dictionary and move most attr within (i.e., d, r, generator))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "tile_type": self.tile_type,
            "initial_shift_fn": self.initial_shift_fn,
            "generator": str(self.generator),
            "generator_args": self.generator_args,
            "metadata": self.metadata,
        }


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


class NoiseProfile:
    # TODO - start with seperating all operations into 3 buckets: 2-qubit (CNOT, SWAP, etc.), 1-qubit (H, Pauli's (X, Y, Z)), Idle/Measurement (M, MX, R, RX)
    #       this could be the de facto "default" noise profile of each qubit but implement it in such a way that the noise profile can be set manually so the profile supports each operation having it's own specific value for pre- and post- operation.
    p = BoundedFloat(0.0, 0.75)

    def __init__(self, p: float = 0.0):
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    # NOTE - mildly pointless right now but plan to expand noise profile so adding this now avoids work later
    def copy(self):
        return NoiseProfile(self.p)

    def to_dict(self):
        return {"p": self.p}


class Qubit:
    def __init__(
        self,
        loc: Optional[Coord] = None,
        noise: Optional[NoiseProfile] = None,
        status: Status = _DEFAULT_STATUS,
        type: CSSType = _DEFAULT_CSSTYPE,
    ):
        self.loc: Optional[Coord] = loc
        self.noise = noise if noise is not None else NoiseProfile()
        self._status = status
        self._type = type

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
        """Resets the qubit's type to the default typing, initially `UNAASSIGNED`"""
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
