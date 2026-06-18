from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Optional, Self, overload, Tuple, Callable, Dict

@dataclass
class TileTag():
    distance: Optional[int] = None
    name: Optional[str] = None
    generator: Optional[Callable] = None 
    metadata = {}

class Status(Enum):
    INACTIVE = 0 # True if not within a logical patch; default state
    LOGICAL = 1 # True if actively used in a loaded logical patch
    ANCILLA = 2 # True if in a logical patch but not actively used by logical circuit

class CSSType(Enum):
    UNASSIGNED = 0
    X_CHECK = 1
    Z_CHECK = 2
    DATA = 3
    BUFFER = 4

type Coord = Tuple[float,float]
type ShiftFunction = Callable[[*tuple[float, ...]], tuple[float, ...]]

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
        if (self.min_value is not None and self.min_value > value) or (self.max_value is not None and self.max_value < value):
            raise ValueError(
                f"Value for '{self.name}' must be between {self.min_value if self.min_value is not None else "-INF"} and {self.max_value if self.max_value is not None else "INF"}. Given {value}."
            )
        setattr(obj, self.private_name, value)


class NoiseProfile:
    p = BoundedFloat(0.0, 0.75)

    def __init__(self, p: float = 0.0):
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class Qubit:
    def __init__(self,
                 loc: Optional[Coord] = None,
                 noise: Optional[NoiseProfile] = None,
                 status: Status = Status.INACTIVE,
                 type: CSSType = CSSType.UNASSIGNED):
        self.loc: Optional[Coord] = loc
        self.noise = noise if noise is not None else NoiseProfile()
        self.status = status
        self.type = type

    def is_active(self) -> bool:
        return False if self.status == Status.INACTIVE else True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loc={self.loc}, status={self.status}, noise={self.noise})"