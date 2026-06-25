from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Optional, Self, overload, Tuple, Callable, Dict, List
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

# TODO - extend the NoiseProfile to StaticNoiseProfile and DynamicProfile, where static has fixed independent noise profiles from the qubits around it and dynamic allows for noise to evolve or change (i.e., noise profile of a qubit changes over some set amount of time or use in operations)
# TODO - create an (or find the exisiting STIM) enum for representing the available noise channels and the available operations (that noise channels are appropriate to apply to). 
#        the noise profile should then map each operation to one (or perhaps multiple, like one channel for before one for after) noise channel. then reference the qubit's specific profile that says what channel and physical error rate should each operation on the qubit use.
class NoiseProfile:
    #TODO - start with seperating all operations into 3 buckets: 2-qubit (CNOT, SWAP, etc.), 1-qubit (H, Pauli's (X, Y, Z)), Idle/Measurement (M, MX, R, RX)
    #       this could be the de facto "default" noise profile of each qubit but implement it in such a way that the noise profile can be set manually so the profile supports each operation having it's own specific value for pre- and post- operation.
    p = BoundedFloat(0.0, 0.75)
    # _op_2_channel = Dict[]
    # _channel_2_per = Dict[]

    def for_operation(self, op_name, targs: "Optional[List[Qubit]]"):
        '''Return the physical noise for performing the gate on this qubit.'''
        return NotImplemented

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