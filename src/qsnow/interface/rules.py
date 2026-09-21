from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Literal, Optional, TypeAlias, Union

from .models import Coord, Coupler, Qubit

# if TYPE_CHECKING:
#     from interface.models import Qubit, CSSType, Status

# ------------------------------------------------------------------
# Relevant gate name refs
# ------------------------------------------------------------------
# TODO - move these into a the default typing/object file for the when refactor to the /rules subfolder
_NOISE_CHANNEL_OPS = [
    "CORRELATED_ERROR",
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "E",
    "ELSE_CORRELATED_ERROR",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
]
_MULTI_Q_NOISE_CHANNEL_OPS = ["CORRELATED_ERROR", "DEPOLARIZE2", "PAULI_CHANNEL_2"]
_TWO_CLIFFORD_OPS = [
    "CNOT",
    "CX",
    "CY",
    "CZ",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
]
_SINGLE_CLIFFORD_OPS = [
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "S_DAG",
]
_PAULI_OPS = ["I", "X", "Y", "Z"]
_MEASURE_OPS = ["M", "MPP", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ"]
_RESETS_OPS = ["R", "RX", "RY", "RZ", "MR", "MRX", "MRY", "MRZ"]
_ANNOTATION_OPS = [
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "TICK",
    "REPEAT",
]

# ------------------------------------------------------------------
# Operational bool helpers
# ------------------------------------------------------------------


def is_noise_channel(name) -> bool:
    return name in _NOISE_CHANNEL_OPS


def is_two_qubit_noise_channel(name) -> bool:
    return name in _MULTI_Q_NOISE_CHANNEL_OPS


def is_two_qubit(name) -> bool:
    return name in _TWO_CLIFFORD_OPS


def is_single_qubit(name) -> bool:
    return name in _SINGLE_CLIFFORD_OPS or name in _PAULI_OPS


def is_reset(name) -> bool:
    return name in _RESETS_OPS


def is_measurement(name) -> bool:
    return name in _MEASURE_OPS


def is_collapsing(name) -> bool:
    return name in _RESETS_OPS or name in _MEASURE_OPS


def is_annotation(name) -> bool:
    return name in _ANNOTATION_OPS


# ------------------------------------------------------------------
# Rule triggers -> When to trigger the rule
# ------------------------------------------------------------------


# TODO - move these to a _default_triggers/_default_filters file or something along those lines
def _blank_trigger(targ_indexes: List[List[int]], qubits: Dict[int, Qubit]) -> bool:
    return False


def _any_qubits_trigger(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> bool:
    return True


def _all_qubits_trigger(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> bool:
    return all(all(k in qubits for k in i) for i in targ_indexes)


def _data_trigger(targ_indexes: List[List[int]], qubits: Dict[int, Qubit]) -> bool:
    return all(all(qubits.get(t, Qubit()).is_data() for t in i) for i in targ_indexes)


def _all_measures_trigger(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> bool:
    b = all(all(qubits.get(t, Qubit()).is_measure() for t in i) for i in targ_indexes)
    # print(f'triggered on all_measures -> {b}')
    return b
    # return all(all(qubits.get(t, Qubit()).is_measure() for t in i) for i in targ_indexes)


def _x_measures_trigger(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> bool:
    b = all(all(qubits.get(t, Qubit()).is_x_measure() for t in i) for i in targ_indexes)
    # print(f'triggered on all_measures -> {b}')
    return b


def _z_measures_trigger(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> bool:
    return all(
        all(qubits.get(t, Qubit()).is_z_measure() for t in i) for i in targ_indexes
    )


# -----------------------------------------------------------------------
# Rule filters -> What the qubits specific channels should be applied to
# -----------------------------------------------------------------------


def _blank_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return []


def _all_qubits_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return [[k] for k in qubits]


def _active_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return targ_indexes


def _idle_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    # print(f'{[item for sublist in targ_indexes for item in sublist]}')
    # print(f"idle qubits:{[[k] for k in qubits if k not in [item for sublist in targ_indexes for item in sublist]]}")
    return [
        [k]
        for k in qubits
        if k not in [item for sublist in targ_indexes for item in sublist]
    ]


def _all_measures_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return [[k] for k, v in qubits.items() if v.is_measure()]


def _data_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return [[k] for k, v in qubits.items() if v.is_data()]


def _x_measures_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return [[k] for k, v in qubits.items() if v.is_x_measure()]


def _z_measures_filter(
    targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
) -> List[List[int]]:
    return [[k] for k, v in qubits.items() if v.is_z_measure()]


# -----------------------------------------------------------------------
# Rate sources -> Where a channel's error rate is read from
# -----------------------------------------------------------------------
# A channel rule already says *what* channel and *which* qubits; a source says where its
# rate comes from. Keeping that a named, swappable axis is what lets an edge-owned rate
# (a Coupler) coexist with the endpoint-derived rates every single-qubit rule still wants.


def _qubit_mean_source(targets, qubits, coupler_at) -> float:
    """The mean rate of the qubits in the group - the historical, and default, behavior."""
    return mean([qubits[i].noise.p for i in targets])


def _qubit_max_source(targets, qubits, coupler_at) -> float:
    """The worst rate among the qubits in the group."""
    return max(qubits[i].noise.p for i in targets)


def _coupler_source(targets, qubits, coupler_at) -> float:
    """
    The rate owned by the coupler joining a pair of qubits.

    Raises rather than falling back to an endpoint average: a two-qubit gate on a pair the
    chip does not physically couple is a modelling error, and silently averaging it would
    hide exactly the defect this source exists to expose. Register a custom source if you
    want lenient behavior.
    """
    if len(targets) != 2:
        raise ValueError(
            f"The 'coupler' rate source needs exactly 2 target qubits, got {len(targets)}. "
            "Pair it with a filter that preserves operand groups, such as 'active'."
        )
    a, b = (qubits[i].loc for i in targets)
    coupler = coupler_at(a, b)
    if coupler is None:
        raise ValueError(
            f"No coupler joins {a} and {b}, so the 'coupler' rate source cannot price an "
            "operation across them. The gate spans a non-adjacent pair."
        )
    return coupler.noise.p


# ------------------------------------------------------------------
# Basic rule setups
# ------------------------------------------------------------------


# TODO - move these to a rule_types or something along those lines
TriggerFunc: TypeAlias = Callable[
    [List[List[int]], Dict[int, Qubit]], bool
]  # specifies when to trigger rule
FilterFunc: TypeAlias = Callable[
    [List[List[int]], Dict[int, Qubit]], List[List[int]]
]  # specifies what (qubits) to trigger the rule on
CouplerLookup: TypeAlias = Callable[
    [Coord, Coord], Optional[Coupler]
]  # resolves the coupler joining two chip coordinates, or None
RateFunc: TypeAlias = Callable[
    [List[int], Dict[int, Qubit], CouplerLookup], float
]  # specifies where the rule's error rate is read from


@dataclass
class Filter:
    name: str = "empty"
    func: FilterFunc = _blank_filter


@dataclass
class Trigger:
    name: str = "empty"
    func: TriggerFunc = _blank_trigger


@dataclass
class Source:
    # Defaults to the qubit mean, so an unrecognized source name degrades to the
    # historical behavior rather than to silence (as unknown triggers/filters do).
    name: str = "qubit_mean"
    func: RateFunc = _qubit_mean_source


@dataclass
class ChannelRule:
    channel: Literal[
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    ]
    filter: Union[
        Literal[
            "active",
            "idle",
            "all_qubits",
            "all_measures",
            "data",
            "x_measures",
            "z_measures",
        ],
        str,
    ]  # This filter determines what qubits this channel will be applied to
    scalar: float = 1.0
    name: Optional[str] = None
    # This source determines where the unscaled error rate is read from. Declared last so
    # no positional caller of the fields above breaks.
    source: Union[
        Literal["qubit_mean", "qubit_max", "coupler"],
        str,
    ] = "qubit_mean"

    def __repr__(self) -> str:
        return f"ChannelRule({f'name={self.name},' if self.name is not None else ''}channel={self.channel}, on_qubits={self.filter}, rate={self.scalar}p, from={self.source})"


@dataclass
class InjectionRule:
    operation: str
    trigger: Union[
        Literal[
            "any", "all_qubits", "all_measures", "data", "x_measures", "z_measures"
        ],
        str,
    ]  # This filter determines IF the before or after channels will be applied to this operation
    before: List[ChannelRule] = field(default_factory=list)
    after: List[ChannelRule] = field(default_factory=list)
    exclusive: bool = False  # if true, once this rule fires (its trigger passes) no lower priority rules are checked for that instruction
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"InjectionRule({f'name={self.name},' if self.name is not None else ''}gate={self.operation}, on_qubits={self.trigger}, before = [{', '.join([str(r) for r in self.before])}], after = [{','.join([str(r) for r in self.after])}])"


_DEFAULT_FILTERS: List[Filter] = [
    Filter("active", _active_filter),
    Filter("idle", _idle_filter),
    Filter("all_qubits", _all_qubits_filter),
    Filter("all_measures", _all_measures_filter),
    Filter("data", _data_filter),
    Filter("x_measures", _x_measures_filter),
    Filter("z_measures", _z_measures_filter),
]
_DEFAULT_SOURCES: List[Source] = [
    Source("qubit_mean", _qubit_mean_source),
    Source("qubit_max", _qubit_max_source),
    Source("coupler", _coupler_source),
]
_DEFAULT_TRIGGERS: List[Trigger] = [
    Trigger("any", _any_qubits_trigger),
    Trigger("all_qubits", _all_qubits_trigger),
    Trigger("all_measures", _all_measures_trigger),
    Trigger("data", _data_trigger),
    Trigger("x_measures", _x_measures_trigger),
    Trigger("z_measures", _z_measures_trigger),
]


class Ruleset:
    def __init__(self, injection_rules: Optional[List[InjectionRule]] = None):
        # Copied, so neither a constructor default nor a caller's list is ever aliased
        # between rulesets: adding a rule to one tile must not add it to every other.
        self._rules: List[InjectionRule] = (
            list(injection_rules) if injection_rules else []
        )
        self._filters: Dict[str, Filter] = {f.name: f for f in _DEFAULT_FILTERS}
        self._triggers: Dict[str, Trigger] = {t.name: t for t in _DEFAULT_TRIGGERS}
        self._sources: Dict[str, Source] = {s.name: s for s in _DEFAULT_SOURCES}

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------
    @property
    def triggers(self) -> List[Trigger]:
        return list(self._triggers.values())

    def add_trigger(self, new_trigger: Trigger):
        self._triggers[new_trigger.name] = new_trigger

    def remove_trigger(self, trigger_name: str):
        self._triggers.pop(trigger_name)

    def check_trigger(
        self, name: str, targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
    ) -> bool:
        return self._triggers.get(name, Trigger()).func(targ_indexes, qubits)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    @property
    def filters(self) -> List[Filter]:
        return list(self._filters.values())

    def add_filter(self, new_filter: Filter):
        self._filters[new_filter.name] = new_filter

    def remove_filter(self, filter_name: str):
        self._filters.pop(filter_name)

    def apply_filter(
        self, name: str, targ_indexes: List[List[int]], qubits: Dict[int, Qubit]
    ) -> List[List[int]]:
        return self._filters.get(name, Filter()).func(targ_indexes, qubits)

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    @property
    def sources(self) -> List[Source]:
        return list(self._sources.values())

    def add_source(self, new_source: Source):
        self._sources[new_source.name] = new_source

    def remove_source(self, source_name: str):
        self._sources.pop(source_name)

    def rate_for(
        self,
        channel_rule: ChannelRule,
        targets: List[int],
        qubits: Dict[int, Qubit],
        coupler_at: CouplerLookup,
    ) -> float:
        """
        The error rate a channel rule emits: its source's value, times its scalar.

        The single place a rate is decided, so scaling and sourcing stay one concern that
        a ruleset owns rather than something the circuit emitter open-codes.
        """
        return (
            self._sources.get(channel_rule.source, Source()).func(
                targets, qubits, coupler_at
            )
            * channel_rule.scalar
        )

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------
    @property
    def rules(self):
        return self._rules

    def add_rule(
        self, injection_rule: InjectionRule, priority_index: Optional[int] = None
    ) -> None:
        if not (self._validate_rule_index(priority_index)):
            raise ValueError(f"""Priority index is larger than the # of rules. Given {priority_index} but only {len(self._rules)} # of rules.
                             \n To add rule to the end, priority index should be set to `None` (default for parameter).""")
        if priority_index is not None:
            self._rules.insert(priority_index, injection_rule)
        else:
            self._rules.append(injection_rule)

    def pop_rule(self, priority_index: int) -> InjectionRule:
        if not (self._validate_rule_index(priority_index)):
            raise ValueError(
                f"Priority index is larger than the # of rules. Given {priority_index} but only {len(self._rules)} # of rules"
            )
        return self._rules.pop(priority_index)

    def _validate_rule_index(self, index: Optional[int]) -> bool:
        # `is None`, not truthiness: index 0 is the highest priority, not "no index".
        return index is None or index < len(self._rules)

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "\n".join([f"{i}:{r}" for i, r in enumerate(self.rules)])
