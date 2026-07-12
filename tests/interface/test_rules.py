import pytest

from qsnow.interface.models import CSSType, Qubit
from qsnow.interface.rules import (
    ChannelRule,
    InjectionRule,
    Ruleset,
    _all_qubits_trigger,
    _data_trigger,
    is_measurement,
    is_reset,
    is_two_qubit,
)


class TestOperationalHelpers:
    @pytest.mark.parametrize("op", ["M", "MR", "MX"])
    def test_is_measurement(self, op):
        assert is_measurement(op) is True

    @pytest.mark.parametrize("op", ["R", "RX", "MR"])
    def test_is_reset(self, op):
        assert is_reset(op) is True

    def test_is_two_qubit(self):
        assert is_two_qubit("CX") is True
        assert is_two_qubit("H") is False


class TestTriggers:
    def test_all_qubits_trigger_true_when_all_present(self):
        qubits = {1: Qubit(), 2: Qubit()}
        assert _all_qubits_trigger([[1, 2]], qubits) is True

    def test_all_qubits_trigger_false_when_missing(self):
        qubits = {1: Qubit()}
        assert _all_qubits_trigger([[1, 2]], qubits) is False

    def test_data_trigger_requires_all_data_type(self):
        qubits = {1: Qubit(type=CSSType.DATA), 2: Qubit(type=CSSType.X_CHECK)}
        assert _data_trigger([[1, 2]], qubits) is False
        qubits[2] = Qubit(type=CSSType.DATA)
        assert _data_trigger([[1, 2]], qubits) is True


class TestRuleset:
    # NOTE: Ruleset(injection_rules=[]) is passed explicitly in every test here rather than
    # relying on the constructor default, since `Ruleset.__init__`'s `injection_rules: List = []`
    # default argument is mutable and shared across instances that omit it.

    def test_add_and_list_rules(self):
        ruleset = Ruleset(injection_rules=[])
        rule = InjectionRule(
            "R", "all_qubits", after=[ChannelRule("X_ERROR", "all_qubits")]
        )

        ruleset.add_rule(rule)

        assert ruleset.rules == [rule]

    def test_pop_rule_removes_and_returns(self):
        ruleset = Ruleset(injection_rules=[])
        rule = InjectionRule("R", "all_qubits")
        ruleset.add_rule(rule)

        popped = ruleset.pop_rule(0)

        assert popped is rule
        assert ruleset.rules == []

    def test_pop_rule_invalid_index_raises(self):
        # NOTE: index 0 is not usable here - `_validate_rule_index` does
        # `index and index >= len(self._rules)`, and `0 and ...` short-circuits
        # to falsy regardless of list length, so index=0 always "validates" as in-bounds.
        ruleset = Ruleset(injection_rules=[])
        with pytest.raises(ValueError):
            ruleset.pop_rule(1)
