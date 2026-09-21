import pytest

from qsnow.interface.models import Coupler, CSSType, NoiseProfile, Qubit
from qsnow.interface.rules import (
    ChannelRule,
    InjectionRule,
    Ruleset,
    Source,
    _all_qubits_trigger,
    _coupler_source,
    _data_trigger,
    _qubit_max_source,
    _qubit_mean_source,
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
    def test_default_rules_are_not_shared_between_rulesets(self):
        a, b = Ruleset(), Ruleset()
        a.add_rule(InjectionRule("R", "all_qubits"))

        assert b.rules == []

    def test_constructor_copies_the_callers_list(self):
        rules = [InjectionRule("R", "all_qubits")]
        ruleset = Ruleset(rules)
        ruleset.add_rule(InjectionRule("H", "any"))

        assert len(rules) == 1

    def test_add_rule_at_priority_zero_inserts_first(self):
        first, second = InjectionRule("R", "all_qubits"), InjectionRule("H", "any")
        ruleset = Ruleset([first])

        ruleset.add_rule(second, priority_index=0)

        assert ruleset.rules == [second, first]

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

    @pytest.mark.parametrize("index", [0, 1])
    def test_pop_rule_invalid_index_raises(self, index):
        ruleset = Ruleset(injection_rules=[])
        with pytest.raises(ValueError):
            ruleset.pop_rule(index)


def _pair(p_a=0.01, p_b=0.03):
    """Two adjacent qubits and the coupler joining them, with no chip involved."""
    qubits = {
        0: Qubit(loc=(0, 0), noise=NoiseProfile(p_a)),
        1: Qubit(loc=(1, 1), noise=NoiseProfile(p_b)),
    }
    coupler = Coupler(((0, 0), (1, 1)), NoiseProfile(0.2))
    return qubits, lambda a, b: coupler if {a, b} == {(0, 0), (1, 1)} else None


class TestRateSources:
    def test_qubit_mean_averages_the_group(self):
        qubits, coupler_at = _pair()

        assert _qubit_mean_source([0, 1], qubits, coupler_at) == pytest.approx(0.02)

    def test_qubit_max_takes_the_worst_endpoint(self):
        qubits, coupler_at = _pair()

        assert _qubit_max_source([0, 1], qubits, coupler_at) == 0.03

    def test_coupler_reads_the_edge_not_the_endpoints(self):
        qubits, coupler_at = _pair()

        assert _coupler_source([0, 1], qubits, coupler_at) == 0.2

    def test_coupler_rejects_a_group_that_is_not_a_pair(self):
        qubits, coupler_at = _pair()

        with pytest.raises(ValueError, match="exactly 2 target qubits"):
            _coupler_source([0], qubits, coupler_at)

    def test_coupler_raises_on_a_non_adjacent_pair(self):
        """A gate across an uncoupled pair is a modelling error, not something to
        quietly average away - that would hide the defect the source exists to expose."""
        qubits, coupler_at = _pair()
        qubits[1].loc = (5, 5)

        with pytest.raises(ValueError, match="No coupler joins"):
            _coupler_source([0, 1], qubits, coupler_at)


class TestRulesetSources:
    def test_default_sources_are_registered(self):
        assert {s.name for s in Ruleset().sources} == {
            "qubit_mean",
            "qubit_max",
            "coupler",
        }

    def test_channel_rule_defaults_to_the_qubit_mean(self):
        assert ChannelRule("DEPOLARIZE1", "active").source == "qubit_mean"

    def test_rate_for_applies_the_scalar(self):
        qubits, coupler_at = _pair()
        rule = ChannelRule("DEPOLARIZE2", "active", scalar=2.0, source="coupler")

        assert Ruleset().rate_for(rule, [0, 1], qubits, coupler_at) == pytest.approx(
            0.4
        )

    def test_rate_for_routes_to_the_named_source(self):
        qubits, coupler_at = _pair()
        ruleset = Ruleset()

        mean_rate = ruleset.rate_for(
            ChannelRule("DEPOLARIZE2", "active"), [0, 1], qubits, coupler_at
        )
        coupler_rate = ruleset.rate_for(
            ChannelRule("DEPOLARIZE2", "active", source="coupler"),
            [0, 1],
            qubits,
            coupler_at,
        )

        assert mean_rate == pytest.approx(0.02)
        assert coupler_rate == 0.2

    def test_unknown_source_degrades_to_the_qubit_mean(self):
        qubits, coupler_at = _pair()
        rule = ChannelRule("DEPOLARIZE2", "active", source="nonexistent")

        assert Ruleset().rate_for(rule, [0, 1], qubits, coupler_at) == pytest.approx(
            0.02
        )

    def test_custom_source_can_be_registered(self):
        qubits, coupler_at = _pair()
        ruleset = Ruleset()
        ruleset.add_source(Source("always_half", lambda t, q, c: 0.5))

        rule = ChannelRule("DEPOLARIZE2", "active", source="always_half")

        assert ruleset.rate_for(rule, [0, 1], qubits, coupler_at) == 0.5

    def test_source_can_be_removed(self):
        ruleset = Ruleset()
        ruleset.remove_source("coupler")

        assert "coupler" not in {s.name for s in ruleset.sources}
