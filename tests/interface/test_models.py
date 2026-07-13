import pytest

from qsnow.interface.models import CSSType, NoiseProfile, Qubit, Status, TileTag


class TestNoiseProfile:
    def test_default_p_is_zero(self):
        assert NoiseProfile().p == 0.0

    def test_accepts_value_within_bounds(self):
        profile = NoiseProfile(p=0.5)
        assert profile.p == 0.5

    @pytest.mark.parametrize("bad_p", [-0.01, 0.76])
    def test_rejects_value_outside_bounds(self, bad_p):
        with pytest.raises(ValueError):
            NoiseProfile(p=bad_p)

    def test_copy_noise_profile(self, p = 0.5):
        np = NoiseProfile(p)
        assert np.copy() != np
        assert np.copy().to_dict() == np.to_dict()

class TestQubit:
    def test_defaults(self):
        qubit = Qubit()
        assert qubit.status == Status.INACTIVE
        assert qubit.type == CSSType.UNASSIGNED
        assert qubit.is_active() is False

    def test_is_active_when_status_not_inactive(self):
        qubit = Qubit(status=Status.LOGICAL)
        assert qubit.is_active() is True

    @pytest.mark.parametrize(
        "css_type,predicate",
        [
            (CSSType.X_CHECK, "is_x_measure"),
            (CSSType.Z_CHECK, "is_z_measure"),
            (CSSType.DATA, "is_data"),
        ],
    )
    def test_type_predicates(self, css_type, predicate):
        qubit = Qubit(type=css_type)
        assert getattr(qubit, predicate)() is True

    def test_is_measure_true_for_either_check_type(self):
        assert Qubit(type=CSSType.X_CHECK).is_measure() is True
        assert Qubit(type=CSSType.Z_CHECK).is_measure() is True
        assert Qubit(type=CSSType.DATA).is_measure() is False

    def test_reset_restores_defaults_but_keeps_noise(self):
        qubit = Qubit(status=Status.LOGICAL, type=CSSType.DATA)
        qubit.noise.p = 0.2

        qubit.reset()

        assert qubit.status == Status.INACTIVE
        assert qubit.type == CSSType.UNASSIGNED
        assert qubit.noise.p == 0.2


class TestTileTag:
    def test_to_dict_includes_expected_keys(self):
        tag = TileTag(name="d3", distance=3, rounds=1)
        result = tag.to_dict()
        assert result["name"] == "d3"
        assert result["metadata"] == {}
