import pytest

from qsnow.interface.models import CSSType, NoiseProfile, Qubit, Status, Tag, TileSpec


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

    def test_copy_noise_profile(self, p=0.5):
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


class TestTag:
    def test_defaults(self):
        tag = Tag()
        assert tag.name is None
        assert tag.desc is None
        assert tag.metadata == {}

    def test_default_metadata_not_shared_between_instances(self):
        assert Tag().metadata is not Tag().metadata


class TestTileSpec:
    def test_defaults(self):
        spec = TileSpec()
        assert spec.tile_type is None
        assert spec.distance is None
        assert spec.rounds is None
        assert spec.generator_args == {}

    def test_default_generator_args_not_shared_between_instances(self):
        assert TileSpec().generator_args is not TileSpec().generator_args
