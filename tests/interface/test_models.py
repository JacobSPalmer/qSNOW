import pytest

from qsnow.interface.models import (
    Coupler,
    CSSType,
    NoiseProfile,
    Qubit,
    Status,
    Tag,
    TileSpec,
    coupler_key,
)


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


class TestCouplerKey:
    def test_key_is_order_independent(self):
        assert coupler_key((1, 1), (0, 0)) == coupler_key((0, 0), (1, 1))

    def test_key_is_sorted(self):
        assert coupler_key((2, 2), (1, 1)) == ((1, 1), (2, 2))


class TestCoupler:
    def test_ends_are_canonical_regardless_of_construction_order(self):
        assert Coupler(((2, 2), (1, 1))).ends == Coupler(((1, 1), (2, 2))).ends

    def test_defaults_to_a_zero_noise_profile(self):
        assert Coupler(((0, 0), (1, 1))).noise.p == 0.0

    def test_accepts_a_noise_profile(self):
        assert Coupler(((0, 0), (1, 1)), NoiseProfile(0.03)).noise.p == 0.03

    def test_midpoint_sits_between_the_endpoints(self):
        assert Coupler(((0, 0), (2, 2))).midpoint == (1.0, 1.0)

    def test_other_returns_the_opposite_endpoint(self):
        coupler = Coupler(((0, 0), (1, 1)))

        assert coupler.other((0, 0)) == (1, 1)
        assert coupler.other((1, 1)) == (0, 0)

    def test_other_rejects_a_coordinate_it_does_not_join(self):
        with pytest.raises(KeyError, match="not an endpoint"):
            Coupler(((0, 0), (1, 1))).other((5, 5))

    def test_contains_its_endpoints_only(self):
        coupler = Coupler(((0, 0), (1, 1)))

        assert (0, 0) in coupler
        assert (1, 1) in coupler
        assert (2, 2) not in coupler

    def test_copy_is_independent(self):
        original = Coupler(((0, 0), (1, 1)), NoiseProfile(0.04))
        clone = original.copy()
        clone.noise.p = 0.5

        assert original.noise.p == 0.04
        assert clone.ends == original.ends
