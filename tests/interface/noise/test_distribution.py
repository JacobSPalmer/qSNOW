from pathlib import Path
from statistics import mean

import numpy as np
import pytest
from scipy import stats

from qsnow.interface.chip import Chip
from qsnow.interface.models import NoiseProfile
from qsnow.interface.noise import (
    ContourDistribution,
    Custom,
    GaussianFieldDistribution,
    NoiseDistribution,
    NormalContour,
    RandomGaussian,
    RandomUniform,
    SkewContour,
    Uniform,
    p_bounds,
    register_distribution,
)

BASELINES = Path(__file__).parent / "baselines"

SAMPLED = [
    RandomUniform((0.01, 0.05), seed=11),
    RandomGaussian(0.01, 0.002, seed=7),
    NormalContour(0.01, 0.003, seed=3),
    SkewContour(0.01, 0.003, 1.5, seed=3),
]


def _rates(profiles):
    return np.array([n.p for n in profiles.values()])


class TestRecipe:
    @pytest.mark.parametrize("dist", SAMPLED + [Uniform(0.01), Custom()])
    def test_round_trips_through_its_record(self, dist):
        assert NoiseDistribution.from_dict(dist.as_dict()) == dist

    def test_record_names_match_the_historic_labels(self):
        assert RandomGaussian(0.01, 0.002).as_dict()["name"] == "gaussian"
        assert NormalContour(0.01, 0.003).as_dict()["name"] == "derived contour"
        assert SkewContour(0.01, 0.003, 1.0).as_dict()["name"] == "skewed contour"
        assert RandomUniform((0.01, 0.05)).as_dict()["name"] == "uniform random"
        assert Uniform(0.01).as_dict() == {"name": "uniform homogeneous", "p": 0.01}
        assert Custom().as_dict() == {"name": "custom"}

    def test_record_lists_positional_params_before_keyword_ones(self):
        record = SkewContour(0.01, 0.003, 1.5, seed=3).as_dict()
        assert list(record) == ["name", "location", "deviation", "skew", "center", "slope", "seed"]

    def test_randomized_distributions_resolve_a_seed(self):
        assert RandomGaussian(0.01, 0.002).seed is not None
        assert SkewContour(0.01, 0.003, 1.0).seed is not None

    def test_deterministic_distributions_keep_no_seed(self):
        assert Uniform(0.01).seed is None
        assert Custom().seed is None

    def test_unknown_name_raises_listing_the_registry(self):
        with pytest.raises(ValueError, match="gaussian"):
            NoiseDistribution.from_dict({"name": "no such thing"})

    def test_json_lists_are_coerced_back_to_tuples(self):
        dist = NoiseDistribution.from_dict({"name": "uniform random", "bounds": [0.01, 0.05], "seed": 1})
        assert dist == RandomUniform((0.01, 0.05), seed=1)

    def test_abstract_classes_cannot_be_instantiated(self):
        for cls in (NoiseDistribution, GaussianFieldDistribution, ContourDistribution):
            with pytest.raises(TypeError):
                cls()

    def test_a_user_subclass_registers_and_round_trips(self):
        from dataclasses import dataclass
        from typing import ClassVar

        @register_distribution
        @dataclass
        class Doubled(Uniform):
            name: ClassVar[str] = "test doubled"

            def sites(self, chip):
                return {c: NoiseProfile(2 * self.p) for c in chip.noise_map}

        assert NoiseDistribution.from_dict(Doubled(0.01).as_dict()) == Doubled(0.01)


class TestProducers:
    @pytest.mark.parametrize("dist", SAMPLED + [Uniform(0.01)])
    def test_sites_cover_every_site_with_fresh_profiles_in_range(self, chip, dist):
        out = dist.sites(chip)
        lo, hi = p_bounds()
        assert out.keys() == chip.noise_map.keys()
        assert all(isinstance(n, NoiseProfile) and lo <= n.p <= hi for n in out.values())
        assert len({id(n) for n in out.values()}) == len(out)

    @pytest.mark.parametrize("dist", SAMPLED + [Uniform(0.01)])
    def test_couplers_cover_every_coupler(self, chip, dist):
        out = dist.couplers(chip)
        assert out.keys() == chip.coupler_map.keys()
        assert all(isinstance(n, NoiseProfile) for n in out.values())

    @pytest.mark.parametrize("dist", SAMPLED)
    def test_same_object_produces_the_same_landscape_twice(self, chip, dist):
        assert _rates(dist.sites(chip)).tolist() == _rates(dist.sites(chip)).tolist()

    def test_custom_returns_its_maps_as_fresh_profiles(self, chip):
        source = NoiseProfile(0.02)
        dist = Custom(noise_map={(0, 0): source, (2, 2): 0.03})
        out = dist.sites(chip)
        assert out[(0, 0)].p == 0.02 and out[(0, 0)] is not source
        assert out[(2, 2)].p == 0.03

    def test_custom_without_a_map_raises(self, chip):
        with pytest.raises(ValueError, match="site map"):
            Custom().sites(chip)
        with pytest.raises(ValueError, match="coupler map"):
            Custom().couplers(chip)

    @pytest.mark.parametrize("dist", [Uniform(0.01), Custom(coupler_map={})])
    def test_distributions_without_a_field_reject_correlation(self, chip, dist):
        with pytest.raises(ValueError, match="correlat"):
            dist.couplers(chip, correlation=0.5)

    def test_iid_field_path_agrees_with_the_rvs_path_in_distribution(self):
        # RandomGaussian draws with scipy rvs normally and by inverse transform when a
        # correlation is asked; the two must describe the same marginal
        big = Chip(20, 20)
        big.generate_noise(SkewContour(0.01, 0.003, 1.5, seed=3))
        dist = RandomGaussian(0.05, 0.01, seed=5)
        plain = _rates(dist.couplers(big))
        blended = _rates(dist.couplers(big, correlation=0.0))
        assert plain.mean() == pytest.approx(blended.mean(), rel=0.05)
        assert plain.std() == pytest.approx(blended.std(), rel=0.15)


class TestSeededBaselines:
    """Seeds recorded by the pre-class generators must reproduce identical chips."""

    @pytest.mark.parametrize(
        "dist, filename",
        [
            (RandomUniform((0.01, 0.05), seed=11), "random_uniform_6x6_seed11.npy"),
            (RandomGaussian(0.01, 0.002, seed=7), "random_gaussian_6x6_seed7.npy"),
            (NormalContour(0.01, 0.003, seed=3), "normal_contour_6x6_seed3.npy"),
            (SkewContour(0.01, 0.003, 1.5, seed=3), "skew_contour_6x6_seed3.npy"),
            (SkewContour(0.01, 0.003, 1.5, "median", seed=3), "skew_contour_median_6x6_seed3.npy"),
        ],
    )
    def test_sites_match_the_recorded_baseline(self, dist, filename):
        chip = Chip(6, 6)
        chip.generate_noise(dist)
        expected = np.load(BASELINES / filename)
        assert np.array_equal(np.array([q.noise.p for q in chip.qubits]), expected)

    def test_correlated_couplers_match_the_recorded_baseline(self):
        chip = Chip(6, 6)
        chip.generate_noise(SkewContour(0.01, 0.003, 1.5, seed=3))
        chip.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.6)
        expected = np.load(BASELINES / "skew_contour_couplers_6x6_seed4_rho0p6.npy")
        assert np.array_equal(np.array([c.noise.p for c in chip.couplers]), expected)


class TestCrossCorrelation:
    @pytest.fixture
    def landscape(self) -> Chip:
        c = Chip(16, 16)
        c.generate_noise(SkewContour(0.01, 0.003, 1.5, seed=3))
        return c

    @staticmethod
    def _endpoint_mean_correlation(chip: Chip) -> float:
        own = [c.noise.p for c in chip.couplers]
        ends = [mean(chip.loc(e).noise.p for e in c.ends) for c in chip.couplers]
        return stats.spearmanr(own, ends).statistic

    def test_full_correlation_reproduces_the_endpoint_mean_ordering(self, landscape):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=1.0)
        assert self._endpoint_mean_correlation(landscape) == pytest.approx(1.0)

    def test_zero_correlation_ignores_the_site_landscape(self, landscape):
        dist = SkewContour(0.05, 0.01, 1.0, seed=4)
        landscape.generate_coupler_noise(dist, correlation=0.0)
        first = [c.noise.p for c in landscape.couplers]
        landscape.generate_noise(SkewContour(0.02, 0.005, 0.5, seed=99))
        landscape.generate_coupler_noise(dist, correlation=0.0)
        assert [c.noise.p for c in landscape.couplers] == pytest.approx(first)

    def test_correlation_orders_the_coupling_strength(self, landscape):
        observed = []
        for rho in (0.0, 0.5, 1.0):
            landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=rho)
            observed.append(self._endpoint_mean_correlation(landscape))
        assert observed[0] < observed[1] < observed[2]
        assert 0.3 < observed[1] < 0.8

    def test_couplers_take_their_own_marginal(self, landscape):
        landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, seed=4), correlation=0.5)
        p = [c.noise.p for c in landscape.couplers]
        assert np.mean(p) == pytest.approx(0.05, rel=0.1)
        assert stats.skew(p) > 0.3

    def test_an_iid_distribution_can_be_correlated_too(self, landscape):
        landscape.generate_coupler_noise(RandomGaussian(0.05, 0.01, seed=4), correlation=1.0)
        assert self._endpoint_mean_correlation(landscape) == pytest.approx(1.0)

    def test_requires_a_site_landscape_with_variance(self, chip):
        chip.generate_noise(Uniform(0.01))
        with pytest.raises(ValueError, match="variance"):
            chip.generate_coupler_noise(SkewContour(0.05, 0.01, seed=4), correlation=0.5)

    def test_slope_below_three_is_rejected(self, landscape):
        with pytest.raises(ValueError, match="slope"):
            landscape.generate_coupler_noise(SkewContour(0.05, 0.01, 1.0, slope=2, seed=4))
