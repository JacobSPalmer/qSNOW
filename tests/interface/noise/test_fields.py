import numpy as np
import pytest
from scipy.stats import lognorm, norm

from qsnow.interface.models import NoiseProfile
from qsnow.interface.noise.fields import (
    log_skewed_target,
    P_FLOOR,
    blend_latents,
    normal_scores,
    rank_percentiles,
    p_bounds,
    quantile_map,
    skewed_target,
    standardize,
)


class TestPBounds:
    def test_bounds_follow_the_noise_profile_ceiling(self):
        assert p_bounds() == (P_FLOOR, NoiseProfile.p.max_value)


class TestSkewedTarget:
    def test_mean_centre_matches_the_requested_moments(self):
        mean, var, skew = skewed_target(0.01, 0.003, 1.5).stats(moments="mvs")
        assert mean == pytest.approx(0.01)
        assert np.sqrt(var) == pytest.approx(0.003)
        assert skew == pytest.approx(1.5)

    def test_median_centre_pins_the_median(self):
        target = skewed_target(0.01, 0.003, 1.5, center="median")
        assert target.median() == pytest.approx(0.01)
        assert target.mean() > 0.01  # right-skewed: mean sits above the median

    def test_zero_skew_is_the_normal_distribution(self):
        target = skewed_target(0.01, 0.003, 0.0)
        q = np.linspace(0.05, 0.95, 7)
        assert target.ppf(q) == pytest.approx(norm(0.01, 0.003).ppf(q))

    @pytest.mark.parametrize("deviation", [0.0, -0.001])
    def test_rejects_non_positive_deviation(self, deviation):
        with pytest.raises(ValueError, match="deviation"):
            skewed_target(0.01, deviation, 1.0)

    def test_rejects_unknown_centre(self):
        with pytest.raises(ValueError, match="center"):
            skewed_target(0.01, 0.003, 1.0, center="mode")


class TestQuantileMap:
    @pytest.fixture
    def field(self):
        return np.random.default_rng(0).normal(0.02, 0.004, size=300)

    def test_output_stays_inside_bounds(self, field):
        mapped = quantile_map(field, skewed_target(0.01, 0.01, 3.0))
        lo, hi = p_bounds()
        assert np.all((mapped >= lo) & (mapped <= hi))

    def test_preserves_the_ordering_of_the_field(self, field):
        mapped = quantile_map(field, skewed_target(0.01, 0.003, 1.5))
        assert np.array_equal(np.argsort(mapped), np.argsort(field))

    def test_sorted_output_is_exactly_the_target_quantiles(self, field):
        target = skewed_target(0.01, 0.003, 1.5)
        mapped = quantile_map(field, target)
        lo, hi = p_bounds()
        f_lo, f_hi = target.cdf(lo), target.cdf(hi)
        u = (np.arange(1, field.size + 1) - 0.5) / field.size
        assert np.sort(mapped) == pytest.approx(target.ppf(f_lo + u * (f_hi - f_lo)))

    def test_depends_on_the_field_only_through_its_ranks(self, field):
        target = skewed_target(0.01, 0.003, 1.5)
        assert np.array_equal(quantile_map(field, target), quantile_map(np.exp(field), target))
        assert np.array_equal(quantile_map(field, target), quantile_map(3.0 * field + 1.0, target))

    def test_explicit_bounds_override_the_default(self, field):
        mapped = quantile_map(field, skewed_target(0.01, 0.003, 1.5), bounds=(0.008, 0.012))
        assert mapped.min() >= 0.008 and mapped.max() <= 0.012


class TestRankPercentiles:
    def test_are_the_shifted_ranks_over_n(self):
        assert rank_percentiles([0.3, 0.1, 0.2, 0.4]) == pytest.approx([0.625, 0.125, 0.375, 0.875])

    def test_stay_strictly_inside_the_unit_interval(self):
        u = rank_percentiles(np.random.default_rng(0).normal(size=1000))
        assert u.min() > 0 and u.max() < 1

    def test_ties_share_a_percentile(self):
        u = rank_percentiles([1.0, 2.0, 2.0, 3.0])
        assert u[1] == u[2]

    def test_rejects_a_constant_field(self):
        with pytest.raises(ValueError, match="variance"):
            rank_percentiles(np.full(5, 0.3))


class TestNormalScores:
    @pytest.fixture
    def field(self):
        return np.random.default_rng(1).gamma(2.0, size=500)

    def test_preserves_the_ordering(self, field):
        assert np.array_equal(np.argsort(normal_scores(field)), np.argsort(field))

    def test_is_standard_normal_shaped(self, field):
        z = normal_scores(field)
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0, rel=0.02)


class TestBlendLatents:
    @pytest.fixture
    def latents(self):
        rng = np.random.default_rng(2)
        return rng.normal(size=5000), rng.normal(size=5000)

    def test_rho_one_is_the_standardised_first_latent(self, latents):
        a, b = latents
        assert blend_latents(a, b, 1.0) == pytest.approx(standardize(a))

    def test_rho_zero_is_the_standardised_second_latent(self, latents):
        a, b = latents
        assert blend_latents(a, b, 0.0) == pytest.approx(standardize(b))

    def test_correlation_with_the_first_latent_is_rho(self, latents):
        a, b = latents
        z = blend_latents(a, b, 0.6)
        assert np.corrcoef(z, a)[0, 1] == pytest.approx(0.6, abs=0.05)
        assert z.std() == pytest.approx(1.0, rel=0.05)

    @pytest.mark.parametrize("rho", [-0.1, 1.1])
    def test_rejects_rho_outside_unit_interval(self, latents, rho):
        with pytest.raises(ValueError, match="correlation"):
            blend_latents(*latents, rho)


class TestLogSkewedTarget:
    def test_median_centre_pins_the_median_in_rate_space(self):
        assert log_skewed_target(0.02, 0.3, 1.2).ppf(0.5) == pytest.approx(0.02)

    def test_geometric_centre_pins_the_geometric_mean(self):
        target = log_skewed_target(0.02, 0.3, 0.0, center="geometric")
        assert target.ppf(0.5) == pytest.approx(0.02)  # symmetric in log space: median == geometric mean

    def test_zero_skew_is_log_normal(self):
        target = log_skewed_target(0.02, 0.3, 0.0, center="geometric")
        q = np.linspace(0.05, 0.95, 7)
        assert target.ppf(q) == pytest.approx(lognorm(s=0.3 * np.log(10), scale=0.02).ppf(q))

    def test_cdf_and_ppf_are_inverse(self):
        target = log_skewed_target(0.02, 0.3, 1.5)
        u = np.linspace(0.01, 0.99, 9)
        assert target.cdf(target.ppf(u)) == pytest.approx(u)

    @pytest.mark.parametrize("kwargs", [dict(location=0.0), dict(location=-0.01), dict(deviation=0.0)])
    def test_rejects_non_positive_inputs(self, kwargs):
        params = dict(location=0.02, deviation=0.3, skew=1.0); params.update(kwargs)
        with pytest.raises(ValueError):
            log_skewed_target(**params)

    def test_rejects_unknown_centre(self):
        with pytest.raises(ValueError, match="center"):
            log_skewed_target(0.02, 0.3, 1.0, center="mean")
