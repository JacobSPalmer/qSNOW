"""
The numerics behind the noise distributions: Gaussian fields and marginal transforms.

A `GaussianFieldDistribution` builds a landscape in two steps - a Gaussian field that
carries only spatial structure, then a reshaping of its values into the requested
marginal. The field builders here (`iid_gaussian_field`, `correlated_gaussian_field`)
produce the first; the marginal transforms (`quantile_map`, `scale_gaussian_contour`)
produce the second by quantile mapping (the "normal score transform" of geostatistics,
NORTA's rank-based form): replace each value by its rank percentile, then pull those
percentiles through the target distribution's quantile function. `normal_scores` and
`blend_latents` are the pieces of the cross-correlated coupler path. Method walkthrough:
`writeups/noise_generation_walkthrough.tex`.

Everything except the two field builders is a pure function of arrays and frozen scipy
distributions, testable without a chip.
"""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING, Dict, Literal, Tuple

import numpy as np
from scipy.stats import norm, pearson3, rankdata

from ..models import Coord, NoiseProfile

if TYPE_CHECKING:
    from ..chip import Chip

__all__ = [
    "P_FLOOR",
    "Center",
    "p_bounds",
    "skewed_target",
    "log_skewed_target",
    "LogCenter",
    "standardize",
    "rank_percentiles",
    "normal_scores",
    "blend_latents",
    "quantile_map",
    "scale_gaussian_contour",
    "iid_gaussian_field",
    "correlated_gaussian_field",
]

Center = Literal["mean", "median"]
LogCenter = Literal["median", "geometric"]

#: Smallest physical error rate a generator will assign. `NoiseProfile.p` admits 0.0, but
#: a qubit with *no* noise is never what a landscape generator means; the truncated
#: Gaussian generator has always used this floor.
P_FLOOR = 1e-10


def p_bounds() -> Tuple[float, float]:
    """The `(lower, upper)` range a generated physical error rate may occupy."""
    return (P_FLOOR, NoiseProfile.p.max_value)


def skewed_target(
    location: float, deviation: float, skew: float, *, center: Center = "mean"
):
    """
    A frozen Pearson type III distribution with the requested shape.

    Pearson III is parameterised directly by its mean, standard deviation and skewness
    (`scipy.stats.pearson3`); `skew > 0` gives a right-skewed (long upper tail)
    distribution and `skew == 0` is exactly the normal. `center` selects which statistic
    `location` pins: the mean, or the median (the "typical qubit" of a skewed landscape).
    """
    if deviation <= 0:
        raise ValueError(f"deviation must be positive, given {deviation}.")
    if center == "mean":
        loc = location
    elif center == "median":
        loc = location - pearson3(skew, loc=0.0, scale=deviation).median()
    else:
        raise ValueError(f"center must be 'mean' or 'median', given {center!r}.")
    return pearson3(skew, loc=loc, scale=deviation)


def standardize(values) -> np.ndarray:
    """Centre by the sample mean and scale by the sample deviation.

    Raises on a constant field: there is no scale to standardise by, and every
    consumer here (a quantile map, a latent blend) needs one.
    """
    v = np.asarray(values, dtype=float)
    if np.ptp(v) == 0:  # exact, unlike `std()`, which carries rounding residue
        raise ValueError("Cannot standardize a field with no variance.")
    return (v - v.mean()) / v.std()


def rank_percentiles(values) -> np.ndarray:
    """
    Each entry's percentile by rank, `(rank − ½) / n`, with average ranks for ties.

    Depends on `values` only through their order, so any monotone transform of the
    field gives the same result. The `−½` keeps the extremes off 0 and 1, where a
    quantile function is infinite. This is the one place the package turns a field into
    percentiles: `quantile_map` pulls them through a target, `normal_scores` through
    `Φ⁻¹`. Raises on a constant field, which has no order to use.
    """
    v = np.asarray(values, dtype=float)
    if np.ptp(v) == 0:  # exact, unlike `std()`, which carries rounding residue
        raise ValueError("Cannot rank a field with no variance.")
    return (rankdata(v, method="average") - 0.5) / v.size


def normal_scores(values) -> np.ndarray:
    """
    The rank-based Gaussian latent behind `values` (Gaussian anamorphosis):
    `Φ⁻¹` of `rank_percentiles`.

    Recovers a standard-normal latent from *any* landscape - a generated one, or
    measured rates imported onto the chip - which is what lets a coupler map be
    correlated with whatever site map the chip currently holds.
    """
    return norm.ppf(rank_percentiles(values))


def blend_latents(a, b, correlation: float) -> np.ndarray:
    """
    `ρ·a + √(1 − ρ²)·b` over standardised `a` and `b`: a unit-variance latent whose
    correlation with `a` is exactly `ρ` when `a` and `b` are independent.

    The two-variable linear model of coregionalization (the Cholesky factor of a 2×2
    correlation matrix). `ρ = 1` returns `a`'s standardisation, `ρ = 0` returns `b`'s.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must lie in [0, 1], given {correlation}.")
    return correlation * standardize(a) + np.sqrt(1.0 - correlation**2) * standardize(b)


class _LogSpace:
    """A distribution on log10(rate), exposed in rate space.

    Only `cdf` and `ppf` are needed by `quantile_map`, so the wrapper is deliberately
    minimal: `cdf(p) = base.cdf(log10 p)` and `ppf(u) = 10 ** base.ppf(u)`.
    """

    def __init__(self, base):
        self.base = base

    def cdf(self, p):
        return self.base.cdf(np.log10(p))

    def ppf(self, u):
        return 10.0 ** self.base.ppf(u)


def log_skewed_target(
    location: float, deviation: float, skew: float, *, center: LogCenter = "median"
) -> _LogSpace:
    """
    A log-Pearson III target: Pearson III on log10(rate), returned in rate space.

    This is the "LP3" distribution of flood-frequency analysis (USGS Bulletin 17B/17C),
    and it is the shape of the working-component bulk of calibration data: measured
    error rates form a slightly right-skewed bell on a *log* axis. `deviation` is the
    spread of log10(rate) in decades, `skew` the skewness of log10(rate) (0 gives a
    log-normal), and `location` is a rate: the median (`center="median"`) or the
    geometric mean `10 ** mean(log10 rate)` (`center="geometric"`). See `LogSkewContour`
    for which sample statistics to feed it.
    """
    if location <= 0:
        raise ValueError(f"location must be a positive rate, given {location}.")
    if deviation <= 0:
        raise ValueError(f"deviation must be positive (decades of log10 rate), given {deviation}.")
    if center == "median":
        loc = np.log10(location) - pearson3(skew, loc=0.0, scale=deviation).median()
    elif center == "geometric":
        loc = np.log10(location)
    else:
        raise ValueError(f"center must be 'median' or 'geometric', given {center!r}.")
    return _LogSpace(pearson3(skew, loc=loc, scale=deviation))


def quantile_map(
    values, target, bounds: Tuple[float, float] | None = None
) -> np.ndarray:
    """
    Map a field onto `target`'s marginal, preserving the field's ordering.

    Each value's rank percentile (`rank_percentiles`) is pulled through `target.ppf`
    restricted to `bounds` (default `p_bounds()`). Restricting the *percentiles* to
    `[F(lo), F(hi)]` rather than clipping the output is the same truncation `truncnorm`
    applies, so no mass piles up at either bound. The map is monotone, so ranks - and
    hence the field's spatial contours - are unchanged.

    Because the percentiles come from ranks, the sorted output is exactly the target's
    quantiles at `(i − ½) / n` for every input of size `n`: the field decides only where
    each value sits, so a fit of a generated landscape returns the parameters it was
    given. (Standardising the field by its sample moments and applying `Φ` instead let
    the field's own finite-sample shape through; on a small correlated chip that moved
    the fitted skew by tens of percent between seeds.)
    """
    lo, hi = bounds if bounds is not None else p_bounds()
    u = rank_percentiles(values)
    f_lo, f_hi = target.cdf(lo), target.cdf(hi)
    return target.ppf(f_lo + u * (f_hi - f_lo))


def scale_gaussian_contour(
    values, deviation: float, alpha_ratio: float = 0.1
) -> np.ndarray:
    """
    The legacy contour reshape: rescale a Gaussian field to `deviation` about its own
    mean, with an exponential soft clip that keeps the lower tail above a fence of
    `alpha_ratio * mean` so no site reaches an absolute zero rate.

    Kept verbatim from the original `generate_derived_contour_noise`, arithmetic and
    order included, because its seeded output is part of the DATE record.
    """
    orig_values = np.array(values, dtype=float)
    deviation_factor = deviation / np.std(orig_values)

    target_mean = np.mean(orig_values)
    scaled_linear = target_mean + deviation_factor * (orig_values - target_mean)

    # exponential soft-clipping to the lower tail near zero
    alpha = target_mean * alpha_ratio

    # smooth C1-continuous blending function
    final_values = np.where(
        scaled_linear >= alpha,
        scaled_linear,
        alpha * np.exp((scaled_linear - alpha) / alpha),
    )

    # shift slightly to correct any minor mean drift caused by the tail smoothing
    mean_drift = np.mean(final_values) - target_mean
    final_values = final_values - mean_drift

    # if the shift pushed anything below alpha, clamp it smoothly
    final_values = np.where(
        final_values >= alpha,
        final_values,
        alpha * np.exp((final_values - alpha) / alpha),
    )
    return final_values


def iid_gaussian_field(chip: "Chip", seed: int) -> Dict[Coord, float]:
    """One independent standard-normal value per site: a field with no spatial structure."""
    from numpy.random import default_rng

    coords = list(chip.noise_map.keys())
    return dict(zip(coords, default_rng(seed).standard_normal(len(coords))))


def correlated_gaussian_field(
    chip: "Chip", mean_: float, deviation: float, seed: int, slope: int
) -> Dict[Coord, float]:
    """
    A spatially correlated Gaussian value per site, before any reshaping.

    White Gaussian noise (`RandomGaussian(mean_, deviation, seed)`) is drawn on a chip
    padded by `slope + 1` unit cells, then box-averaged over an `SCTile(slope)` footprint
    at every valid placement and cropped back to `chip`'s extent. The moving average is
    what gives the landscape its correlation length (`slope`); every contour
    distribution shares it, so their peaks and valleys coincide for the same seed. The
    sampling order and the box arithmetic are part of the seeded contract and must not
    change casually (`tests/interface/noise/baselines/`).
    """
    from ..chip import Chip
    from ..codes import SCTile
    from .distribution import RandomGaussian

    if slope < 3:
        raise ValueError(
            f"slope must be >= 3 (it is the distance of the surface-code tile whose "
            f"footprint sets the correlation length; smaller tiles have no valid "
            f"placements), given {slope}."
        )

    buffer_l, buffer_h = chip.unit_dims
    buffer_chip = Chip(buffer_l + slope + 1, buffer_h + slope + 1, lattice=chip.lattice)
    buffer_chip.set_noise_map(RandomGaussian(mean_, deviation, seed=seed).sites(buffer_chip))

    tile = SCTile(slope)
    field: Dict[Coord, float] = {}
    for origin in buffer_chip.candidate_placements(tile):
        if origin[0] >= chip.length or origin[1] >= chip.height:
            continue
        _, bound = buffer_chip.footprint_for(origin, tile.length, tile.height)
        field[origin] = mean(
            q.noise.p for q in buffer_chip.select_rect(*origin, *bound).values()
        )

    missing = chip.noise_map.keys() - field.keys()
    if missing:  # loud, rather than leaving those sites at the default rate
        raise RuntimeError(
            f"Correlated field left {len(missing)} site(s) uncovered (slope={slope})."
        )
    return field
