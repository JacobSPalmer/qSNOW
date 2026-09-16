"""
The numerics behind the noise distributions: Gaussian fields and marginal transforms.

A `GaussianFieldDistribution` builds a landscape in two steps - a Gaussian field that
carries only spatial structure, then a reshaping of its values into the requested
marginal. The field builders here (`iid_gaussian_field`, `correlated_gaussian_field`)
produce the first; the marginal transforms (`quantile_map`, `scale_gaussian_contour`)
produce the second, using the NORTA / Gaussian-copula construction (Cario & Nelson,
1997; the "normal score transform" of geostatistics): standardise the field, push it
through the standard normal CDF to uniforms, then pull those uniforms through the
target distribution's quantile function. `normal_scores` and `blend_latents` are the
pieces of the cross-correlated coupler path. Method walkthrough:
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
    "standardize",
    "normal_scores",
    "blend_latents",
    "quantile_map",
    "scale_gaussian_contour",
    "iid_gaussian_field",
    "correlated_gaussian_field",
]

Center = Literal["mean", "median"]

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


def normal_scores(values) -> np.ndarray:
    """
    The rank-based Gaussian latent behind `values` (Gaussian anamorphosis).

    `Φ⁻¹((rank − ½) / n)` with average ranks for ties. Depends on `values` only through
    their order, so it recovers a standard-normal latent from *any* landscape - a
    generated one, or measured rates imported onto the chip - which is what lets a
    coupler map be correlated with whatever site map the chip currently holds.
    """
    v = np.asarray(values, dtype=float)
    if np.ptp(v) == 0:
        raise ValueError("Cannot take normal scores of a field with no variance.")
    ranks = rankdata(v, method="average")
    return norm.ppf((ranks - 0.5) / v.size)


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


def quantile_map(
    values, target, bounds: Tuple[float, float] | None = None
) -> np.ndarray:
    """
    Map a Gaussian field onto `target`'s marginal, preserving the field's ordering.

    `values` are standardised with their own sample mean and deviation, converted to
    uniforms through the standard normal CDF, and pulled through `target.ppf` restricted
    to `bounds` (default `p_bounds()`). Restricting the *uniforms* to `[F(lo), F(hi)]`
    rather than clipping the output is the same truncation `truncnorm` applies, so no
    mass piles up at either bound. The map is monotone, so ranks - and hence the field's
    spatial contours - are unchanged.
    """
    lo, hi = bounds if bounds is not None else p_bounds()
    u = norm.cdf(standardize(values))
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
