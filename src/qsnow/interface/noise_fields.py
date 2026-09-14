"""
Marginal transforms for spatially correlated noise landscapes.

A contour generator on `Chip` produces a *Gaussian* field whose spatial structure comes
from box-averaging white noise. The functions here give that field a different marginal
distribution without disturbing its structure, using the NORTA / Gaussian-copula
construction (Cario & Nelson, 1997; the "normal score transform" of geostatistics):
standardise the field, push it through the standard normal CDF to uniforms, then pull
those uniforms through the target distribution's quantile function.

Everything in this module is a pure function of arrays and frozen scipy distributions,
so it can be unit-tested without building a chip.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy.stats import norm, pearson3

from .models import NoiseProfile

__all__ = ["P_FLOOR", "Center", "p_bounds", "skewed_target", "quantile_map"]

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
    v = np.asarray(values, dtype=float)
    if np.ptp(v) == 0:  # exact, unlike `std()`, which carries rounding residue
        raise ValueError("Cannot map a field with no variance onto a target marginal.")
    u = norm.cdf((v - v.mean()) / v.std())
    f_lo, f_hi = target.cdf(lo), target.cdf(hi)
    return target.ppf(f_lo + u * (f_hi - f_lo))
