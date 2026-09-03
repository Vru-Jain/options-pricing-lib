"""options_pricing: a small, independently-cross-checked European/American
option pricing library.

Every pricer in this package is validated against at least one other,
independent method:

- Black-Scholes-Merton closed form  <-> dual-number autodiff Greeks
                                     <-> central finite-difference Greeks
- CRR binomial tree (European)      <-> Black-Scholes closed form (as N -> inf)
- Monte Carlo (antithetic)          <-> Black-Scholes closed form
- Implied vol solver                <-> round-trips price -> sigma -> price

See tests/test_cross_validation.py for the centerpiece agreement test.
"""

from options_pricing.black_scholes import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    norm_cdf,
    norm_pdf,
)
from options_pricing.dual import Dual
from options_pricing.binomial import crr_price
from options_pricing.monte_carlo import MonteCarloResult, mc_price
from options_pricing.implied_vol import implied_volatility

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "norm_cdf",
    "norm_pdf",
    "Dual",
    "crr_price",
    "MonteCarloResult",
    "mc_price",
    "implied_volatility",
]
