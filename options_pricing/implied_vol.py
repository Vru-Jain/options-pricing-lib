"""Implied volatility solver: given a market price, solve for the sigma
that reproduces it under Black-Scholes.

Primary method is Newton-Raphson using the closed-form vega as the exact
derivative of price w.r.t. sigma (fast, quadratic convergence). Vega decays
toward zero for deep ITM/OTM options, which can stall or destabilize pure
Newton (near-zero derivative => huge steps or division blow-up), so on any
sign of trouble -- non-convergence, a step landing outside sane bounds, or a
vega too small to trust -- we fall back to bisection, which only relies on
Black-Scholes price being monotonically increasing in sigma and so cannot
diverge.
"""

import math

from options_pricing.black_scholes import bs_price, bs_vega

_VEGA_FLOOR = 1e-8
_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0


def _bisection(market_price, S, K, r, q, T, option_type, tol, max_iter):
    lo, hi = _SIGMA_LO, _SIGMA_HI
    price_lo = bs_price(S, K, r, q, lo, T, option_type)
    price_hi = bs_price(S, K, r, q, hi, T, option_type)
    if not (price_lo - market_price <= 0.0 <= price_hi - market_price):
        raise ValueError(
            f"market_price={market_price} not reachable for sigma in "
            f"[{_SIGMA_LO}, {_SIGMA_HI}] (bracket prices [{price_lo}, {price_hi}]); "
            "check for an arbitrage-inconsistent input"
        )
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = bs_price(S, K, r, q, mid, T, option_type)
        diff = price_mid - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0.0:
            hi = mid
        else:
            lo = mid
    return mid


def implied_volatility(market_price: float, S: float, K: float, r: float,
                        q: float, T: float, option_type: str = "call",
                        initial_guess: float = 0.2, tol: float = 1e-8,
                        max_iter: int = 100) -> float:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if market_price <= 0.0:
        raise ValueError("market_price must be positive")

    sigma = initial_guess
    for _ in range(max_iter):
        if sigma <= _SIGMA_LO or sigma >= _SIGMA_HI or not math.isfinite(sigma):
            break  # Newton wandered out of a sane range -> bisection
        price = bs_price(S, K, r, q, sigma, T, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, r, q, sigma, T, option_type)
        if vega < _VEGA_FLOOR:
            break  # derivative too flat to trust -> bisection
        sigma = sigma - diff / vega

    return _bisection(market_price, S, K, r, q, T, option_type, tol, max_iter=200)
