"""Black-Scholes-Merton closed-form pricer and Greeks for European options
on an underlying paying a continuous dividend yield q.

Formulas (Hull, "Options, Futures, and Other Derivatives"):

    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    Call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
    Put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

N is the standard normal CDF; phi is the standard normal PDF. All Greeks are
reported per unit (per 1.00 change in the underlying/vol/rate, per 1 year of
time), not per-percent or per-day -- callers rescale as needed.
"""

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def norm_cdf(x: float) -> float:
    """Standard normal CDF N(x), via the stdlib (equivalent to the
    erf-based 0.5*(1+erf(x/sqrt(2))) formula the spec calls out)."""
    return _NORMAL.cdf(x)


def norm_pdf(x: float) -> float:
    """Standard normal PDF phi(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float):
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    """Black-Scholes-Merton price of a European call or put."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return math.exp(-q * T) * norm_cdf(d1)
    elif option_type == "put":
        return math.exp(-q * T) * (norm_cdf(d1) - 1.0)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_gamma(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    # Same for calls and puts.
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, r: float, q: float, sigma: float, T: float,
            option_type: str = "call") -> float:
    # Same for calls and puts. Per unit of sigma (not per 1% vol point).
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)


def bs_theta(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: str = "call") -> float:
    # Per year. Divide by 365 for a per-calendar-day figure.
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    common = -(S * norm_pdf(d1) * sigma * math.exp(-q * T)) / (2.0 * math.sqrt(T))
    if option_type == "call":
        return common - r * K * math.exp(-r * T) * norm_cdf(d2) + q * S * math.exp(-q * T) * norm_cdf(d1)
    elif option_type == "put":
        return common + r * K * math.exp(-r * T) * norm_cdf(-d2) - q * S * math.exp(-q * T) * norm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_rho(S: float, K: float, r: float, q: float, sigma: float, T: float,
           option_type: str = "call") -> float:
    # Per unit of r (not per 1% rate point).
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    if option_type == "call":
        return K * T * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return -K * T * math.exp(-r * T) * norm_cdf(-d2)
    raise ValueError("option_type must be 'call' or 'put'")
