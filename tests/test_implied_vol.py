import itertools

import pytest

from options_pricing.black_scholes import bs_price
from options_pricing.implied_vol import implied_volatility

S, K, r, q, T = 100.0, 100.0, 0.05, 0.01, 1.0


@pytest.mark.parametrize("K,sigma,option_type", list(itertools.product(
    (80.0, 100.0, 120.0), (0.1, 0.2, 0.5, 0.9), ("call", "put"))))
def test_round_trip_recovers_original_sigma(K, sigma, option_type):
    market_price = bs_price(S, K, r, q, sigma, T, option_type)
    recovered = implied_volatility(market_price, S, K, r, q, T, option_type)
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_round_trip_low_vega_regime_still_recovers():
    # Deep ITM, moderate maturity: vega is small (~1.8e-4) here -- far
    # smaller than at-the-money -- which is exactly the regime where a
    # naive fixed-step Newton update can overshoot wildly. The price still
    # carries (barely) enough sigma-sensitivity to be inverted; verifies
    # the solver handles a near-flat-vega case, not just the friendly one.
    S_, K_, sigma_, T_ = 100.0, 50.0, 0.15, 1.0
    market_price = bs_price(S_, K_, r, q, sigma_, T_, "call")
    recovered = implied_volatility(market_price, S_, K_, r, q, T_, "call")
    assert recovered == pytest.approx(sigma_, abs=1e-3)


def test_extreme_initial_guess_forces_bisection_fallback_and_still_recovers():
    # An initial guess pinned near the upper sigma bound sends the very
    # first Newton step wildly off (or out of [SIGMA_LO, SIGMA_HI]
    # entirely) for a plain at-the-money option -- this is what should
    # trip the "sigma wandered out of range" guard and hand off to
    # bisection, which must still land on the right answer.
    sigma_ = 0.2
    market_price = bs_price(S, K, r, q, sigma_, T, "call")
    recovered = implied_volatility(market_price, S, K, r, q, T, "call",
                                    initial_guess=4.99)
    assert recovered == pytest.approx(sigma_, abs=1e-4)


def test_negative_or_zero_market_price_raises():
    with pytest.raises(ValueError):
        implied_volatility(0.0, S, K, r, q, T, "call")
    with pytest.raises(ValueError):
        implied_volatility(-1.0, S, K, r, q, T, "call")


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        implied_volatility(5.0, S, K, r, q, T, "collar")


def test_unreachable_price_raises():
    # A price far above the sigma=5.0 upper bound's max price is unreachable.
    with pytest.raises(ValueError):
        implied_volatility(1e6, S, K, r, q, T, "call")
