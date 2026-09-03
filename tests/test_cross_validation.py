"""Centerpiece correctness test: three *independent* ways of computing the
same Greeks -- closed-form calculus, forward-mode dual-number autodiff, and
central finite differences on the closed-form price -- must all agree.

No single method is treated as ground truth; agreement across three
independently-derived implementations is the actual evidence of
correctness. A grid of realistic (S, K, r, q, sigma, T) combinations is
checked, spanning ITM/ATM/OTM, short/long maturities, and with/without a
dividend yield.
"""

import itertools

import pytest

from options_pricing.black_scholes import bs_delta, bs_vega, bs_price
from options_pricing.dual import dual_delta, dual_vega

TOL = 1e-4
H = 1e-4  # finite-difference step


def central_diff_delta(S, K, r, q, sigma, T, option_type):
    up = bs_price(S + H, K, r, q, sigma, T, option_type)
    down = bs_price(S - H, K, r, q, sigma, T, option_type)
    return (up - down) / (2.0 * H)


def central_diff_vega(S, K, r, q, sigma, T, option_type):
    up = bs_price(S, K, r, q, sigma + H, T, option_type)
    down = bs_price(S, K, r, q, sigma - H, T, option_type)
    return (up - down) / (2.0 * H)


# Realistic grid: spot fixed at 100, strikes spanning deep OTM to deep ITM,
# a range of rates/yields/vols/maturities.
STRIKES = (70.0, 90.0, 100.0, 110.0, 130.0)
RATES = (0.0, 0.03, 0.07)
DIVS = (0.0, 0.02)
VOLS = (0.10, 0.20, 0.45)
MATURITIES = (0.1, 1.0, 3.0)
OPTION_TYPES = ("call", "put")

GRID = list(itertools.product(STRIKES, RATES, DIVS, VOLS, MATURITIES, OPTION_TYPES))


@pytest.mark.parametrize("K,r,q,sigma,T,option_type", GRID)
def test_delta_three_methods_agree(K, r, q, sigma, T, option_type):
    S = 100.0
    closed_form = bs_delta(S, K, r, q, sigma, T, option_type)
    autodiff = dual_delta(S, K, r, q, sigma, T, option_type)
    finite_diff = central_diff_delta(S, K, r, q, sigma, T, option_type)

    assert closed_form == pytest.approx(autodiff, abs=TOL)
    assert closed_form == pytest.approx(finite_diff, abs=TOL)
    assert autodiff == pytest.approx(finite_diff, abs=TOL)


@pytest.mark.parametrize("K,r,q,sigma,T,option_type", GRID)
def test_vega_three_methods_agree(K, r, q, sigma, T, option_type):
    S = 100.0
    closed_form = bs_vega(S, K, r, q, sigma, T, option_type)
    autodiff = dual_vega(S, K, r, q, sigma, T, option_type)
    finite_diff = central_diff_vega(S, K, r, q, sigma, T, option_type)

    assert closed_form == pytest.approx(autodiff, abs=TOL)
    assert closed_form == pytest.approx(finite_diff, abs=TOL)
    assert autodiff == pytest.approx(finite_diff, abs=TOL)
