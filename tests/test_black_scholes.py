import math

import pytest

from options_pricing.black_scholes import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho, norm_cdf, norm_pdf,
)


def test_norm_cdf_matches_erf_formula():
    # Spec's alternative formula: N(x) = 0.5*(1 + erf(x/sqrt(2)))
    for x in (-2.5, -1.0, 0.0, 0.37, 1.96, 3.1):
        expected = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert norm_cdf(x) == pytest.approx(expected, abs=1e-15)


def test_norm_pdf_known_value():
    # phi(0) = 1/sqrt(2*pi)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-15)


def test_known_textbook_value_S100_K100_r5_q0_sigma20_T1():
    """S=100, K=100, r=0.05, q=0, sigma=0.2, T=1 -> call ~= 10.4506
    (standard textbook reference value, e.g. Hull)."""
    price = bs_price(S=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity_with_dividend_yield():
    # C - P = S*exp(-qT) - K*exp(-rT), holds regardless of sigma.
    S, K, r, q, sigma, T = 105.0, 95.0, 0.03, 0.015, 0.25, 0.75
    call = bs_price(S, K, r, q, sigma, T, "call")
    put = bs_price(S, K, r, q, sigma, T, "put")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_deep_itm_call_converges_to_intrinsic_forward_value():
    # Deep ITM call with tiny vol should trade very close to its forward intrinsic value.
    S, K, r, q, sigma, T = 1000.0, 100.0, 0.05, 0.0, 0.0001, 1.0
    price = bs_price(S, K, r, q, sigma, T, "call")
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert price == pytest.approx(expected, rel=1e-6)


def test_deep_otm_put_near_zero():
    S, K, r, q, sigma, T = 1000.0, 10.0, 0.05, 0.0, 0.2, 0.1
    price = bs_price(S, K, r, q, sigma, T, "put")
    assert price < 1e-6


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "straddle")


def test_gamma_and_vega_identical_for_call_and_put():
    S, K, r, q, sigma, T = 90.0, 110.0, 0.02, 0.01, 0.35, 2.0
    assert bs_gamma(S, K, r, q, sigma, T, "call") == pytest.approx(
        bs_gamma(S, K, r, q, sigma, T, "put"), abs=1e-12)
    assert bs_vega(S, K, r, q, sigma, T, "call") == pytest.approx(
        bs_vega(S, K, r, q, sigma, T, "put"), abs=1e-12)


def test_delta_bounds():
    S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0
    call_delta = bs_delta(S, K, r, q, sigma, T, "call")
    put_delta = bs_delta(S, K, r, q, sigma, T, "put")
    assert 0.0 <= call_delta <= math.exp(-q * T)
    assert -math.exp(-q * T) <= put_delta <= 0.0
    # call delta - put delta == exp(-qT)
    assert call_delta - put_delta == pytest.approx(math.exp(-q * T), abs=1e-10)


def test_rho_signs():
    S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0
    assert bs_rho(S, K, r, q, sigma, T, "call") > 0.0
    assert bs_rho(S, K, r, q, sigma, T, "put") < 0.0
