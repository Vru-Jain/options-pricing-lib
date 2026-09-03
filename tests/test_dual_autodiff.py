import math

import pytest

from options_pricing.dual import (
    Dual, dexp, dlog, dsqrt, dnorm_cdf, dnorm_pdf, bs_price_dual, dual_delta, dual_vega,
)
from options_pricing.black_scholes import bs_price, bs_delta, bs_vega, norm_cdf


# --- arithmetic building blocks: check both value and derivative ---

def test_add_sub():
    x = Dual(3.0, 1.0)  # x, dx/dx = 1
    y = Dual(5.0, 0.0)  # constant
    s = x + y
    assert s.val == pytest.approx(8.0)
    assert s.eps == pytest.approx(1.0)  # d(x+5)/dx = 1
    d = x - y
    assert d.val == pytest.approx(-2.0)
    assert d.eps == pytest.approx(1.0)  # d(x-5)/dx = 1


def test_mul_product_rule():
    x = Dual(3.0, 1.0)
    # f(x) = x * x -> f'(x) = 2x = 6
    f = x * x
    assert f.val == pytest.approx(9.0)
    assert f.eps == pytest.approx(6.0)


def test_div_quotient_rule():
    x = Dual(4.0, 1.0)
    # f(x) = x / 2 -> f'(x) = 0.5
    f = x / 2.0
    assert f.val == pytest.approx(2.0)
    assert f.eps == pytest.approx(0.5)
    # f(x) = 10 / x -> f'(x) = -10/x^2 = -0.625
    g = 10.0 / x
    assert g.val == pytest.approx(2.5)
    assert g.eps == pytest.approx(-0.625)


def test_neg_and_rsub():
    x = Dual(3.0, 1.0)
    assert (-x).val == pytest.approx(-3.0)
    assert (-x).eps == pytest.approx(-1.0)
    r = 5.0 - x  # f(x) = 5 - x -> f'(x) = -1
    assert r.val == pytest.approx(2.0)
    assert r.eps == pytest.approx(-1.0)


def test_exp_derivative():
    x = Dual(2.0, 1.0)
    f = dexp(x)
    assert f.val == pytest.approx(math.exp(2.0))
    assert f.eps == pytest.approx(math.exp(2.0))  # d/dx exp(x) = exp(x)


def test_log_derivative():
    x = Dual(4.0, 1.0)
    f = dlog(x)
    assert f.val == pytest.approx(math.log(4.0))
    assert f.eps == pytest.approx(0.25)  # d/dx ln(x) = 1/x


def test_sqrt_derivative():
    x = Dual(9.0, 1.0)
    f = dsqrt(x)
    assert f.val == pytest.approx(3.0)
    assert f.eps == pytest.approx(1.0 / 6.0)  # d/dx sqrt(x) = 1/(2 sqrt(x))


def test_norm_cdf_derivative_is_pdf():
    x = Dual(0.5, 1.0)
    f = dnorm_cdf(x)
    assert f.val == pytest.approx(norm_cdf(0.5))
    assert f.eps == pytest.approx(dnorm_pdf(0.5))  # d/dx N(x) = phi(x)


def test_composite_function_chain_rule():
    # f(x) = exp(sqrt(x)), f'(x) = exp(sqrt(x)) / (2*sqrt(x))
    x = Dual(4.0, 1.0)
    f = dexp(dsqrt(x))
    expected_val = math.exp(2.0)
    expected_deriv = math.exp(2.0) / (2.0 * 2.0)
    assert f.val == pytest.approx(expected_val)
    assert f.eps == pytest.approx(expected_deriv, rel=1e-10)


# --- the actual signature use case: BS Greeks via dual numbers ---

def test_bs_price_dual_matches_plain_price():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.04, 0.01, 0.22, 0.5
    for opt in ("call", "put"):
        dual_result = bs_price_dual(S, K, r, q, sigma, T, opt)
        plain = bs_price(S, K, r, q, sigma, T, opt)
        assert dual_result.val == pytest.approx(plain, rel=1e-12)


def test_dual_delta_matches_closed_form():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.04, 0.01, 0.22, 0.5
    for opt in ("call", "put"):
        d = dual_delta(S, K, r, q, sigma, T, opt)
        cf = bs_delta(S, K, r, q, sigma, T, opt)
        assert d == pytest.approx(cf, abs=1e-10)


def test_dual_vega_matches_closed_form():
    S, K, r, q, sigma, T = 100.0, 105.0, 0.04, 0.01, 0.22, 0.5
    for opt in ("call", "put"):
        v = dual_vega(S, K, r, q, sigma, T, opt)
        cf = bs_vega(S, K, r, q, sigma, T, opt)
        assert v == pytest.approx(cf, abs=1e-8)
