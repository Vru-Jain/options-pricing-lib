import pytest

from options_pricing.monte_carlo import mc_price
from options_pricing.black_scholes import bs_price

S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0


def test_mc_confidence_interval_contains_black_scholes_price_call():
    bs = bs_price(S, K, r, q, sigma, T, "call")
    result = mc_price(S, K, r, q, sigma, T, "call", n_paths=200_000, seed=42)
    lo, hi = result.ci95
    assert lo <= bs <= hi


def test_mc_confidence_interval_contains_black_scholes_price_put():
    bs = bs_price(S, K, r, q, sigma, T, "put")
    result = mc_price(S, K, r, q, sigma, T, "put", n_paths=200_000, seed=42)
    lo, hi = result.ci95
    assert lo <= bs <= hi


def test_mc_deterministic_with_fixed_seed():
    r1 = mc_price(S, K, r, q, sigma, T, "call", n_paths=10_000, seed=7)
    r2 = mc_price(S, K, r, q, sigma, T, "call", n_paths=10_000, seed=7)
    assert r1.price == r2.price
    assert r1.std_error == r2.std_error


def test_mc_std_error_shrinks_with_more_paths():
    small = mc_price(S, K, r, q, sigma, T, "call", n_paths=5_000, seed=1)
    large = mc_price(S, K, r, q, sigma, T, "call", n_paths=200_000, seed=1)
    assert large.std_error < small.std_error


def test_mc_price_is_nonnegative():
    result = mc_price(S, K, r, q, sigma, T, "put", n_paths=5_000, seed=3)
    assert result.price >= 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        mc_price(S, K, r, q, sigma, T, "spread", n_paths=1000)
    with pytest.raises(ValueError):
        mc_price(S, K, r, q, sigma, T, "call", n_paths=1)
