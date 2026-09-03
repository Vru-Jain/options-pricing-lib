import pytest

from options_pricing.binomial import crr_price
from options_pricing.black_scholes import bs_price

S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.2, 1.0


def test_european_call_converges_to_black_scholes_as_steps_grow():
    bs = bs_price(S, K, r, q, sigma, T, "call")
    steps = (50, 200, 500, 2000)
    gaps = [abs(crr_price(S, K, r, q, sigma, T, n, "call", "european") - bs) for n in steps]

    # Gap must shrink monotonically as the tree gets finer...
    for a, b in zip(gaps, gaps[1:]):
        assert b < a
    # ...and land close for a fine enough tree.
    assert gaps[-1] < 1e-2


def test_european_put_converges_to_black_scholes_as_steps_grow():
    bs = bs_price(S, K, r, q, sigma, T, "put")
    steps = (50, 200, 500, 2000)
    gaps = [abs(crr_price(S, K, r, q, sigma, T, n, "put", "european") - bs) for n in steps]

    for a, b in zip(gaps, gaps[1:]):
        assert b < a
    assert gaps[-1] < 1e-2


def test_american_call_no_dividend_equals_european_call():
    # With q=0, early exercise of an American call is never optimal, so
    # American and European call prices must coincide.
    euro = crr_price(S, K, r, 0.0, sigma, T, 500, "call", "european")
    amer = crr_price(S, K, r, 0.0, sigma, T, 500, "call", "american")
    assert amer == pytest.approx(euro, abs=1e-9)


def test_american_put_worth_at_least_as_much_as_european_put():
    # Early exercise optionality can only add value.
    euro = crr_price(S, K, r, q, sigma, T, 500, "put", "european")
    amer = crr_price(S, K, r, q, sigma, T, 500, "put", "american")
    assert amer >= euro - 1e-12


def test_american_put_deep_itm_has_positive_early_exercise_premium():
    deep_itm_S = 60.0
    euro = crr_price(deep_itm_S, K, r, q, sigma, T, 500, "put", "european")
    amer = crr_price(deep_itm_S, K, r, q, sigma, T, 500, "put", "american")
    assert amer > euro + 1e-6


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        crr_price(S, K, r, q, sigma, T, 0, "call", "european")
    with pytest.raises(ValueError):
        crr_price(S, K, r, q, sigma, T, 100, "strangle", "european")
    with pytest.raises(ValueError):
        crr_price(S, K, r, q, sigma, T, 100, "call", "bermudan")
