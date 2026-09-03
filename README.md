# options-pricing-lib

A European/American option pricing library in Python, built to be *visibly, verifiably
correct*: every pricer's output is cross-checked against at least one independent method or
a known textbook value, not just unit-tested against itself.

## What's inside

- **Black-Scholes-Merton closed form** (`options_pricing/black_scholes.py`) -- European
  calls/puts with a continuous dividend yield `q`, plus all five closed-form Greeks
  (delta, gamma, vega, theta, rho).
- **Forward-mode automatic differentiation from scratch** (`options_pricing/dual.py`) -- a
  `Dual` number class (`a + b*eps`, `eps**2 = 0`) with hand-written derivative rules for
  `+ - * /`, `exp`, `log`, `sqrt`, and the normal CDF. The Black-Scholes formula is
  reimplemented purely in `Dual` arithmetic and used to compute delta and vega by seeding
  `S` or `sigma` with a unit dual part -- no autodiff library involved.
- **CRR binomial tree** (`options_pricing/binomial.py`) -- Cox-Ross-Rubinstein, European or
  American exercise, O(N^2) backward induction.
- **Monte Carlo with antithetic variates** (`options_pricing/monte_carlo.py`) -- European
  pricer reporting both the price estimate and its standard error / 95% CI.
- **Implied volatility solver** (`options_pricing/implied_vol.py`) -- Newton-Raphson using
  closed-form vega, with an automatic bisection fallback for the near-zero-vega regime
  (deep ITM/OTM) where pure Newton is unreliable.
- **`demo.py`** -- runs all five pricers on one parameter set and prints a comparison table.

## The correctness argument

The centerpiece test (`tests/test_cross_validation.py`) computes delta and vega three
independent ways -- closed-form calculus, dual-number autodiff, and central finite
differences on the closed-form price -- across a 540-combination grid of strikes, rates,
dividend yields, vols, and maturities, and asserts all three agree within `1e-4`. No single
method is trusted as ground truth; the agreement itself is the evidence.

On top of that:

- `tests/test_black_scholes.py` reproduces the standard textbook reference value
  (S=K=100, r=0.05, q=0, sigma=0.2, T=1 -> call ≈ **10.4506**) to within `1e-3`, and checks
  put-call parity.
- `tests/test_binomial.py` checks the CRR European price gap to Black-Scholes shrinks
  monotonically as steps grow from 50 to 2000, and that American puts are worth at least as
  much as their European counterparts (with a concrete deep-ITM early-exercise premium).
- `tests/test_monte_carlo.py` checks the Monte Carlo 95% CI contains the closed-form price
  (fixed seed, so this is deterministic, not flaky).
- `tests/test_implied_vol.py` round-trips price -> implied sigma -> back to within `1e-4`
  across a grid of strikes/vols, plus a low-vega and an extreme-initial-guess case that
  specifically exercise the bisection fallback.

## Setup / run / test

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

.venv/Scripts/python.exe demo.py          # comparison table across all 5 pricers
.venv/Scripts/python.exe -m pytest        # 1143 tests
```

## Later

Not built here -- listed, not stubbed:

- Longstaff-Schwartz American Monte Carlo
- Stochastic volatility (Heston / local vol)
- Rates / FX products
- A C++ port for speed
