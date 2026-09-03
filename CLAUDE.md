# options-pricing-lib

A European/American option pricing library: Black-Scholes closed form, a from-scratch dual-number
autodiff implementation, CRR binomial tree, Monte Carlo, and an implied vol solver -- cross-validated
against each other, not just unit-tested in isolation.

## Commands — exact, run these, don't guess

| Action | Command |
|---|---|
| Setup | `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt` |
| Run | `.venv/Scripts/python.exe demo.py` |
| Test | `.venv/Scripts/python.exe -m pytest` |
| Lint + format | none configured — see Later in README.md |

## Definition of done — every task, every model, including subagents

- [ ] Tests pass — you ran the test command THIS session and saw it pass
- [ ] No new dependency without one sentence in the commit body saying why
- [ ] `git diff` reviewed before committing: no secrets, no files unrelated to the task
- [ ] Commands table above still accurate

## Scope rules

- Implement only what the current request names. New ideas become one line under "Later" in README.md.
- Smallest change that fully solves the problem. No drive-by refactors.
- Bug fixes: reproduce first; the commit body states the root cause.

## Reporting

End every task with: what ran, what passed, what failed (with output), and what was NOT verified. Never
say "production-ready" or "comprehensive" — list the commands that actually ran instead.

## Conventions

- **Plain `float` throughout, deliberately.** This is continuous-time stochastic calculus (Black-Scholes
  PDE, GBM paths, dual-number derivatives) -- values are inherently real-valued. This is NOT the same
  situation as an order book's discrete tick prices, where integers are the right representation. Do not
  "fix" this into integers or `Decimal` in a future session; it would be a regression, not a hardening.
- **Greeks are per unit, not per convention.** `bs_vega` is per 1.00 change in sigma (not per 1 vol
  point / 0.01), `bs_rho` is per 1.00 change in r (not per 1% / 0.01), and `bs_theta` is per 1 year (not
  per calendar day — divide by 365 at the call site if you want that).
- **`Dual` (`options_pricing/dual.py`) is the signature piece of this repo.** It is a real from-scratch
  forward-mode AD implementation (operator overloads + hand-coded chain rules for `exp`/`log`/`sqrt`/
  normal CDF), not a stub or a wrapper around a library. Keep it that way — do not replace it with
  `scipy`/`jax`/`autograd` even if one gets added for something else later.
- **CRR binomial tree is O(N^2)** (rebuilds the array levels back-to-front, no vectorization). This is
  fine through N=2000 (the convergence test's ceiling) but would need a numpy rewrite before pushing much
  further — not a bug, a known scaling limit.
- **The implied vol solver's Newton step is guarded, not raw**: it aborts to bisection whenever sigma
  wanders outside `[1e-6, 5.0]` or vega drops below `1e-8`, specifically because deep ITM/OTM vega can be
  near-zero and blow up a raw Newton step. If you touch `implied_vol.py`, keep that guard — see
  `tests/test_implied_vol.py::test_extreme_initial_guess_forces_bisection_fallback_and_still_recovers`
  for why it exists.
- **Monte Carlo uses antithetic variates with a fixed default seed (42) at the call site**, not baked
  into the library as a global default behavior beyond the function signature's default arg — pass your
  own `seed` for independent replicates. Tests rely on the fixed seed for determinism, not for tolerance.

## Boundaries

- No numpy/scipy/pandas — stdlib `math`/`statistics`/`random`/`dataclasses` only, per the project brief.
  If a future task needs vectorized performance (e.g., pricing a large options chain at once), that's a
  deliberate dependency add with a one-line justification in the commit body, not a silent import.
- No live market data, no calibration to a vol surface — this library prices from given parameters only.
