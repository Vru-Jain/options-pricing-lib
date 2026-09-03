"""Runs all five pricers on the same set of parameters and prints a
comparison table -- a quick visual sanity check that every method agrees.

Run: .venv/Scripts/python.exe demo.py
"""

from options_pricing.black_scholes import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho
from options_pricing.dual import dual_delta, dual_vega
from options_pricing.binomial import crr_price
from options_pricing.monte_carlo import mc_price
from options_pricing.implied_vol import implied_volatility

# A single realistic parameter set, priced every way this library knows how.
S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    print(f"Parameters: S={S} K={K} r={r} q={q} sigma={sigma} T={T}")

    section("1. Black-Scholes-Merton closed form")
    for opt in ("call", "put"):
        price = bs_price(S, K, r, q, sigma, T, opt)
        print(f"  {opt:>4}: price={price:.6f}  delta={bs_delta(S, K, r, q, sigma, T, opt):.6f}  "
              f"gamma={bs_gamma(S, K, r, q, sigma, T, opt):.6f}  vega={bs_vega(S, K, r, q, sigma, T, opt):.6f}  "
              f"theta={bs_theta(S, K, r, q, sigma, T, opt):.6f}  rho={bs_rho(S, K, r, q, sigma, T, opt):.6f}")

    section("2. Dual-number (forward-mode autodiff) Greeks")
    for opt in ("call", "put"):
        d = dual_delta(S, K, r, q, sigma, T, opt)
        v = dual_vega(S, K, r, q, sigma, T, opt)
        cf_d = bs_delta(S, K, r, q, sigma, T, opt)
        cf_v = bs_vega(S, K, r, q, sigma, T, opt)
        print(f"  {opt:>4}: delta={d:.6f} (closed-form {cf_d:.6f}, diff {abs(d - cf_d):.2e})  "
              f"vega={v:.6f} (closed-form {cf_v:.6f}, diff {abs(v - cf_v):.2e})")

    section("3. CRR binomial tree (European) -> converges to Black-Scholes")
    bs_call = bs_price(S, K, r, q, sigma, T, "call")
    for n_steps in (50, 200, 500, 2000):
        crr_call = crr_price(S, K, r, q, sigma, T, n_steps, "call", "european")
        print(f"  N={n_steps:>5}: price={crr_call:.6f}  gap vs BS={abs(crr_call - bs_call):.2e}")

    section("3b. CRR binomial tree (American put, early-exercise premium)")
    euro_put = crr_price(S, K, r, q, sigma, T, 1000, "put", "european")
    amer_put = crr_price(S, K, r, q, sigma, T, 1000, "put", "american")
    print(f"  European put: {euro_put:.6f}   American put: {amer_put:.6f}   "
          f"early-exercise premium: {amer_put - euro_put:.6f}")

    section("4. Monte Carlo (antithetic variates, 200,000 paths)")
    for opt in ("call", "put"):
        mc = mc_price(S, K, r, q, sigma, T, opt, n_paths=200_000, seed=42)
        cf = bs_price(S, K, r, q, sigma, T, opt)
        lo, hi = mc.ci95
        print(f"  {opt:>4}: price={mc.price:.6f} +/- {mc.std_error:.6f}  95% CI=[{lo:.6f}, {hi:.6f}]  "
              f"closed-form={cf:.6f}  (in CI: {lo <= cf <= hi})")

    section("5. Implied volatility solver (round-trip)")
    for opt in ("call", "put"):
        market_price = bs_price(S, K, r, q, sigma, T, opt)
        recovered = implied_volatility(market_price, S, K, r, q, T, opt)
        print(f"  {opt:>4}: true sigma={sigma:.6f}  priced at {market_price:.6f}  "
              f"recovered sigma={recovered:.6f}  diff={abs(recovered - sigma):.2e}")

    section("Known textbook value check")
    ref_call = bs_price(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    print(f"  S=K=100 r=0.05 q=0 sigma=0.2 T=1 call = {ref_call:.6f}  (reference ~10.4506)")


if __name__ == "__main__":
    main()
