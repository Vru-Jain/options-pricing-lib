"""Monte Carlo European option pricer using antithetic variates.

Terminal stock price under the risk-neutral GBM dynamics:

    S_T = S * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),   Z ~ N(0,1)

For each standard normal draw Z we also price its antithetic pair -Z; this
cancels first-order sampling error and roughly halves variance for a given
draw count versus plain Monte Carlo.
"""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self):
        """95% confidence interval (Gaussian approximation, ~1.96 sigma)."""
        half_width = 1.959963984540054 * self.std_error
        return (self.price - half_width, self.price + half_width)


def mc_price(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: str = "call", n_paths: int = 100_000,
             seed: int = 42) -> MonteCarloResult:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2")

    rng = random.Random(seed)
    half = n_paths // 2

    drift = (r - q - 0.5 * sigma * sigma) * T
    vol_term = sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    payoffs = []
    for _ in range(half):
        z = rng.gauss(0.0, 1.0)
        for zz in (z, -z):  # antithetic pair
            ST = S * math.exp(drift + vol_term * zz)
            payoff = max(ST - K, 0.0) if option_type == "call" else max(K - ST, 0.0)
            payoffs.append(disc * payoff)

    n = len(payoffs)
    mean = sum(payoffs) / n
    variance = sum((x - mean) ** 2 for x in payoffs) / (n - 1)
    std_error = math.sqrt(variance / n)

    return MonteCarloResult(price=mean, std_error=std_error, n_paths=n)
