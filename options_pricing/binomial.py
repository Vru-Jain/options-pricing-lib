"""Cox-Ross-Rubinstein (CRR) binomial tree pricer, European or American
exercise.

    dt = T / N
    u  = exp(sigma * sqrt(dt))
    d  = 1 / u
    p  = (exp((r - q) * dt) - d) / (u - d)      risk-neutral up probability

Backward induction discounts each step at exp(-r*dt). American nodes take
max(continuation value, intrinsic value) at every step (early exercise).
"""

import math


def crr_price(S: float, K: float, r: float, q: float, sigma: float, T: float,
              N: int, option_type: str = "call", exercise: str = "european") -> float:
    if N < 1:
        raise ValueError("N must be >= 1")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")

    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"risk-neutral probability p={p:.6f} outside [0,1]; "
            "parameters imply an arbitrage-inconsistent tree (dt too large "
            "relative to sigma/r/q)"
        )

    def intrinsic(spot: float) -> float:
        return max(spot - K, 0.0) if option_type == "call" else max(K - spot, 0.0)

    # Terminal layer: stock prices S * u^j * d^(N-j) for j up-moves, j=0..N.
    values = [intrinsic(S * (u ** j) * (d ** (N - j))) for j in range(N + 1)]

    # Backward induction. At step i there are i+1 nodes (j = 0..i up-moves).
    for i in range(N - 1, -1, -1):
        next_values = values
        values = [0.0] * (i + 1)
        for j in range(i + 1):
            continuation = disc * (p * next_values[j + 1] + (1.0 - p) * next_values[j])
            if exercise == "american":
                spot = S * (u ** j) * (d ** (i - j))
                values[j] = max(continuation, intrinsic(spot))
            else:
                values[j] = continuation

    return values[0]
