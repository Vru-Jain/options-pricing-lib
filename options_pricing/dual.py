"""Forward-mode automatic differentiation via dual numbers, built from
scratch (no autodiff library).

A dual number is a + b*eps with the algebraic rule eps**2 = 0. Carrying a
value through arithmetic in this ring simultaneously carries its exact
derivative: if x = a + 1*eps (a "seeded" variable) and f is built purely out
of +, -, *, /, exp, log, sqrt and our normal CDF, then

    f(x) = f(a) + f'(a)*eps

comes out for free, term by term, via the chain rule baked into each
operator/function below. No numerical step size, no truncation error -- the
result is exact up to floating point.

This module also provides `bs_price_dual`, the Black-Scholes formula
re-implemented purely in terms of Dual arithmetic, used to compute delta and
vega by seeding S or sigma with a unit dual part.
"""

import math
from statistics import NormalDist

_NORMAL = NormalDist()


class Dual:
    """A dual number a + b*eps, eps**2 = 0."""

    __slots__ = ("val", "eps")

    def __init__(self, val: float, eps: float = 0.0):
        self.val = float(val)
        self.eps = float(eps)

    @staticmethod
    def _coerce(x) -> "Dual":
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __repr__(self):
        return f"Dual({self.val!r}, {self.eps!r})"

    # --- arithmetic: each rule is the ordinary calculus product/quotient
    # rule, applied symbolically to the (value, derivative) pair. ---

    def __add__(self, other):
        o = Dual._coerce(other)
        return Dual(self.val + o.val, self.eps + o.eps)

    __radd__ = __add__

    def __sub__(self, other):
        o = Dual._coerce(other)
        return Dual(self.val - o.val, self.eps - o.eps)

    def __rsub__(self, other):
        return Dual._coerce(other) - self

    def __mul__(self, other):
        o = Dual._coerce(other)
        # (a + b eps)(c + d eps) = ac + (ad + bc) eps
        return Dual(self.val * o.val, self.eps * o.val + self.val * o.eps)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = Dual._coerce(other)
        # (a + b eps)/(c + d eps) = a/c + (bc - ad)/c^2 eps   (quotient rule)
        return Dual(self.val / o.val,
                     (self.eps * o.val - self.val * o.eps) / (o.val * o.val))

    def __rtruediv__(self, other):
        return Dual._coerce(other) / self

    def __neg__(self):
        return Dual(-self.val, -self.eps)

    def __pos__(self):
        return Dual(self.val, self.eps)


# --- elementary functions, each carrying its own derivative rule ---

def dexp(x) -> Dual:
    x = Dual._coerce(x)
    v = math.exp(x.val)
    return Dual(v, v * x.eps)  # d/dx exp(x) = exp(x)


def dlog(x) -> Dual:
    x = Dual._coerce(x)
    return Dual(math.log(x.val), x.eps / x.val)  # d/dx ln(x) = 1/x


def dsqrt(x) -> Dual:
    x = Dual._coerce(x)
    v = math.sqrt(x.val)
    return Dual(v, x.eps / (2.0 * v))  # d/dx sqrt(x) = 1/(2 sqrt(x))


def dnorm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def dnorm_cdf(x) -> Dual:
    """Dual-aware standard normal CDF. d/dx N(x) = phi(x), so the dual part
    is carried through by the chain rule using the (plain-float) normal
    PDF at the value part."""
    x = Dual._coerce(x)
    v = _NORMAL.cdf(x.val)
    return Dual(v, dnorm_pdf(x.val) * x.eps)


def bs_price_dual(S, K, r, q, sigma, T, option_type: str = "call") -> Dual:
    """Black-Scholes price computed entirely in Dual arithmetic. Any of the
    six inputs may be a `Dual` (seeded with eps=1 to differentiate) or a
    plain float/int (treated as a constant, eps=0); the rest are coerced
    automatically.

    Seed S with Dual(S, 1.0) to read delta off the result's `.eps`; seed
    sigma with Dual(sigma, 1.0) to read vega.
    """
    S = Dual._coerce(S)
    K = Dual._coerce(K)
    r = Dual._coerce(r)
    q = Dual._coerce(q)
    sigma = Dual._coerce(sigma)
    T = Dual._coerce(T)

    sqrtT = dsqrt(T)
    d1 = (dlog(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        return S * dexp(-(q * T)) * dnorm_cdf(d1) - K * dexp(-(r * T)) * dnorm_cdf(d2)
    elif option_type == "put":
        return K * dexp(-(r * T)) * dnorm_cdf(-d2) - S * dexp(-(q * T)) * dnorm_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def dual_delta(S, K, r, q, sigma, T, option_type: str = "call") -> float:
    """Delta via forward-mode autodiff: seed S with a unit dual part."""
    result = bs_price_dual(Dual(S, 1.0), K, r, q, sigma, T, option_type)
    return result.eps


def dual_vega(S, K, r, q, sigma, T, option_type: str = "call") -> float:
    """Vega via forward-mode autodiff: seed sigma with a unit dual part."""
    result = bs_price_dual(S, K, r, q, Dual(sigma, 1.0), T, option_type)
    return result.eps
