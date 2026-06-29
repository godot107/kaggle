"""Turn an elasticity estimate into a margin-maximizing price.

For constant-elasticity demand  q(p) = A * p**eps  (eps < -1), profit margin
    pi(p) = (p - c) * q(p)
is maximized at the closed form
    p* = c * eps / (eps + 1)
The markup factor eps/(eps+1) > 1 shrinks toward 1 as demand gets more elastic
(|eps| large) -- elastic products tolerate less markup.

`optimal_price` returns the closed form; `optimize_numeric` maximizes the same
objective numerically as an independent check; `margin_uplift` scores a
recommended price against the observed one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def optimal_price(cost: float, epsilon: float) -> float:
    """Closed-form margin-maximizing price. Requires epsilon < -1 (elastic)."""
    if epsilon >= -1:
        raise ValueError(
            f"epsilon={epsilon:.3f} is not elastic (need eps < -1); "
            "no interior optimum -- demand isn't price-sensitive enough."
        )
    return cost * epsilon / (epsilon + 1)


def demand(price, A: float, epsilon: float):
    """Constant-elasticity demand q = A * p**eps."""
    return A * np.power(price, epsilon)


def fit_intercept(price_obs: float, qty_obs: float, epsilon: float) -> float:
    """Back out A so the demand curve passes through one observed (p, q) point."""
    return qty_obs / np.power(price_obs, epsilon)


@dataclass
class OptimizeResult:
    cost: float
    epsilon: float
    price_star: float
    margin_star: float
    price_obs: float
    margin_obs: float

    @property
    def uplift_pct(self) -> float:
        return 100.0 * (self.margin_star - self.margin_obs) / self.margin_obs


def margin_uplift(
    cost: float,
    epsilon: float,
    price_obs: float,
    qty_obs: float,
) -> OptimizeResult:
    """Counterfactual: margin at p* vs at the observed price, same demand curve.

    Anchors the constant-elasticity curve to the observed (price, qty) point,
    then compares (p-c)*q at the optimal price against the status quo.
    """
    A = fit_intercept(price_obs, qty_obs, epsilon)
    p_star = optimal_price(cost, epsilon)
    margin = lambda p: (p - cost) * demand(p, A, epsilon)
    return OptimizeResult(
        cost=cost, epsilon=epsilon,
        price_star=p_star, margin_star=float(margin(p_star)),
        price_obs=price_obs, margin_obs=float(margin(price_obs)),
    )


def optimize_numeric(cost: float, epsilon: float, A: float,
                     bounds: tuple[float, float] = (1e-3, 1e6)) -> float:
    """Numeric margin maximization -- should match `optimal_price` closely."""
    neg_margin = lambda p: -((p - cost) * demand(p, A, epsilon))
    res = minimize_scalar(neg_margin, bounds=bounds, method="bounded")
    return float(res.x)


if __name__ == "__main__":
    # sanity: closed form vs numeric for c=20, eps=-2  -> p* = 40
    c, eps = 20.0, -2.0
    p_cf = optimal_price(c, eps)
    p_num = optimize_numeric(c, eps, A=fit_intercept(45.0, 10.0, eps))
    print(f"closed-form p*={p_cf:.3f}  numeric p*={p_num:.3f}")
