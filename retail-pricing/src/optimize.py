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
import pandas as pd
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


def per_product_counterfactual(
    df: pd.DataFrame,
    eps_table: pd.DataFrame,
    cost_frac: float = 0.6,
    eps_floor: float = -6.0,
    max_price_change: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """Apply each product's own optimal price; aggregate margin uplift vs status quo.

    For each product we take its (shrunk) elasticity, set unit cost as a fraction
    of its reference price, anchor a constant-elasticity demand curve at the
    product's (median price, median qty), and compare margin at the optimal price
    `p* = c·ε/(ε+1)` against the status-quo reference.

    Two guardrails make this stable and realistic -- without them, noisy per-product
    elasticities over-extrapolate the constant-elasticity curve and blow up:
      * `eps_floor`: cap extreme (noisy) elasticities (e.g. ε=-20 -> -6).
      * `max_price_change`: clip the recommended price to +/- this band around the
        reference, keeping it near the observed price support (what real pricing
        teams do). Products that aren't elastic (ε >= -1) **hold price**.

    Aggregate uplift is weighted by each product's number of observed periods.

    Assumptions (stated, not hidden): constant per-product elasticity; unit cost =
    `cost_frac` x reference price (data has no cost); demand anchored in-sample. The
    guardrail does real work -- report it, don't hide it. Sweep `cost_frac`.
    """
    ref = df.groupby("product_id").agg(
        p_ref=("unit_price", "median"),
        q_ref=("qty", "median"),
        n=("qty", "size"),
    ).reset_index()
    m = ref.merge(eps_table[["product_id", "eps"]], on="product_id", how="left")

    out = []
    for r in m.itertuples():
        eps = max(r.eps, eps_floor)  # cap extreme/noisy elasticities
        c = cost_frac * r.p_ref
        optimizable = eps < -1
        if optimizable:
            lo, hi = r.p_ref * (1 - max_price_change), r.p_ref * (1 + max_price_change)
            p_unc = optimal_price(c, eps)
            p_star = min(max(p_unc, lo), hi)            # clip to guardrail band
            at_guardrail = not (lo < p_unc < hi)
        else:
            p_star, at_guardrail = r.p_ref, False        # not elastic -> hold price
        A = fit_intercept(r.p_ref, r.q_ref, eps)
        m_obs = (r.p_ref - c) * r.q_ref
        m_star = (p_star - c) * demand(p_star, A, eps)
        out.append({
            "product_id": r.product_id, "n": r.n, "eps": eps,
            "p_ref": r.p_ref, "cost": c, "p_star": p_star,
            "optimizable": optimizable, "at_guardrail": at_guardrail,
            "price_change_pct": 100 * (p_star - r.p_ref) / r.p_ref,
            "margin_obs": m_obs, "margin_star": m_star,
            "uplift_pct": 100 * (m_star - m_obs) / m_obs,
        })
    pp = pd.DataFrame(out)

    w = pp["n"]
    summary = {
        "cost_frac": cost_frac,
        "n_products": len(pp),
        "n_optimizable": int(pp["optimizable"].sum()),
        "n_at_guardrail": int(pp["at_guardrail"].sum()),
        "share_price_increase": float((pp["p_star"] > pp["p_ref"]).mean()),
        "weighted_uplift_pct": float(
            100 * (w * (pp["margin_star"] - pp["margin_obs"])).sum()
            / (w * pp["margin_obs"]).sum()
        ),
        "median_uplift_pct": float(pp["uplift_pct"].median()),
    }
    return pp, summary


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
