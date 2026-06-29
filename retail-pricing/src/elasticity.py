"""Estimate price elasticity of demand three ways, from biased to causal.

All three regress log(qty) on log(price). They differ in how they handle
*price endogeneity* -- the fact that retailers set prices in response to demand,
which biases a naive slope toward zero (too inelastic).

    naive_ols        plain OLS log-log. The biased foil.
    fixed_effects    product (and optional time) fixed effects -- removes
                     time-invariant per-product confounding.
    iv_2sls          instrument log(price) with lagged own price and competitor
                     prices -- targets the causal demand slope.

Each returns an `ElasticityResult` with the point estimate and a 95% CI so the
naive-vs-corrected gap can be read off directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS

from .data import COMP_PRICE_COLS


@dataclass
class ElasticityResult:
    method: str
    epsilon: float          # elasticity (slope on log price; expect < 0)
    ci_low: float
    ci_high: float
    n: int
    note: str = ""

    def __str__(self) -> str:
        return (
            f"{self.method:<14} epsilon = {self.epsilon:+.3f}  "
            f"95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]  (n={self.n})"
            + (f"  | {self.note}" if self.note else "")
        )


def naive_ols(df: pd.DataFrame) -> ElasticityResult:
    """Plain OLS: log_q ~ log_p. Biased toward zero by price endogeneity."""
    m = smf.ols("log_q ~ log_p", data=df).fit(cov_type="HC1")
    ci = m.conf_int().loc["log_p"]
    return ElasticityResult(
        "naive_ols", m.params["log_p"], ci[0], ci[1], int(m.nobs),
        note="biased foil",
    )


def fixed_effects(df: pd.DataFrame, time_effects: bool = True) -> ElasticityResult:
    """Product fixed effects (optionally + time effects), clustered by product.

    Removes time-invariant per-product confounders (brand, category, quality).
    """
    panel = df.set_index(["product_id", "period"])
    mod = PanelOLS.from_formula(
        "log_q ~ log_p + EntityEffects" + (" + TimeEffects" if time_effects else ""),
        data=panel,
        drop_absorbed=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    ci = res.conf_int().loc["log_p"]
    return ElasticityResult(
        "fixed_effects", res.params["log_p"], ci["lower"], ci["upper"],
        int(res.nobs),
        note="product" + ("+time" if time_effects else "") + " FE, clustered",
    )


def iv_2sls(
    df: pd.DataFrame,
    instruments: list[str] | None = None,
) -> ElasticityResult:
    """Instrument log price with competitor prices (2SLS).

    Identifying assumption: instruments shift the firm's *cost/competition*
    (and hence its price) but don't directly move this product's demand shock.

    We default to **competitor prices only** and deliberately exclude lagged own
    price. Empirically lag_price predicts current price almost perfectly (sticky
    prices, first-stage R^2 ~ 0.99) -- a very *strong* instrument that is also
    *invalid*: lagged price carries lagged demand shocks, violating exclusion.
    Strength is necessary but not sufficient (Angrist & Pischke, *Mastering
    'Metrics*, ch. 3); validity is the binding constraint here.

    The reported note carries the first-stage F so weak-instrument problems are
    visible (rule of thumb: F < 10 is weak).
    """
    if instruments is None:
        instruments = [f"log_{c}" for c in COMP_PRICE_COLS]

    data = df.dropna(subset=["log_q", "log_p", *instruments])
    formula = "log_q ~ 1 + [log_p ~ " + " + ".join(instruments) + "]"
    res = IV2SLS.from_formula(formula, data=data).fit(cov_type="robust")
    ci = res.conf_int().loc["log_p"]
    first_stage_f = float(res.first_stage.diagnostics["f.stat"].iloc[0])
    return ElasticityResult(
        "iv_2sls", res.params["log_p"], ci["lower"], ci["upper"], int(res.nobs),
        note=f"IV {', '.join(instruments)}; first-stage F={first_stage_f:.0f}",
    )


def compare_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run all three estimators and return a tidy comparison table."""
    rows = [naive_ols(df), fixed_effects(df), iv_2sls(df)]
    return pd.DataFrame(
        [{"method": r.method, "epsilon": r.epsilon, "ci_low": r.ci_low,
          "ci_high": r.ci_high, "n": r.n, "note": r.note} for r in rows]
    )


if __name__ == "__main__":
    from .data import load_prepared

    d = load_prepared()
    for r in (naive_ols(d), fixed_effects(d), iv_2sls(d)):
        print(r)
