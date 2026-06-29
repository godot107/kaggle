# Retail Pricing Optimization

Estimate **price elasticity of demand**, correct for the fact that retailers set prices
strategically (price endogeneity), and use the corrected elasticity to recommend
**margin-maximizing** prices.

> Trello: Personal ML Projects → "Retail Pricing Optimization" (card aJq22DBv)

## The core idea
Optimize *margin* `(p − c)·q(p)`, not gross revenue. A naive `quantity ~ price` regression
gives a biased (too-flat) elasticity because price responds to demand. The **gap between the
naive and causally-corrected elasticity** — and the margin you'd leave on the table by pricing
on the wrong number — is the headline result.

Sanity check: for constant-elasticity demand `q = A·p^ε` (ε < −1), the margin-maximizing
price has a closed form: **`p* = c · ε/(ε+1)`**.

## Data
Not committed. Fetch into `data/raw/`.

- **v1 prototype:** [Kaggle — Retail Price Optimization (Suddharshan)](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization)
  ```bash
  kaggle datasets download -d suddharshan/retail-price-optimization -p data/raw --unzip
  ```
- **v2 "real":** [dunnhumby — Breakfast at the Frat](https://www.dunnhumby.com/source-files/)
  (store×week×product with price, base_price, feature/display promo flags → instruments).

## Plan
1. **EDA** — `log(quantity)` vs `log(price)`; promo vs non-promo splits.
2. **Naive model** — OLS log-log elasticity (biased baseline / foil).
3. **Causal model** — fixed effects / promo IV / DoubleML; report ε with confidence intervals.
4. **Cost + optimize** — assume unit cost `c`; maximize `(p−c)·q(p)`; check vs `p* = c·ε/(ε+1)`.
5. **Validate** — counterfactual margin uplift vs actual prices; state assumptions explicitly.
6. **Writeup** — narrative: naive ε vs corrected ε → margin cost of pricing on the wrong number.

## Layout
```
src/                 reusable, importable pipeline
  data.py            load_prepared(): clean panel + logs + markdown proxy
  elasticity.py      naive_ols / fixed_effects / iv_2sls -> ElasticityResult
  optimize.py        optimal_price p*=c·ε/(ε+1), margin_uplift, numeric check
notebooks/           01_eda -> 02_elasticity -> 03_optimize (import from src/)
kaggle/              self-contained notebook for publishing on Kaggle
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.elasticity      # smoke-test the estimators on the fetched data
jupyter lab                   # notebooks/01_eda.ipynb
```

## Publish on Kaggle
`kaggle/retail-pricing-optimization.ipynb` is **self-contained** (no `src/` import;
auto-detects the data at `/kaggle/input/...`). The dataset is attached via
`kaggle/kernel-metadata.json`.
```bash
cd kaggle && kaggle kernels push   # williemaize/retail-pricing-optimization
```
Starts **private** (`is_private: true`); flip to `false` in the metadata (or the Kaggle UI)
when ready to publish for view.

## Results (corrected for price endogeneity)
| Estimator | ε | 95% CI | Read |
|---|---|---|---|
| naive OLS | −0.13 | [−0.24, −0.02] | biased foil — looks inelastic |
| product+time FE | **−2.69** | [−3.62, −1.75] | within-product: genuinely elastic |
| IV (competitor prices) | −0.16 | [−0.35, +0.04] | strong first stage (robust F≈395), brushes 0 |

The naive→FE gap is the headline: pricing on ε=−0.13 vs −2.69 is the difference between
"raise price without limit" (no interior optimum) and a finite margin-maximizing `p*`.

### Per-product counterfactual validation (notebook 04)
Each product gets its own elasticity (partial-pooled toward the pooled estimate to tame the
~13-month noise), its margin-maximizing price, and a counterfactual margin uplift vs the prices
actually charged. Guardrails — cap extreme ε, clip price moves to ±25% — keep recommendations
near the observed price support (without them the constant-elasticity curve over-extrapolates).

| cost assumption | optimizable | weighted margin uplift |
|---|---|---|
| c = 40% of price | 34/52 | +45% |
| c = 60% of price | 34/52 | **+17%** |
| c = 70% of price | 34/52 | +8% |

Aggregate uplift stays **positive across the whole cost sweep**; at a central assumption the model
mostly recommends modest cuts (23 cut / 18 hold / 11 raise). Trust the direction and rough
magnitude, not three sig figs.

## Assumptions / caveats (keep honest)
- Constant-elasticity demand is a simplification; check robustness to functional form.
- Unit cost `c` is assumed (run a few scenarios) unless the data provides it.
- Static, single-product optimization — ignores competitor reaction and cross-product
  cannibalization. Those are v2.
