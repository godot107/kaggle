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

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab  # notebooks/01_eda.ipynb
```

## Assumptions / caveats (keep honest)
- Constant-elasticity demand is a simplification; check robustness to functional form.
- Unit cost `c` is assumed (run a few scenarios) unless the data provides it.
- Static, single-product optimization — ignores competitor reaction and cross-product
  cannibalization. Those are v2.
