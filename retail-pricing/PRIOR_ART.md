# Prior-art scan — dataset `suddharshan/retail-price-optimization`

Existing community notebooks for the same data. Read the top few before building:
their consensus approach is the baseline to beat, and the most-commented ones are
where the sharp critiques live (open the ref to read comments — not in the API).

## Highest-rated (by votes)
| votes | title | author | ref (open for code + comments) | last run |
|---|---|---|---|---|
| 380 | 🛒Retail Price Optimization | Harsh Singh | [harshsingh2209/retail-price-optimization](https://www.kaggle.com/code/harshsingh2209/retail-price-optimization) | 2023-04-23 |
| 57 | Retail_Price_Optimization_ML | İsmail ACAR | [smailaar/retail-price-optimization-ml](https://www.kaggle.com/code/smailaar/retail-price-optimization-ml) | 2023-01-06 |
| 33 | Retail Price Optimization | Isharab Ahmed | [isharab/retail-price-optimization](https://www.kaggle.com/code/isharab/retail-price-optimization) | 2023-05-29 |
| 18 | price_optimization | Thanhdz01 | [thanhdz01/price-optimization](https://www.kaggle.com/code/thanhdz01/price-optimization) | 2023-12-27 |
| 14 | Ecommerce pricing (including features description) | Alex1iv | [alex1iv/ecommerce-pricing-including-features-description](https://www.kaggle.com/code/alex1iv/ecommerce-pricing-including-features-description) | 2023-07-03 |
| 13 | pricing optimization | Balaji Pandi | [rbalajipandi/pricing-optimization](https://www.kaggle.com/code/rbalajipandi/pricing-optimization) | 2024-10-28 |
| 13 | Sale Prices Elasticity, Descriptive Statistics | Dr. Sam Fouad | [eslamfouad/sale-prices-elasticity-descriptive-statistics](https://www.kaggle.com/code/eslamfouad/sale-prices-elasticity-descriptive-statistics) | 2023-06-28 |
| 12 | Retail Price Optiomization | Pablo Veinberg | [pveinberg/retail-price-optiomization](https://www.kaggle.com/code/pveinberg/retail-price-optiomization) | 2024-05-04 |

## Most-discussed (by comment count — look here for constructive criticism)
| votes | title | author | ref (open for code + comments) | last run |
|---|---|---|---|---|
| 380 | 🛒Retail Price Optimization | Harsh Singh | [harshsingh2209/retail-price-optimization](https://www.kaggle.com/code/harshsingh2209/retail-price-optimization) | 2023-04-23 |
| 33 | Retail Price Optimization | Isharab Ahmed | [isharab/retail-price-optimization](https://www.kaggle.com/code/isharab/retail-price-optimization) | 2023-05-29 |
| 14 | Ecommerce pricing (including features description) | Alex1iv | [alex1iv/ecommerce-pricing-including-features-description](https://www.kaggle.com/code/alex1iv/ecommerce-pricing-including-features-description) | 2023-07-03 |
| 1 | Retail Pricing Optimization | WillieMaize | [williemaize/retail-pricing-optimization](https://www.kaggle.com/code/williemaize/retail-pricing-optimization) | 2026-06-29 |
| 3 | Retail Price Optimization Engine | Deepak Rana | [deepaksacso/retail-price-optimization-engine](https://www.kaggle.com/code/deepaksacso/retail-price-optimization-engine) | 2026-01-24 |
| 0 | Retail Price Optimization | Bijay-Odyssey | [bijaybeezoe/retail-price-optimization](https://www.kaggle.com/code/bijaybeezoe/retail-price-optimization) | 2025-11-26 |
| 5 | Retail Price Optimization & Demand Modeling | Bijay-Odyssey | [bijaybeezoe/retail-price-optimization-demand-modeling](https://www.kaggle.com/code/bijaybeezoe/retail-price-optimization-demand-modeling) | 2025-11-26 |
| 4 | price_optimization | VICTOR IHEANACHO | [victoriheanacho/price-optimization](https://www.kaggle.com/code/victoriheanacho/price-optimization) | 2025-11-06 |

## Synthesize (worked example — read the top notebook's code via `kaggle kernels pull`)
- **Consensus approach / baseline:** The top notebook (Harsh Singh, 380 votes) and most others
  frame this as a **supervised regression to predict `total_price`** (RandomForest, sometimes
  OLS/XGBoost) + EDA + SHAP/permutation-importance explainability. It's a *predict-the-price*
  framing, not a *set-the-optimal-price* one.
- **What the crowd does NOT do:** estimate **price elasticity**, address **price endogeneity**,
  or use **causal methods** (IV / fixed effects). "Optimization" rarely means maximizing
  `(p−c)·q(p)`; it usually means predicting the price the data already shows.
- **What to adopt:** their solid EDA + competitor-comparison framing, and SHAP as an
  interpretability layer are worth borrowing for presentation.
- **My differentiated angle (the edge):** treat it as a **causal demand problem** — naive vs
  FE vs IV elasticity (the endogeneity story the crowd skips), then **margin** optimization
  `p*=c·ε/(ε+1)` with a per-product counterfactual. That's a genuine step beyond the top voted
  notebooks, not a reskin.
- **Comments to still read in-browser** (not in API): open the most-discussed refs above and
  note any critiques (data leakage, target definition, cost assumptions) before finalizing.
- **Credit / license notes:** dataset is CC0; notebooks are individually licensed — credit any
  specific idea you adopt, adapt rather than copy.
