"""Load and prepare the Suddharshan retail-price-optimization panel.

The raw file is a product x month panel. We add the columns the elasticity
models need (logs, a time index, a markdown proxy) and keep the originals.

Fetch first (nothing is committed):
    kaggle datasets download -d suddharshan/retail-price-optimization -p data/raw --unzip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# repo-root-relative default so it works from notebooks/ or src/ or the project root
DEFAULT_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "retail_price.csv"

# competitor price columns -> candidate IV instruments for unit_price
COMP_PRICE_COLS = ["comp_1", "comp_2", "comp_3"]


def load_raw(path: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    """Read the raw CSV, failing loudly with the fetch command if it's missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            "  kaggle datasets download -d suddharshan/retail-price-optimization "
            "-p data/raw --unzip"
        )
    return pd.read_csv(path)


def load_prepared(path: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    """Return the panel with log columns, a datetime period, and a markdown proxy.

    Added columns:
        period          integer month index (year*12+month) for time effects
        log_q           log(qty)            -- demand
        log_p           log(unit_price)     -- own price
        log_lag_price   log(lag_price)      -- IV candidate (own price, lagged)
        log_comp_{1,2,3} log(comp_n)        -- IV candidates (competitor prices)
        is_markdown     unit_price < product's median price (promo/clearance proxy)

    Note: this dataset has no explicit promo flag, so `is_markdown` is a proxy.
    Treat promo-split results as suggestive, not definitive.
    """
    df = load_raw(path).copy()

    # integer month index for panel time effects. We deliberately avoid
    # pd.to_datetime here: its datetime `take` path segfaults under pandas
    # 3.0.4, and an ordered integer is all the panel time index needs.
    df["period"] = (df["year"] * 12 + df["month"]).astype(int)

    df["log_q"] = np.log(df["qty"])
    df["log_p"] = np.log(df["unit_price"])
    df["log_lag_price"] = np.log(df["lag_price"])
    for col in COMP_PRICE_COLS:
        df[f"log_{col}"] = np.log(df[col])

    # markdown proxy: priced below this product's own median (within-product discount)
    product_median = df.groupby("product_id")["unit_price"].transform("median")
    df["is_markdown"] = df["unit_price"] < product_median

    return df


if __name__ == "__main__":
    d = load_prepared()
    print(f"rows={len(d)}  products={d.product_id.nunique()}  months={d.period.nunique()}")
    print(d[["product_id", "period", "unit_price", "qty", "log_p", "log_q", "is_markdown"]].head())
