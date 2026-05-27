"""
Adds baseline_ticket_price and log_price_premium features to the cleaned dataset.

Pipeline:
  1. Load testing_sample_raw.csv (unstandardized prices)
  2. Compute artist-level expanding median price (no leakage), with fallbacks
  3. Compute log_price_premium = log(actual_price / baseline_price)
  4. Z-score the new features using the existing scaler convention
  5. Merge by eventid into test_sample_cleaned_apr8.csv
  6. Save test_sample_cleaned_apr8_v2.csv and scaler_stats_apr8_v2.json

Run from the model/ directory or anywhere — uses absolute paths.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/ethanc/RnA/data")

RAW_PATH = DATA_DIR / "testing_sample_raw.csv"
CLEANED_PATH = DATA_DIR / "test_sample_cleaned_apr8.csv"
SCALER_PATH = DATA_DIR / "scaler_stats_apr8.json"

OUT_CLEANED = DATA_DIR / "test_sample_cleaned_apr8_v2.csv"
OUT_SCALER = DATA_DIR / "scaler_stats_apr8_v2.json"

MIN_HIST = 3  # min prior events to use artist's own historical median
PRICE_RATIO_FLOOR = 0.01  # prevents log(0) on extreme outliers

# 1. Load raw (unstandardized) data
raw = pd.read_csv(RAW_PATH, index_col=0)
raw["event_date"] = pd.to_datetime(raw["event_date"])

# Dedupe by eventid — the raw file has ~36 events that appear in multiple rows
# (same show counted twice). Keep one row per event so the expanding median
# doesn't double-count.
raw = raw.drop_duplicates(subset="eventid", keep="first")
raw = raw.sort_values(["headliner", "event_date"]).reset_index(drop=True)

# 2. Artist-level expanding median (shift excludes current event → no leakage)
raw["artist_hist_median"] = (
    raw.groupby("headliner")["ticket_price_avg"]
    .transform(lambda x: x.expanding().median().shift(1))
)
raw["artist_hist_count"] = (
    raw.groupby("headliner")["ticket_price_avg"]
    .transform(lambda x: x.expanding().count().shift(1))
    .fillna(0)
    .astype(int)
)

# Fallback 1: genre + market median
gm_median = (
    raw.groupby(["genre", "market"])["ticket_price_avg"]
    .median()
    .rename("genre_market_median")
)
raw = raw.merge(gm_median, on=["genre", "market"], how="left")

# Fallback 2: global median
global_median = float(raw["ticket_price_avg"].median())

raw["baseline_ticket_price"] = np.where(
    raw["artist_hist_count"] >= MIN_HIST,
    raw["artist_hist_median"],
    raw["genre_market_median"].fillna(global_median),
)

# 3. log_price_premium = log(actual / baseline)
ratio = (raw["ticket_price_avg"] / raw["baseline_ticket_price"]).clip(
    lower=PRICE_RATIO_FLOOR
)
raw["log_price_premium"] = np.log(ratio)

print(f"Raw rows: {len(raw)}")
print("\nbaseline_ticket_price summary:")
print(raw["baseline_ticket_price"].describe())
print("\nlog_price_premium summary:")
print(raw["log_price_premium"].describe())

# 4. Load existing cleaned data + scaler stats
cleaned = pd.read_csv(CLEANED_PATH)
with open(SCALER_PATH) as f:
    scaler_stats = json.load(f)

# Compute z-score stats for new features and append to scaler_stats
for col in ["baseline_ticket_price", "log_price_premium"]:
    s = raw[col].dropna()
    scaler_stats[col] = {"mean": float(s.mean()), "std": float(s.std())}

# Standardize the new features
raw["baseline_ticket_price_z"] = (
    raw["baseline_ticket_price"] - scaler_stats["baseline_ticket_price"]["mean"]
) / scaler_stats["baseline_ticket_price"]["std"]

raw["log_price_premium_z"] = (
    raw["log_price_premium"] - scaler_stats["log_price_premium"]["mean"]
) / scaler_stats["log_price_premium"]["std"]

# 5. Merge into cleaned dataset by eventid
new_features = raw[
    ["eventid", "baseline_ticket_price_z", "log_price_premium_z"]
].rename(
    columns={
        "baseline_ticket_price_z": "baseline_ticket_price",
        "log_price_premium_z": "log_price_premium",
    }
)
cleaned = cleaned.merge(new_features, on="eventid", how="left")

n_missing = cleaned[["baseline_ticket_price", "log_price_premium"]].isna().any(axis=1).sum()
print(f"\nRows with missing new features (will fill with 0 = mean): {n_missing}")
cleaned[["baseline_ticket_price", "log_price_premium"]] = cleaned[
    ["baseline_ticket_price", "log_price_premium"]
].fillna(0)

# 6. Save outputs
cleaned.to_csv(OUT_CLEANED, index=False)
with open(OUT_SCALER, "w") as f:
    json.dump(scaler_stats, f, indent=2)

print(f"\nSaved: {OUT_CLEANED}")
print(f"Saved: {OUT_SCALER}")
print(f"\nNew cleaned shape: {cleaned.shape}")
