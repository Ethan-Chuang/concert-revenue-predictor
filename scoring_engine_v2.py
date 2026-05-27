# -*- coding: utf-8 -*-
"""
Stage 2 Scoring Engine v2 (with baseline_ticket_price + log_price_premium)

Key change from the original scoring engine: whenever a candidate price is
substituted into the feature row (for the 3x3 grid, the 5x5 grid, or the
elasticity-based scoring), log_price_premium is recomputed for that candidate.
This lets the model see *how far this price deviates from the artist's
baseline*, which is the signal that disentangles popularity from pricing.
"""

import json
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================
# Load v2 data + v2 model + v2 scaler stats
# =============================================================
DF_PATH = "/Users/ethanc/RnA/data/test_sample_cleaned_apr8_v2.csv"
SCALER_PATH = "/Users/ethanc/RnA/data/scaler_stats_apr8_v2.json"
MODEL_PATH = "/Users/ethanc/RnA/data/stage1_ticket_sales_model_v2.json"

df = pd.read_csv(DF_PATH)
with open(SCALER_PATH) as f:
    scaler_stats = json.load(f)

model = XGBRegressor()
model.load_model(MODEL_PATH)
model_features = model.get_booster().feature_names


def to_real(scaled_val, col):
    """Inverse z-score: scaled → original units."""
    return scaled_val * scaler_stats[col]["std"] + scaler_stats[col]["mean"]


def to_scaled(real_val, col):
    """Z-score: original units → scaled."""
    return (real_val - scaler_stats[col]["mean"]) / scaler_stats[col]["std"]


def set_price_and_premium(row, candidate_price_z):
    """
    Update both ticket_price_avg AND log_price_premium for a candidate price.

    baseline_ticket_price is artist-level (fixed in the row), so log_price_premium
    = log(candidate_price / baseline) and must be recomputed each time the
    candidate price changes.
    """
    row["ticket_price_avg"] = candidate_price_z
    candidate_price_real = to_real(candidate_price_z, "ticket_price_avg")
    baseline_real = to_real(row["baseline_ticket_price"], "baseline_ticket_price")
    ratio = max(candidate_price_real / baseline_real, 0.01)
    row["log_price_premium"] = to_scaled(np.log(ratio), "log_price_premium")
    return row


# =============================================================
# Peer grouping (unchanged from original)
# =============================================================
def simplify_genre(genre_str):
    if pd.isna(genre_str):
        return "Other"
    if "Latin" in genre_str:
        return "Latin"
    if "Pop / Rock" in genre_str:
        return "Pop / Rock"
    if "Country" in genre_str:
        return "Country"
    if "Rap" in genre_str or "HipHop" in genre_str:
        return "Rap / HipHop"
    if "Dance" in genre_str or "Electronic" in genre_str:
        return "Dance / Electronic"
    if "Asian Pop" in genre_str:
        return "Asian Pop"
    return "Other"


df["genre_group"] = df["genre"].apply(simplify_genre)
df["demand_tier"] = pd.qcut(df["avg_tickets_sold"], q=3, labels=["Low", "Mid", "High"])


def find_peers(event_idx, data, min_peers=10):
    genre = data.loc[event_idx, "genre_group"]
    tier = data.loc[event_idx, "demand_tier"]
    peers = data[(data["genre_group"] == genre) & (data["demand_tier"] == tier)]
    match_type = "genre + tier"
    if len(peers) < min_peers:
        peers = data[data["genre_group"] == genre]
        match_type = "genre only (fallback)"
    peers = peers.drop(index=event_idx, errors="ignore")
    return peers, {
        "genre": genre,
        "tier": tier,
        "match_type": match_type,
        "num_peers": len(peers),
    }


def build_grid(peers, percentiles=[0.25, 0.50, 0.75]):
    price_levels = peers["ticket_price_avg"].quantile(percentiles).values
    capacity_levels = peers["avg_event_capacity"].quantile(percentiles).values
    n = len(percentiles)
    if n == 3:
        plabels, clabels = ["Low", "Mid", "High"], ["Small", "Mid", "Large"]
    elif n == 5:
        plabels = ["Very Low", "Low", "Mid", "High", "Very High"]
        clabels = ["Very Small", "Small", "Mid", "Large", "Very Large"]
    else:
        plabels = [f"P{int(p*100)}" for p in percentiles]
        clabels = [f"C{int(p*100)}" for p in percentiles]
    return price_levels, capacity_levels, {
        "price_labels": plabels,
        "capacity_labels": clabels,
        "price_values": price_levels,
        "capacity_values": capacity_levels,
        "percentiles_used": percentiles,
    }


# =============================================================
# 3x3 grid scoring for a single event
# =============================================================
test_idx = 29
target_event_df = df.loc[test_idx]
X_event = pd.DataFrame([target_event_df[model_features]])
pred_scaled = model.predict(X_event)[0]
pred_tickets = to_real(pred_scaled, "avg_tickets_sold")

actual_price = to_real(target_event_df["ticket_price_avg"], "ticket_price_avg")
actual_capacity = to_real(target_event_df["avg_event_capacity"], "avg_event_capacity")
actual_tickets = to_real(target_event_df["avg_tickets_sold"], "avg_tickets_sold")
print(f"Artist:           {target_event_df['headliner']}")
print(f"Venue:            {target_event_df['venue']}  ({actual_capacity:,.0f} seats)")
print(f"Avg ticket price: ${actual_price:,.0f}")
print(f"Actual tickets:   {actual_tickets:,.0f}")
print(f"Predicted tickets:{pred_tickets:,.0f}")

peers, peer_info = find_peers(test_idx, df)
price_s, cap_s, grid_info = build_grid(peers)
print(f"\nPeer group: {peer_info['num_peers']} events ({peer_info['match_type']})")

# Build 9 candidate rows — recompute log_price_premium for each
rows = []
for cap in cap_s:
    for price in price_s:
        row = target_event_df[model_features].copy()
        row["avg_event_capacity"] = cap
        row = set_price_and_premium(row, price)
        rows.append(row)

X_grid = pd.DataFrame(rows, columns=model_features)
preds_scaled = model.predict(X_grid)
preds_tickets = np.array([to_real(p, "avg_tickets_sold") for p in preds_scaled])
preds_tickets = np.clip(preds_tickets, 0, None).reshape(3, 3)

real_prices = [to_real(p, "ticket_price_avg") for p in price_s]
real_caps = [to_real(c, "avg_event_capacity") for c in cap_s]

# Cap at capacity, compute revenue, find best
tickets_sold = np.array(
    [[min(preds_tickets[i, j], real_caps[i]) for j in range(3)] for i in range(3)]
)
fill_rates = (tickets_sold / np.array(real_caps).reshape(3, 1)) * 100
revenues = np.array(
    [[real_prices[j] * tickets_sold[i, j] for j in range(3)] for i in range(3)]
)
best_i, best_j = np.unravel_index(np.argmax(revenues), revenues.shape)

print(f"\nRecommended: {grid_info['price_labels'][best_j]} price (${real_prices[best_j]:,.0f}) × "
      f"{grid_info['capacity_labels'][best_i]} venue ({real_caps[best_i]:,.0f} seats)")
print(f"Expected revenue: ${revenues[best_i, best_j]:,.0f}")


# =============================================================
# Validation: how often does the engine match/beat actual revenue?
# =============================================================
validation_results = []
for idx in df.index:
    genre = df.loc[idx, "genre_group"]
    tier = df.loc[idx, "demand_tier"]
    peers = df[(df["genre_group"] == genre) & (df["demand_tier"] == tier)].drop(
        index=idx, errors="ignore"
    )
    if len(peers) < 10:
        peers = df[df["genre_group"] == genre].drop(index=idx, errors="ignore")
    if len(peers) < 5:
        continue

    prices = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
    caps = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values

    best_rev = -np.inf
    for cap in caps:
        for price in prices:
            row = df.loc[idx][model_features].copy()
            row["avg_event_capacity"] = cap
            row = set_price_and_premium(row, price)
            e = pd.DataFrame([row])
            pred = max(model.predict(e)[0], 0)
            pred = min(pred, cap) if cap > 0 else pred
            rev = price * pred
            if rev > best_rev:
                best_rev = rev

    actual_rev = df.loc[idx, "ticket_price_avg"] * df.loc[idx, "avg_tickets_sold"]
    validation_results.append(
        {
            "genre": genre,
            "tier": tier,
            "actual_revenue": actual_rev,
            "recommended_revenue": best_rev,
            "improved": best_rev >= actual_rev,
        }
    )

val = pd.DataFrame(validation_results)
print(f"\n=== Validation ===")
print(f"Events scored: {len(val)}")
print(f"Recommendation matches/beats actual: {val['improved'].sum()}/{len(val)} ({val['improved'].mean()*100:.0f}%)")
print("By genre:")
for g in val["genre"].unique():
    sub = val[val["genre"] == g]
    print(f"  {g:<22}: {sub['improved'].mean()*100:>3.0f}%  (n={len(sub)})")


# =============================================================
# Bias check: what fraction of recommendations land at each price tier?
# =============================================================
bias_results = []
for idx in df.index:
    genre = df.loc[idx, "genre_group"]
    tier = df.loc[idx, "demand_tier"]
    peers = df[(df["genre_group"] == genre) & (df["demand_tier"] == tier)].drop(
        index=idx, errors="ignore"
    )
    if len(peers) < 10:
        peers = df[df["genre_group"] == genre].drop(index=idx, errors="ignore")
    if len(peers) < 5:
        continue

    prices = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
    caps = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values

    best_rev = -np.inf
    best_pi = None
    for ci, cap in enumerate(caps):
        for pi, price in enumerate(prices):
            row = df.loc[idx][model_features].copy()
            row["avg_event_capacity"] = cap
            row = set_price_and_premium(row, price)
            e = pd.DataFrame([row])
            pred = max(model.predict(e)[0], 0)
            pred = min(pred, cap) if cap > 0 else pred
            rev = price * pred
            if rev > best_rev:
                best_rev = rev
                best_pi = pi
    bias_results.append({"best_price_tier": best_pi})

bias_df = pd.DataFrame(bias_results)
tier_labels = ["Low", "Mid", "High"]
print("\n=== Price-tier bias (v2 model) ===")
for i, label in enumerate(tier_labels):
    pct = (bias_df["best_price_tier"] == i).mean() * 100
    print(f"  {label} price:  {pct:.0f}%")


# =============================================================
# Elasticity-adjusted scoring
# =============================================================
def score_event_with_elasticity(idx, df, model, model_features, elasticity=-0.5):
    """
    Score one event with price elasticity adjustment. Prediction is done at the
    peer median price (the most reliable point), then demand is adjusted up/down
    based on candidate price. log_price_premium is set to match whichever price
    the model is being evaluated at.
    """
    genre = df.loc[idx, "genre_group"]
    tier = df.loc[idx, "demand_tier"]
    peers = df[(df["genre_group"] == genre) & (df["demand_tier"] == tier)].drop(
        index=idx, errors="ignore"
    )
    if len(peers) < 10:
        peers = df[df["genre_group"] == genre].drop(index=idx, errors="ignore")
    if len(peers) < 5:
        return None

    prices_z = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
    caps_z = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values
    median_price_z = prices_z[1]

    prices_real = [to_real(p, "ticket_price_avg") for p in prices_z]
    median_price_real = prices_real[1]

    best_rev = -np.inf
    best_pi, best_ci = None, None

    for ci, cap_z in enumerate(caps_z):
        cap_real = to_real(cap_z, "avg_event_capacity")
        for pi, price_z in enumerate(prices_z):
            # Predict at MEDIAN price to avoid extrapolation
            row = df.loc[idx][model_features].copy()
            row["avg_event_capacity"] = cap_z
            row = set_price_and_premium(row, median_price_z)
            e = pd.DataFrame([row])
            pred_z = max(model.predict(e)[0], 0)
            pred_tickets = max(to_real(pred_z, "avg_tickets_sold"), 0)

            # Apply elasticity around the median
            price_pct_change = (prices_real[pi] - median_price_real) / median_price_real
            multiplier = max(1 + elasticity * price_pct_change, 0.1)
            pred_adjusted = pred_tickets * multiplier
            pred_adjusted = min(pred_adjusted, cap_real) if cap_real > 0 else pred_adjusted

            rev = prices_real[pi] * pred_adjusted
            if rev > best_rev:
                best_rev = rev
                best_pi = pi
                best_ci = ci

    return {"best_price_tier": best_pi, "best_cap_tier": best_ci, "best_rev": best_rev}


result = score_event_with_elasticity(test_idx, df, model, model_features, elasticity=-0.5)
print(f"\nEvent {test_idx} with elasticity=-0.5:")
print(f"  Recommended: {tier_labels[result['best_price_tier']]} price × {tier_labels[result['best_cap_tier']]} venue")
print(f"  Expected revenue: ${result['best_rev']:,.0f}")
