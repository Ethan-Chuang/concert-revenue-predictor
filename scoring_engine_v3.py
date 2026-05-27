# -*- coding: utf-8 -*-
"""
Stage 2 Scoring Engine v3 — predicts FILL RATE, not raw tickets.

Pipeline change vs v2:
  - Model now predicts avg_capacity_sold (fill rate %)
  - Revenue = candidate_price × (predicted_fill_rate / 100) × candidate_capacity
  - Predicted fill rate is clipped to [0, 100] before computing tickets

Same `set_price_and_premium` helper as v2 — log_price_premium is still
recomputed for each candidate price.
"""

import json
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DF_PATH = "/Users/ethanc/RnA/data/test_sample_cleaned_apr8_v2.csv"
SCALER_PATH = "/Users/ethanc/RnA/data/scaler_stats_apr8_v2.json"
MODEL_PATH = "/Users/ethanc/RnA/data/stage1_fill_rate_model_v3.json"

df = pd.read_csv(DF_PATH)
with open(SCALER_PATH) as f:
    scaler_stats = json.load(f)
model = XGBRegressor()
model.load_model(MODEL_PATH)
model_features = model.get_booster().feature_names


def to_real(scaled_val, col):
    return scaled_val * scaler_stats[col]["std"] + scaler_stats[col]["mean"]


def to_scaled(real_val, col):
    return (real_val - scaler_stats[col]["mean"]) / scaler_stats[col]["std"]


def set_price_and_premium(row, candidate_price_z):
    row["ticket_price_avg"] = candidate_price_z
    candidate_price_real = to_real(candidate_price_z, "ticket_price_avg")
    baseline_real = to_real(row["baseline_ticket_price"], "baseline_ticket_price")
    ratio = max(candidate_price_real / baseline_real, 0.01)
    row["log_price_premium"] = to_scaled(np.log(ratio), "log_price_premium")
    return row


def predict_revenue(row, candidate_price_z, candidate_cap_z):
    """Predict revenue in real dollars for a (price, capacity) candidate."""
    row = row.copy()
    row["avg_event_capacity"] = candidate_cap_z
    row = set_price_and_premium(row, candidate_price_z)

    e = pd.DataFrame([row])
    fill_z = model.predict(e)[0]
    fill_real = max(min(to_real(fill_z, "avg_capacity_sold"), 100.0), 0.0)

    cap_real = to_real(candidate_cap_z, "avg_event_capacity")
    price_real = to_real(candidate_price_z, "ticket_price_avg")

    tickets = (fill_real / 100.0) * cap_real
    revenue = price_real * tickets
    return revenue, tickets, fill_real


# Peer grouping
def simplify_genre(g):
    if pd.isna(g):
        return "Other"
    for k in ["Latin", "Pop / Rock", "Country", "Asian Pop"]:
        if k in g:
            return k
    if "Rap" in g or "HipHop" in g:
        return "Rap / HipHop"
    if "Dance" in g or "Electronic" in g:
        return "Dance / Electronic"
    return "Other"


df["genre_group"] = df["genre"].apply(simplify_genre)
df["demand_tier"] = pd.qcut(df["avg_tickets_sold"], q=3, labels=["Low", "Mid", "High"])


def find_peers(event_idx, data, min_peers=10):
    genre = data.loc[event_idx, "genre_group"]
    tier = data.loc[event_idx, "demand_tier"]
    peers = data[(data["genre_group"] == genre) & (data["demand_tier"] == tier)]
    if len(peers) < min_peers:
        peers = data[data["genre_group"] == genre]
    return peers.drop(index=event_idx, errors="ignore")


# Single-event demo
test_idx = 29
target = df.loc[test_idx]
actual_price = to_real(target["ticket_price_avg"], "ticket_price_avg")
actual_cap = to_real(target["avg_event_capacity"], "avg_event_capacity")
actual_tix = to_real(target["avg_tickets_sold"], "avg_tickets_sold")
print(f"Artist: {target['headliner']}")
print(f"Venue: {target['venue']} ({actual_cap:,.0f} seats)")
print(f"Actual: ${actual_price:,.0f} × {actual_tix:,.0f} tickets = ${actual_price*actual_tix:,.0f} revenue")

peers = find_peers(test_idx, df)
price_s = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
cap_s = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values

real_prices = [to_real(p, "ticket_price_avg") for p in price_s]
real_caps = [to_real(c, "avg_event_capacity") for c in cap_s]
print(f"\n3x3 grid for event {test_idx} (Pop/Rock, Mid tier):")
print(f"  Prices: {[f'${p:.0f}' for p in real_prices]}")
print(f"  Caps:   {[f'{c:,.0f}' for c in real_caps]}")

revenues = np.zeros((3, 3))
fills = np.zeros((3, 3))
for i, cap_z in enumerate(cap_s):
    for j, price_z in enumerate(price_s):
        rev, _, fill = predict_revenue(target[model_features], price_z, cap_z)
        revenues[i, j] = rev
        fills[i, j] = fill

best_i, best_j = np.unravel_index(np.argmax(revenues), revenues.shape)
tier_labels = ["Low", "Mid", "High"]
cap_labels = ["Small", "Mid", "Large"]

print("\nRevenue grid:")
print(f"{'':20} " + " ".join(f"{t:>10}(${p:.0f})" for t, p in zip(tier_labels, real_prices)))
for i, c in enumerate(real_caps):
    line = f"{cap_labels[i]:>5}({c:,.0f}): "
    for j in range(3):
        marker = " *" if (i, j) == (best_i, best_j) else "  "
        line += f"  ${revenues[i,j]:>9,.0f} ({fills[i,j]:>3.0f}%){marker}"
    print(line)

print(f"\n* Recommended: {tier_labels[best_j]} price (${real_prices[best_j]:,.0f}) × "
      f"{cap_labels[best_i]} venue ({real_caps[best_i]:,.0f} seats)")
print(f"  Expected revenue: ${revenues[best_i, best_j]:,.0f}")


# Validation across all events
print("\n=== Validation ===")
val = []
for idx in df.index:
    genre = df.loc[idx, "genre_group"]
    tier = df.loc[idx, "demand_tier"]
    peers = df[(df["genre_group"] == genre) & (df["demand_tier"] == tier)].drop(index=idx, errors="ignore")
    if len(peers) < 10:
        peers = df[df["genre_group"] == genre].drop(index=idx, errors="ignore")
    if len(peers) < 5:
        continue

    prices = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
    caps = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values

    best_rev = -np.inf
    best_pi = None
    target_row = df.loc[idx][model_features]
    for ci, cap_z in enumerate(caps):
        for pi, price_z in enumerate(prices):
            rev, _, _ = predict_revenue(target_row, price_z, cap_z)
            if rev > best_rev:
                best_rev = rev
                best_pi = pi

    actual_p = to_real(df.loc[idx, "ticket_price_avg"], "ticket_price_avg")
    actual_t = to_real(df.loc[idx, "avg_tickets_sold"], "avg_tickets_sold")
    actual_rev = actual_p * actual_t

    val.append({"genre": genre, "tier": tier,
                "actual_revenue": actual_rev,
                "recommended_revenue": best_rev,
                "best_price_tier": best_pi,
                "improved": best_rev >= actual_rev})

val = pd.DataFrame(val)
print(f"Events scored: {len(val)}")
print(f"Recommendation matches/beats actual: {val['improved'].sum()}/{len(val)} ({val['improved'].mean()*100:.0f}%)")
print("By genre:")
for g in val["genre"].unique():
    sub = val[val["genre"] == g]
    print(f"  {g:<22}: {sub['improved'].mean()*100:>3.0f}%  (n={len(sub)})")

print("\n=== Price-tier bias (v3, fill rate model) ===")
for i, label in enumerate(tier_labels):
    pct = (val["best_price_tier"] == i).mean() * 100
    print(f"  {label} price:  {pct:.0f}%")
