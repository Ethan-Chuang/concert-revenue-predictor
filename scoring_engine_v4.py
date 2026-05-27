# -*- coding: utf-8 -*-
"""
Stage 2 Scoring Engine v4 — fill rate model + assumed elasticity + fill cap.

The within-artist elasticity fits (see fit_elasticity_v4.py) came out nearly
all positive due to (a) endogenous pricing decisions by promoters and (b)
censoring at 100% fill rate. The Pollstar data structurally can't reveal
price elasticity, so v4 uses a sensible assumed elasticity from the
concert economics literature (default -0.4) instead.

Recipe:
  1. Predict base fill rate at the PEER MEDIAN price using the v3 model
     (avoids extrapolating into price regions the model didn't train on).
  2. Apply constant elasticity adjustment for each candidate price:
        fill_adj = base_fill * (candidate_price / median_price) ^ elasticity
  3. Cap fill_adj at MAX_FILL (default 0.90) — realistic plausibility ceiling.
  4. Revenue = candidate_price × fill_adj × candidate_capacity.

This combines the cleaner v3 architecture (fill rate target, decoupled venue)
with an explicit pricing model the team can document and defend.
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

# Hyperparams of the scoring engine (NOT the model)
ELASTICITY = -0.4    # canonical concert demand elasticity
MAX_FILL = 0.90      # plausibility cap on predicted fill rate

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
    cand_real = to_real(candidate_price_z, "ticket_price_avg")
    base_real = to_real(row["baseline_ticket_price"], "baseline_ticket_price")
    ratio = max(cand_real / base_real, 0.01)
    row["log_price_premium"] = to_scaled(np.log(ratio), "log_price_premium")
    return row


def predict_base_fill(row, median_price_z, candidate_cap_z):
    """Predict fill rate at the peer median price (the most reliable point)."""
    row = row.copy()
    row["avg_event_capacity"] = candidate_cap_z
    row = set_price_and_premium(row, median_price_z)
    e = pd.DataFrame([row])
    fill_z = model.predict(e)[0]
    fill_real = max(min(to_real(fill_z, "avg_capacity_sold"), 100.0), 0.0)
    return fill_real / 100.0  # return as fraction [0, 1]


def score_revenue(row, prices_z, cap_z, elasticity=ELASTICITY, max_fill=MAX_FILL):
    """
    For all candidate prices at a given capacity, compute (price, revenue, fill).

    Steps:
      - predict base fill rate at median price
      - adjust for each candidate price via constant-elasticity formula
      - cap at max_fill, then compute revenue
    """
    prices_real = [to_real(p, "ticket_price_avg") for p in prices_z]
    median_price_real = prices_real[1]  # median is the middle of qcut [0.25, 0.5, 0.75]
    cap_real = to_real(cap_z, "avg_event_capacity")

    base_fill = predict_base_fill(row, prices_z[1], cap_z)

    results = []
    for pi, p_real in enumerate(prices_real):
        # constant elasticity demand curve, centered at median
        ratio = (p_real / median_price_real) ** elasticity
        fill_adj = min(base_fill * ratio, max_fill)
        fill_adj = max(fill_adj, 0.0)
        tickets = fill_adj * cap_real
        revenue = p_real * tickets
        results.append((pi, p_real, revenue, fill_adj, tickets))
    return results


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


def find_peers(idx, data, min_peers=10):
    g = data.loc[idx, "genre_group"]
    t = data.loc[idx, "demand_tier"]
    peers = data[(data["genre_group"] == g) & (data["demand_tier"] == t)]
    if len(peers) < min_peers:
        peers = data[data["genre_group"] == g]
    return peers.drop(index=idx, errors="ignore")


# ===== Single-event demo =====
test_idx = 29
target = df.loc[test_idx]
peers = find_peers(test_idx, df)
prices_z = peers["ticket_price_avg"].quantile([0.25, 0.50, 0.75]).values
caps_z = peers["avg_event_capacity"].quantile([0.25, 0.50, 0.75]).values

print(f"=== Event {test_idx}: {target['headliner']} ===")
print(f"Actual: ${to_real(target['ticket_price_avg'], 'ticket_price_avg'):.0f} × "
      f"{to_real(target['avg_tickets_sold'], 'avg_tickets_sold'):.0f} tickets")
print(f"\nElasticity = {ELASTICITY}, max_fill cap = {MAX_FILL*100:.0f}%\n")

tier_labels = ["Low", "Mid", "High"]
cap_labels = ["Small", "Mid", "Large"]
real_prices = [to_real(p, "ticket_price_avg") for p in prices_z]
real_caps = [to_real(c, "avg_event_capacity") for c in caps_z]

grid_rev = np.zeros((3, 3))
grid_fill = np.zeros((3, 3))
for i, cap_z in enumerate(caps_z):
    results = score_revenue(target[model_features], prices_z, cap_z)
    for pi, p_real, rev, fill, tix in results:
        grid_rev[i, pi] = rev
        grid_fill[i, pi] = fill * 100

best_i, best_j = np.unravel_index(np.argmax(grid_rev), grid_rev.shape)
print(f"{'':18} " + " ".join(f"{t:>9}(${p:.0f})" for t, p in zip(tier_labels, real_prices)))
for i, c in enumerate(real_caps):
    line = f"{cap_labels[i]:>5}({c:,.0f}): "
    for j in range(3):
        m = " *" if (i, j) == (best_i, best_j) else "  "
        line += f"  ${grid_rev[i,j]:>9,.0f} ({grid_fill[i,j]:>3.0f}%){m}"
    print(line)
print(f"\n* Recommended: {tier_labels[best_j]} price (${real_prices[best_j]:,.0f}) × "
      f"{cap_labels[best_i]} venue ({real_caps[best_i]:,.0f} seats)")
print(f"  Expected revenue: ${grid_rev[best_i, best_j]:,.0f}")


# ===== Bias sweep across elasticity values =====
print("\n=== Bias sweep across elasticity assumptions ===")
print("(percent of recommendations at each price tier)")
print(f"{'elasticity':>10}  {'Low':>5}  {'Mid':>5}  {'High':>5}  {'match%':>7}")
for elast in [-0.1, -0.3, -0.4, -0.6, -0.8, -1.0]:
    bias = [0, 0, 0]
    n = 0
    n_beats = 0
    for idx in df.index:
        peers = find_peers(idx, df)
        if len(peers) < 5:
            continue
        prices_z = peers["ticket_price_avg"].quantile([0.25, 0.5, 0.75]).values
        caps_z = peers["avg_event_capacity"].quantile([0.25, 0.5, 0.75]).values

        target_row = df.loc[idx][model_features]
        best_rev = -np.inf
        best_pi = None
        for cap_z in caps_z:
            results = score_revenue(target_row, prices_z, cap_z, elasticity=elast)
            for pi, p_real, rev, _, _ in results:
                if rev > best_rev:
                    best_rev = rev
                    best_pi = pi
        if best_pi is None:
            continue
        bias[best_pi] += 1
        n += 1
        actual_p = to_real(df.loc[idx, "ticket_price_avg"], "ticket_price_avg")
        actual_t = to_real(df.loc[idx, "avg_tickets_sold"], "avg_tickets_sold")
        if best_rev >= actual_p * actual_t:
            n_beats += 1

    pct = [100 * b / n for b in bias]
    print(f"{elast:>10.1f}  {pct[0]:>4.0f}%  {pct[1]:>4.0f}%  {pct[2]:>4.0f}%  {100*n_beats/n:>6.0f}%")
