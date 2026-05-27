# -*- coding: utf-8 -*-
"""
Streamlit dashboard for the original Stage 1 model + Stage 2 scoring engine.

Uses:
  - Model:   stage1_ticket_sales_model.json  (predicts avg_tickets_sold)
  - Data:    test_sample_cleaned_apr8.csv    (49-feature dataset)
  - Scaler:  scaler_stats_apr8.json
  - Scoring: 3x3 / 5x5 grid + optional elasticity adjustment
             (taken from "(updated) scoring_engine_part2.py")

Run:
    cd /Users/ethanc/RnA
    source venv/bin/activate
    streamlit run model/dashboard.py
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from xgboost import XGBRegressor

# =====================================================================
# 1. Page config
# =====================================================================
st.set_page_config(page_title="Concert Revenue Predictor", layout="wide")

# =====================================================================
# 2. Paths — resolved relative to this file so the app works both
# locally and on Streamlit Cloud (which sets CWD to the repo root,
# not to model/).
# =====================================================================
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DF_PATH = DATA_DIR / "test_sample_cleaned_apr8.csv"
SCALER_PATH = DATA_DIR / "scaler_stats_apr8.json"
MODEL_PATH = DATA_DIR / "stage1_ticket_sales_model.json"


# =====================================================================
# 3. Loaders (cached)
# =====================================================================
def _simplify_genre(g):
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


@st.cache_data
def load_data():
    df = pd.read_csv(DF_PATH)
    with open(SCALER_PATH) as f:
        scaler_stats = json.load(f)
    df["genre_group"] = df["genre"].apply(_simplify_genre)
    df["demand_tier"] = pd.qcut(
        df["avg_tickets_sold"], q=3, labels=["Low", "Mid", "High"]
    )
    return df, scaler_stats


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


# =====================================================================
# 4. Helpers — original scoring logic
# =====================================================================
def to_real(scaled_val, col, scaler_stats):
    return scaled_val * scaler_stats[col]["std"] + scaler_stats[col]["mean"]


def to_scaled(real_val, col, scaler_stats):
    return (real_val - scaler_stats[col]["mean"]) / scaler_stats[col]["std"]


def score_grid(row, prices_z, caps_z, model, scaler_stats, elasticity=0.0):
    """
    Original Stage 2 scoring (from scoring_engine_part2.py).

    Predicts TICKETS at the median peer price (avoids extrapolation), then
    applies an elasticity multiplier for each candidate price:
        adjusted_tickets = pred_tickets × (1 + elasticity × pct_change_from_median)
    Capped at venue capacity. Revenue = candidate_price × tickets.

    With elasticity=0.0 the multiplier is always 1 (no adjustment).
    """
    prices_real = [to_real(p, "ticket_price_avg", scaler_stats) for p in prices_z]
    caps_real = [to_real(c, "avg_event_capacity", scaler_stats) for c in caps_z]
    median_idx = len(prices_z) // 2
    median_price_real = prices_real[median_idx]
    median_price_z = prices_z[median_idx]

    n_caps, n_prices = len(caps_z), len(prices_z)
    revenues = np.zeros((n_caps, n_prices))
    tickets_arr = np.zeros((n_caps, n_prices))
    fills = np.zeros((n_caps, n_prices))

    for i, cap_z in enumerate(caps_z):
        cap_real = caps_real[i]
        # Predict at the MEDIAN price (most reliable point) for this capacity
        e = row.copy()
        e["ticket_price_avg"] = median_price_z
        e["avg_event_capacity"] = cap_z
        pred_z = max(float(model.predict(pd.DataFrame([e]))[0]), 0)
        pred_tickets = max(to_real(pred_z, "avg_tickets_sold", scaler_stats), 0)

        for j, p_real in enumerate(prices_real):
            pct_change = (p_real - median_price_real) / median_price_real
            multiplier = max(1 + elasticity * pct_change, 0.1)
            adjusted = pred_tickets * multiplier
            tickets = min(adjusted, cap_real) if cap_real > 0 else adjusted
            tickets_arr[i, j] = tickets
            fills[i, j] = (tickets / cap_real * 100) if cap_real > 0 else 0
            revenues[i, j] = p_real * tickets
    return revenues, tickets_arr, fills, prices_real, caps_real


def make_heatmap(revenues, fills, prices_real, caps_real, best_idx):
    n = revenues.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(revenues, cmap="YlGn", aspect="auto",
              vmin=revenues.min() * 0.85, vmax=revenues.max() * 1.05)

    for i in range(n):
        for j in range(n):
            is_best = (i, j) == best_idx
            txt = "white" if revenues[i, j] > revenues.mean() else "#222"
            ax.text(j, i - 0.15, f"${revenues[i, j]:,.0f}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold" if is_best else "normal", color=txt)
            ax.text(j, i + 0.20, f"{fills[i, j]:.0f}% fill",
                    ha="center", va="center", fontsize=8, color=txt)

    ax.add_patch(patches.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=3, edgecolor="gold", facecolor="none"))

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"${p:,.0f}" for p in prices_real])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{c:,.0f} seats" for c in caps_real])
    ax.set_xlabel("Ticket Price")
    ax.set_ylabel("Venue Capacity")
    return fig


def find_peers(idx, data, min_peers=10):
    g = data.loc[idx, "genre_group"]
    t = data.loc[idx, "demand_tier"]
    peers = data[(data["genre_group"] == g) & (data["demand_tier"] == t)]
    if len(peers) < min_peers:
        peers = data[data["genre_group"] == g]
    return peers.drop(index=idx, errors="ignore")


# =====================================================================
# 4b. Custom artist row builder (tab 2)
# =====================================================================
ONEHOT_GROUPS = {
    "city_cleaned_": ["boston", "chicago", "new_york", "portland", "washington", "other"],
    "state_cleaned_": ["california", "florida", "massachusetts", "new_york", "texas", "other"],
    "market_cleaned_": ["boston_manchester", "chicago", "los_angeles", "new_york",
                        "washington_dc_hagerstown", "other"],
    "genre_cleaned_": ["country", "dance_electronic", "latin", "pop_rock", "rap_hiphop", "other"],
}


def build_custom_row(inputs, scaler_stats, model_features, df):
    """Construct a single feature row from user inputs for the original 49-feature model."""
    row = df[model_features].median().to_dict()

    # One-hot overrides
    for prefix, options in ONEHOT_GROUPS.items():
        for opt in options:
            col = prefix + opt
            if col in row:
                row[col] = 0
        chosen = inputs.get(prefix.rstrip("_").replace("_cleaned", ""))
        if chosen:
            target_col = prefix + chosen
            if target_col in row:
                row[target_col] = 1

    # Numeric features: convert real → z-scored
    raw_to_scaled = {
        "ticket_price_avg": inputs["price"],
        "avg_event_capacity": inputs["capacity"],
        "wiki_avg_views_30d": inputs["wiki_views"],
        "historical_concerts": inputs["hist_concerts"],
        "past_year_avg_tickets": inputs["past_year_tix"],
        "career_age": inputs["career_age"],
        "days_since_last_album": inputs["days_since_album"],
    }
    for col, val in raw_to_scaled.items():
        if col in scaler_stats:
            row[col] = to_scaled(val, col, scaler_stats)

    # Date-derived features (not z-scored)
    d = inputs["event_date"]
    row["year"] = d.year
    row["year_offset"] = d.year - 2020
    row["month_sin"] = np.sin(2 * np.pi * d.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * d.month / 12)
    dow = d.weekday()
    row["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    row["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    row["lockdown"] = int(date(2020, 3, 15) <= d <= date(2021, 7, 1))
    row["album_release_last_12m"] = int(inputs["days_since_album"] <= 365)

    for col in ["is_missing_support", "is_missing_genre", "is_missing_album_dates"]:
        if col in row:
            row[col] = 0

    # Look up census features for chosen market
    market_col = "market_cleaned_" + inputs["market"]
    if market_col in df.columns:
        market_rows = df[df[market_col] == 1]
        if len(market_rows) > 0:
            for col in ["market_population", "median_income", "population"]:
                if col in row:
                    row[col] = float(market_rows[col].median())

    return pd.Series(row)[model_features]


# =====================================================================
# 4c. Whole-dataset predictions (cached) for the Model Performance tab
# =====================================================================
@st.cache_data
def compute_predictions(_model, _df, _scaler_stats, _features):
    """Predict tickets for every event, derive revenue using actual price."""
    X = _df[list(_features)]
    pred_tix_z = _model.predict(X)

    tix_m = _scaler_stats["avg_tickets_sold"]["mean"]
    tix_s = _scaler_stats["avg_tickets_sold"]["std"]
    cap_m = _scaler_stats["avg_event_capacity"]["mean"]
    cap_s = _scaler_stats["avg_event_capacity"]["std"]
    price_m = _scaler_stats["ticket_price_avg"]["mean"]
    price_s = _scaler_stats["ticket_price_avg"]["std"]

    pred_tix = np.clip(pred_tix_z * tix_s + tix_m, 0, None)
    actual_tix = _df["avg_tickets_sold"].values * tix_s + tix_m
    actual_cap = _df["avg_event_capacity"].values * cap_s + cap_m
    actual_price = _df["ticket_price_avg"].values * price_s + price_m

    # Cap predicted tickets at actual capacity (model can over-predict)
    pred_tix_capped = np.minimum(pred_tix, actual_cap)
    pred_rev = actual_price * pred_tix_capped
    actual_rev = actual_price * actual_tix

    # Derived fill rates for context
    actual_fill = (actual_tix / actual_cap) * 100
    pred_fill = (pred_tix_capped / actual_cap) * 100

    return pd.DataFrame({
        "eventid": _df["eventid"].values,
        "headliner": _df["headliner"].values,
        "genre_group": _df["genre_group"].values,
        "demand_tier": _df["demand_tier"].values,
        "actual_tickets": actual_tix,
        "pred_tickets": pred_tix_capped,
        "actual_fill": actual_fill,
        "pred_fill": pred_fill,
        "actual_revenue": actual_rev,
        "pred_revenue": pred_rev,
        "actual_price": actual_price,
        "actual_capacity": actual_cap,
    })


def make_pred_vs_actual(ax, actual, pred, title, unit="", log=False):
    ax.scatter(actual, pred, alpha=0.25, s=12, color="#2D8B4E", edgecolor="none")
    lo = max(min(actual.min(), pred.min()), 1) if log else min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, alpha=0.7, label="perfect")
    ax.set_xlabel(f"Actual {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.set_title(title)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    return ax


def make_distribution_with_marker(values, marker, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(values, bins=30, alpha=0.75, color="#4A6FA5", edgecolor="white")
    ax.axvline(marker, color="red", linewidth=2.5,
               label=f"This event: {marker:,.0f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# events")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return fig


# =====================================================================
# 5. Load data + model
# =====================================================================
df, scaler_stats = load_data()
model = load_model()
model_features = model.get_booster().feature_names


# =====================================================================
# 6. UI
# =====================================================================
st.title("Concert Revenue Predictor")
st.caption(
    "Stage 1 ticket-demand model + Stage 2 grid scoring engine "
    "(original artifacts only)."
)

tab1, tab2, tab3 = st.tabs(["Existing event", "Custom artist", "Model performance"])


# ---- Sidebar (shared by tab 1; elasticity also used by tab 2) ----
with st.sidebar:
    st.header("Event (tab 1)")
    artists = sorted(df["headliner"].dropna().unique())
    default_artist = "Lianne La Havas" if "Lianne La Havas" in artists else artists[0]
    artist = st.selectbox("Headliner", artists, index=artists.index(default_artist))
    artist_events = df[df["headliner"] == artist].sort_values("event_date")
    event_labels = [
        f"{r.event_date}  —  {r.venue} ({r.market})"
        for r in artist_events.itertuples()
    ]
    label_to_idx = dict(zip(event_labels, artist_events.index))
    chosen_label = st.selectbox("Show", event_labels)
    event_idx = label_to_idx[chosen_label]

    st.divider()
    st.header("Scoring params")
    elasticity = st.slider(
        "Price elasticity", -2.0, 0.5, -0.5, 0.1,
        help="0 = use model's direct ticket prediction. Negative = demand falls "
             "as price rises (concert literature: -0.3 to -0.7).",
    )
    grid_size = st.radio("Grid size", ["3x3", "5x5"], horizontal=True)


# =====================================================================
# TAB 1 — existing event
# =====================================================================
with tab1:
    target = df.loc[event_idx]
    peers = find_peers(event_idx, df)

    qs = [0.25, 0.50, 0.75] if grid_size == "3x3" else [0.10, 0.30, 0.50, 0.70, 0.90]
    prices_z = peers["ticket_price_avg"].quantile(qs).values
    caps_z = peers["avg_event_capacity"].quantile(qs).values

    revenues, tickets_arr, fills, prices_real, caps_real = score_grid(
        target[model_features], prices_z, caps_z, model, scaler_stats,
        elasticity=elasticity,
    )
    best_idx = np.unravel_index(np.argmax(revenues), revenues.shape)

    actual_price = to_real(target["ticket_price_avg"], "ticket_price_avg", scaler_stats)
    actual_tix = to_real(target["avg_tickets_sold"], "avg_tickets_sold", scaler_stats)
    actual_cap = to_real(target["avg_event_capacity"], "avg_event_capacity", scaler_stats)
    actual_rev = actual_price * actual_tix
    rec_rev = revenues[best_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual revenue", f"${actual_rev:,.0f}",
              help=f"At ${actual_price:.0f} × {actual_tix:,.0f} tickets")
    c2.metric("Recommended revenue", f"${rec_rev:,.0f}")
    delta_pct = (rec_rev - actual_rev) / actual_rev * 100 if actual_rev else 0
    c3.metric("Uplift vs actual", f"{delta_pct:+.1f}%")

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Revenue grid  ({grid_size})")
        st.pyplot(make_heatmap(revenues, fills, prices_real, caps_real, best_idx))

    with right:
        st.subheader("Event")
        st.write(f"**{target['headliner']}**")
        st.write(f"Venue: {target['venue']}")
        st.write(f"Market: {target['market']}")
        st.write(f"Date: {target['event_date']}")
        st.write(f"Genre: {target['genre']}")
        st.write(f"Capacity: {actual_cap:,.0f} seats")

        st.divider()
        st.subheader("Recommendation")
        rec_price = prices_real[best_idx[1]]
        rec_cap = caps_real[best_idx[0]]
        rec_tix = tickets_arr[best_idx]
        rec_fill = fills[best_idx]
        st.write(f"Price: **${rec_price:,.0f}**")
        st.write(f"Venue capacity: **{rec_cap:,.0f} seats**")
        st.write(f"Predicted tickets: **{rec_tix:,.0f}**")
        st.write(f"Predicted fill: **{rec_fill:.0f}%**")

    st.divider()
    st.subheader("Where this event sits among its peers")
    pc1, pc2 = st.columns(2)
    with pc1:
        peer_tickets = peers["avg_tickets_sold"].apply(
            lambda z: to_real(z, "avg_tickets_sold", scaler_stats)
        ).values
        st.pyplot(make_distribution_with_marker(
            peer_tickets, actual_tix,
            f"Peer tickets sold  (n={len(peer_tickets)})",
            "Tickets sold",
        ))
    with pc2:
        peer_prices = peers["ticket_price_avg"].apply(
            lambda z: to_real(z, "ticket_price_avg", scaler_stats)
        ).values
        st.pyplot(make_distribution_with_marker(
            peer_prices, actual_price,
            f"Peer ticket prices  (n={len(peer_prices)})",
            "Ticket price ($)",
        ))

    st.divider()
    st.subheader("How the recommended price tier shifts with elasticity")
    sweep_rows = []
    for e in [0.0, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0]:
        rev_s, _, _, prices_s, _ = score_grid(
            target[model_features], prices_z, caps_z, model, scaler_stats,
            elasticity=e,
        )
        b = np.unravel_index(np.argmax(rev_s), rev_s.shape)
        sweep_rows.append({
            "elasticity": e,
            "best_price": f"${prices_s[b[1]]:,.0f}",
            "best_revenue": f"${rev_s[b]:,.0f}",
        })
    st.dataframe(pd.DataFrame(sweep_rows), hide_index=True, use_container_width=True)


# =====================================================================
# TAB 2 — custom artist input form
# =====================================================================
with tab2:
    st.subheader("Predict revenue for a hypothetical artist/event")
    st.caption(
        "Fill in the form below. Google Trends and other unfilled signals "
        "default to dataset medians."
    )

    with st.form("custom_predict"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Artist**")
            artist_name = st.text_input("Name (display only)", "")
            genre = st.selectbox(
                "Genre",
                ["pop_rock", "country", "latin", "rap_hiphop", "dance_electronic", "other"],
            )
            career_age = st.number_input("Career age (years)", 0, 60, 10)
            hist_concerts = st.number_input("Total historical concerts", 0, 5000, 50)
            wiki_views = st.number_input("Avg daily Wikipedia views (30d)", 0, 200000, 500)
            past_year_tix = st.number_input("Past year avg tickets per show", 0, 100000, 2000)
            days_since_album = st.number_input("Days since last album", 0, 5000, 365)

        with c2:
            st.markdown("**Event**")
            event_date_in = st.date_input("Event date", date(2025, 6, 15))
            market = st.selectbox(
                "Market",
                ["new_york", "chicago", "los_angeles", "boston_manchester",
                 "washington_dc_hagerstown", "other"],
            )
            city = st.selectbox(
                "City",
                ["new_york", "chicago", "boston", "portland", "washington", "other"],
            )
            state = st.selectbox(
                "State",
                ["california", "florida", "massachusetts", "new_york", "texas", "other"],
            )
            capacity = st.number_input("Venue capacity (seats)", 50, 100000, 2000)

        with c3:
            st.markdown("**Pricing**")
            price = st.number_input("Ticket price ($)", 5, 1000, 50)
            st.markdown("---")
            st.markdown("**Output**")
            sweep_caps = st.checkbox("Also sweep across 3 venue sizes", value=True)
            st.caption(
                f"Elasticity = {elasticity} (from sidebar). Only matters for the "
                f"sweep below; the single-point prediction is the raw model output."
            )

        submitted = st.form_submit_button("Predict revenue", type="primary")

    if submitted:
        inputs = {
            "genre": genre, "city": city, "state": state, "market": market,
            "career_age": career_age, "hist_concerts": hist_concerts,
            "wiki_views": wiki_views, "past_year_tix": past_year_tix,
            "days_since_album": days_since_album,
            "event_date": event_date_in,
            "capacity": capacity, "price": price,
        }

        row = build_custom_row(inputs, scaler_stats, model_features, df)

        # Single-point prediction
        pred_z = max(float(model.predict(pd.DataFrame([row]))[0]), 0)
        pred_tickets = max(to_real(pred_z, "avg_tickets_sold", scaler_stats), 0)
        tickets = min(pred_tickets, capacity)
        revenue = price * tickets
        fill = tickets / capacity * 100 if capacity > 0 else 0

        st.divider()
        st.subheader(f"Prediction{' for ' + artist_name if artist_name else ''}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted tickets", f"{tickets:,.0f}")
        m2.metric("Implied fill rate", f"{fill:.0f}%")
        m3.metric("Predicted revenue", f"${revenue:,.0f}")
        m4.metric("Capacity", f"{capacity:,.0f}")

        # Context histograms
        st.divider()
        st.subheader("Context: how this prediction compares to the dataset")
        all_preds = compute_predictions(model, df, scaler_stats, tuple(model_features))
        ctx1, ctx2 = st.columns(2)
        with ctx1:
            st.pyplot(make_distribution_with_marker(
                all_preds["pred_tickets"].values, tickets,
                "Predicted tickets vs all 1808 events",
                "Predicted tickets",
            ))
        with ctx2:
            st.pyplot(make_distribution_with_marker(
                all_preds["pred_revenue"].values, revenue,
                "Predicted revenue vs all 1808 events",
                "Predicted revenue ($)",
            ))

        if sweep_caps:
            st.divider()
            st.subheader("Revenue grid: 3 prices × 3 venue sizes")
            prices_real_grid = np.array([price * 0.7, price, price * 1.4])
            caps_real_grid = np.array([capacity * 0.6, capacity, capacity * 1.5])
            prices_z_grid = np.array(
                [to_scaled(p, "ticket_price_avg", scaler_stats) for p in prices_real_grid]
            )
            caps_z_grid = np.array(
                [to_scaled(c, "avg_event_capacity", scaler_stats) for c in caps_real_grid]
            )

            revs, _, fls, pr, cr = score_grid(
                row, prices_z_grid, caps_z_grid, model, scaler_stats,
                elasticity=elasticity,
            )
            best = np.unravel_index(np.argmax(revs), revs.shape)

            colL, colR = st.columns([2, 1])
            with colL:
                st.pyplot(make_heatmap(revs, fls, pr, cr, best))
            with colR:
                st.markdown("**Recommendation**")
                st.write(f"Price: **${pr[best[1]]:,.0f}**")
                st.write(f"Capacity: **{cr[best[0]]:,.0f} seats**")
                st.write(f"Fill: **{fls[best]:.0f}%**")
                st.write(f"Revenue: **${revs[best]:,.0f}**")


# =====================================================================
# TAB 3 — model performance
# =====================================================================
with tab3:
    st.subheader("How the model performs across all 1808 events")
    st.caption(
        "Predictions made on each event using its actual features. Tickets are "
        "capped at actual capacity; revenue = predicted_tickets × actual_price."
    )

    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score
    )
    preds = compute_predictions(model, df, scaler_stats, tuple(model_features))

    tix_r2 = r2_score(preds["actual_tickets"], preds["pred_tickets"])
    tix_mae = mean_absolute_error(preds["actual_tickets"], preds["pred_tickets"])
    tix_rmse = float(np.sqrt(mean_squared_error(preds["actual_tickets"], preds["pred_tickets"])))
    rev_r2 = r2_score(preds["actual_revenue"], preds["pred_revenue"])
    rev_mae = mean_absolute_error(preds["actual_revenue"], preds["pred_revenue"])
    rev_mape = float(np.mean(
        np.abs(preds["pred_revenue"] - preds["actual_revenue"])
        / preds["actual_revenue"].clip(lower=1)
    ) * 100)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tickets R²", f"{tix_r2:.3f}")
    m2.metric("Tickets MAE", f"{tix_mae:,.0f}")
    m3.metric("Tickets RMSE", f"{tix_rmse:,.0f}")
    m4.metric("Revenue R²", f"{rev_r2:.3f}")
    m5.metric("Revenue MAE", f"${rev_mae:,.0f}")
    m6.metric("Revenue MAPE", f"{rev_mape:.0f}%")

    # Pred vs actual scatters
    st.divider()
    st.subheader("Predicted vs actual")
    pc1, pc2 = st.columns(2)
    with pc1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ok = (preds["actual_tickets"] > 0) & (preds["pred_tickets"] > 0)
        make_pred_vs_actual(ax, preds.loc[ok, "actual_tickets"],
                            preds.loc[ok, "pred_tickets"],
                            "Tickets (log scale)", unit="tickets", log=True)
        st.pyplot(fig)
    with pc2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ok = (preds["actual_revenue"] > 0) & (preds["pred_revenue"] > 0)
        make_pred_vs_actual(ax, preds.loc[ok, "actual_revenue"],
                            preds.loc[ok, "pred_revenue"],
                            "Revenue (log scale)", unit="($)", log=True)
        st.pyplot(fig)

    # Residual analysis
    st.divider()
    st.subheader("Residual analysis")
    rc1, rc2 = st.columns(2)
    residuals = preds["pred_tickets"] - preds["actual_tickets"]
    with rc1:
        fig, ax = plt.subplots(figsize=(6, 4))
        # clip extreme residuals for plotting only
        clipped = residuals.clip(residuals.quantile(0.01), residuals.quantile(0.99))
        ax.hist(clipped, bins=50, color="#4A6FA5", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(residuals.mean(), color="orange", linewidth=1.5,
                   label=f"mean: {residuals.mean():+.0f}")
        ax.set_xlabel("Predicted − Actual tickets (clipped 1-99 %ile)")
        ax.set_ylabel("# events")
        ax.set_title("Residual distribution")
        ax.legend()
        st.pyplot(fig)
    with rc2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(preds["pred_tickets"], residuals, alpha=0.25, s=12,
                   color="#2D8B4E", edgecolor="none")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted tickets")
        ax.set_ylabel("Residual (pred − actual)")
        ax.set_xscale("log")
        ax.set_title("Residuals vs predicted")
        st.pyplot(fig)

    # Performance by genre
    st.divider()
    st.subheader("Performance by genre")
    rows = []
    for g, sub in preds.groupby("genre_group"):
        if len(sub) < 10:
            continue
        rows.append({
            "genre": g,
            "n_events": len(sub),
            "tickets_R²": r2_score(sub["actual_tickets"], sub["pred_tickets"]),
            "tickets_MAE": mean_absolute_error(sub["actual_tickets"], sub["pred_tickets"]),
            "mean_actual_tix": sub["actual_tickets"].mean(),
            "mean_pred_tix": sub["pred_tickets"].mean(),
        })
    gp = pd.DataFrame(rows).sort_values("n_events", ascending=False)

    gc1, gc2 = st.columns(2)
    with gc1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(gp["genre"], gp["tickets_R²"], color="#2D8B4E")
        ax.set_xlabel("R²")
        ax.set_title("Tickets R² by genre")
        ax.axvline(tix_r2, color="orange", linestyle="--",
                   label=f"overall: {tix_r2:.2f}")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    with gc2:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(gp))
        w = 0.4
        ax.bar(x - w/2, gp["mean_actual_tix"], w, label="Actual", color="#4A6FA5")
        ax.bar(x + w/2, gp["mean_pred_tix"], w, label="Predicted", color="#E67E22")
        ax.set_xticks(x)
        ax.set_xticklabels(gp["genre"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean tickets")
        ax.set_title("Average tickets by genre")
        ax.legend()
        st.pyplot(fig)
    st.dataframe(gp.round(2), hide_index=True, use_container_width=True)

    # Performance by demand tier
    st.divider()
    st.subheader("Performance by demand tier")
    tier_rows = []
    for t in ["Low", "Mid", "High"]:
        sub = preds[preds["demand_tier"] == t]
        if len(sub) == 0:
            continue
        tier_rows.append({
            "tier": t,
            "n_events": len(sub),
            "tickets_R²": r2_score(sub["actual_tickets"], sub["pred_tickets"]),
            "tickets_MAE": mean_absolute_error(sub["actual_tickets"], sub["pred_tickets"]),
            "mean_actual_tix": sub["actual_tickets"].mean(),
            "mean_pred_tix": sub["pred_tickets"].mean(),
        })
    st.dataframe(pd.DataFrame(tier_rows).round(2), hide_index=True, use_container_width=True)

    # Feature importance
    st.divider()
    st.subheader("Top 20 feature importances")
    imp = pd.Series(model.feature_importances_, index=model_features).sort_values()
    top = imp.tail(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top.values, color="#4A6FA5")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 20 features")
    st.pyplot(fig)

    # Worst predictions
    st.divider()
    st.subheader("Where the model misses most")
    st.caption("The 10 events with the largest ticket prediction errors.")
    preds["abs_err"] = (preds["pred_tickets"] - preds["actual_tickets"]).abs()
    worst = preds.nlargest(10, "abs_err")[
        ["headliner", "genre_group", "actual_tickets", "pred_tickets",
         "actual_revenue", "pred_revenue"]
    ].round(0)
    st.dataframe(worst, hide_index=True, use_container_width=True)
