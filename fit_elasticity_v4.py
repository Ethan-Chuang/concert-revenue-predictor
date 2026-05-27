"""
Fit per-genre price elasticity from within-artist variation.

The idea: instead of comparing different artists (where popularity confounds
price), look only at the SAME artist appearing at different prices across
events. The slope of (fill rate residual) on (price residual), after
subtracting each artist's own mean, gives elasticity free of artist-quality
confounding.

Math:
    log(fill_ia)  = α_a + β · log(price_ia) + ε_ia
    α_a = artist fixed effect (cancels out via demeaning)

    After demeaning by artist:
        (log_fill_ia - mean_a(log_fill)) = β · (log_price_ia - mean_a(log_price)) + ε

OLS through origin on the demeaned variables → β = elasticity.

Output: /Users/ethanc/RnA/data/genre_elasticities_v4.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/ethanc/RnA/data")
RAW_PATH = DATA_DIR / "testing_sample_raw.csv"
OUT_PATH = DATA_DIR / "genre_elasticities_v4.json"

MIN_EVENTS_PER_ARTIST = 3  # artist needs this many events with price variation
MIN_EVENTS_PER_GENRE = 30  # genre needs this many events after filtering


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


# Load & basic filtering
raw = pd.read_csv(RAW_PATH, index_col=0)
raw = raw.drop_duplicates(subset="eventid", keep="first")
print(f"Raw rows (deduped): {len(raw)}")

raw["fill_rate"] = raw["avg_capacity_sold"] / 100.0
raw = raw[(raw["fill_rate"] > 0.01) & (raw["fill_rate"] <= 1.5)]
raw = raw[raw["ticket_price_avg"] > 0]
print(f"After filtering fill/price: {len(raw)}")

raw["log_price"] = np.log(raw["ticket_price_avg"])
raw["log_fill"] = np.log(raw["fill_rate"])
raw["genre_group"] = raw["genre"].apply(simplify_genre)


def fit_within_artist(sub):
    """Demean by artist and run OLS through origin on demeaned values."""
    sub = sub.copy()
    artist_means = sub.groupby("headliner")[["log_price", "log_fill"]].transform("mean")
    sub["lp_dm"] = sub["log_price"] - artist_means["log_price"]
    sub["lf_dm"] = sub["log_fill"] - artist_means["log_fill"]

    # Drop rows where the artist had no price variation (demeaned price ≈ 0)
    valid = sub[sub["lp_dm"].abs() > 1e-6]
    if len(valid) < MIN_EVENTS_PER_GENRE:
        return None

    x = valid["lp_dm"].values
    y = valid["lf_dm"].values
    denom = float(x @ x)
    if denom == 0:
        return None
    beta = float(x @ y) / denom

    # R² for the regression through origin
    y_hat = beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "beta": beta,
        "r2_through_origin": r2,
        "n_events": int(len(valid)),
        "n_artists": int(sub["headliner"].nunique()),
    }


# Overall elasticity across all genres
all_artists = raw.groupby("headliner").size()
multi_artists = all_artists[all_artists >= MIN_EVENTS_PER_ARTIST].index
overall_sub = raw[raw["headliner"].isin(multi_artists)]
overall = fit_within_artist(overall_sub)
print(f"\n=== Overall within-artist elasticity ===")
print(f"  β = {overall['beta']:+.3f}")
print(f"  n_events = {overall['n_events']}, n_artists = {overall['n_artists']}")
print(f"  R² (through origin) = {overall['r2_through_origin']:.4f}")

# Per-genre
print("\n=== Genre-specific elasticities ===")
genre_results = {}
for genre, group in raw.groupby("genre_group"):
    artist_counts = group.groupby("headliner").size()
    multi = artist_counts[artist_counts >= MIN_EVENTS_PER_ARTIST].index
    sub = group[group["headliner"].isin(multi)]
    if len(sub) < MIN_EVENTS_PER_GENRE:
        continue
    res = fit_within_artist(sub)
    if res is None:
        continue
    genre_results[genre] = res
    print(f"  {genre:<22} β={res['beta']:+.3f}  "
          f"n_events={res['n_events']:>4}  n_artists={res['n_artists']:>3}  "
          f"R²={res['r2_through_origin']:.3f}")

# Save
output = {
    "overall": overall,
    "by_genre": genre_results,
    "min_events_per_artist": MIN_EVENTS_PER_ARTIST,
    "min_events_per_genre": MIN_EVENTS_PER_GENRE,
    "method": "OLS within-artist demeaning, log-log",
}
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
