"""
Retrain Stage 1 with FILL RATE (avg_capacity_sold) as the target instead of
raw ticket count.

This decouples artist popularity from venue size: the model now predicts
"what % of a venue would this artist fill" rather than "how many tickets in
this specific venue." Revenue is computed downstream as
  revenue = candidate_price × (predicted_fill_rate × candidate_capacity)

Features stay the same as v2 (51 total) — including avg_event_capacity, so
the model can still learn patterns like "small artists fill big venues less."

Output: /Users/ethanc/RnA/data/stage1_fill_rate_model_v3.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path("/Users/ethanc/RnA/data")
CLEANED_PATH = DATA_DIR / "test_sample_cleaned_apr8_v2.csv"
V2_MODEL_PATH = DATA_DIR / "stage1_ticket_sales_model_v2.json"
OUT_MODEL_PATH = DATA_DIR / "stage1_fill_rate_model_v3.json"

TARGET = "avg_capacity_sold"  # fill rate, z-scored in the cleaned dataset
NEW_FEATURES = ["baseline_ticket_price", "log_price_premium"]

# Load data
df = pd.read_csv(CLEANED_PATH)
print(f"Loaded: {df.shape}")

# Use the same feature list as v2
v2_model = XGBRegressor()
v2_model.load_model(V2_MODEL_PATH)
features = v2_model.get_booster().feature_names

missing = [f for f in features if f not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")
print(f"Training with {len(features)} features (same as v2)")

X = df[features].copy()
y = df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.7,
    reg_lambda=5.0,
    random_state=42,
    min_child_weight=5,
)
model.fit(X_train, y_train)

print("\n=== v3 model performance (target=fill rate) ===")
for name, X_, y_ in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
    pred = model.predict(X_)
    print(
        f"{name} -> MAE: {mean_absolute_error(y_, pred):.3f}  "
        f"RMSE: {np.sqrt(mean_squared_error(y_, pred)):.3f}  "
        f"R^2: {r2_score(y_, pred):.4f}"
    )

imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n=== Top 15 features ===")
print(imp.head(15).to_string())

print("\n=== New features rank ===")
for f in NEW_FEATURES:
    rank = list(imp.index).index(f) + 1
    print(f"  {f}: rank {rank} / {len(imp)}  (importance={imp[f]:.4f})")

model.save_model(OUT_MODEL_PATH)
print(f"\nSaved model: {OUT_MODEL_PATH}")
