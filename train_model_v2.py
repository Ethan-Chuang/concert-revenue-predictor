"""
Retrain the Stage 1 XGBoost model with two new features:
  - baseline_ticket_price
  - log_price_premium

The feature set matches the deployed model + these two additions, so the
scoring engine can drop-in load the new model.

Output: /Users/ethanc/RnA/data/stage1_ticket_sales_model_v2.json
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path("/Users/ethanc/RnA/data")
CLEANED_PATH = DATA_DIR / "test_sample_cleaned_apr8_v2.csv"
DEPLOYED_MODEL_PATH = DATA_DIR / "stage1_ticket_sales_model.json"
OUT_MODEL_PATH = DATA_DIR / "stage1_ticket_sales_model_v2.json"

NEW_FEATURES = ["baseline_ticket_price", "log_price_premium"]
TARGET = "avg_tickets_sold"

# 1. Load data
df = pd.read_csv(CLEANED_PATH)
print(f"Loaded: {df.shape}")

# 2. Get the deployed model's feature list, then add the two new features
deployed = XGBRegressor()
deployed.load_model(DEPLOYED_MODEL_PATH)
v1_features = deployed.get_booster().feature_names
v2_features = v1_features + NEW_FEATURES

missing = [f for f in v2_features if f not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in cleaned dataset: {missing}")

print(f"Training with {len(v2_features)} features ({len(v1_features)} original + {len(NEW_FEATURES)} new)")

# 3. Train/test split
X = df[v2_features].copy()
y = df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train XGBoost with the same hyperparams as Two_stage_model.ipynb
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

# 5. Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n=== v2 model performance ===")
print(
    f"Train -> MAE: {mean_absolute_error(y_train, y_train_pred):.3f}  "
    f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.3f}  "
    f"R^2: {r2_score(y_train, y_train_pred):.4f}"
)
print(
    f"Test  -> MAE: {mean_absolute_error(y_test, y_test_pred):.3f}  "
    f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}  "
    f"R^2: {r2_score(y_test, y_test_pred):.4f}"
)

# 6. Feature importance — make sure the new features are actually being used
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n=== Top 15 features (importance) ===")
print(imp.head(15).to_string())
print("\n=== New features rank ===")
for f in NEW_FEATURES:
    rank = list(imp.index).index(f) + 1
    print(f"  {f}: rank {rank} / {len(imp)}  (importance={imp[f]:.4f})")

# 7. Save
model.save_model(OUT_MODEL_PATH)
print(f"\nSaved model: {OUT_MODEL_PATH}")
