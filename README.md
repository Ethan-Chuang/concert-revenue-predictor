# Concert Revenue Predictor — A Round Entertainment

Streamlit dashboard wrapping the two-stage concert revenue model:
**Stage 1** predicts ticket demand from artist + market signals,
**Stage 2** scores price × venue combinations to recommend the
revenue-maximizing combo.

## Run locally

```bash
git clone <this-repo-url>
cd <repo-folder>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run model/dashboard.py
```

The dashboard opens at <http://localhost:8501>.

## Repo layout

```
data/
  test_sample_cleaned_apr8.csv      cleaned dataset (1808 events × 49 features)
  scaler_stats_apr8.json            z-score mean/std for inverse-transforms
  stage1_ticket_sales_model.json    trained XGBoost model
model/
  dashboard.py                      Streamlit app (the entry point)
  two_stage_model.py                model training script
  scoring_engine_part2.py           Stage 2 grid scoring engine
  ...
requirements.txt
.gitignore
```

The three files under `data/` are the only inputs the dashboard needs.

## Dashboard tabs

1. **Existing event** — pick an artist + show, see the 3×3 (or 5×5)
   revenue grid with a recommendation overlay.
2. **Custom artist** — form for entering a hypothetical artist + event
   and getting a prediction.
3. **Model performance** — predicted vs actual scatter, residuals,
   feature importance, per-genre + per-tier breakdowns.

Sidebar controls (elasticity, grid size) apply across tabs.

## Data note

The cleaned dataset is derived from Pollstar and enriched with Google
Trends, Wikipedia, MusicBrainz, and US Census features. This repo is
**private** — do not make it public without confirming the data licensing
with the team. Large raw files (`pollstar_post_2020.csv`,
`pollstar_combined_dataset.csv`, `pollstar_feature_engineered.csv`)
are gitignored.
