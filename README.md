# 🚗 NYC Taxi Demand Forecasting

An end-to-end demand forecasting pipeline for NYC Yellow Taxi trip data, with borough-level hourly forecasting, MLflow experiment tracking, and an interactive Streamlit dashboard.

The project ingests raw NYC TLC trip data, engineers time-based and lag features, trains and compares multiple forecasting models (XGBoost, LightGBM, CatBoost, ARIMA, SARIMA), and serves the results through a multi-page Streamlit app for exploration and forecasting.

## Features

- **Data preprocessing & sampling** — cleans raw trip data, maps zones to boroughs, buckets trip distances, and aggregates trips into hourly borough-level demand counts
- **Feature engineering** — holiday flags, day-of-week/weekend/hour features, and lag features (`Lag1` through `Lag336`) for short- and long-horizon patterns
- **Model training & comparison**
  - Gradient boosting: XGBoost, LightGBM, CatBoost (with Optuna-tuned XGBoost params via `config.yaml`)
  - Statistical models: ARIMA and SARIMA via grid search over orders/seasonal orders
  - Best model per borough selected on RMSE and persisted with `joblib`
- **Experiment tracking** — all runs, metrics (MAE, MSE, RMSE, R², MAPE), parameters, feature importances, and diagnostic plots logged to MLflow (SQLite backend)
- **Error analysis** — residual distributions, actual-vs-predicted plots, and largest-error diagnostics logged as artifacts
- **Interactive dashboard** (Streamlit, multi-page)
  - **Home** — landing page and navigation
  - **Data Insights** — ride volume by vendor, hourly/weekly/monthly trends, weekday vs. weekend demand, congestion surcharge patterns, pickup hotspots, choropleth demand map, trip distance distribution
  - **Forecasting** — pick a borough, model, and horizon (next hour up to next month) to generate live forecasts, view demand heatmaps, and inspect model performance metrics

## Project Structure

```
.
├── Home.py                                # Streamlit app entry point
├── pages/
│   ├── Data Insights.py                   # Exploratory data visualizations
│   └── Forecasting.py                     # Interactive forecasting UI
├── stratified_sampling_.py                # Raw parquet -> hourly borough-level CSV
├── Gradient_Boost_model_training.py       # XGBoost/LightGBM/CatBoost training + MLflow
├── New_Gradient_Boost_model_training.py   # Extended version w/ residual & error-analysis plots
├── stats_model_training.py                # ARIMA/SARIMA grid search + MLflow
├── config.yaml                            # Tuned hyperparameters (e.g. Optuna XGBoost params)
├── Data/                                  # Raw parquet files, sampled/combined CSVs, lookups
├── models/                                # Persisted best models per borough (.pkl)
├── artifacts/                             # Feature importances, error metrics, residuals, plots
├── logs/                                  # Per-script log files
└── mlflow.db                              # MLflow tracking store (SQLite)
```

> Note: `Home.py`, `Data Insights.py`, and `Forecasting.py` are Streamlit multi-page app files — Streamlit expects the sub-pages inside a `pages/` directory alongside the entry script.

## Tech Stack

- **Modeling:** scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels (ARIMA/SARIMAX)
- **Experiment tracking:** MLflow (SQLite backend)
- **Dashboard:** Streamlit, Plotly, DuckDB
- **Data processing:** pandas, NumPy, `holidays`
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Other:** PyYAML, joblib, pathlib

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> A `requirements.txt` isn't included yet — you can generate one with `pip freeze > requirements.txt` once your environment is set up, or let me know and I'll draft one from the imports used across these scripts.

### Data

Place raw NYC TLC Yellow Taxi trip parquet files (named `yellow_tripdata_*.parquet`) along with `location_ids.csv` and `taxi_zone_lookup.csv` inside a `Data/` directory at the project root. Data can be downloaded from the [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) page.

## Usage

**1. Preprocess and sample raw data**
```bash
python stratified_sampling_.py
```
Produces `Data/combined_sampled_data.csv` — hourly trip counts by borough.

**2. Train gradient boosting models**
```bash
python Gradient_Boost_model_training.py
# or the extended version with error-analysis plots
python New_Gradient_Boost_model_training.py
```

**3. Train ARIMA/SARIMA models**
```bash
python stats_model_training.py
```

**4. View experiment results in MLflow**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**5. Launch the dashboard**
```bash
streamlit run Home.py
```

## Results

_Add key metrics here once finalized, e.g. best-performing model per borough and its RMSE/R²._

## Roadmap / Ideas

- [ ] Add `requirements.txt` / `environment.yml`
- [ ] Add Route Recommendation page (referenced in `Home.py` but not yet implemented)
- [ ] Add screenshots of the dashboard
- [ ] Containerize with Docker for reproducible deployment

## Author

Developed by Shafeeq