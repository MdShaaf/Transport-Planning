# 🚕 NYC Taxi Demand Forecasting

An end-to-end **time-series forecasting and analytics platform** for predicting hourly NYC Yellow Taxi demand at the borough level.

The project transforms raw NYC TLC trip records into an hourly demand dataset, engineers temporal and lag-based features, compares machine-learning and statistical forecasting approaches, tracks experiments with MLflow, and exposes the results through an interactive Streamlit dashboard.

The objective is to provide a practical forecasting system that can help identify **when and where taxi demand is likely to increase**, supporting better vehicle allocation and operational planning.

---

## 🎯 Business Problem

Taxi demand varies significantly by:

- Hour of the day
- Day of the week
- Weekday vs. weekend
- Holidays
- Recent historical demand
- Borough/location

A forecasting system can help operators and drivers anticipate demand peaks and make better decisions about **where and when vehicle capacity is likely to be required**.

This project addresses the problem as a **borough-level hourly demand forecasting task**.

### Forecasting Target

> **Number of taxi trips per borough per hour**

The raw trip-level data is transformed into an aggregated time series:

```text
Borough × Hour → Taxi Trip Demand
```

---

## 🧠 Solution Overview

```text
NYC TLC Yellow Taxi Data
          │
          ▼
   Data Cleaning
          │
          ▼
 Zone → Borough Mapping
          │
          ▼
 Hourly Demand Aggregation
          │
          ▼
   Feature Engineering
          │
          ├───────────────┐
          ▼               ▼
  Machine Learning   Statistical Models
  XGBoost             ARIMA
  LightGBM            SARIMA
  CatBoost
          │               │
          └───────┬───────┘
                  ▼
        Model Evaluation
                  │
                  ▼
          MLflow Tracking
                  │
                  ▼
       Best Model Selection
                  │
                  ▼
       Persisted Model (.pkl)
                  │
                  ▼
       Streamlit Dashboard
```

---

## 📊 Data

The project uses **NYC TLC Yellow Taxi Trip Record Data**.

Raw trip records are processed to create an hourly borough-level forecasting dataset.

### Data processing

1. Load raw Parquet trip records
2. Clean and standardize trip information
3. Map taxi zones to boroughs
4. Process trip distance and related fields
5. Aggregate trips into hourly demand
6. Combine the processed data
7. Generate time-series forecasting features

### Final forecasting granularity

```text
Borough + Timestamp
        ↓
Hourly Taxi Demand
```

---

## ⚙️ Feature Engineering

The forecasting models use temporal and historical demand information.

### Time-based features

- Hour of day
- Day of week
- Month
- Weekend indicator
- Holiday indicator

### Lag features

Historical demand is incorporated through lag variables:

```text
Lag1     → Previous hour
Lag2     → Two hours earlier
Lag3     → Three hours earlier
Lag6     → Six hours earlier
Lag12    → Twelve hours earlier
Lag24    → Same hour previous day
Lag48    → Two days earlier
Lag72    → Three days earlier
Lag168   → Same hour previous week
Lag336   → Same hour two weeks earlier
```

These features allow the models to capture short-term demand persistence as well as daily and weekly seasonality.

---

## 🤖 Modeling Approach

Two forecasting approaches were evaluated.

### 1. Machine Learning

Three gradient-boosting algorithms were compared:

- **XGBoost** (with Optuna-tuned hyperparameters stored in `config.yaml`)
- **LightGBM**
- **CatBoost**

### 2. Statistical Forecasting

Traditional time-series models were also evaluated:

- **ARIMA**
- **SARIMA**

This provides a comparison between feature-based machine learning and classical time-series methods.

---

## 🧪 Model Evaluation

Models are evaluated using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination
- **MAPE** (percentage error) where available

RMSE is used as the primary selection metric because larger errors during demand peaks are operationally more costly.

---

## 🏆 Results

Logged MLflow runs identify **CatBoost** as the best-performing gradient-boosting model for each borough in the primary experiments.

Representative results from the logged borough-level runs:

| Borough   | Best Model | MAE   | RMSE  | R²    | Percentage Error |
|-----------|------------|-------|-------|-------|------------------|
| Manhattan | CatBoost   | 30.62 | 44.56 | 0.972 | —                |
| Brooklyn  | CatBoost   | 25.18 | 34.39 | 0.907 | 0.200%           |
| Bronx     | CatBoost   | 9.32  | 12.99 | 0.886 | 0.319%           |
| Queens    | CatBoost   | 7.71  | 10.16 | 0.838 | —                |

**Note:** Multiple experimental runs exist in the MLflow tracking database (different preprocessing / data scales). The table above uses consistent logged CatBoost results from the main gradient-boosting experiments.

### Statistical baseline

Best logged statistical model results:

| Model  | MAE    | RMSE   | Configuration              |
|--------|--------|--------|----------------------------|
| SARIMA | 119.82 | 158.89 | `(0,1,0)(1,1,1,24)`        |
| ARIMA  | higher | higher | simple differenced orders  |

Gradient-boosting models substantially outperformed the classical statistical baselines on the same task.

---

## 🔍 Key Findings

1. **CatBoost was the strongest gradient-boosting model** across the logged borough-level experiments.
2. **Performance varies by borough**. Manhattan shows the highest R²; Bronx and Queens are more difficult.
3. **Machine-learning models with lag + calendar features** significantly outperformed pure ARIMA/SARIMA on this dataset.
4. Error analysis (residuals, actual-vs-predicted plots, largest-error diagnostics) is logged as MLflow artifacts to understand *when* the models fail, not only overall accuracy.

---

## 🧪 Experiment Tracking with MLflow

All major modeling runs are tracked with MLflow (SQLite backend), including:

- Model name and borough
- Hyperparameters
- MAE / MSE / RMSE / R² / MAPE
- Feature importance
- Residual and diagnostic plots
- Best-model selection tags

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 📊 Interactive Streamlit Dashboard

Multi-page Streamlit application:

### 🏠 Home
Project overview and navigation.

### 📊 Data Insights
Exploratory analysis including:
- Ride volume by vendor
- Hourly / weekly / monthly trends
- Weekday vs weekend demand
- Congestion surcharge patterns
- Pickup hotspots and borough maps
- Trip distance distributions

### 🔮 Forecasting
Users can select borough, model, and forecast horizon.  
**Current implementation fully supports CatBoost recursive forecasting**; other model options are present in the UI but not all are fully wired for multi-step prediction.

Supported horizons include next hour up to next month.

---

## 🗂️ Project Structure

```text
.
├── src/
│   ├── Home.py
│   ├── Pages/
│   │   ├── Data Insights.py
│   │   └── Forecasting.py
│   ├── model_comparison.py 
│   ├── stratified_sampling.py      
│   ├── Gradient_Boost_model_training.py
│   └── stats_model_training.py
│
├── config.yaml
├── requirements.txt
├── artifacts/                       # metrics, residuals, plots
├── models/                          # persisted .pkl models
├── logs/
├── mlflow.db
└── Data/                            # raw parquet + processed CSVs (not committed)
```

> **Note:** The repository layout currently places the main scripts and Streamlit pages under `src/`. Some filenames and folder names contain typos or inconsistencies.

---

## 🛠️ Tech Stack

**Modeling:** scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels (ARIMA/SARIMA)  
**Tracking:** MLflow + SQLite  
**Dashboard:** Streamlit, Plotly, DuckDB  
**Data:** pandas, NumPy, holidays  
**Other:** joblib, PyYAML, Optuna (for XGBoost tuning)

---

## 🚀 Setup

```bash
git clone <your-repo-url>
cd <repo-name>

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **Warning:** The current `requirements.txt` contains duplicate / conflicting package versions. Clean it before production use.

### Data

Place NYC TLC Yellow Taxi Parquet files and lookup tables in a `Data/` directory:

```text
Data/
├── yellow_tripdata_*.parquet
├── location_ids.csv
└── taxi_zone_lookup.csv
```

Data source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## ▶️ Running the Project

```bash
# 1. Preprocess
python src/stratified_sampling\ .py

# 2. Train gradient boosting models
python src/Gradient_Boost_model_training.py
# or the version with extra error analysis
python src/New_Gradient_Boost_model_training.py

# 3. Train statistical models
python src/stats_model_training.py

# 4. MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 5. Dashboard (run from the directory containing the Streamlit entry point)
streamlit run src/Home.py
```

---

## 💡 Project Scope

This project demonstrates a complete applied data science workflow:

```text
Raw Data → Cleaning → Feature Engineering → 
Model Comparison → Experiment Tracking → 
Error Analysis → Model Persistence → Interactive Dashboard
```

It is intentionally designed as an end-to-end prototype rather than a production system.

---

## 🔮 Future Improvements

- [ ] Clean and pin `requirements.txt` / add environment.yml
- [ ] Standardize project layout and remove filename inconsistencies
- [ ] Complete multi-model support in the forecasting page
- [ ] Add proper time-series cross-validation and multi-step evaluation
- [ ] Add prediction intervals
- [ ] Dockerize the application
- [ ] Add model monitoring / drift detection
- [ ] External features (weather, events) where they improve accuracy
- [ ] Automated retraining pipeline

---

## 👨‍💻 Author

**Shafeeq**  
Data Science | Machine Learning | Time-Series Forecasting
```