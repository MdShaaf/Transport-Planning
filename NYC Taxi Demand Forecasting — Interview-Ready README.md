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

The raw dataset is therefore converted from millions of individual trip records into a much more manageable forecasting dataset.

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

Historical demand is incorporated through lag variables such as:

```text
Lag1     → Previous hour
Lag2     → Two hours earlier
Lag3     → Three hours earlier
Lag6     → Six hours earlier
Lag24    → Same hour previous day
Lag168   → Same hour previous week
Lag336   → Same hour two weeks earlier
```

These features allow the machine-learning models to capture:

- Short-term demand persistence
- Daily seasonality
- Weekly seasonality
- Repeating demand patterns

---

## 🤖 Modeling Approach

Two different forecasting approaches were evaluated.

### 1. Machine Learning

The project compares three gradient-boosting algorithms:

- **XGBoost**
- **LightGBM**
- **CatBoost**

Hyperparameter optimization was also performed for XGBoost using **Optuna**, with the resulting parameters stored in `config.yaml`.

### 2. Statistical Forecasting

Traditional time-series models were also evaluated:

- **ARIMA**
- **SARIMA**

SARIMA was evaluated with seasonal components to capture repeating hourly demand patterns.

This provides a useful comparison between:

```text
Feature-based ML
        vs.
Traditional time-series forecasting
```

---

## 🧪 Model Evaluation

The models are evaluated using forecasting error metrics including:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Explained variance
- **Percentage Error** where available

### Why RMSE?

RMSE is useful for demand forecasting because it penalizes larger errors more strongly.

Large forecasting errors during demand peaks can be operationally more important than many small errors, making RMSE a useful model-selection metric.

---

## 🏆 Results

The MLflow experiment results show **CatBoostRegressor as the winning gradient-boosting model across the borough-level experiments**.

| Borough | Best Model | MAE | RMSE | R² | Percentage Error |
|---|---|---:|---:|---:|---:|
| Manhattan | CatBoost | 30.62 | 44.56 | **0.972** | — |
| Queens | CatBoost | 50.21 | 69.69 | **0.917** | 0.623% |
| Brooklyn | CatBoost | 25.18 | 34.39 | **0.907** | 0.200% |
| Bronx | CatBoost | 9.32 | 12.99 | **0.886** | 0.319% |

The logged MLflow experiments identify CatBoost as the winner for Manhattan, Queens, Brooklyn, and Bronx. 
### Overall benchmark

A separate logged gradient-boosting run reports:

| Metric | Result |
|---|---:|
| Best Model | CatBoostRegressor |
| MAE | 34.42 |
| RMSE | 49.61 |
| R² | **0.971** |

The MLflow run explicitly identifies CatBoost as the winning model for this benchmark.

### Statistical baseline

The statistical-model search produced the following best logged results:

| Model | MAE | RMSE | Configuration |
|---|---:|---:|---|
| SARIMA | 119.82 | 158.89 | `(0,1,0)(1,1,1,24)` |
| ARIMA | 278.85 | 359.55 | `(0,1,0)` |

The logged search therefore showed substantially lower forecasting error for the best SARIMA configuration than the best ARIMA configuration.

> **Note:** The MLflow summary contains multiple runs produced during experimentation. The results above use the explicitly logged borough-level gradient-boosting benchmark runs rather than mixing metrics from different evaluation runs.

---

## 🔍 Key Findings

### 1. CatBoost consistently performed best among the gradient-boosting models

CatBoost was selected as the winner in the logged borough-level gradient-boosting experiments.

This suggests that the combination of engineered temporal and lag features was particularly effective with CatBoost for this forecasting problem.

### 2. Model performance varies by borough

The forecasting problem is not equally difficult across all boroughs.

For example:

- Manhattan achieved R² ≈ **0.972**
- Queens achieved R² ≈ **0.917**
- Brooklyn achieved R² ≈ **0.907**
- Bronx achieved R² ≈ **0.886**

This supports evaluating and selecting models at the **borough level**, rather than assuming one model behaves identically across all locations. 
### 3. Machine-learning models provided a strong alternative to traditional statistical models

The project deliberately evaluates both tree-based machine-learning models and statistical forecasting models.

This makes it possible to compare:

```text
Historical time-series structure
            vs.
Historical structure + engineered features
```

rather than assuming that one forecasting methodology is universally superior.

---

## 📈 Error Analysis

Model evaluation goes beyond aggregate metrics.

The project also generates:

- Residual distributions
- Actual vs. predicted plots
- Largest-error diagnostics
- Feature importance visualizations

This helps identify **where the model performs poorly**, rather than relying only on a single R² or RMSE value.

The goal is to answer:

> "When does the model fail, and how large are those failures?"

rather than simply:

> "What is the model's accuracy?"

---

## 🧪 Experiment Tracking with MLflow

Every major modeling experiment is tracked using **MLflow**.

Tracked information includes:

- Model name
- Borough
- Hyperparameters
- MAE
- MSE
- RMSE
- R²
- Percentage error
- Model-selection results
- Diagnostic artifacts

The project uses an SQLite MLflow backend.

Example:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This provides a reproducible experiment history instead of relying on manually recorded model results.

---

## 📊 Interactive Streamlit Dashboard

The project includes a multi-page Streamlit application.

### 🏠 Home

Provides project overview and navigation.

### 📊 Data Insights

The dashboard provides exploratory analysis including:

- Ride volume by vendor
- Hourly demand trends
- Weekly demand patterns
- Monthly trends
- Weekday vs. weekend demand
- Congestion surcharge patterns
- Pickup hotspots
- Borough demand maps
- Trip-distance distributions

### 🔮 Forecasting

Users can select:

- Borough
- Forecasting model
- Forecast horizon

Supported horizons include:

```text
Next Hour
Next 6 Hours
Next 12 Hours
Next Day
Next Week
Next Month
```

The forecasting page provides:

- Forecast trend
- Forecast table
- Demand heatmaps
- Model performance metrics
- Residual analysis

---

## 🗂️ Project Structure

```text
.
├── Home.py
├── pages/
│   ├── Data Insights.py
│   └── Forecasting.py
│
├── stratified_sampling_.py
├── Gradient_Boost_model_training.py
├── New_Gradient_Boost_model_training.py
├── stats_model_training.py
│
├── config.yaml
├── requirements.txt
│
├── Data/
│   ├── raw parquet files
│   ├── sampled/combined datasets
│   └── lookup files
│
├── models/
│   └── persisted models
│
├── artifacts/
│   ├── feature importance
│   ├── metrics
│   ├── residuals
│   └── diagnostic plots
│
├── logs/
│   └── training logs
│
└── mlflow.db
```

---

## 🛠️ Tech Stack

### Machine Learning

- Python
- scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Time Series

- statsmodels
- ARIMA
- SARIMA

### Data Processing

- pandas
- NumPy
- DuckDB
- holidays

### Experiment Tracking

- MLflow
- SQLite

### Visualization & Dashboard

- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Model Persistence

- joblib

### Configuration & Optimization

- PyYAML
- Optuna

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Data Setup

Place the NYC TLC Yellow Taxi data inside the `Data/` directory.

Expected files include:

```text
Data/
├── yellow_tripdata_*.parquet
├── location_ids.csv
└── taxi_zone_lookup.csv
```

NYC TLC trip data can be obtained from the official NYC Taxi & Limousine Commission data portal.

Because the raw TLC dataset is large, the project includes a preprocessing/sampling step before model training.

---

## ▶️ Running the Project

### Step 1 — Preprocess the raw data

```bash
python stratified_sampling_.py
```

This produces the processed hourly borough-level demand dataset.

### Step 2 — Train gradient-boosting models

```bash
python Gradient_Boost_model_training.py
```

For the extended error-analysis workflow:

```bash
python New_Gradient_Boost_model_training.py
```

### Step 3 — Train statistical models

```bash
python stats_model_training.py
```

This performs ARIMA/SARIMA model searches.

### Step 4 — Launch MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Step 5 — Launch the dashboard

```bash
streamlit run Home.py
```

---

## 💡 Why This Project Is More Than a Model

This project was designed as an **end-to-end Data Science workflow**, rather than a standalone model-training exercise.

It covers:

```text
Raw Data
   ↓
Data Engineering
   ↓
Exploratory Analysis
   ↓
Feature Engineering
   ↓
Time-Series Modeling
   ↓
Model Comparison
   ↓
Experiment Tracking
   ↓
Error Analysis
   ↓
Model Persistence
   ↓
Interactive Deployment
```

This allows the project to demonstrate the complete lifecycle from **raw data to an interactive forecasting application**.

---

## 🔮 Future Improvements

- [ ] Add automated retraining pipeline
- [ ] Add model monitoring and drift detection
- [ ] Add prediction intervals / uncertainty estimates
- [ ] Improve multi-step forecasting strategy
- [ ] Add Docker-based deployment
- [ ] Add automated CI/CD
- [ ] Add additional external features where they demonstrate measurable forecasting value
- [ ] Add a business-oriented demand opportunity/recommendation layer
- [ ] Add automated MLflow model registration and versioning

---

## 👨‍💻 Author

**Shafeeq**

Data Science | Machine Learning | Time-Series Forecasting | Engineering Analytics