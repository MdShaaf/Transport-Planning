# 🚗 Transport Analytics Dashboard

A Python-based **transport planning and analytics dashboard** built using **NYC Taxi trip data**.  
This application provides **data insights**, **demand forecasting**, and **profit-based route recommendations** through an interactive **Streamlit dashboard**.

The project focuses on **practical analytics, interpretable models, and real-world decision support**, making it suitable for **portfolio presentation and interviews**.

---

## 📌 Project Overview

Urban transportation systems generate massive amounts of trip data. This project analyzes real-world taxi data to:

- Understand **travel demand patterns**
- Forecast **future transport demand**
- Recommend **profitable zones** for routing decisions

Since the original NYC Taxi dataset is very large, **stratified sampling** was applied to preserve representative spatial and temporal patterns while keeping computation efficient.

---

## 🧠 Key Features

### 📊 Data Insights
- Hourly, weekly, and monthly ride trends
- Pickup demand by borough and taxi zone
- Trip distance and passenger distribution
- Weekday vs weekend demand comparison

### 📈 Demand Forecasting
Multiple forecasting models were implemented and evaluated:
- Naive baseline model
- SARIMA / SARIMAX
- Prophet
- **XGBoost (final selected model)**

Models were compared using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)

**XGBoost achieved the best overall performance**, effectively capturing non-linear patterns and external influences.

### 🗺️ Route Recommendation (Profit Optimization)
A rule-based recommendation system that:
- Calculates trip profit using  
  **Profit = (Fare + Tips) − Fuel Cost**
- Aggregates profit by **zone and hour**
- Compares nearby zones for the next time period
- Recommends zones with **higher expected profitability**

This approach prioritizes **interpretability over black-box routing algorithms**.

---

## 🛠️ Tech Stack

- **Language:** Python 3.13.1  
- **Framework:** Streamlit  
- **Data Processing:** Pandas, NumPy, DuckDB  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Time Series:** Statsmodels, pmdarima, Prophet  
- **Machine Learning:** Scikit-learn, XGBoost  
- **External Data:** Meteostat (weather), Holidays  

---

## 📂 Project Structure

```

transport-planning-streamlit/
│── Home.py (for streamlit)
│── data/
│── notebooks/
│── src/
│── requirements.txt
│── README.md

````
## ▶️ How to Run the Project

### 1️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run Home.py
```
## 📊 Dataset

* **Source:** NYC Taxi Trip Data
* **Year:** 2025
* **Processing:**

  * Original dataset is large-scale
  * Reduced using **stratified sampling** to retain representative demand and spatial patterns
* **Purpose:** Efficient analytics without compromising data integrity

---

## 📈 Model Summary

| Model       | Performance               |
| ----------- | ------------------------- |
| Naive       | Baseline                  |
| SARIMA      | Captures seasonality      |
| Prophet     | Smooth trend modeling     |
| **XGBoost** | **Best overall accuracy** |

✅ **Final Model Selected:** XGBoost


## ⚠️ Limitations

* Uses historical data (not real-time traffic)
* Road network constraints are not explicitly modeled
* Fuel cost assumed as a fixed value
* Zone-based recommendations (not turn-by-turn navigation)

---

## 🚀 Future Improvements

* Integration with real road networks (OpenStreetMap)
* Traffic-aware and time-dependent routing
* Real-time data ingestion
* Advanced spatial clustering
* Cloud deployment for scalable analytics

---

## 👤 Author

Developed by **Shafeeq**
```
