import pandas as pd
# import streamlit as st
# import duckdb
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
import json
import os
import time
# from prophet import Prophet
import pickle
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from meteostat import Point, Hourly
# import holidays
from xgboost import XGBRegressor
import numpy as np
import logging
from pathlib import Path
import yaml
from statsmodels.tsa.arima.model import ARIMA
import mlflow

# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------
project_dir = Path().resolve()
data_dir = project_dir / "Data"
model_dir = project_dir / "models"
artifact_dir = project_dir / "artifacts"
log_dir = project_dir / "logs"                 # was missing -> caused NameError
plot_dir = artifact_dir / "plots"

for i in (data_dir, model_dir, artifact_dir, log_dir, plot_dir):
    os.makedirs(i, exist_ok=True)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# avoid duplicate handlers if script is re-run in same session
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
yaml_path = project_dir / 'config.yaml'
config = {}
try:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Error while importing yaml {e}")

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
logger.info("Reading the resampled file")
df = pd.read_csv(data_dir / "combined_sampled_data.csv", parse_dates=True)
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])


def train_stats_models(data, borough):
    """
    Trains ARIMA and SARIMA models for a single borough, logs to MLflow,
    saves each model to disk, saves an actual-vs-forecast plot, and
    returns a dict of error metrics for both models.
    """
    results = {"borough": borough}

    try:
        data = data[data['Borough'] == borough]
        data = data[['tpep_pickup_datetime', 'Trips']]
        data = data.set_index('tpep_pickup_datetime')
        data = data.sort_index()

        idx = int(len(data) * 0.8)
        train = data.iloc[:idx]
        test = data.iloc[idx:]

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("NYC Taxi Demand")

        # ---------------- ARIMA ----------------
        arima_order = (0, 1, 0)
        arima_forecast = None
        try:
            with mlflow.start_run(run_name=f"ARIMA_{borough}"):
                logger.info(f"[{borough}] Starting ARIMA training")

                arima_model = ARIMA(train, order=arima_order)
                arima_model = arima_model.fit()

                forecast = arima_model.get_forecast(len(test))
                arima_forecast = forecast.predicted_mean

                arima_mae = mean_absolute_error(test, arima_forecast)
                arima_mse = mean_squared_error(test, arima_forecast)
                arima_rmse = root_mean_squared_error(test, arima_forecast)

                arima_metrics = {
                    "mae": arima_mae,
                    "mse": arima_mse,
                    "rmse": arima_rmse,
                    "aic": arima_model.aic,
                    "bic": arima_model.bic,
                }

                # log params/metrics/model INSIDE the active run
                mlflow.log_param("order", arima_order)
                mlflow.log_metrics(arima_metrics)
                mlflow.statsmodels.log_model(arima_model, "arima_model")

                # save model locally too
                arima_path = model_dir / f"arima_{borough}.pkl"
                with open(arima_path, "wb") as f:
                    pickle.dump(arima_model, f)

                logger.info(f"[{borough}] ARIMA training complete | RMSE={arima_rmse:.3f}")

            results.update({
                "arima_mae": arima_mae,
                "arima_mse": arima_mse,
                "arima_rmse": arima_rmse,
                "arima_aic": arima_model.aic,
                "arima_bic": arima_model.bic,
            })

        except Exception as e:
            logger.error(f"[{borough}] Error while training ARIMA model: {e}")
            print(f"[{borough}] ARIMA error: {e}")

        # ---------------- SARIMA ----------------
        sarima_order = (0, 1, 0)
        seasonal_order = (1, 1, 1, 24)
        sarima_forecast = None
        try:
            with mlflow.start_run(run_name=f"SARIMA_{borough}"):
                logger.info(f"[{borough}] Starting SARIMA training")

                sarima_model = SARIMAX(
                    train,
                    order=sarima_order,
                    seasonal_order=seasonal_order,
                    enforce_invertibility=False,
                    enforce_stationarity=False,
                )
                sarima_model = sarima_model.fit(disp=False)

                forecast = sarima_model.get_forecast(len(test))
                sarima_forecast = forecast.predicted_mean

                sarima_mae = mean_absolute_error(test, sarima_forecast)
                sarima_mse = mean_squared_error(test, sarima_forecast)
                sarima_rmse = root_mean_squared_error(test, sarima_forecast)

                sarima_metrics = {
                    "mae": sarima_mae,
                    "mse": sarima_mse,
                    "rmse": sarima_rmse,
                    "aic": sarima_model.aic,
                    "bic": sarima_model.bic,
                }

                mlflow.log_params({"order": sarima_order, "seasonal_order": seasonal_order})
                mlflow.log_metrics(sarima_metrics)
                mlflow.statsmodels.log_model(sarima_model, "sarima_model")

                sarima_path = model_dir / f"sarima_{borough}.pkl"
                with open(sarima_path, "wb") as f:
                    pickle.dump(sarima_model, f)

                logger.info(f"[{borough}] SARIMA training complete | RMSE={sarima_rmse:.3f}")

            results.update({
                "sarima_mae": sarima_mae,
                "sarima_mse": sarima_mse,
                "sarima_rmse": sarima_rmse,
                "sarima_aic": sarima_model.aic,
                "sarima_bic": sarima_model.bic,
            })

        except Exception as e:
            logger.error(f"[{borough}] Error while training SARIMA model: {e}")
            print(f"[{borough}] SARIMA error: {e}")

        # ---------------- Plot: actual vs forecast ----------------
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(train.index[-200:], train['Trips'].iloc[-200:], label="Train (tail)")
            plt.plot(test.index, test['Trips'], label="Actual", color="black")
            if arima_forecast is not None:
                plt.plot(test.index, arima_forecast, label="ARIMA Forecast", linestyle="--")
            if sarima_forecast is not None:
                plt.plot(test.index, sarima_forecast, label="SARIMA Forecast", linestyle="--")
            plt.title(f"{borough} - Actual vs Forecast")
            plt.xlabel("Datetime")
            plt.ylabel("Trips")
            plt.legend()
            plt.tight_layout()

            plot_path = plot_dir / f"{borough}_forecast.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"[{borough}] Saved forecast plot to {plot_path}")

        except Exception as e:
            logger.error(f"[{borough}] Error while plotting forecast: {e}")
            print(f"[{borough}] Plot error: {e}")

    except Exception as e:
        logger.error(f"[{borough}] Error while training models: {e}")
        print(f"[{borough}] Fatal error: {e}")

    return results


# ------------------------------------------------------------------
# Run for all boroughs and collect a combined error-metrics table
# ------------------------------------------------------------------
boroughs = ['Manhattan', 'Queens', 'Brooklyn', 'Bronx']
all_results = []

for b in boroughs:
    borough_df = df[df['Borough'] == b]
    res = train_stats_models(borough_df, b)
    all_results.append(res)

metrics_df = pd.DataFrame(all_results)
metrics_path = artifact_dir / "borough_error_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

logger.info(f"Saved combined error metrics to {metrics_path}")
print("\n=== Error metrics by borough ===")
print(metrics_df.to_string(index=False))