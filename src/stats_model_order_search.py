import pandas as pd
# import streamlit as st
# import duckdb
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
import json
import os
import time
# from prophet import Prophet
import pickle
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from meteostat import Point, Hourly
# import holidays
from xgboost import XGBRegressor
import numpy as np
import os
import logging
from pathlib import Path
import yaml
from statsmodels.tsa.arima.model import ARIMA
import mlflow

#directories
project_dir = Path().resolve()
data_dir = project_dir/"Data"


# Setting up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
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

logger.addHandler(console_handler)
logger.addHandler(file_handler)
##importing yaml
yaml_path = project_dir/'config.yaml'
try:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logger.error(f"Error while importing yaml {e}")


##Reading the data for model training

try:
    # logger.info(f"Reading the data from{data_dir/"preprocessed_data.csv"}")
    # df = pd.read_csv(data_dir/"preprocessed_data.csv",parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'])
    # tsa_data = df.set_index('tpep_pickup_datetime')
    # hourly_demand =pd.DataFrame(tsa_data.resample(config['resample_type']).size().rename("Trips"))
    # logger.info(f"The data is resampled for{config['resample_type']}")
    logger.info(f"Reading the resampled file")
    df_resampled = pd.read_csv(data_dir/"hourly_demand.csv",parse_dates=True)
    df_resampled['tpep_pickup_datetime'] = pd.to_datetime(df_resampled['tpep_pickup_datetime'])
    df_resampled=df_resampled.set_index('tpep_pickup_datetime')
    df_resampled=df_resampled.sort_index()
    logger.info(f"input shape{df_resampled.shape}")
    idx=int(len(df_resampled)*0.8)
    train=df_resampled.iloc[:idx]
    test=df_resampled.iloc[idx:]
    # 1. Set the tracking URI to a local directory in your workspace
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NYC Taxi Demand")
    best_arima_rmse = float("inf")
    best_arima_order = None
    best_arima_model = None
    best_arima_metrics = None
    try:
        with mlflow.start_run(run_name="ARIMA Search"):
            logger.info("Starting Training of  ARIMA model")
            for order in [(0,1,0),(1,1,0),(1,1,1),(0,1,1),(2,1,1),(2,1,2)]:
                with mlflow.start_run(nested=True, run_name=f"ARIMA{order}"):
                    model=ARIMA(train,order=order)
                    model=model.fit()
                    forecast = model.get_forecast(len(test))
                    forecast=forecast.predicted_mean
                    mae=mean_absolute_error(test,forecast)
                    mse=mean_squared_error(test,forecast)
                    rmse=root_mean_squared_error(test,forecast)
                    if rmse < best_arima_rmse:
                        best_arima_rmse = rmse
                        best_arima_order = order
                        best_arima_model = model

                        metrics= {
                        "best_mae" :mae,
                        "best_mse": mse,
                        "best_rmse": rmse,
                        "best_aic" : model.aic,
                        "best_bic":model.bic}
            mlflow.log_param("best order",best_arima_order)
            mlflow.log_metrics(metrics)
            mlflow.statsmodels.log_model(best_arima_model, "best_model")
            logger.info("Training ARIMA is complete")
    except Exception as e:
        logger.error(f"Error while training arima model {e}")
        raise
    try:
        best_sarima_rmse = float("inf")
        best_sarima_order = None
        best_sarima_seasonal_order = None
        best_sarima_model = None
        with mlflow.start_run(run_name="SARIMA Search"):
            logger.info("Starting Training of  SARIMA model")
            for sarima_order in [(0,1,0),(1,1,0),(1,1,1),(0,1,1)]:
                for seasonal in[(1,1,1,24),(2,1,1,24),(2,1,2,24)]:
                    with mlflow.start_run(nested=True,run_name=f"SARIMA{sarima_order}{seasonal}"):
                        df= train.tail(24*120)
                        sarima_model = SARIMAX(train,order=sarima_order, seasonal_order=seasonal,enforce_invertibility=False, enforce_stationarity=False)
                        sarima_model= sarima_model.fit()
                        forecast = sarima_model.get_forecast(len(test))
                        forecast=forecast.predicted_mean
                        sarima_mae=mean_absolute_error(test,forecast)
                        sarima_mse=mean_squared_error(test,forecast)
                        sarima_rmse=root_mean_squared_error(test,forecast)
                        mlflow.log_params({"order":sarima_order,"seasonal_order":seasonal})
                        mlflow.log_metrics({"MAE":sarima_mae,
                                            "MSE":sarima_mse,
                                            "RMSE":sarima_rmse,
                                            "AIC":sarima_model.aic,
                                            "BIC":sarima_model.bic})

                        if sarima_rmse < best_sarima_rmse:
                            best_sarima_rmse = sarima_rmse
                            best_sarima_order = sarima_order
                            best_sarima_seasonal_order = seasonal
                            best_sarima_model = sarima_model
                            sarima_metrics= {
                            "best_mae" :sarima_mae,
                            "best_mse": sarima_mse,
                            "best_rmse": sarima_rmse,
                            "best_aic" : sarima_model.aic,
                            "best_bic":sarima_model.bic}
            mlflow.log_params({"best order":best_sarima_order, "best seasonal order":best_sarima_seasonal_order})
            mlflow.log_metrics(sarima_metrics)
            mlflow.statsmodels.log_model(best_sarima_model, "best_model")
            logger.info("Training  SARIMA is complete") 

    except Exception as e:
        logger.error(f"Error while training Sarima model {e}")

except Exception as e:
    logger.error(f"Error while training the model{e}")



