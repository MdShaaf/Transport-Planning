import json
import os
import time
# from prophet import Prophet
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import holidays
from xgboost import XGBRegressor
import numpy as np
import os
import logging
from pathlib import Path
import yaml
import mlflow
import seaborn as sns
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
#directories
project_dir = Path().resolve()
data_dir = project_dir/"Data"
model_dir = project_dir/"models"
import holidays
import joblib
# Setting up logging
dirs = ['logs','models', 'artifacts']
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

log_dir = project_dir/'logs'
artifacts_dir = project_dir/'artifacts'
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('Gradient_Boost_model_training')
logger.setLevel(logging.DEBUG)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 

log_file_path = os.path.join(log_dir, 'Gradient_Boost_model_training.log')  
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

def train_model(df, borough): 
    try:        
        #### Borough wise Data 
        ####------------------
        logger.info(f"Training for the borough {borough}")
       
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df= df.sort_values(by='tpep_pickup_datetime')
        logger.info(f"Beginning data processing and Feature Enginnering for {borough}")
        holidays_usa=holidays.US()

        df['Is_Holiday'] = df['tpep_pickup_datetime'].dt.date.apply(lambda d: d in holidays_usa).astype(int)
        df['Day of the week']=df['tpep_pickup_datetime'].dt.day_of_week
        df['Month']=df['tpep_pickup_datetime'].dt.month
        df['Is weekend']=(df['Day of the week']>=5).astype(int)
        df['Hour of the Day']=df['tpep_pickup_datetime'].dt.hour
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168,336]
        for i in lags:
            df[f"Lag{i}"]=df['Trips'].shift(i)
        df = df.dropna(axis=0)
        # logger.info(f"Shape of data before {df_resampled.shape}, shape after data processing {df.shape}")
        logger.info(f" The features are {df.columns}")
        idx=int(len(df)*0.8)
        train=df.iloc[:idx]
        test=df.iloc[idx:]
        test.to_csv(data_dir/"test_split_data.csv")
        X_train=train.drop(columns=['Borough','Trips','tpep_pickup_datetime'],axis=1)
        y_train=train['Trips']
        X_test=test.drop(columns=['Borough','Trips','tpep_pickup_datetime'],axis=1)
        y_test=test['Trips']

    except Exception as e:
        logger.info(f"Error during the Data Processing {e}")
        raise


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NYC Taxi Demand")
    try:
        best_rmse = float('inf')
        best_mae= None
        best_model=None
        best_mse = None
        logger.info(f"Beginning the model training zone {borough}")
        with mlflow.start_run(run_name=f"Gradient Boosting Models for {borough}"):
            logger.info("Starting Training of  Gradient Bossting Models")
            models = {'XGBRegressor':XGBRegressor(**config['best_params']), ### From Optuna
                        'LGBMRegressor':LGBMRegressor(),
                        'CatBoostRegressor':CatBoostRegressor(verbose=0)}
            for model_name, model in models.items():
                trip_stats = df["Trips"].describe()

                mlflow.log_params({
                    "trip_count": int(trip_stats["count"]),
                    "trip_mean": float(trip_stats["mean"]),
                    "trip_std": float(trip_stats["std"]),
                    "trip_min": float(trip_stats["min"]),
                    "trip_Q_25": float(trip_stats["25%"]),
                    "trip_Q_50": float(trip_stats["50%"]),
                    "trip_Q_75": float(trip_stats["75%"]),
                    "trip_max": float(trip_stats["max"]),
                })
                with mlflow.start_run(nested=True, run_name=f"{borough}_{model_name}"):
                    logger.info(f"Trainig the {model_name} model")
                    # Log hyperparameters
                    mlflow.log_param("borough", borough)
                    mlflow.log_params(model.get_params())

                    model=model.fit(X_train, y_train)
                    predicted = model.predict(X_test)
                    mae=mean_absolute_error(y_test,predicted)
                    mse=mean_squared_error(y_test,predicted)
                    rmse=root_mean_squared_error(y_test,predicted)
                    r2 = r2_score(y_test,predicted)
                    percentage_error = mean_absolute_percentage_error(y_test,predicted)


                    # Log metrics
                    metrics= {
                        'MAE':mae,
                        'MSE':mse,
                        'RMSE':rmse,
                        'r2_score':r2,
                        'percentage_error':percentage_error
                    }
                    mlflow.log_metrics(metrics)
                    if rmse< best_rmse:
                        best_rmse = rmse
                        best_mae= mae
                        best_mse = mse
                        best_model=model
                        best_r2=r2
                        best_percentage_error = percentage_error
                        best_model_name = model_name
                        best_metrics= {
                            "best_mae" :best_mae,
                            "best_mse": best_mse,
                            "best_rmse": best_rmse,
                            "best_r2_score":best_r2,
                            'best_percentage_error':best_percentage_error
                            }
                logger.info(f"Trainig completed for the {model_name,borough} model")
            features_imp = (pd.DataFrame({'features': X_train.columns,
                                        'importance': best_model.feature_importances_})
                            .sort_values(by='importance', ascending=False)) 
            # Log feature importance as artifact     
            feature_artifact_path =artifacts_dir/f"{borough}_feature_importance.csv"
            features_imp.to_csv(feature_artifact_path)
            mlflow.log_artifact(feature_artifact_path)
            mlflow.set_tag("Best Model", best_model_name)
            mlflow.log_metrics(best_metrics)
            df_metrics = pd.DataFrame([best_metrics])
            df_metrics.to_csv(artifacts_dir/f"Error_metrics_for_{borough}.csv", index=False)

            mlflow.log_params(best_model.get_params())

            ##Logging Model
            if best_model_name == 'CatBoostRegressor':
                mlflow.catboost.log_model(best_model, f"{best_model_name}")
            else:
                mlflow.sklearn.log_model(best_model, f"{best_model_name}")
            mlflow.log_metrics(best_metrics)
            mlflow.log_param("Best Model", best_model_name)
            mlflow.set_tag("Winner", best_model_name)
            joblib.dump(best_model, model_dir/f"{borough}_{best_model_name}.pkl")


            ###----------------------------------------------
            #####Error Analysis
            ###-----------------------------------------------
            logger.info(f"Beginning Error Analysis for {borough}")
            residuals = pd.DataFrame({"Actuals":y_test, "Predicted":(best_model.predict(X_test))})
            residuals['Error']= residuals["Actuals"]-residuals["Predicted"]
            residuals_path =artifacts_dir/f"{borough}_Residuals.csv"
            residuals.to_csv(residuals_path,index=True)
            errors_stats=residuals['Error'].describe()
            errors_stats_df = errors_stats.reset_index()
            errors_stats_df.columns = ["Statistic", "Value"]
            mlflow.log_text(errors_stats_df.to_string(), "Errors_stats.txt")

            total = len(residuals)

            positive_errors = (residuals["Error"] > 0).sum() / total
            negative_errors = (residuals["Error"] < 0).sum() / total
            errors_dict= pd.DataFrame({'Positive Errors':[positive_errors], "Negative Erros":[negative_errors]})
            mlflow.log_text(errors_dict.to_string(index=False), "Errors_Bias.txt")
            ##Residuals data frame merging with date and time
            residuals = residuals.join(df, how='left')
            # residuals_df = residuals.merge(df, how='left', left_on='Unnamed: 0', right_index=True)


            ####Plotting Errors
            sns.set_style("whitegrid")
            sns.set_context("talk")   # larger fonts for presentations

            plt.figure(figsize=(12,8))  # bigger canvas

            logger.info(f"Plotting Actual vs Predicted plot for {borough}")

            # Plot Actuals
            sns.lineplot(data=residuals, x='tpep_pickup_datetime', y='Actuals',
                        label='Actual', color='steelblue', linewidth=2)

            # Optional: add Predicted for comparison
            sns.lineplot(data=residuals, x='tpep_pickup_datetime', y='Predicted',
                        label='Predicted', color='darkorange', linewidth=2, linestyle='--')

            # Titles and labels
            plt.title("Actual vs Predicted Over Time", fontsize=18, fontweight='bold')
            plt.xlabel("Pickup Time", fontsize=14)
            plt.ylabel("Values", fontsize=14)

            # Legend styling
            plt.legend(title="Legend", fontsize=12)

            # Rotate x-axis ticks if datetime
            plt.xticks(rotation=45)

            # Tight layout for neatness
            plt.tight_layout()
            plot_actual_predicted_path = artifacts_dir/f"{borough}_actual_vs_predicted.png"
            plt.savefig(plot_actual_predicted_path,dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(plot_actual_predicted_path,artifact_path="plots")

            logger.info(f"Plotting histplot for {borough} ")
            histplot_path = artifacts_dir/f"{borough}_Errors_histplot.png"
            plt.figure(figsize=(8,6))
            sns.histplot(residuals['Error'],bins=50)
            plt.tight_layout()
            plt.savefig(histplot_path)
            plt.close()
            mlflow.log_artifact(histplot_path,artifact_path="plots")


            ####LArgest Errors 
            largest_errors = residuals.reindex(residuals['Error'].abs().sort_values(ascending=False).index)
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=largest_errors.head(50), x="Actuals", y="Predicted", 
                            color="red", s=80, label="Largest Errors")

            # Add diagonal line (perfect prediction reference)
            max_val = max(largest_errors["Actuals"].max(), largest_errors["Predicted"].max())
            plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", label="Perfect Fit")

            plt.title("Actual vs Predicted (Top Errors)", fontsize=16, fontweight="bold")
            plt.xlabel("Actuals", fontsize=14)
            plt.ylabel("Predicted", fontsize=14)
            plt.legend()
            plt.tight_layout()

            largest_errors_path=artifacts_dir/f"{borough}_Largest_Errors.png"
            plt.savefig(largest_errors_path)
            plt.close()
            mlflow.log_artifact(largest_errors_path,artifact_path="plots")

    except Exception as e:
        logger.error(f"Error while training the models: {e}")


df = pd.read_csv(data_dir/"combined_sampled_data.csv",parse_dates=True)
boroughs = ['Manhattan','Queens','Brooklyn','Bronx']
for borough in boroughs:
    borough_df = df[df["Borough"] == borough].copy()
    train_model(borough_df, borough)