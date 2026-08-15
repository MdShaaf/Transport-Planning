"""
NYC Taxi Demand Forecasting
Model Comparison Experiment

Compares:
    - ARIMA
    - SARIMA
    - Best ML model per borough

For each borough:
    - Identifies the best statistical baseline using RMSE
    - Compares it with the best ML model
    - Calculates RMSE improvement
    - Records R²

Outputs:
    - model_comparison.csv
    - model_comparison.png
    - rmse_improvement.png

MLflow:
    Logs a run named "Model Comparison" under the existing
    "NYC Taxi Demand" experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================
project_dir = Path().resolve()
model_dir = project_dir/"models"
DATA_DIR = project_dir/"artifacts"
OUTPUT_DIR = project_dir/"artifacts"

BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx"]


# ============================================================
# 2. MLflow experiment
# ============================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("NYC Taxi Demand")


# ============================================================
# 3. Load ARIMA / SARIMA baseline results
# ============================================================

baseline_file = DATA_DIR / "borough_error_metrics.csv"

if not baseline_file.exists():
    raise FileNotFoundError(f"Missing file: {baseline_file}")

baseline = pd.read_csv(baseline_file)

baseline_results = []

for _, row in baseline.iterrows():

    baseline_results.extend([
        {
            "Borough": row["borough"],
            "Model": "ARIMA",
            "RMSE": row["arima_rmse"],
            "MAE": row["arima_mae"],
        },
        {
            "Borough": row["borough"],
            "Model": "SARIMA",
            "RMSE": row["sarima_rmse"],
            "MAE": row["sarima_mae"],
        },
    ])

baseline_df = pd.DataFrame(baseline_results)


# ============================================================
# 4. Load best ML model results
# ============================================================

ml_results = []

for borough in BOROUGHS:

    file = DATA_DIR / f"Error_metrics_for_{borough}.csv"

    if not file.exists():
        print(f"Warning: {file.name} not found")
        continue

    result = pd.read_csv(file).iloc[0]

    ml_results.append({
        "Borough": borough,
        "Best ML Model": "CatBoost",
        "ML RMSE": result["best_rmse"],
        "ML MAE": result["best_mae"],
        "ML R2": result["best_r2_score"],
    })


ml_df = pd.DataFrame(ml_results)

if ml_df.empty:
    raise ValueError("No ML model result files were found.")


# ============================================================
# 5. Identify strongest statistical baseline
# ============================================================
# Lower RMSE = better model

best_baseline = (
    baseline_df
    .sort_values("RMSE")
    .groupby("Borough", as_index=False)
    .first()
)

best_baseline = best_baseline.rename(columns={
    "Model": "Best Baseline",
    "RMSE": "Baseline RMSE",
    "MAE": "Baseline MAE",
})


# ============================================================
# 6. Combine results
# ============================================================

comparison = ml_df.merge(
    best_baseline[
        [
            "Borough",
            "Best Baseline",
            "Baseline RMSE",
            "Baseline MAE",
        ]
    ],
    on="Borough",
    how="left",
)


# ============================================================
# 7. Calculate RMSE improvement
# ============================================================

comparison["RMSE Improvement (%)"] = (
    (comparison["Baseline RMSE"] - comparison["ML RMSE"])
    / comparison["Baseline RMSE"]
    * 100
)


# ============================================================
# 8. Final comparison table
# ============================================================

comparison = comparison[
    [
        "Borough",
        "Best Baseline",
        "Baseline RMSE",
        "Best ML Model",
        "ML RMSE",
        "RMSE Improvement (%)",
        "ML R2",
    ]
]

comparison = comparison.round({
    "Baseline RMSE": 2,
    "ML RMSE": 2,
    "RMSE Improvement (%)": 1,
    "ML R2": 3,
})


# ============================================================
# 9. Save comparison table
# ============================================================

comparison_file = OUTPUT_DIR / "model_comparison.csv"
comparison.to_csv(comparison_file, index=False)


# ============================================================
# 10. Print results
# ============================================================

print("\n" + "=" * 75)
print("NYC TAXI DEMAND - MODEL COMPARISON")
print("=" * 75)

print(comparison.to_string(index=False))

print("\nInterpretation:")
print(
    "The best ML model is compared against the strongest statistical "
    "baseline (lowest RMSE) for each borough."
)

print(f"\nSaved: {comparison_file}")


# ============================================================
# 11. MLflow logging
# ============================================================

with mlflow.start_run(run_name="Model Comparison"):

    mlflow.set_tags({
        "stage": "model_comparison",
        "baseline_models": "ARIMA, SARIMA",
        "ml_models": "XGBoost, LightGBM, CatBoost",
        "best_ml_model": "CatBoost",
        "selection_metric": "RMSE",
    })

    for _, row in comparison.iterrows():

        borough = row["Borough"]

        mlflow.log_metric(
            f"{borough}_baseline_rmse",
            float(row["Baseline RMSE"])
        )

        mlflow.log_metric(
            f"{borough}_ml_rmse",
            float(row["ML RMSE"])
        )

        mlflow.log_metric(
            f"{borough}_rmse_improvement_pct",
            float(row["RMSE Improvement (%)"])
        )

        mlflow.log_metric(
            f"{borough}_ml_r2",
            float(row["ML R2"])
        )

    # Summary metrics across all boroughs - useful for comparing
    # "Model Comparison" runs against each other over time
    mlflow.log_metric(
        "avg_rmse_improvement_pct",
        float(comparison["RMSE Improvement (%)"].mean())
    )

    mlflow.log_metric(
        "avg_ml_r2",
        float(comparison["ML R2"].mean())
    )

    mlflow.log_metric(
        "min_rmse_improvement_pct",
        float(comparison["RMSE Improvement (%)"].min())
    )

    # Log the final comparison table
    mlflow.log_artifact(str(comparison_file))

    print("\nMLflow comparison run logged successfully.")


# ============================================================
# 12. Chart 1 - Baseline vs ML RMSE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(comparison))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    comparison["Baseline RMSE"],
    width,
    label="Best Baseline",
)

ax.bar(
    [i + width / 2 for i in x],
    comparison["ML RMSE"],
    width,
    label="Best ML Model",
)

ax.set_xticks(x)
ax.set_xticklabels(comparison["Borough"])

ax.set_ylabel("RMSE")
ax.set_xlabel("Borough")

ax.set_title(
    "NYC Taxi Demand: Baseline vs Best ML Model"
)

ax.legend()

plt.tight_layout()

rmse_chart = OUTPUT_DIR / "model_comparison.png"

plt.savefig(
    rmse_chart,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 13. Chart 2 - RMSE improvement
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    comparison["Borough"],
    comparison["RMSE Improvement (%)"],
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_ylabel("RMSE Improvement (%)")
ax.set_xlabel("Borough")

ax.set_title(
    "Best ML Model vs Best Statistical Baseline"
)

for bar, value in zip(
    bars,
    comparison["RMSE Improvement (%)"],
):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

improvement_chart = OUTPUT_DIR / "rmse_improvement.png"

plt.savefig(
    improvement_chart,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 14. Finish
# ============================================================

print(f"Saved: {rmse_chart}")
print(f"Saved: {improvement_chart}")

print("\n" + "=" * 75)
print("MODEL COMPARISON COMPLETE")
print("=" * 75)