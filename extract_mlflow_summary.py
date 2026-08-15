"""
Extracts a clean summary of MLflow experiment runs from a local
sqlite mlflow.db tracking store, for feeding into an AI (e.g. to
draft a README Results section) or just for your own review.

Usage:
    python extract_mlflow_summary.py

Run this from the same directory as your mlflow.db (or edit
TRACKING_URI below). Produces:
    - mlflow_summary.json  (full structured data)
    - mlflow_summary.md    (human/AI-readable markdown table)
"""

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
OUTPUT_JSON = Path("mlflow_summary.json")
OUTPUT_MD = Path("mlflow_summary.md")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

summary = []

for exp in client.search_experiments():
    exp_data = {"experiment_name": exp.name, "experiment_id": exp.experiment_id, "runs": []}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.RMSE ASC", "metrics.best_rmse ASC"],
    )

    for run in runs:
        info = run.info
        data = run.data

        # Skip runs with no metrics logged (e.g. empty parent runs)
        if not data.metrics:
            continue

        exp_data["runs"].append({
            "run_name": data.tags.get("mlflow.runName", info.run_id),
            "status": info.status,
            "metrics": data.metrics,
            "params": {k: v for k, v in data.params.items() if not k.startswith("trip_")},
            "tags": {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
        })

    if exp_data["runs"]:
        summary.append(exp_data)

# Write JSON (full detail, good for feeding to an AI)
OUTPUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

# Write Markdown (quick human scan / drop straight into a README)
lines = ["# MLflow Run Summary\n"]
for exp in summary:
    lines.append(f"## {exp['experiment_name']}\n")
    for run in exp["runs"]:
        lines.append(f"### {run['run_name']} ({run['status']})")
        if run["metrics"]:
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k, v in sorted(run["metrics"].items()):
                lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        if run["tags"]:
            lines.append(f"\n**Tags:** {run['tags']}")
        lines.append("")
OUTPUT_MD.write_text("\n".join(lines))

print(f"Wrote {OUTPUT_JSON} and {OUTPUT_MD}")
print(f"Found {sum(len(e['runs']) for e in summary)} runs across {len(summary)} experiments.")