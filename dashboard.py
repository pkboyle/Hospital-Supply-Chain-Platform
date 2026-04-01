"""
dashboard.py
------------
Flask application serving the operator-facing supply chain risk dashboard.
Loads the trained model and inventory data to provide:
  - Real-time stockout risk scores by facility and product
  - Supplier delay alerts
  - REST API endpoints for programmatic access

Run: python dashboard.py
Then open: http://localhost:5000
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Load artifacts ────────────────────────────────────────────────────────────
MODEL_PATH    = "models/stockout_model.pkl"
DATA_PATH     = "data/ml_dataset.csv"
METADATA_PATH = "data/metadata.json"
METRICS_PATH  = "outputs/metrics.json"

FEATURES = [
    "closing_stock", "days_of_supply", "avg_consumption_4w",
    "avg_received_4w", "stockout_rate_4w", "below_safety_rate_4w",
    "avg_supplier_reliability", "avg_delay_days", "fulfillment_rate",
    "facility_beds", "week_of_year", "critical_item",
]

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
    for col in ["avg_supplier_reliability", "avg_delay_days", "fulfillment_rate"]:
        df[col] = df[col].fillna(df[col].median())
    df["critical_item"] = df["critical_item"].astype(int)

    with open(METADATA_PATH) as f:
        meta = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    # Use most recent snapshot per facility/product as "current" state
    latest = (
        df.sort_values("snapshot_date")
        .groupby(["facility_id", "product_id"])
        .last()
        .reset_index()
    )
    latest["risk_score"] = model.predict_proba(latest[FEATURES])[:, 1]
    latest["risk_level"] = pd.cut(
        latest["risk_score"],
        bins=[-0.001, 0.25, 0.55, 1.001],
        labels=["Low", "Medium", "High"],
    )

    return model, df, latest, meta, metrics


try:
    model, df, latest, meta, metrics = load_artifacts()
    print("Artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please run  python generate_data.py  then  python train_model.py  first.")
    model = df = latest = meta = metrics = None


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/api/facilities")
def get_facilities():
    if latest is None:
        return jsonify({"error": "Model not loaded"}), 503

    summary = (
        latest.groupby("facility_id")
        .agg(
            high_risk_items=("risk_level", lambda x: (x == "High").sum()),
            medium_risk_items=("risk_level", lambda x: (x == "Medium").sum()),
            mean_risk_score=("risk_score", "mean"),
            critical_stockouts=("stockout", lambda x:
                latest.loc[x.index][latest.loc[x.index, "critical_item"] == 1]["stockout"].sum()
            ),
        )
        .reset_index()
    )
    summary["facility_name"] = summary["facility_id"].map(
        {k: v["name"] for k, v in meta["facilities"].items()}
    )
    summary["facility_beds"] = summary["facility_id"].map(
        {k: v["beds"] for k, v in meta["facilities"].items()}
    )
    summary = summary.sort_values("mean_risk_score", ascending=False)
    return jsonify(summary.to_dict(orient="records"))


@app.route("/api/risk")
def get_risk():
    if latest is None:
        return jsonify({"error": "Model not loaded"}), 503

    facility_id = request.args.get("facility_id")
    risk_level  = request.args.get("risk_level")        # High / Medium / Low
    critical    = request.args.get("critical_only")     # true / false

    filtered = latest.copy()
    if facility_id:
        filtered = filtered[filtered["facility_id"] == facility_id]
    if risk_level:
        filtered = filtered[filtered["risk_level"] == risk_level]
    if critical == "true":
        filtered = filtered[filtered["critical_item"] == 1]

    filtered = filtered.sort_values("risk_score", ascending=False)

    result = filtered[[
        "facility_id", "product_id", "closing_stock",
        "days_of_supply", "risk_score", "risk_level",
        "critical_item", "stockout_rate_4w", "avg_delay_days",
    ]].copy()
    result["risk_score"] = result["risk_score"].round(4)
    result["days_of_supply"] = result["days_of_supply"].round(1)
    result["risk_level"] = result["risk_level"].astype(str)

    return jsonify(result.head(100).to_dict(orient="records"))


@app.route("/api/alerts")
def get_alerts():
    """Return high-priority alerts: high-risk critical items."""
    if latest is None:
        return jsonify({"error": "Model not loaded"}), 503

    alerts = latest[
        (latest["risk_level"] == "High") &
        (latest["critical_item"] == 1)
    ].copy()

    alerts["facility_name"] = alerts["facility_id"].map(
        {k: v["name"] for k, v in meta["facilities"].items()}
    )
    alerts["product_name"] = alerts["product_id"].map(
        {k: v["name"] for k, v in meta["products"].items()}
    )
    alerts = alerts.sort_values("risk_score", ascending=False)

    result = alerts[[
        "facility_id", "facility_name", "product_id", "product_name",
        "risk_score", "closing_stock", "days_of_supply",
        "avg_delay_days", "stockout_rate_4w",
    ]].copy()
    result["risk_score"]    = result["risk_score"].round(4)
    result["days_of_supply"]= result["days_of_supply"].round(1)

    return jsonify({
        "total_alerts": len(result),
        "alerts": result.head(50).to_dict(orient="records"),
    })


@app.route("/api/supplier_delays")
def get_supplier_delays():
    if df is None:
        return jsonify({"error": "Data not loaded"}), 503

    orders = pd.read_csv("data/orders.csv")
    delayed = orders[orders["delay_days"] > 0]
    supplier_summary = (
        delayed.groupby("supplier_id")
        .agg(
            delayed_orders=("order_id", "count"),
            avg_delay_days=("delay_days", "mean"),
            max_delay_days=("delay_days", "max"),
            fulfillment_rate=("fulfilled", "mean"),
        )
        .reset_index()
        .sort_values("avg_delay_days", ascending=False)
    )
    supplier_summary["supplier_name"] = supplier_summary["supplier_id"].map(
        {k: v["name"] for k, v in meta["suppliers"].items()}
    )
    supplier_summary["avg_delay_days"] = supplier_summary["avg_delay_days"].round(2)
    supplier_summary["fulfillment_rate"] = supplier_summary["fulfillment_rate"].round(3)

    return jsonify(supplier_summary.to_dict(orient="records"))


# ── UI ────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print("\n Hospital Supply Chain Risk Dashboard")
    print(" ─────────────────────────────────────")
    print(" Running at: http://localhost:5000\n")
    app.run(debug=True, port=5000)
