"""
train_models.py
Trains and saves the three AI models from Phase 6:
  1. Energy Prediction     -> RandomForestRegressor
  2. Occupancy Prediction  -> DecisionTreeClassifier
  3. Fault/Anomaly Detection -> IsolationForest

Run this AFTER simulator.py has generated historical data
(`python simulator.py historical 10`).
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

from database import get_connection

MODELS_DIR = Path(os.environ.get("DT_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataframe():
    """All stored readings. If per-room readings from the building simulation
    exist, only those are used (rows from the pre-3D single-zone version, which
    have no room_id, are skipped so the two data distributions are not mixed)."""
    with get_connection() as conn:
        n_room = conn.execute("SELECT COUNT(*) FROM readings WHERE room_id IS NOT NULL").fetchone()[0]
        where = "WHERE room_id IS NOT NULL" if n_room else ""
        df = pd.read_sql_query(f"SELECT * FROM readings {where} ORDER BY id", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    return df


def train_energy_model(df):
    print("\n--- Training Energy Prediction model (RandomForestRegressor) ---")
    features = ["temperature", "humidity", "occupancy", "hour", "weekday"]
    X = df[features]
    y = df["power_consumption"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE on held-out test set: {mae:.4f} kW")

    with open(MODELS_DIR / "energy_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"Saved -> {MODELS_DIR / 'energy_model.pkl'}")
    return mae


def train_occupancy_model(df):
    print("\n--- Training Occupancy Prediction model (DecisionTreeClassifier) ---")
    features = ["temperature", "light_level", "hour", "weekday"]
    X = df[features]
    y = df["occupancy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on held-out test set: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Empty", "Occupied"]))

    with open(MODELS_DIR / "occupancy_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"Saved -> {MODELS_DIR / 'occupancy_model.pkl'}")
    return acc


def train_fault_model(df):
    print("\n--- Training Fault Detection model (IsolationForest) ---")
    features = ["temperature", "humidity", "power_consumption"]
    X = df[features]

    # contamination = expected proportion of anomalies (matches simulator's ~2% rate)
    model = IsolationForest(contamination=0.02, random_state=42, n_estimators=150)
    model.fit(X)

    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    predicted_anomaly = (preds == -1).astype(int)
    agreement = (predicted_anomaly == df["is_anomaly"]).mean()
    print(f"Agreement with simulator's injected anomaly labels: {agreement:.4f}")
    print(f"Flagged {predicted_anomaly.sum()} anomalies out of {len(df)} readings "
          f"(ground truth had {df['is_anomaly'].sum()}).")

    with open(MODELS_DIR / "fault_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"Saved -> {MODELS_DIR / 'fault_model.pkl'}")
    return agreement


def main():
    df = load_dataframe()
    print(f"Loaded {len(df)} readings for training.")
    if len(df) < 500:
        print("WARNING: very little data. Run "
              "`python simulator.py historical 10` first for better models.")

    mae = train_energy_model(df)
    acc = train_occupancy_model(df)
    agreement = train_fault_model(df)
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_readings": int(len(df)),
        "n_rooms": int(df["room_id"].nunique()) if "room_id" in df else None,
        "n_injected_anomalies": int(df["is_anomaly"].sum()),
        "energy_mae_kw_test": round(float(mae), 4),
        "occupancy_accuracy_test": round(float(acc), 4),
        "fault_label_agreement_all": round(float(agreement), 4),
        "note": "Computed on simulated data by this script; not real-building performance.",
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nAll models trained and saved to /models")
    return metrics


if __name__ == "__main__":
    main()
