"""Reproduces the evidence for rule_engine.FAULT_PROBABILITY_THRESHOLD.

Simulates 20 days of an INDEPENDENT building run (seed 7, not the training seed),
scores every reading with the trained IsolationForest exactly as the backend does
(p = 0.5 - decision_function), and prints, per fault type, the share of readings
above several thresholds.

Usage (from the project root, after the models exist):
    python tools/fault_threshold_report.py
"""
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import rule_engine  # noqa: E402
from simulation import BuildingSimulation  # noqa: E402

models_dir = Path(os.environ.get("DT_MODELS_DIR", ROOT / "models"))
bundle = pickle.load(open(models_dir / "fault_model.pkl", "rb"))

sim = BuildingSimulation(seed=7, start=datetime(2026, 9, 1))
sim.advance(6 * 3600)
rows = []
for _ in range(20 * 288):
    sim.advance(300)
    for r in sim.readings():
        st = sim.rooms[r["room_id"]]
        r["kind"] = st.scenario["kind"] if st.scenario else "normal"
        rows.append(r)
        sim.apply_actions(r["room_id"], [(d["action"], d["reason"]) for d in rule_engine.decide(r)])
df = pd.DataFrame(rows)
df["p"] = np.clip(0.5 - bundle["model"].decision_function(df[bundle["features"]]), 0, 1)
print(f"{len(df)} readings, by type:", df["kind"].value_counts().to_dict())
print("\nshare of readings with fault probability above threshold")
table = {th: df.assign(f=df["p"] > th).groupby("kind")["f"].mean() for th in (0.5, 0.52, 0.55, 0.6, 0.7)}
print(pd.DataFrame(table).round(3).to_string())
print(f"\ncurrent rule_engine.FAULT_PROBABILITY_THRESHOLD = {rule_engine.FAULT_PROBABILITY_THRESHOLD}")
print(f"highest fault probability of a normal reading: {df.loc[df.kind == 'normal', 'p'].max():.3f}")
