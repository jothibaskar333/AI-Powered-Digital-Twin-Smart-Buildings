"""
main.py
FastAPI backend for the AI-Powered Digital Twin.

Runs a background thread that, every DT_TICK_SECONDS (default 5 s of wall time):
  1. advances the simulated building (simulation.py) by tick x speed seconds of
     simulated time and measures one reading per room
  2. stores the readings in SQLite
  3. runs each reading through the fault model (IsolationForest) and the rule
     engine (Phase 7 optimization)
  4. applies the rule engine's HVAC / lighting decisions back to the simulated
     building (closed loop), when automatic control is enabled

The energy (RandomForest) and occupancy (DecisionTree) models are applied on
request by the /api/predict/... and /api/rooms/live endpoints, using the stored
readings.

The 3D digital twin (frontend/twin) is served by this app at /twin and reads the
same endpoints as the Streamlit dashboard, so both always show the same data.

Run with:  uvicorn main:app --port 8000
"""

import os
import pickle
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import building as B
import rule_engine
from database import (
    init_db, insert_readings, get_latest_reading, get_recent_readings,
    get_recent_alerts, get_recent_actions, get_connection, get_latest_readings_by_room,
    count_room_readings, last_reading_time,
)
from rule_engine import evaluate, room_status
from simulation import BuildingSimulation

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("DT_MODELS_DIR", ROOT / "models"))
TWIN_DIR = ROOT / "frontend" / "twin"
TICK_SECONDS = float(os.environ.get("DT_TICK_SECONDS", "5"))
BOOTSTRAP = os.environ.get("DT_BOOTSTRAP", "1") == "1"
HISTORY_DAYS = int(os.environ.get("DT_HISTORY_DAYS", "10"))
SIM_SEED = int(os.environ.get("DT_SIM_SEED", "42"))
FAULT_FRACTION = float(os.environ.get("DT_SIM_FAULT_FRACTION", "0.02"))
MODEL_NAMES = ["energy_model", "occupancy_model", "fault_model"]

app = FastAPI(title="AI Digital Twin - Smart Building API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_models = {}
_model_errors = {}
_sim: Optional[BuildingSimulation] = None
_sim_lock = threading.RLock()
_loop = {"started": False, "ticks": 0, "last_tick": None, "last_error": None}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_models():
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"{name}.pkl"
        _models[name] = None
        _model_errors.pop(name, None)
        if not path.exists():
            _model_errors[name] = f"{path.name} not found - run train_models.py"
            continue
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
                raise ValueError("unexpected file format")
            _models[name] = bundle
        except Exception as exc:  # incompatible scikit-learn version, corrupt file, ...
            _model_errors[name] = f"could not load {path.name}: {exc}"
    return _models


def _models_need_training():
    """True if a model file is missing, or the models were trained before the
    building simulation existed (no per-room training metadata)."""
    if any(not (MODELS_DIR / f"{n}.pkl").exists() for n in MODEL_NAMES):
        return True
    meta = MODELS_DIR / "metrics.json"
    if not meta.exists():
        return True
    try:
        import json
        return not json.loads(meta.read_text()).get("n_rooms")
    except Exception:
        return True


def _score_to_probability(score: float) -> float:
    """Shared conversion: IsolationForest decision_function score -> 0-1
    'fault probability'. Lower (more negative) score = more anomalous."""
    return round(max(0.0, min(1.0, 0.5 - score)), 3)


def _predict_fault_probability(reading: dict) -> float | None:
    bundle = _models.get("fault_model")
    if bundle is None or not reading:
        return None
    model, features = bundle["model"], bundle["features"]
    row = pd.DataFrame([[reading[f] for f in features]], columns=features)
    score = model.decision_function(row)[0]
    return _score_to_probability(score)


def _predict_power(reading: dict) -> float | None:
    bundle = _models.get("energy_model")
    if bundle is None or not reading:
        return None
    model, features = bundle["model"], bundle["features"]
    ts = pd.to_datetime(reading["timestamp"])
    row = {"temperature": reading["temperature"], "humidity": reading["humidity"],
           "occupancy": reading["occupancy"], "hour": ts.hour, "weekday": ts.weekday()}
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    return round(float(model.predict(X)[0]), 3)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def _create_simulation():
    last = last_reading_time()
    start = datetime.now().replace(microsecond=0)
    if last:
        try:
            start = max(start, datetime.fromisoformat(last) + timedelta(seconds=1))
        except ValueError:
            pass
    sim = BuildingSimulation(seed=SIM_SEED, start=start - timedelta(hours=6), fault_fraction=FAULT_FRACTION)
    sim.advance(6 * 3600)        # warm-up, nothing stored
    return sim


def simulation_tick():
    """One step of the live pipeline. Returns the readings stored (empty if paused)."""
    with _sim_lock:
        if _sim is None or not _sim.running:
            return []
        _sim.advance(TICK_SECONDS * _sim.speed)
        readings = _sim.readings()
        insert_readings(readings)
        for reading in readings:
            fault_prob = _predict_fault_probability(reading)
            actions = evaluate(reading, fault_probability=fault_prob, room_id=reading["room_id"])
            _sim.apply_actions(reading["room_id"], actions)
        _loop["ticks"] += 1
        _loop["last_tick"] = datetime.now().isoformat(timespec="seconds")
        return readings


def background_loop(interval_seconds=TICK_SECONDS):
    while True:
        t0 = time.time()
        try:
            simulation_tick()
            _loop["last_error"] = None
        except Exception as exc:
            _loop["last_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        time.sleep(max(0.0, interval_seconds - (time.time() - t0)))


def bootstrap():
    """Make a fresh installation usable: create history and train the models if
    they are missing. Controlled by DT_BOOTSTRAP (default 1)."""
    init_db()
    if BOOTSTRAP:
        if count_room_readings() == 0:
            from simulator import generate_historical
            print(f"[bootstrap] no per-room history found - simulating {HISTORY_DAYS} days ...")
            generate_historical(days=HISTORY_DAYS)
        if _models_need_training():
            import train_models
            print("[bootstrap] training the three models on the simulated history ...")
            train_models.main()
    load_models()


@app.on_event("startup")
def startup():
    global _sim
    bootstrap()
    with _sim_lock:
        if _sim is None:
            _sim = _create_simulation()
    if not _loop["started"]:
        _loop["started"] = True
        thread = threading.Thread(target=background_loop, daemon=True, name="simulation-loop")
        thread.start()


def _require_sim():
    if _sim is None:
        raise HTTPException(status_code=503, detail="simulation not started yet")
    return _sim


def _check_room(room_id):
    if room_id is not None and room_id not in B.ROOM_BY_ID:
        raise HTTPException(status_code=404, detail=f"unknown room_id {room_id}")


# ---------------------------------------------------------------------------
# Original endpoints (now with optional room_id)
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Digital Twin backend",
            "models_loaded": {n: _models.get(n) is not None for n in MODEL_NAMES},
            "model_errors": _model_errors, "loop": _loop}


@app.get("/api/live")
def api_live(room_id: str = None):
    _check_room(room_id)
    reading = get_latest_reading(room_id)
    return reading or {}


@app.get("/api/history")
def api_history(limit: int = 200, room_id: str = None):
    _check_room(room_id)
    return get_recent_readings(limit, room_id)


@app.get("/api/alerts")
def api_alerts(limit: int = 20, room_id: str = None):
    _check_room(room_id)
    return get_recent_alerts(limit, room_id)


@app.get("/api/actions")
def api_actions(limit: int = 20, room_id: str = None):
    _check_room(room_id)
    return get_recent_actions(limit, room_id)


@app.get("/api/predict/energy")
def predict_energy(room_id: str = None):
    _check_room(room_id)
    reading = get_latest_reading(room_id)
    bundle = _models.get("energy_model")
    if not reading or bundle is None:
        return {"error": "no data or model not trained yet"}
    model, features = bundle["model"], bundle["features"]
    ts = pd.to_datetime(reading["timestamp"])
    row = {
        "temperature": reading["temperature"],
        "humidity": reading["humidity"],
        "occupancy": reading["occupancy"],
        "hour": ts.hour,
        "weekday": ts.weekday(),
    }
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    predicted_kw = float(model.predict(X)[0])
    return {"predicted_power_kw": round(predicted_kw, 3), "based_on": row, "room_id": reading.get("room_id")}


@app.get("/api/predict/occupancy")
def predict_occupancy(hour: int = None, weekday: int = None, room_id: str = None):
    _check_room(room_id)
    bundle = _models.get("occupancy_model")
    reading = get_latest_reading(room_id)
    if bundle is None or not reading:
        return {"error": "no data or model not trained yet"}
    model, features = bundle["model"], bundle["features"]
    # time features come from the reading's (simulated) timestamp unless overridden
    ts = pd.to_datetime(reading["timestamp"])
    row = {
        "temperature": reading["temperature"],
        "light_level": reading["light_level"],
        "hour": hour if hour is not None else ts.hour,
        "weekday": weekday if weekday is not None else ts.weekday(),
    }
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"predicted_occupied": bool(pred), "probability": round(proba, 3), "based_on": row,
            "room_id": reading.get("room_id")}


@app.get("/api/predict/fault")
def predict_fault(room_id: str = None):
    _check_room(room_id)
    reading = get_latest_reading(room_id)
    if not reading:
        return {"error": "no data yet"}
    prob = _predict_fault_probability(reading)
    return {"fault_probability": prob, "based_on": reading}


@app.get("/api/predict/energy/history")
def predict_energy_history(limit: int = 100, room_id: str = None):
    """Applies the already-trained Random Forest energy model to recent
    historical readings, so the dashboard can plot actual vs predicted
    power over time without re-implementing any model logic itself."""
    _check_room(room_id)
    bundle = _models.get("energy_model")
    readings = get_recent_readings(limit, room_id)
    if bundle is None or not readings:
        return {"data": [], "mae_kw": None}
    model, features = bundle["model"], bundle["features"]
    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    X = df[features]
    df["predicted_power_kw"] = model.predict(X).round(3)
    mae = float((df["predicted_power_kw"] - df["power_consumption"]).abs().mean())
    out = df[["timestamp", "power_consumption", "predicted_power_kw"]].rename(
        columns={"power_consumption": "actual_power_kw"}
    )
    out["timestamp"] = out["timestamp"].astype(str)
    return {"data": out.to_dict(orient="records"), "mae_kw": round(mae, 4)}


@app.get("/api/predict/occupancy/history")
def predict_occupancy_history(limit: int = 100, room_id: str = None):
    """Applies the trained Decision Tree occupancy model to recent
    historical readings for an actual-vs-predicted comparison view."""
    _check_room(room_id)
    bundle = _models.get("occupancy_model")
    readings = get_recent_readings(limit, room_id)
    if bundle is None or not readings:
        return {"data": [], "accuracy": None}
    model, features = bundle["model"], bundle["features"]
    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    X = df[features]
    df["predicted_occupied"] = model.predict(X).astype(int)
    df["predicted_probability"] = model.predict_proba(X)[:, 1].round(3)
    accuracy = float((df["predicted_occupied"] == df["occupancy"]).mean())
    out = df[["timestamp", "occupancy", "predicted_occupied", "predicted_probability"]].rename(
        columns={"occupancy": "actual_occupancy"}
    )
    out["timestamp"] = out["timestamp"].astype(str)
    return {"data": out.to_dict(orient="records"), "accuracy": round(accuracy, 4)}


@app.get("/api/predict/fault/history")
def predict_fault_history(limit: int = 100, room_id: str = None):
    """Applies the trained Isolation Forest fault model to recent historical
    readings, producing a fault-probability timeline + flagged anomalies."""
    _check_room(room_id)
    bundle = _models.get("fault_model")
    readings = get_recent_readings(limit, room_id)
    if bundle is None or not readings:
        return {"data": [], "agreement": None}
    model, features = bundle["model"], bundle["features"]
    df = pd.DataFrame(readings)
    X = df[features]
    scores = model.decision_function(X)
    df["fault_probability"] = [_score_to_probability(s) for s in scores]
    df["predicted_anomaly"] = (model.predict(X) == -1).astype(int)
    agreement = float((df["predicted_anomaly"] == df["is_anomaly"]).mean())
    out = df[["timestamp", "is_anomaly", "predicted_anomaly", "fault_probability",
              "temperature", "humidity", "power_consumption"]].rename(
        columns={"is_anomaly": "actual_anomaly_flag"}
    )
    return {"data": out.to_dict(orient="records"), "agreement": round(agreement, 4)}


@app.get("/api/analytics/daily_energy")
def daily_energy(room_id: str = None):
    """Energy use per (simulated) day in kWh, for one room or the whole building.
    Each power sample is multiplied by the time until that room's next sample
    (capped at 15 minutes), i.e. power is integrated over time. The earlier
    version summed raw kW samples, which is not an energy value."""
    _check_room(room_id)
    w, args = ("WHERE room_id = ?", (room_id,)) if room_id else ("WHERE room_id IS NOT NULL", ())
    with get_connection() as conn:
        df = pd.read_sql_query(f"SELECT room_id, timestamp, power_consumption FROM readings {w}", conn, params=args)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["room_id", "timestamp"])
    dt_h = df.groupby("room_id")["timestamp"].diff(-1).abs().dt.total_seconds() / 3600
    df["hours"] = dt_h.fillna(0).clip(upper=0.25)
    df["energy_kwh"] = df["power_consumption"] * df["hours"]
    df["date"] = df["timestamp"].dt.date.astype(str)
    daily = df.groupby("date")["energy_kwh"].sum().round(3).reset_index()
    return daily.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Building / 3D twin endpoints
# ---------------------------------------------------------------------------

@app.get("/api/building")
def api_building():
    """Static building description used by the 3D twin and the dashboard."""
    return {
        "name": "Simulated Smart Building",
        "floors": sorted({r["floor"] for r in B.ROOMS}),
        "floor_names": {r["floor"]: B.floor_name(r["floor"]) for r in B.ROOMS},
        "floor_height": B.FLOOR_HEIGHT, "width": B.BUILDING_WIDTH, "depth": B.BUILDING_DEPTH,
        "room_depth": B.ROOM_DEPTH,
        "rooms": [{k: r[k] for k in ("id", "name", "floor", "x", "z", "w", "d", "furniture", "profile")} for r in B.ROOMS],
        "sensors": B.SENSORS,
        "thresholds": rule_engine.THRESHOLDS,
        "scenarios": B.SCENARIOS,
        "tick_seconds": TICK_SECONDS,
        "data_source": "simulated",
    }


@app.get("/api/rooms/live")
def rooms_live():
    """Latest reading of every room together with the AI outputs computed from
    it by the trained models and the status derived by the rule thresholds.
    This is the single source the 3D twin renders."""
    latest = get_latest_readings_by_room()
    with _sim_lock:
        sim_status = _sim.status() if _sim else None
    rooms = []
    for cfg in B.ROOMS:
        reading = latest.get(cfg["id"])
        fault_prob = _predict_fault_probability(reading) if reading else None
        status, reasons = room_status(reading, fault_prob)
        rooms.append({
            "room_id": cfg["id"], "name": cfg["name"], "floor": cfg["floor"],
            "reading": reading,
            "fault_probability": fault_prob,
            "predicted_power_kw": _predict_power(reading) if reading else None,
            "status": status, "reasons": reasons,
            "state": sim_status["rooms"][cfg["id"]] if sim_status else None,
        })
    sim = None
    if sim_status:
        sim = {k: v for k, v in sim_status.items() if k != "rooms"}
    return {"rooms": rooms, "simulation": sim, "models_loaded": {n: _models.get(n) is not None for n in MODEL_NAMES},
            "model_errors": _model_errors}


class SpeedBody(BaseModel):
    speed: int


class ToggleBody(BaseModel):
    enabled: bool


class ScenarioBody(BaseModel):
    room_id: str
    kind: str
    duration_minutes: float = 15


class RoomBody(BaseModel):
    room_id: str


class RoomControlBody(BaseModel):
    room_id: str
    hvac_mode: Optional[str] = None
    lights_mode: Optional[str] = None


@app.get("/api/simulation/status")
def simulation_status():
    with _sim_lock:
        return _require_sim().status()


@app.post("/api/simulation/start")
def simulation_start():
    with _sim_lock:
        _require_sim().running = True
        return _sim.status()


@app.post("/api/simulation/pause")
def simulation_pause():
    with _sim_lock:
        _require_sim().running = False
        return _sim.status()


@app.post("/api/simulation/reset")
def simulation_reset():
    """Re-initialise all rooms from the seed (stored history is kept)."""
    with _sim_lock:
        _require_sim().reset()
        rule_engine.reset_state()
        return _sim.status()


@app.post("/api/simulation/speed")
def simulation_speed(body: SpeedBody):
    with _sim_lock:
        try:
            _require_sim().set_speed(body.speed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _sim.status()


@app.post("/api/simulation/auto_control")
def simulation_auto_control(body: ToggleBody):
    with _sim_lock:
        _require_sim().auto_control = bool(body.enabled)
        return _sim.status()


@app.post("/api/simulation/scenario")
def simulation_scenario(body: ScenarioBody):
    _check_room(body.room_id)
    with _sim_lock:
        try:
            _require_sim().trigger_scenario(body.room_id, body.kind, body.duration_minutes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _sim.status()


@app.post("/api/simulation/scenario/clear")
def simulation_scenario_clear(body: RoomBody):
    _check_room(body.room_id)
    with _sim_lock:
        _require_sim().clear_scenario(body.room_id)
        return _sim.status()


@app.post("/api/simulation/room_control")
def simulation_room_control(body: RoomControlBody):
    _check_room(body.room_id)
    with _sim_lock:
        try:
            _require_sim().set_room_modes(body.room_id, body.hvac_mode, body.lights_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _sim.status()


# 3D digital twin (static files). Mounted last so it never shadows API routes.
if TWIN_DIR.exists():
    app.mount("/twin", StaticFiles(directory=str(TWIN_DIR), html=True), name="twin")
