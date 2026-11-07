"""
simulator.py
Simulated IoT sensor layer (stands in for DHT22 / PIR / LDR / current sensor + ESP32).

3D-twin build: readings are no longer drawn independently for a single zone.
They are measured from the state of the simulated building (simulation.py),
room by room, using the rooms defined in building.py.

Two modes:
  1. generate_historical(days)  -> bulk-generate past data for AI model training
  2. run_live(interval_seconds) -> stand-alone live stream without the backend
                                   (normally the backend runs the live simulation
                                   itself; do not run both at the same time)

Usage:
    python simulator.py historical 10      # 10 days x 5-min readings for every room
    python simulator.py live
"""

import os
import sys
import time
from datetime import datetime, timedelta

from database import init_db, insert_readings, insert_alert
from simulation import BuildingSimulation
import rule_engine

SEED = int(os.environ.get("DT_SIM_SEED", "42"))
FAULT_FRACTION = float(os.environ.get("DT_SIM_FAULT_FRACTION", "0.02"))


def generate_historical(days=10, interval_minutes=5, seed=SEED, end=None, verbose=True):
    """Simulate `days` of building operation ending at `end` (default: now) and
    store one reading per room every `interval_minutes`.

    The same rule logic the live system uses (rule_engine.decide) drives the
    simulated HVAC and lighting, so the history contains the same kinds of load
    the live building produces. No alerts/actions are written for history."""
    init_db()
    end = (end or datetime.now()).replace(microsecond=0)
    start = end - timedelta(days=days)
    sim = BuildingSimulation(seed=seed, start=start - timedelta(hours=6), fault_fraction=FAULT_FRACTION)
    sim.advance(6 * 3600)                      # warm-up so the first stored readings are settled
    steps = int(days * 24 * 60 / interval_minutes)
    batch, count = [], 0
    for _ in range(steps):
        sim.advance(interval_minutes * 60)
        for reading in sim.readings():
            batch.append(reading)
            decisions = rule_engine.decide(reading)
            sim.apply_actions(reading["room_id"], [(d["action"], d["reason"]) for d in decisions])
        if len(batch) >= 2000:
            insert_readings(batch); count += len(batch); batch = []
    if batch:
        insert_readings(batch); count += len(batch)
    if verbose:
        print(f"Generated {count} historical readings ({len(sim.rooms)} rooms) spanning {days} days.")
    return count


def run_live(interval_seconds=5, max_iterations=None, seed=SEED):
    """Continuously simulate the building in real time and store readings, like
    ESP32 nodes streaming to the cloud. Writes a sensor_anomaly alert for each
    reading taken while a fault episode is active (as the original simulator did)."""
    init_db()
    sim = BuildingSimulation(seed=seed, start=datetime.now() - timedelta(hours=6), fault_fraction=FAULT_FRACTION)
    sim.advance(6 * 3600)
    print(f"Live simulation started ({len(sim.rooms)} rooms). Writing readings every {interval_seconds}s. "
          f"Press Ctrl+C to stop.")
    i = 0
    try:
        while True:
            readings = sim.readings()
            insert_readings(readings)
            for reading in readings:
                decisions = rule_engine.decide(reading)
                sim.apply_actions(reading["room_id"], [(d["action"], d["reason"]) for d in decisions])
                if reading["is_anomaly"]:
                    insert_alert(timestamp=reading["timestamp"], alert_type="sensor_anomaly",
                                 message=f"[{reading['room_id']}] Anomalous reading detected: "
                                         f"temp={reading['temperature']}C, power={reading['power_consumption']}kW",
                                 severity="warning", room_id=reading["room_id"])
            print(f"[{readings[0]['timestamp']}] {len(readings)} room readings stored")
            i += 1
            if max_iterations and i >= max_iterations:
                break
            time.sleep(interval_seconds)
            sim.advance(interval_seconds)
    except KeyboardInterrupt:
        print("\nLive simulation stopped.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        generate_historical(days=days)
    else:
        run_live(interval_seconds=5)
