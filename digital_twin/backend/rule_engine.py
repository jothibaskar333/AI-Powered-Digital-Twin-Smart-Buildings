"""
rule_engine.py
Optimization / Feedback engine (Phase 7). Pure rule-based logic - no ML needed here,
by design, per the original project plan. Consumes the latest sensor reading (+ AI
predictions where useful) and decides on automated actions and alerts.

Changes in the 3D-twin build:
  * decide() holds the rule logic as a pure function (no database writes), so the
    simulator can use exactly the same rules when it generates history.
  * evaluate() keeps its original behaviour and return value, and adds an optional
    room_id. To stop the log filling with identical rows every few seconds, a
    decision is written to the database only when it changes for that room
    (HVAC ON/OFF switch, lights newly dimmed, an alert condition newly starting).
  * The predicted-fault threshold is a named constant (see FAULT_PROBABILITY_THRESHOLD).
  * room_status() turns a reading into the normal / warning / critical status used
    by the 3D twin and the dashboard, using only the thresholds below.
"""

from datetime import datetime
from database import insert_action, insert_alert

COMFORT_TEMP_THRESHOLD = 26.0       # deg C
BRIGHTNESS_THRESHOLD = 700          # lux-equivalent
POWER_ALERT_THRESHOLD = 2.0         # kW
# Fault probability = 0.5 - IsolationForest decision score (see main.py), so the
# model's own outlier boundary is 0.5. The original threshold of 0.7 was never
# reached by the model on simulated data, so rule 4 could not fire. 0.52 was chosen
# from the score distribution of the retrained model on an independent simulated
# run (README, "Fault threshold"): it keeps false alarms to about 0.4% of normal
# readings while catching most power-surge and sensor-dropout faults.
FAULT_PROBABILITY_THRESHOLD = 0.52

# last logged state per room, used to log changes only
_last = {}


def reset_state():
    _last.clear()


def decide(reading: dict, fault_probability: float = None):
    """Pure rule evaluation. Returns a list of dicts:
    {"type": "action"|"alert", "action": str, "reason": str, "alert_type": str|None, "severity": str|None}"""
    out = []
    occupancy = reading["occupancy"]
    temperature = reading["temperature"]
    light_level = reading["light_level"]
    power = reading["power_consumption"]

    # Rule 1: HVAC control
    if occupancy == 0 and temperature < COMFORT_TEMP_THRESHOLD:
        out.append({"type": "action", "action": "HVAC OFF",
                    "reason": f"Room empty and temp {temperature}C is within comfort range"})
    elif occupancy == 1 and temperature >= COMFORT_TEMP_THRESHOLD:
        out.append({"type": "action", "action": "HVAC ON",
                    "reason": f"Room occupied and temp {temperature}C exceeds comfort threshold"})

    # Rule 2: Lighting control
    if light_level > BRIGHTNESS_THRESHOLD:
        out.append({"type": "action", "action": "DIM LIGHTS",
                    "reason": f"Ambient brightness {light_level} exceeds threshold"})

    # Rule 3: Power alert
    if power > POWER_ALERT_THRESHOLD:
        out.append({"type": "alert", "action": "ALERT: HIGH POWER", "reason": f"{power}kW draw",
                    "alert_type": "high_power", "severity": "high",
                    "message": f"Power consumption {power}kW exceeds {POWER_ALERT_THRESHOLD}kW threshold"})

    # Rule 4: Predicted fault probability from the AI fault-detection model
    if fault_probability is not None and fault_probability > FAULT_PROBABILITY_THRESHOLD:
        out.append({"type": "alert", "action": "ALERT: PREDICTED FAULT",
                    "reason": f"{fault_probability:.0%} probability",
                    "alert_type": "predicted_fault", "severity": "critical",
                    "message": f"AI model flags {fault_probability:.0%} fault probability"})
    return out


def evaluate(reading: dict, fault_probability: float = None, room_id: str = None):
    """Runs the rule set against one reading, logs changed decisions, and returns
    the list of (action, reason) decisions (all of them, logged or not)."""
    now = reading.get("timestamp", datetime.now().isoformat(timespec="seconds"))
    room_id = room_id if room_id is not None else reading.get("room_id")
    prev = _last.setdefault(room_id, {"hvac": None, "dim": False, "high_power": False, "predicted_fault": False})
    decisions = decide(reading, fault_probability)
    kinds = {d["action"] for d in decisions}
    prefix = f"[{room_id}] " if room_id else ""

    for d in decisions:
        if d["type"] == "action" and d["action"].startswith("HVAC"):
            if prev["hvac"] != d["action"]:
                insert_action(now, d["action"], prefix + d["reason"], room_id=room_id)
                prev["hvac"] = d["action"]
        elif d["action"] == "DIM LIGHTS":
            if not prev["dim"]:
                insert_action(now, d["action"], prefix + d["reason"], room_id=room_id)
        elif d["type"] == "alert":
            if not prev[d["alert_type"]]:
                insert_alert(now, d["alert_type"], prefix + d["message"], d["severity"], room_id=room_id)
    prev["dim"] = "DIM LIGHTS" in kinds
    prev["high_power"] = "ALERT: HIGH POWER" in kinds
    prev["predicted_fault"] = "ALERT: PREDICTED FAULT" in kinds
    return [(d["action"], d["reason"]) for d in decisions]


def room_status(reading: dict, fault_probability: float = None):
    """Status used to colour rooms in the 3D twin and in the dashboard's room table.
    critical : rule 4 condition (AI fault probability above threshold)
    warning  : rule 3 condition (power above threshold), or the room is occupied and
               at or above the comfort temperature (the rule-1 HVAC ON condition)
    normal   : none of the above
    unknown  : no reading yet"""
    if not reading:
        return "unknown", ["No reading yet"]
    reasons, status = [], "normal"
    if fault_probability is not None and fault_probability > FAULT_PROBABILITY_THRESHOLD:
        status = "critical"
        reasons.append(f"AI fault probability {fault_probability:.0%} > {FAULT_PROBABILITY_THRESHOLD:.0%}")
    if reading["power_consumption"] > POWER_ALERT_THRESHOLD:
        status = "critical" if status == "critical" else "warning"
        reasons.append(f"Power {reading['power_consumption']} kW > {POWER_ALERT_THRESHOLD} kW")
    if reading["occupancy"] == 1 and reading["temperature"] >= COMFORT_TEMP_THRESHOLD:
        status = "critical" if status == "critical" else "warning"
        reasons.append(f"Occupied and {reading['temperature']} °C ≥ {COMFORT_TEMP_THRESHOLD} °C")
    if not reasons:
        reasons.append("All readings within thresholds")
    return status, reasons


THRESHOLDS = {
    "comfort_temp_c": COMFORT_TEMP_THRESHOLD,
    "brightness_lux": BRIGHTNESS_THRESHOLD,
    "power_alert_kw": POWER_ALERT_THRESHOLD,
    "fault_probability": FAULT_PROBABILITY_THRESHOLD,
}


if __name__ == "__main__":
    # quick manual test
    from database import init_db
    init_db()
    test_reading = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "occupancy": 0,
        "temperature": 24.5,
        "light_level": 50,
        "power_consumption": 0.4,
    }
    print(evaluate(test_reading))

    test_reading_2 = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "occupancy": 1,
        "temperature": 29.0,
        "light_level": 900,
        "power_consumption": 3.5,
    }
    print(evaluate(test_reading_2, fault_probability=0.85))
