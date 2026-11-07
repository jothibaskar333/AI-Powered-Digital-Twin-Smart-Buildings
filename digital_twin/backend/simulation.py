"""
simulation.py
Stateful, seedable simulation of the smart building described in building.py.

The building is the source of all simulated sensor readings. Each room keeps a
state (occupied or not, indoor temperature, humidity, HVAC on/off, lights
on/dim/off, any active fault scenario) that evolves continuously over simulated
time. Sensor readings are *measured* from that state, with small sensor noise.

Models used (simple, physically inspired, NOT a validated building-physics model):
  * Occupancy   - two-state Markov process whose long-run probability follows the
                  room's daily profile (building.PROFILES); mean switching time ~15 min.
  * Temperature - first-order relaxation towards an equilibrium set by the outdoor
                  temperature, the room offset and internal gains (people, lights),
                  plus HVAC cooling towards a set-point when the HVAC is running.
  * Humidity    - first-order relaxation towards a daily baseline, raised by
                  occupants and lowered while the HVAC is cooling.
  * Light level - daylight through the room's glazing + artificial lighting
                  (on 300 lux, dimmed 150 lux, off 0) + sensor noise.
  * Power       - room base load + occupant devices + lighting + HVAC + any fault.

The outdoor temperature, humidity and daylight curves and the load values are the
ones the original simulator.py used, so the readings stay on the same scale the
project's three ML models were designed for.

Faults: the four anomaly types of the original simulator (temp_spike, power_surge,
sensor_dropout, hvac_fault) occur as random episodes (about 2% of the time by
default) and can also be triggered on demand for a chosen room. While a fault is
active the room's readings carry is_anomaly = 1 (ground-truth label, used only for
evaluation - the live system never reads it to make decisions).
"""

import math
import random
from datetime import datetime, timedelta

import building as B

SUBSTEP_SECONDS = 60.0          # integration step (simulated seconds)
OCC_SWITCH_RATE = 1.0 / 15.0    # per simulated minute
TAU_ENV_MIN = 90.0              # room <-> outdoor thermal time constant (minutes)
TAU_HVAC_MIN = 20.0             # HVAC cooling time constant (minutes)
TAU_SPIKE_MIN = 15.0            # time constant while a local heat source is active
TAU_HUM_MIN = 45.0
HVAC_SETPOINT = 23.0            # deg C
ARTIFICIAL_LUX = {"off": 0.0, "dim": 150.0, "on": 300.0}
LIGHTING_KW = {"off": 0.0, "dim": 0.05, "on": 0.10}
DEVICES_KW = 0.40               # occupant devices (0.4 + 0.1 lighting = original 0.5 kW occupied load)
HVAC_KW = 0.80                  # original simulator's HVAC load
SPEEDS = [1, 10, 60, 360]       # allowed simulation speed multipliers
FAULT_KINDS = list(B.SCENARIOS)
DEFAULT_MAGNITUDE = {"temp_spike": 9.0, "power_surge": 3.0, "hvac_fault": 2.25, "sensor_dropout": 0.0}


def outdoor_temperature(t: datetime) -> float:
    h = t.hour + t.minute / 60 + t.second / 3600
    return 27 + 5 * math.sin((h - 6) / 24 * 2 * math.pi)


def baseline_humidity(t: datetime) -> float:
    h = t.hour + t.minute / 60
    return 55 + 10 * math.sin((h - 3) / 24 * 2 * math.pi)


def daylight(t: datetime) -> float:
    h = t.hour + t.minute / 60
    return max(0.0, 800 * math.sin((h - 6) / 12 * math.pi)) if 6 <= h <= 18 else 0.0


def occupancy_target(profile: str, t: datetime) -> float:
    table = B.PROFILES[profile]["weekend" if t.weekday() >= 5 else "weekday"]
    h = t.hour + t.minute / 60
    for start, end, p in table:
        if start <= h < end:
            return p
    return table[-1][2]


class RoomState:
    def __init__(self, cfg, t, rng):
        self.cfg = cfg
        self.id = cfg["id"]
        p = occupancy_target(cfg["profile"], t)
        self.occupied = rng.random() < p
        self.temp = outdoor_temperature(t) + cfg["temp_offset"] + (1.5 if self.occupied else 0)
        self.hum = baseline_humidity(t) + cfg["hum_offset"]
        self.hvac_on = False
        self.lights = "on" if self.occupied else "off"
        self.dim_requested = False
        self.hvac_mode = "auto"     # auto | on | off
        self.lights_mode = "auto"   # auto | on | dim | off
        self.scenario = None        # dict(kind, until, magnitude, source)

    def fault(self, kind):
        return self.scenario is not None and self.scenario["kind"] == kind

    def as_dict(self, now):
        sc = None
        if self.scenario:
            remaining = max(0, int((self.scenario["until"] - now).total_seconds() // 60))
            sc = {"kind": self.scenario["kind"], "source": self.scenario["source"],
                  "remaining_minutes": remaining, "description": B.SCENARIOS[self.scenario["kind"]]}
        hvac = "fault" if self.fault("hvac_fault") else ("on" if self.hvac_on else "off")
        return {"room_id": self.id, "occupied": bool(self.occupied), "hvac": hvac, "lights": self.lights,
                "hvac_mode": self.hvac_mode, "lights_mode": self.lights_mode, "scenario": sc}


class BuildingSimulation:
    """Owns the simulated building. Not thread-safe; the backend wraps calls in a lock."""

    def __init__(self, seed=42, start=None, fault_fraction=0.02, random_faults=True):
        self.seed = seed
        self.fault_fraction = fault_fraction
        self.random_faults = random_faults
        self.sim_time = (start or datetime.now()).replace(microsecond=0)
        self.running = True
        self.speed = 1
        self.auto_control = True
        self._init_rooms()

    # ------------------------------------------------------------------ setup
    def _init_rooms(self):
        self.rng = random.Random(self.seed)
        self.rooms = {cfg["id"]: RoomState(cfg, self.sim_time, self.rng) for cfg in B.ROOMS}

    def reset(self):
        """Re-initialise every room from the seed. The simulated clock keeps running
        forward so stored history stays in time order."""
        self._init_rooms()
        self.running = True

    # ------------------------------------------------------------------ dynamics
    def advance(self, sim_seconds: float):
        remaining = float(sim_seconds)
        while remaining > 1e-9:
            dt = min(SUBSTEP_SECONDS, remaining)
            self._substep(dt)
            remaining -= dt

    def _substep(self, dt):
        t = self.sim_time
        dt_min = dt / 60.0
        t_out = outdoor_temperature(t)
        h_base = baseline_humidity(t)
        dl = daylight(t)
        for r in self.rooms.values():
            cfg = r.cfg
            # fault episodes
            if r.scenario and t >= r.scenario["until"]:
                r.scenario = None
            if r.scenario is None and self.random_faults and self.fault_fraction > 0:
                rate = self.fault_fraction / 20.0 / max(1e-6, 1 - self.fault_fraction)  # per minute
                if self.rng.random() < 1 - math.exp(-rate * dt_min):
                    kind = self.rng.choice(FAULT_KINDS)
                    self._start_scenario(r, kind, self.rng.uniform(10, 30), self._random_magnitude(kind), "random")
            # occupancy (Markov process with profile-dependent rates)
            p = occupancy_target(cfg["profile"], t)
            if r.occupied:
                if self.rng.random() < 1 - math.exp(-OCC_SWITCH_RATE * (1 - p) * dt_min):
                    r.occupied = False
            else:
                if self.rng.random() < 1 - math.exp(-OCC_SWITCH_RATE * p * dt_min):
                    r.occupied = True
            # lighting controller
            if r.lights_mode == "auto":
                if not r.occupied:
                    r.lights, r.dim_requested = "off", False
                else:
                    if r.dim_requested and dl * cfg["window"] + ARTIFICIAL_LUX["dim"] < 450:
                        r.dim_requested = False          # too dark to stay dimmed
                    r.lights = "dim" if r.dim_requested else "on"
            else:
                r.lights = r.lights_mode
            # HVAC manual modes
            if r.hvac_mode == "on":
                r.hvac_on = True
            elif r.hvac_mode == "off":
                r.hvac_on = False
            cooling = r.hvac_on and not r.fault("hvac_fault")
            # temperature
            eq = t_out + cfg["temp_offset"] + (1.5 if r.occupied else 0.0) + (0.3 if r.lights == "on" else 0.0)
            tau = TAU_ENV_MIN
            if r.fault("temp_spike"):
                eq += r.scenario["magnitude"]
                tau = TAU_SPIKE_MIN
            if r.fault("hvac_fault"):
                eq += 4.0
            dT = (eq - r.temp) / tau
            if cooling:
                dT += (HVAC_SETPOINT - r.temp) / TAU_HVAC_MIN
            r.temp += dT * dt_min
            # humidity
            heq = h_base + cfg["hum_offset"] + (3.0 if r.occupied else 0.0) - (6.0 if cooling else 0.0)
            r.hum += (heq - r.hum) / TAU_HUM_MIN * dt_min
        self.sim_time = t + timedelta(seconds=dt)

    # ------------------------------------------------------------------ sensing
    def readings(self):
        """Sensor readings for every room at the current simulated time."""
        t = self.sim_time
        dl = daylight(t)
        out = []
        for r in self.rooms.values():
            cfg = r.cfg
            temp = r.temp + self.rng.gauss(0, 0.15)
            hum = min(90.0, max(20.0, r.hum + self.rng.gauss(0, 1.0)))
            if r.fault("sensor_dropout"):
                hum = 0.0
            light = max(0.0, dl * cfg["window"] + ARTIFICIAL_LUX[r.lights] + self.rng.gauss(0, 15))
            power = cfg["base_kw"] + (DEVICES_KW if r.occupied else 0.0) + LIGHTING_KW[r.lights]
            if r.hvac_on:
                power += HVAC_KW
            if r.fault("power_surge") or r.fault("hvac_fault"):
                power += r.scenario["magnitude"]
            power = max(0.1, power + self.rng.gauss(0, 0.04))
            out.append({
                "timestamp": t.isoformat(timespec="seconds"),
                "room_id": r.id,
                "temperature": round(temp, 2),
                "humidity": round(hum, 2),
                "occupancy": int(r.occupied),
                "light_level": round(light, 1),
                "power_consumption": round(power, 3),
                "is_anomaly": int(r.scenario is not None),
            })
        return out

    # ------------------------------------------------------------------ control
    def apply_actions(self, room_id, actions):
        """Apply rule-engine decisions (list of (action, reason)) to a room.
        Only rooms whose HVAC / lighting is in 'auto' mode are affected, and only
        while automatic control is enabled."""
        if not self.auto_control:
            return
        r = self.rooms[room_id]
        for action, _ in actions:
            if action == "HVAC ON" and r.hvac_mode == "auto":
                r.hvac_on = True
            elif action == "HVAC OFF" and r.hvac_mode == "auto":
                r.hvac_on = False
            elif action == "DIM LIGHTS" and r.lights_mode == "auto" and r.occupied:
                r.dim_requested = True
                r.lights = "dim"

    def _random_magnitude(self, kind):
        return {"temp_spike": self.rng.uniform(6, 12), "power_surge": self.rng.uniform(2, 4),
                "hvac_fault": self.rng.uniform(1.5, 3), "sensor_dropout": 0.0}[kind]

    def _start_scenario(self, r, kind, minutes, magnitude, source):
        r.scenario = {"kind": kind, "until": self.sim_time + timedelta(minutes=minutes),
                      "magnitude": magnitude, "source": source}

    def trigger_scenario(self, room_id, kind, duration_minutes=15, magnitude=None):
        if kind not in B.SCENARIOS:
            raise ValueError(f"unknown scenario {kind}")
        r = self.rooms[room_id]
        self._start_scenario(r, kind, float(duration_minutes),
                             DEFAULT_MAGNITUDE[kind] if magnitude is None else float(magnitude), "manual")

    def clear_scenario(self, room_id):
        self.rooms[room_id].scenario = None

    def set_room_modes(self, room_id, hvac_mode=None, lights_mode=None):
        r = self.rooms[room_id]
        if hvac_mode is not None:
            if hvac_mode not in ("auto", "on", "off"):
                raise ValueError("hvac_mode must be auto, on or off")
            r.hvac_mode = hvac_mode
            if hvac_mode != "auto":
                r.hvac_on = hvac_mode == "on"
        if lights_mode is not None:
            if lights_mode not in ("auto", "on", "dim", "off"):
                raise ValueError("lights_mode must be auto, on, dim or off")
            r.lights_mode = lights_mode
            if lights_mode != "auto":
                r.lights = lights_mode

    def set_speed(self, speed):
        speed = int(speed)
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self.speed = speed

    def status(self):
        return {"running": self.running, "speed": self.speed, "speeds": SPEEDS,
                "auto_control": self.auto_control, "random_faults": self.random_faults,
                "seed": self.seed, "sim_time": self.sim_time.isoformat(timespec="seconds"),
                "rooms": {rid: r.as_dict(self.sim_time) for rid, r in self.rooms.items()}}
