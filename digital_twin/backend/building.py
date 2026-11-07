"""
building.py
Static description of the simulated smart building: floors, rooms (monitored
zones), their geometry for the 3D twin, and the behavioural parameters the
simulation engine uses for each room.

This file is the single place where rooms are defined. The backend serves it to
the 3D twin (GET /api/building) so the scene geometry, the room IDs used in the
database and the simulation all come from the same definition.

All parameters are illustrative design values for a simulated building. They are
NOT calibrated against a real building.
"""

# The five sensor types that the existing dashboard displays (unchanged names,
# units and labels). The simulation produces exactly these, per room.
SENSORS = [
    {"key": "temperature", "label": "Temperature", "unit": "°C"},
    {"key": "humidity", "label": "Humidity", "unit": "%"},
    {"key": "occupancy", "label": "Occupancy", "unit": ""},          # 0 = Empty, 1 = Occupied
    {"key": "light_level", "label": "Light Level", "unit": "lux"},
    {"key": "power_consumption", "label": "Power Consumption", "unit": "kW"},
]

FLOOR_HEIGHT = 3.6          # metres, slab to slab
BUILDING_WIDTH = 24.0       # x extent (metres)
BUILDING_DEPTH = 14.0       # z extent (metres); rooms occupy z = 0..10, corridor z = 10..14
ROOM_DEPTH = 10.0

# Occupancy profiles: probability that the room is occupied, by hour of day,
# separately for weekdays and weekends. Used as the long-run target of a
# Markov arrival/departure process (see simulation.py), so occupancy changes
# gradually instead of being re-drawn independently on every reading.
PROFILES = {
    # same shape as the original simulator: 0.85 in working hours, 0.35 around them, 0.05 at night
    "office": {"weekday": [(0, 7, 0.05), (7, 9, 0.35), (9, 19, 0.85), (19, 21, 0.35), (21, 24, 0.05)],
               "weekend": [(0, 24, 0.15)]},
    "lobby": {"weekday": [(0, 7, 0.05), (7, 20, 0.70), (20, 24, 0.10)],
              "weekend": [(0, 9, 0.05), (9, 18, 0.35), (18, 24, 0.05)]},
    "meeting": {"weekday": [(0, 9, 0.02), (9, 12, 0.60), (12, 14, 0.25), (14, 18, 0.60), (18, 24, 0.02)],
                "weekend": [(0, 24, 0.03)]},
    "lab": {"weekday": [(0, 8, 0.10), (8, 20, 0.75), (20, 24, 0.25)],
            "weekend": [(0, 10, 0.05), (10, 18, 0.30), (18, 24, 0.05)]},
    "cafeteria": {"weekday": [(0, 8, 0.02), (8, 11, 0.30), (11, 15, 0.90), (15, 18, 0.30), (18, 24, 0.02)],
                  "weekend": [(0, 11, 0.02), (11, 15, 0.30), (15, 24, 0.02)]},
}

# Rooms. x/z/w/d are the room footprint in metres (x to the east, z to the north);
# floor is 0 (ground) or 1 (first). "window" is the daylight factor of the room's
# glazing (1.0 = full south-facing curtain wall), used for the light-level model.
ROOMS = [
    {"id": "R101", "name": "Lobby", "floor": 0, "x": 0.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "lobby", "window": 1.00, "base_kw": 0.40, "temp_offset": 0.3, "hum_offset": 1.0,
     "furniture": "lobby"},
    {"id": "R102", "name": "Open Office A", "floor": 0, "x": 8.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "office", "window": 0.85, "base_kw": 0.40, "temp_offset": 0.0, "hum_offset": 0.0,
     "furniture": "office"},
    {"id": "R103", "name": "Meeting Room", "floor": 0, "x": 16.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "meeting", "window": 0.70, "base_kw": 0.35, "temp_offset": -0.3, "hum_offset": -1.0,
     "furniture": "meeting"},
    {"id": "R201", "name": "Open Office B", "floor": 1, "x": 0.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "office", "window": 0.90, "base_kw": 0.40, "temp_offset": 0.5, "hum_offset": -0.5,
     "furniture": "office"},
    {"id": "R202", "name": "Research Lab", "floor": 1, "x": 8.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "lab", "window": 0.75, "base_kw": 0.45, "temp_offset": 0.4, "hum_offset": -2.0,
     "furniture": "lab"},
    {"id": "R203", "name": "Cafeteria", "floor": 1, "x": 16.0, "z": 0.0, "w": 8.0, "d": ROOM_DEPTH,
     "profile": "cafeteria", "window": 0.95, "base_kw": 0.40, "temp_offset": 0.2, "hum_offset": 3.0,
     "furniture": "cafeteria"},
]

ROOM_BY_ID = {r["id"]: r for r in ROOMS}

# Controlled test scenarios. These are the same four anomaly types the original
# simulator injected at random, now applied to one room for a set duration.
SCENARIOS = {
    "temp_spike": "Local heat source: temperature rises well above normal",
    "power_surge": "Equipment fault: power draw rises by 2-4 kW",
    "sensor_dropout": "Humidity sensor failure: humidity reads 0",
    "hvac_fault": "HVAC fault: unit draws extra power and stops cooling",
}


def floor_name(floor: int) -> str:
    return "Ground floor" if floor == 0 else f"Floor {floor}"
