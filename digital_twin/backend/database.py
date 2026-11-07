"""
database.py
Simulated 'cloud' data layer (stands in for ThingSpeak / Firebase / ThingsBoard).
Uses SQLite so the whole project runs with zero external accounts or API keys.

3D-twin build: every reading, alert and action now carries the room_id of the
room (zone) it belongs to. Databases created by the earlier version are upgraded
in place by init_db() (ALTER TABLE ... ADD COLUMN); their old rows keep a NULL
room_id and are ignored by the per-room queries.
"""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(os.environ.get("DT_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "digital_twin.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    temperature REAL NOT NULL,
    humidity REAL NOT NULL,
    occupancy INTEGER NOT NULL,
    light_level REAL NOT NULL,
    power_consumption REAL NOT NULL,
    is_anomaly INTEGER DEFAULT 0,
    room_id TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    room_id TEXT
);

CREATE TABLE IF NOT EXISTS actions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    room_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp);
"""

READING_FIELDS = ("timestamp", "temperature", "humidity", "occupancy", "light_level",
                  "power_consumption", "is_anomaly", "room_id")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn):
    """Add room_id to tables created by the pre-3D version of the project."""
    for table in ("readings", "alerts", "actions_log"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "room_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN room_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_room ON readings(room_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_room ON alerts(room_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_room ON actions_log(room_id, id)")


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def insert_reading(timestamp, temperature, humidity, occupancy, light_level,
                   power_consumption, is_anomaly=0, room_id=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO readings
               (timestamp, temperature, humidity, occupancy, light_level,
                power_consumption, is_anomaly, room_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, temperature, humidity, occupancy, light_level,
             power_consumption, is_anomaly, room_id),
        )


def insert_readings(readings):
    """Insert many reading dicts in one transaction (used by the simulation loop)."""
    rows = [tuple(r.get(f, 0 if f == "is_anomaly" else None) for f in READING_FIELDS) for r in readings]
    with get_connection() as conn:
        conn.executemany(
            f"INSERT INTO readings ({', '.join(READING_FIELDS)}) VALUES ({', '.join('?' * len(READING_FIELDS))})",
            rows,
        )


def insert_alert(timestamp, alert_type, message, severity, room_id=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO alerts (timestamp, alert_type, message, severity, room_id)
               VALUES (?, ?, ?, ?, ?)""",
            (timestamp, alert_type, message, severity, room_id),
        )


def insert_action(timestamp, action, reason, room_id=None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO actions_log (timestamp, action, reason, room_id) VALUES (?, ?, ?, ?)",
            (timestamp, action, reason, room_id),
        )


def _where(room_id):
    return ("WHERE room_id = ?", (room_id,)) if room_id else ("", ())


def get_latest_reading(room_id=None):
    w, args = _where(room_id)
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM readings {w} ORDER BY id DESC LIMIT 1", args
        ).fetchone()
        return dict(row) if row else None


def get_latest_readings_by_room():
    """Latest reading of every room, keyed by room_id."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT r.* FROM readings r
               JOIN (SELECT room_id, MAX(id) AS mid FROM readings
                     WHERE room_id IS NOT NULL GROUP BY room_id) m ON r.id = m.mid"""
        ).fetchall()
        return {r["room_id"]: dict(r) for r in rows}


def get_recent_readings(limit=200, room_id=None):
    w, args = _where(room_id)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM readings {w} ORDER BY id DESC LIMIT ?", args + (limit,)
        ).fetchall()
        return [dict(r) for r in rows][::-1]  # chronological order


def get_recent_alerts(limit=20, room_id=None):
    w, args = _where(room_id)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM alerts {w} ORDER BY id DESC LIMIT ?", args + (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_actions(limit=20, room_id=None):
    w, args = _where(room_id)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM actions_log {w} ORDER BY id DESC LIMIT ?", args + (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def count_room_readings():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM readings WHERE room_id IS NOT NULL").fetchone()[0]


def last_reading_time():
    with get_connection() as conn:
        row = conn.execute("SELECT timestamp FROM readings WHERE room_id IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
