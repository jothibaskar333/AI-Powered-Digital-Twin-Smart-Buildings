"""Database migration / per-room storage and rule-engine tests."""
import sqlite3
import unittest

import _util
import database
import rule_engine


def R(occ, t, lux, kw, room="R101", ts="2026-10-07T10:00:00"):
    return {"timestamp": ts, "room_id": room, "occupancy": occ, "temperature": t,
            "light_level": lux, "power_consumption": kw, "humidity": 50.0}


class Database(unittest.TestCase):
    def setUp(self):
        _util.fresh_db(self.id().split(".")[-1])

    def test_migrates_pre_3d_database(self):
        path = _util.TMP / "old_schema.db"
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        con.executescript("""
          CREATE TABLE readings (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, temperature REAL NOT NULL,
            humidity REAL NOT NULL, occupancy INTEGER NOT NULL, light_level REAL NOT NULL, power_consumption REAL NOT NULL,
            is_anomaly INTEGER DEFAULT 0);
          CREATE TABLE alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, alert_type TEXT NOT NULL,
            message TEXT NOT NULL, severity TEXT NOT NULL);
          CREATE TABLE actions_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, action TEXT NOT NULL,
            reason TEXT NOT NULL);
          INSERT INTO readings (timestamp, temperature, humidity, occupancy, light_level, power_consumption)
            VALUES ('2026-09-01T10:00:00', 25, 50, 1, 400, 1.0);""")
        con.commit(); con.close()
        database.DB_PATH = path
        database.init_db()
        con = sqlite3.connect(path)
        for t in ("readings", "alerts", "actions_log"):
            self.assertIn("room_id", {r[1] for r in con.execute(f"PRAGMA table_info({t})")})
        self.assertEqual(con.execute("SELECT COUNT(*) FROM readings").fetchone()[0], 1)   # old row kept
        con.close()
        self.assertEqual(database.count_room_readings(), 0)                                # ...but not used per room
        self.assertEqual(database.get_latest_readings_by_room(), {})

    def test_per_room_queries(self):
        rows = []
        for i in range(5):
            for room in ("R101", "R102"):
                rows.append({"timestamp": f"2026-10-07T10:00:0{i}", "room_id": room, "temperature": 20 + i,
                             "humidity": 50, "occupancy": 1, "light_level": 300, "power_consumption": 1, "is_anomaly": 0})
        database.insert_readings(rows)
        self.assertEqual(database.get_latest_reading("R102")["room_id"], "R102")
        self.assertEqual(len(database.get_recent_readings(100, "R101")), 5)
        latest = database.get_latest_readings_by_room()
        self.assertEqual(set(latest), {"R101", "R102"})
        self.assertEqual(latest["R101"]["temperature"], 24)
        hist = database.get_recent_readings(3, "R101")
        self.assertEqual([h["temperature"] for h in hist], [22, 23, 24])   # chronological


class Rules(unittest.TestCase):
    def setUp(self):
        _util.fresh_db(self.id().split(".")[-1])
        rule_engine.reset_state()

    def test_original_rules_unchanged(self):
        self.assertEqual([a for a, _ in rule_engine.evaluate(R(0, 24.5, 50, 0.4))], ["HVAC OFF"])
        rule_engine.reset_state()
        acts = [a for a, _ in rule_engine.evaluate(R(1, 29, 900, 3.5), fault_probability=0.85)]
        self.assertEqual(acts, ["HVAC ON", "DIM LIGHTS", "ALERT: HIGH POWER", "ALERT: PREDICTED FAULT"])

    def test_boundaries(self):
        th = rule_engine.FAULT_PROBABILITY_THRESHOLD
        acts = [d["action"] for d in rule_engine.decide(R(1, 26.0, 700, 2.0), th)]
        self.assertEqual(acts, ["HVAC ON"])            # strict >, inclusive >= for HVAC ON

    def test_logs_only_changes_per_room(self):
        for _ in range(5):
            rule_engine.evaluate(R(0, 24, 50, 0.4))
        self.assertEqual(len(database.get_recent_actions(50)), 1)
        rule_engine.evaluate(R(1, 27, 50, 0.4))               # switches to HVAC ON
        self.assertEqual([a["action"] for a in database.get_recent_actions(50)], ["HVAC ON", "HVAC OFF"])
        for _ in range(3):
            rule_engine.evaluate(R(1, 27, 50, 3.0))            # high power persists -> one alert
        rule_engine.evaluate(R(1, 27, 50, 0.5))
        rule_engine.evaluate(R(1, 27, 50, 3.1))                # new episode -> second alert
        self.assertEqual(len([a for a in database.get_recent_alerts(50) if a["alert_type"] == "high_power"]), 2)
        rule_engine.evaluate(R(0, 24, 50, 0.4, room="R102"))   # other rooms tracked separately
        self.assertEqual(database.get_recent_actions(1)[0]["room_id"], "R102")

    def test_room_status(self):
        th = rule_engine.FAULT_PROBABILITY_THRESHOLD
        self.assertEqual(rule_engine.room_status(None)[0], "unknown")
        self.assertEqual(rule_engine.room_status(R(0, 24, 50, 0.5), 0.3)[0], "normal")
        self.assertEqual(rule_engine.room_status(R(1, 26, 50, 0.5), 0.3)[0], "warning")
        self.assertEqual(rule_engine.room_status(R(0, 30, 50, 0.5), 0.3)[0], "normal")   # empty room: no comfort issue
        self.assertEqual(rule_engine.room_status(R(0, 24, 50, 2.5), 0.3)[0], "warning")
        self.assertEqual(rule_engine.room_status(R(0, 24, 50, 0.5), th + 0.01)[0], "critical")
        self.assertEqual(rule_engine.room_status(R(0, 24, 50, 0.5), th)[0], "normal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
