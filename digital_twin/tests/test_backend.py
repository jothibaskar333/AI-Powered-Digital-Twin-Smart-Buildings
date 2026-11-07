"""Backend pipeline tests at handler level (no HTTP server, no background thread):
bootstrap -> simulation tick -> storage -> models -> rule engine -> API handlers."""
import json
import pickle
import shutil
import unittest
from datetime import datetime, timedelta

import pandas as pd

import _util
import database
import rule_engine


class Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _util.fresh_db("backend")
        import main
        cls.main = main
        main.bootstrap()                       # simulates DT_HISTORY_DAYS (3) days and trains the 3 models
        main._sim = main._create_simulation()
        rule_engine.reset_state()

    def setUp(self):
        with self.main._sim_lock:
            s = self.main._sim
            s.running, s.speed, s.auto_control = True, 1, True
            for rid in s.rooms:
                s.clear_scenario(rid)
                s.set_room_modes(rid, "auto", "auto")

    def test_bootstrap_history_and_models(self):
        self.assertEqual(database.count_room_readings() >= 3 * 288 * 6, True)
        meta = json.loads((self.main.MODELS_DIR / "metrics.json").read_text())
        self.assertEqual(meta["n_rooms"], 6)
        self.assertTrue(all(self.main._models[n] is not None for n in self.main.MODEL_NAMES), self.main._model_errors)

    def test_tick_stores_one_reading_per_room(self):
        before = database.count_room_readings()
        t0 = self.main._sim.sim_time
        readings = self.main.simulation_tick()
        self.assertEqual(database.count_room_readings(), before + 6)
        self.assertEqual(sorted(r["room_id"] for r in readings), sorted(["R101", "R102", "R103", "R201", "R202", "R203"]))
        self.assertEqual(self.main._sim.sim_time - t0, timedelta(seconds=self.main.TICK_SECONDS))

    def test_pause_stops_readings_and_clock(self):
        self.main.simulation_pause()
        before, t0 = database.count_room_readings(), self.main._sim.sim_time
        self.assertEqual(self.main.simulation_tick(), [])
        self.assertEqual(database.count_room_readings(), before)
        self.assertEqual(self.main._sim.sim_time, t0)
        self.main.simulation_start()
        self.assertEqual(len(self.main.simulation_tick()), 6)

    def test_speed_scales_simulated_time(self):
        self.main.simulation_speed(self.main.SpeedBody(speed=60))
        t0 = self.main._sim.sim_time
        self.main.simulation_tick()
        self.assertEqual(self.main._sim.sim_time - t0, timedelta(seconds=self.main.TICK_SECONDS * 60))

    def test_rooms_live_matches_live_endpoint_and_models(self):
        self.main.simulation_tick()
        data = self.main.rooms_live()
        fault = self.main._models["fault_model"]
        energy = self.main._models["energy_model"]
        for e in data["rooms"]:
            live = self.main.api_live(e["room_id"])
            self.assertEqual(e["reading"], live)                       # same record the dashboard reads
            X = pd.DataFrame([[live[f] for f in fault["features"]]], columns=fault["features"])
            expect = self.main._score_to_probability(fault["model"].decision_function(X)[0])
            self.assertEqual(e["fault_probability"], expect)           # real model output
            self.assertEqual(e["fault_probability"], self.main.predict_fault(e["room_id"])["fault_probability"])
            self.assertEqual(e["predicted_power_kw"], self.main.predict_energy(e["room_id"])["predicted_power_kw"])
            self.assertEqual((e["status"], e["reasons"]), rule_engine.room_status(live, expect))
            ts = pd.to_datetime(live["timestamp"])
            row = {"temperature": live["temperature"], "humidity": live["humidity"], "occupancy": live["occupancy"],
                   "hour": ts.hour, "weekday": ts.weekday()}
            Xe = pd.DataFrame([[row[f] for f in energy["features"]]], columns=energy["features"])
            self.assertAlmostEqual(e["predicted_power_kw"], round(float(energy["model"].predict(Xe)[0]), 3))

    def test_scenario_reaches_storage_rules_and_status(self):
        self.main.simulation_scenario(self.main.ScenarioBody(room_id="R102", kind="power_surge", duration_minutes=10))
        self.main.simulation_tick()
        live = self.main.api_live("R102")
        self.assertGreater(live["power_consumption"], 2.0)
        self.assertEqual(live["is_anomaly"], 1)
        alerts = self.main.api_alerts(5, "R102")
        self.assertTrue(any(a["alert_type"] == "high_power" for a in alerts))
        entry = next(e for e in self.main.rooms_live()["rooms"] if e["room_id"] == "R102")
        self.assertIn(entry["status"], ("warning", "critical"))
        self.assertEqual(entry["state"]["scenario"]["kind"], "power_surge")
        self.main.simulation_scenario_clear(self.main.RoomBody(room_id="R102"))
        self.assertIsNone(self.main.simulation_status()["rooms"]["R102"]["scenario"])

    def test_room_control_and_auto_control(self):
        self.main.simulation_room_control(self.main.RoomControlBody(room_id="R103", hvac_mode="on", lights_mode="off"))
        self.main.simulation_tick()
        st = self.main.simulation_status()["rooms"]["R103"]
        self.assertEqual((st["hvac"], st["lights"]), ("on", "off"))
        self.main.simulation_auto_control(self.main.ToggleBody(enabled=False))
        self.assertFalse(self.main.simulation_status()["auto_control"])

    def test_reset_keeps_history_and_clock(self):
        n = database.count_room_readings(); t = self.main._sim.sim_time
        self.main.simulation_reset()
        self.assertEqual(database.count_room_readings(), n)
        self.assertEqual(self.main._sim.sim_time, t)

    def test_errors(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            self.main.api_live("R999")
        self.assertEqual(cm.exception.status_code, 404)
        with self.assertRaises(HTTPException) as cm:
            self.main.simulation_speed(self.main.SpeedBody(speed=3))
        self.assertEqual(cm.exception.status_code, 400)
        with self.assertRaises(HTTPException) as cm:
            self.main.simulation_scenario(self.main.ScenarioBody(room_id="R101", kind="flood"))
        self.assertEqual(cm.exception.status_code, 400)

    def test_original_endpoints_still_work(self):
        self.main.simulation_tick()
        self.assertEqual(self.main.root()["status"], "ok")
        self.assertIn("temperature", self.main.api_live())
        self.assertEqual(len(self.main.api_history(30)), 30)
        self.assertEqual(len(self.main.api_history(30, "R201")), 30)
        self.assertIsNotNone(self.main.predict_energy_history(50, "R101")["mae_kw"])
        self.assertIsNotNone(self.main.predict_occupancy_history(50, "R101")["accuracy"])
        self.assertIsNotNone(self.main.predict_fault_history(50, "R101")["agreement"])
        occ = self.main.predict_occupancy(hour=3, weekday=6, room_id="R102")
        self.assertEqual((occ["based_on"]["hour"], occ["based_on"]["weekday"]), (3, 6))
        self.assertTrue(len(self.main.daily_energy()) >= 3)
        json.dumps(self.main.rooms_live(), allow_nan=False)            # serialisable

    def test_missing_model_is_reported_not_faked(self):
        path = self.main.MODELS_DIR / "fault_model.pkl"
        backup = path.with_suffix(".bak")
        shutil.move(path, backup)
        try:
            self.main.load_models()
            self.assertIn("fault_model", self.main._model_errors)
            e = self.main.rooms_live()["rooms"][0]
            self.assertIsNone(e["fault_probability"])
            self.assertNotEqual(e["status"], "critical")
            self.assertIsNone(self.main.predict_fault("R101")["fault_probability"])
        finally:
            shutil.move(backup, path)
            self.main.load_models()
        self.assertNotIn("fault_model", self.main._model_errors)


class DailyEnergy(unittest.TestCase):
    def test_power_is_integrated_to_kwh(self):
        _util.fresh_db("energy")
        import main
        t0 = datetime(2026, 10, 7, 10, 0)
        rows = [{"timestamp": (t0 + timedelta(minutes=5 * i)).isoformat(), "room_id": "R101", "temperature": 24,
                 "humidity": 50, "occupancy": 1, "light_level": 300, "power_consumption": 1.0, "is_anomaly": 0}
                for i in range(13)]                                     # 1 kW for one hour
        rows += [{**r, "room_id": "R102", "power_consumption": 2.0} for r in rows]   # 2 kW for one hour
        database.insert_readings(rows)
        self.assertEqual(main.daily_energy("R101"), [{"date": "2026-10-07", "energy_kwh": 1.0}])
        self.assertEqual(main.daily_energy(), [{"date": "2026-10-07", "energy_kwh": 3.0}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
