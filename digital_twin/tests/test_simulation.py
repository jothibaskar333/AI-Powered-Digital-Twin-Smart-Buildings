"""Simulation engine tests: sensor set, room IDs, determinism, gradual change,
controls and fault scenarios.   Run: python -m unittest discover -s tests -v"""
import unittest
from datetime import datetime

import _util  # noqa: F401  (paths / temp dirs)
import building as B
from simulation import BuildingSimulation, SPEEDS

SENSOR_KEYS = {"temperature", "humidity", "occupancy", "light_level", "power_consumption"}
START = datetime(2026, 10, 7, 9, 0, 0)   # a Wednesday morning


def sim_at(start=START, **kw):
    s = BuildingSimulation(seed=42, start=start, **kw)
    s.advance(3600)
    return s


class SensorsAndRooms(unittest.TestCase):
    def test_only_existing_dashboard_sensors(self):
        self.assertEqual({s["key"] for s in B.SENSORS}, SENSOR_KEYS)
        r = sim_at().readings()[0]
        self.assertEqual(set(r) - {"timestamp", "room_id", "is_anomaly"}, SENSOR_KEYS)

    def test_one_reading_per_room_with_correct_ids(self):
        s = sim_at()
        rs = s.readings()
        self.assertEqual([r["room_id"] for r in rs], [r["id"] for r in B.ROOMS])
        self.assertTrue(all(r["timestamp"] == s.sim_time.isoformat(timespec="seconds") for r in rs))

    def test_value_ranges(self):
        s = sim_at(random_faults=False)
        for _ in range(500):
            s.advance(300)
            for r in s.readings():
                self.assertTrue(20 <= r["humidity"] <= 90)
                self.assertGreaterEqual(r["light_level"], 0)
                self.assertGreaterEqual(r["power_consumption"], 0.1)
                self.assertIn(r["occupancy"], (0, 1))
                self.assertTrue(15 < r["temperature"] < 40)
                self.assertEqual(r["is_anomaly"], 0)


class Determinism(unittest.TestCase):
    def test_same_seed_same_readings(self):
        a, b = sim_at(), sim_at()
        for _ in range(50):
            a.advance(300); b.advance(300)
        self.assertEqual(a.readings(), b.readings())

    def test_different_seed_differs(self):
        a = BuildingSimulation(seed=1, start=START); b = BuildingSimulation(seed=2, start=START)
        a.advance(7200); b.advance(7200)
        self.assertNotEqual(a.readings(), b.readings())


class GradualBehaviour(unittest.TestCase):
    def test_temperature_changes_gradually(self):
        s = sim_at(random_faults=False)
        prev = {r["room_id"]: r["temperature"] for r in s.readings()}
        for _ in range(200):
            s.advance(5)
            for r in s.readings():
                # 5 s of simulated time: only sensor noise (sigma 0.15) plus a tiny drift
                self.assertLess(abs(r["temperature"] - prev[r["room_id"]]), 1.2)
                prev[r["room_id"]] = r["temperature"]

    def test_occupancy_is_persistent_not_redrawn(self):
        s = sim_at(random_faults=False)
        changes, total = 0, 0
        last = {r["room_id"]: r["occupancy"] for r in s.readings()}
        for _ in range(720):                 # one simulated hour at 5 s steps
            s.advance(5)
            for r in s.readings():
                changes += r["occupancy"] != last[r["room_id"]]; total += 1
                last[r["room_id"]] = r["occupancy"]
        self.assertLess(changes / total, 0.02)   # independent draws would change ~25-40% of the time

    def test_daily_profile_office_busier_by_day(self):
        s = BuildingSimulation(seed=3, start=datetime(2026, 10, 5, 0, 0), random_faults=False)  # Monday
        day, night = [], []
        for _ in range(5 * 24 * 12):
            s.advance(300)
            r = next(x for x in s.readings() if x["room_id"] == "R102")
            (day if 10 <= s.sim_time.hour < 17 else night).append(r["occupancy"])
        self.assertGreater(sum(day) / len(day), 0.6)
        self.assertLess(sum(night) / len(night), 0.3)


class Controls(unittest.TestCase):
    def test_reset_restores_seeded_state_but_keeps_clock(self):
        s = sim_at(random_faults=False)
        s.trigger_scenario("R101", "power_surge")
        s.set_room_modes("R102", hvac_mode="on")
        t = s.sim_time
        s.reset()
        self.assertEqual(s.sim_time, t)
        self.assertIsNone(s.rooms["R101"].scenario)
        self.assertEqual(s.rooms["R102"].hvac_mode, "auto")

    def test_speed_validation(self):
        s = sim_at()
        for v in SPEEDS:
            s.set_speed(v); self.assertEqual(s.speed, v)
        with self.assertRaises(ValueError):
            s.set_speed(7)

    def test_manual_lights_and_hvac_change_readings(self):
        night = datetime(2026, 10, 7, 23, 0)
        s = sim_at(start=night, random_faults=False)
        before = {r["room_id"]: r for r in s.readings()}
        s.set_room_modes("R103", hvac_mode="on", lights_mode="on")
        s.advance(60)
        after = {r["room_id"]: r for r in s.readings()}
        self.assertGreater(after["R103"]["light_level"] - before["R103"]["light_level"], 200)   # +300 lux artificial light
        self.assertGreater(after["R103"]["power_consumption"] - before["R103"]["power_consumption"], 0.7)  # HVAC 0.8 + lights 0.1
        s.set_room_modes("R103", hvac_mode="auto", lights_mode="auto")
        self.assertEqual(s.rooms["R103"].hvac_mode, "auto")

    def test_invalid_modes_rejected(self):
        s = sim_at()
        with self.assertRaises(ValueError):
            s.set_room_modes("R101", hvac_mode="max")
        with self.assertRaises(ValueError):
            s.trigger_scenario("R101", "earthquake")

    def test_rule_actions_close_the_loop(self):
        s = sim_at(random_faults=False)
        s.apply_actions("R101", [("HVAC ON", "test")])
        self.assertTrue(s.rooms["R101"].hvac_on)
        s.apply_actions("R101", [("HVAC OFF", "test")])
        self.assertFalse(s.rooms["R101"].hvac_on)
        s.auto_control = False
        s.apply_actions("R101", [("HVAC ON", "test")])
        self.assertFalse(s.rooms["R101"].hvac_on)        # ignored when automatic control is off

    def test_hvac_cools_room(self):
        hot = datetime(2026, 10, 7, 14, 0)
        s = sim_at(start=hot, random_faults=False)
        s.set_room_modes("R102", hvac_mode="off")
        s.advance(2 * 3600)
        warm = s.rooms["R102"].temp
        s.set_room_modes("R102", hvac_mode="on")
        s.advance(3600)
        self.assertLess(s.rooms["R102"].temp, warm - 2)


class Scenarios(unittest.TestCase):
    def setUp(self):
        self.s = sim_at(random_faults=False)

    def reading(self, room):
        return next(r for r in self.s.readings() if r["room_id"] == room)

    def test_power_surge(self):
        base = self.reading("R102")["power_consumption"]
        self.s.trigger_scenario("R102", "power_surge", 10)
        self.s.advance(60)
        r = self.reading("R102")
        self.assertGreater(r["power_consumption"], base + 2)
        self.assertEqual(r["is_anomaly"], 1)

    def test_sensor_dropout(self):
        self.s.trigger_scenario("R203", "sensor_dropout", 10)
        self.s.advance(60)
        self.assertEqual(self.reading("R203")["humidity"], 0.0)

    def test_temp_spike_heats_gradually(self):
        t0 = self.s.rooms["R201"].temp
        self.s.trigger_scenario("R201", "temp_spike", 60)
        self.s.advance(5 * 60)
        t5 = self.s.rooms["R201"].temp
        self.s.advance(25 * 60)
        t30 = self.s.rooms["R201"].temp
        self.assertTrue(t0 < t5 < t30)
        self.assertGreater(t30 - t0, 5)

    def test_hvac_fault(self):
        self.s.trigger_scenario("R202", "hvac_fault", 10)
        self.s.advance(60)
        self.assertEqual(self.s.status()["rooms"]["R202"]["hvac"], "fault")

    def test_scenario_expires_and_clear(self):
        self.s.trigger_scenario("R101", "power_surge", 10)
        self.s.advance(11 * 60)
        self.assertIsNone(self.s.rooms["R101"].scenario)
        self.s.trigger_scenario("R101", "power_surge", 10)
        self.s.clear_scenario("R101")
        self.assertIsNone(self.s.rooms["R101"].scenario)

    def test_random_faults_near_configured_fraction(self):
        s = BuildingSimulation(seed=11, start=START)
        n = a = 0
        for _ in range(10 * 24 * 12):
            s.advance(300)
            for r in s.readings():
                n += 1; a += r["is_anomaly"]
        self.assertTrue(0.01 < a / n < 0.035, a / n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
