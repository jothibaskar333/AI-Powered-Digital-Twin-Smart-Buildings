"""End-to-end tests over real HTTP.

Starts the backend with uvicorn on a free port (temporary database / models),
then checks:
  * HTTP API, background loop and simulation controls
  * the 3D twin page in a headless browser (needs Playwright + Chromium; skipped otherwise)
  * the Streamlit dashboard's page code against the same backend, with the
    Streamlit API replaced by a recording stub (no rendering - Streamlit itself is
    not exercised here)
and that the 3D view, the dashboard and the API show identical values.

If the real FastAPI package is not installed, the test-only stand-in in
tests/support is used (see that file).   Run: python -m unittest tests.test_e2e -v
"""
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import _util
import pandas  # noqa: F401  (loaded before sys.modules is patched for the Streamlit stub)
import requests  # noqa: F401

OUT = Path(os.environ.get("DT_TEST_OUTPUT", _util.TMP / "e2e_output"))
OUT.mkdir(parents=True, exist_ok=True)
ROOMS = ["R101", "R102", "R103", "R201", "R202", "R203"]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Server:
    def __init__(self):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        tmp = Path(tempfile.mkdtemp(prefix="dt_e2e_"))
        env = dict(os.environ, DT_DB_PATH=str(tmp / "e2e.db"), DT_MODELS_DIR=str(tmp / "models"),
                   DT_TICK_SECONDS="1", DT_HISTORY_DAYS="2", PYTHONUNBUFFERED="1")
        if _util.USING_FASTAPI_SHIM:
            env["PYTHONPATH"] = str(_util.ROOT / "tests" / "support") + os.pathsep + env.get("PYTHONPATH", "")
        self.log = open(tmp / "server.log", "w")
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(self.port)],
                                     cwd=_util.BACKEND, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.logpath = tmp / "server.log"
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self.get("/")["loop"]["ticks"] > 0:
                    return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("backend did not start:\n" + self.logpath.read_text())

    def req(self, path, data=None, method=None):
        body = json.dumps(data).encode() if data is not None else None
        r = urllib.request.Request(self.base + path, data=body, method=method or ("POST" if body else "GET"),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw) if resp.headers.get_content_type() == "application/json" else raw.decode()

    def get(self, path):
        return self.req(path)

    def post(self, path, data=None):
        return self.req(path, data if data is not None else {}, "POST")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


SERVER = None


def setUpModule():
    global SERVER
    SERVER = Server()


def tearDownModule():
    if SERVER:
        SERVER.stop()


class Http(unittest.TestCase):
    def test_api_and_loop(self):
        b = SERVER.get("/api/building")
        self.assertEqual([r["id"] for r in b["rooms"]], ROOMS)
        self.assertEqual([s["key"] for s in b["sensors"]],
                         ["temperature", "humidity", "occupancy", "light_level", "power_consumption"])
        SERVER.post("/api/simulation/start")
        h0 = SERVER.get("/api/history?limit=100000")
        t0 = SERVER.get("/api/simulation/status")["sim_time"]
        time.sleep(4.2)
        h1 = SERVER.get("/api/history?limit=100000")
        added = len(h1) - len(h0)
        # one loop only: 1 s tick x 6 rooms -> about 24 rows in 4 s (not 48+ as with duplicate loops)
        self.assertTrue(12 <= added <= 36, added)
        self.assertGreater(SERVER.get("/api/simulation/status")["sim_time"], t0)
        live = SERVER.get("/api/rooms/live")
        self.assertEqual([e["room_id"] for e in live["rooms"]], ROOMS)
        self.assertTrue(all(e["reading"] and e["fault_probability"] is not None for e in live["rooms"]))

    def test_controls_and_errors(self):
        self.assertFalse(SERVER.post("/api/simulation/pause")["running"])
        t = SERVER.get("/api/simulation/status")["sim_time"]
        time.sleep(2.2)
        self.assertEqual(SERVER.get("/api/simulation/status")["sim_time"], t)
        self.assertTrue(SERVER.post("/api/simulation/start")["running"])
        self.assertEqual(SERVER.post("/api/simulation/speed", {"speed": 10})["speed"], 10)
        st = SERVER.post("/api/simulation/scenario", {"room_id": "R202", "kind": "sensor_dropout", "duration_minutes": 30})
        self.assertEqual(st["rooms"]["R202"]["scenario"]["kind"], "sensor_dropout")
        time.sleep(2.2)
        self.assertEqual(SERVER.get("/api/live?room_id=R202")["humidity"], 0.0)
        SERVER.post("/api/simulation/scenario/clear", {"room_id": "R202"})
        SERVER.post("/api/simulation/speed", {"speed": 1})
        for path, data, code in [("/api/simulation/speed", {"speed": 5}, 400),
                                 ("/api/simulation/scenario", {"room_id": "X1", "kind": "power_surge"}, 404),
                                 ("/api/simulation/room_control", {"room_id": "R101", "hvac_mode": "max"}, 400),
                                 ("/api/simulation/speed", {"speed": "fast"}, 422)]:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                SERVER.post(path, data)
            self.assertEqual(cm.exception.code, code, path)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            SERVER.get("/api/live?room_id=R999")
        self.assertEqual(cm.exception.code, 404)

    def test_static_twin_served(self):
        html = SERVER.get("/twin/")
        self.assertIn("3D Digital Twin", html)
        self.assertIn("export class Renderer", SERVER.get("/twin/js/gl.js"))


def _stub_streamlit():
    """MagicMock-based stand-in for the parts of the Streamlit API dashboard/app.py uses.
    Records sensor-card values so they can be compared with the API and the 3D view."""
    st = mock.MagicMock(name="streamlit")
    st.cache_data = lambda **kw: (lambda f: f)
    st.session_state = {}
    st.metrics = []

    def metric_recorder(col):
        col.metric.side_effect = lambda label, value=None, *a, **k: st.metrics.append((label, value))
        return col
    st.columns.side_effect = lambda spec, **kw: [metric_recorder(mock.MagicMock()) for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs.side_effect = lambda names: [mock.MagicMock() for _ in names]
    st.radio.return_value = "All rooms"

    class Stop(Exception):
        pass
    st.Stop = Stop
    st.stop.side_effect = Stop
    st.sidebar.toggle.return_value = False
    st.sidebar.slider.return_value = 5
    components = mock.MagicMock(name="components")
    st.components.v1 = components
    return st, components


class DashboardAndTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SERVER.post("/api/simulation/start")
        time.sleep(2.5)
        SERVER.post("/api/simulation/pause")          # freeze values so all three views can be compared exactly
        time.sleep(1.5)
        cls.live = {r: SERVER.get(f"/api/live?room_id={r}") for r in ROOMS}

    @classmethod
    def tearDownClass(cls):
        SERVER.post("/api/simulation/start")

    def run_dashboard_page(self, page, room):
        st, components = _stub_streamlit()
        components_mod = mock.MagicMock(v1=components)
        with mock.patch.dict(sys.modules, {"streamlit": st, "streamlit.components": components_mod,
                                           "streamlit.components.v1": components}), \
             mock.patch.dict(os.environ, {"DT_API_BASE": SERVER.base}):
            sys.path.insert(0, str(_util.ROOT / "dashboard"))
            try:
                sys.modules.pop("app", None)
                app = importlib.import_module("app")
            finally:
                sys.path.pop(0)
            st.session_state["room_id"] = room
            st.metrics.clear()
            try:
                getattr(app, page)()
            except st.Stop:
                self.fail(f"{page} stopped early (backend unavailable?)")
        return st, components

    def test_dashboard_pages_run_against_backend(self):
        for page in ["page_overview", "page_twin", "page_sensors", "page_ai_analytics", "page_alerts",
                     "page_optimization", "page_history"]:
            with self.subTest(page=page):
                self.run_dashboard_page(page, "R102")

    def test_dashboard_cards_match_api(self):
        for room in ("R101", "R203"):
            st, _ = self.run_dashboard_page("page_sensors", room)
            cards = dict(st.metrics[:5])
            rd = self.live[room]
            self.assertEqual(cards["Temperature"], f"{rd['temperature']} °C")
            self.assertEqual(cards["Humidity"], f"{rd['humidity']} %")
            self.assertEqual(cards["Occupancy"], "Occupied" if rd["occupancy"] else "Empty")
            self.assertEqual(cards["Light Level"], f"{rd['light_level']} lux")
            self.assertEqual(cards["Power"], f"{rd['power_consumption']} kW")

    def test_dashboard_embeds_twin(self):
        _, components = self.run_dashboard_page("page_twin", "R202")
        url = components.iframe.call_args.args[0]
        self.assertTrue(url.startswith(SERVER.base + "/twin/") and url.endswith("room=R202"), url)

    def test_twin_in_browser(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Playwright not installed - browser test not run")
        errors = []
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
            except Exception as exc:
                self.skipTest(f"Chromium not available: {exc}")
            page = browser.new_page(viewport={"width": 1400, "height": 860})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(SERVER.base + "/twin/?room=R101")
            page.wait_for_function("window.__twin && window.__twin.live && document.querySelector('#sensor-table td[data-sensor]')")
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "twin_overview.png"))
            self.assertFalse(page.locator("#fatal").is_visible())

            # 1) 3D panel == API == dashboard card (simulation paused)
            def panel():
                return page.eval_on_selector_all("#sensor-table td[data-sensor]", "els => Object.fromEntries(els.map(e => [e.dataset.sensor, e.textContent]))")
            rd = self.live["R101"]
            self.assertEqual(panel(), {"temperature": f"{rd['temperature']} °C", "humidity": f"{rd['humidity']} %",
                                       "occupancy": "Occupied" if rd["occupancy"] else "Empty",
                                       "light_level": f"{rd['light_level']} lux", "power_consumption": f"{rd['power_consumption']} kW"})
            api_rooms = {e["room_id"]: e for e in SERVER.get("/api/rooms/live")["rooms"]}
            page_rooms = {e["room_id"]: e for e in page.evaluate("window.__twin.live.rooms")}
            for rid in ROOMS:
                self.assertEqual(page_rooms[rid]["reading"], api_rooms[rid]["reading"])
                self.assertEqual(page_rooms[rid]["status"], api_rooms[rid]["status"])
                label = page.locator(f".rlabel[data-room={rid}] .v").text_content()
                self.assertEqual(label, f"{api_rooms[rid]['reading']['temperature']} °C")

            # 2) click a room in the 3D scene -> panel shows that room
            label_hidden = "document.querySelectorAll('.rlabel').forEach(e => e.style.visibility = 'hidden')"
            page.evaluate(label_hidden)
            pt = page.evaluate("window.__twin.project('R203')")
            self.assertEqual(page.evaluate(f"window.__twin.pick({pt['x']}, {pt['y']})"), "R203")
            page.mouse.click(pt["x"], pt["y"])
            page.wait_for_timeout(300)
            self.assertEqual(page.evaluate("window.__twin.selected"), "R203")
            self.assertEqual(page.locator("#room-name").text_content(), "Cafeteria")
            self.assertEqual(page.locator("td[data-sensor=temperature]").text_content(), f"{self.live['R203']['temperature']} °C")

            # 3) controls in the 3D page change the backend simulation
            page.select_option("#scenario-kind", "power_surge")
            page.click("#btn-scenario")
            page.wait_for_timeout(500)
            self.assertEqual(SERVER.get("/api/simulation/status")["rooms"]["R203"]["scenario"]["kind"], "power_surge")
            page.click("#btn-run")                      # Start
            page.wait_for_timeout(500)
            self.assertTrue(SERVER.get("/api/simulation/status")["running"])
            page.wait_for_timeout(2600)
            entry = next(e for e in SERVER.get("/api/rooms/live")["rooms"] if e["room_id"] == "R203")
            self.assertGreater(entry["reading"]["power_consumption"], 2.0)
            page.wait_for_function("window.__twin.live.rooms.find(r => r.room_id === 'R203').reading.power_consumption > 2")
            page.wait_for_timeout(300)
            chip = page.locator("#room-status").text_content().strip().lower()
            live_now = page.evaluate("window.__twin.live.rooms.find(r => r.room_id === 'R203')")
            self.assertEqual(chip, live_now["status"])
            self.assertIn(live_now["status"], ("warning", "critical"))
            page.screenshot(path=str(OUT / "twin_scenario.png"))
            page.click("#btn-clear")
            page.click("#speed button[data-v='60']")
            page.wait_for_timeout(400)
            self.assertEqual(SERVER.get("/api/simulation/status")["speed"], 60)
            page.select_option("#hvac-mode", "on")
            page.wait_for_timeout(400)
            self.assertEqual(SERVER.get("/api/simulation/status")["rooms"]["R203"]["hvac_mode"], "on")
            page.select_option("#hvac-mode", "auto")
            page.click("#btn-run")                      # Pause
            page.wait_for_timeout(400)
            self.assertFalse(SERVER.get("/api/simulation/status")["running"])
            page.click("#speed button[data-v='1']")

            # 4) view controls
            page.locator("#floor-view button", has_text="Ground").click()
            page.wait_for_timeout(400)
            visible = page.eval_on_selector_all(".rlabel", "els => els.filter(e => e.style.display !== 'none').map(e => e.dataset.room)")
            self.assertEqual(sorted(visible), ["R101", "R102", "R103"])
            page.screenshot(path=str(OUT / "twin_ground_floor.png"))
            page.locator("#floor-view button", has_text="All").click()
            page.click("#btn-camera")
            page.mouse.move(700, 430); page.mouse.down(); page.mouse.move(820, 470, steps=6); page.mouse.up()
            page.mouse.wheel(0, -400)
            page.set_viewport_size({"width": 1100, "height": 700})
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / "twin_resized.png"))
            browser.close()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
