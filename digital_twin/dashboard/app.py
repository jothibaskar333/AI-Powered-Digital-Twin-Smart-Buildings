"""
dashboard/app.py
Digital Twin dashboard (Module 2 — Layers 5-6: Digital Twin + Application).

Talks to the FastAPI backend (Module 3 / `backend/main.py`) over HTTP only.
This file is purely a visualization layer: it does not run the simulator,
train or call AI models directly, or evaluate rules — it just renders
whatever the backend already exposes. All AI predictions, alerts, and
optimization actions shown here are the backend's real, current output.

Run with (after the backend is up):
    streamlit run dashboard/app.py
(make sure `uvicorn main:app --reload --port 8000` is running from /backend first)

3D-twin build: readings now come from a simulated building with several rooms.
A sidebar selector chooses the room shown on every page; the Overview page also
lists all rooms. The new "3D Digital Twin" page embeds the interactive 3D model
served by the backend at /twin, which reads the same API endpoints, so the
dashboard and the 3D view always show the same values.

Page layout (see README / demo script for the recommended walkthrough order):
    3D Digital Twin     - interactive 3D building (frontend/twin, served by the backend)
    Overview            - landing page, building state at a glance
    Sensor Monitoring   - Module 1 (IoT + Data Simulation Layer)
    AI Analytics        - Module 3 (AI & Analytics Engine)
    Alerts & Faults     - rule-engine alerts, grouped by severity
    Optimization        - Module 4 (Optimization & Feedback Engine)
    Historical Analytics- longer-run trends across sensors / alerts
"""

import os
import time
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

API_BASE = os.environ.get("DT_API_BASE", "http://localhost:8000").rstrip("/")
TWIN_URL = os.environ.get("DT_TWIN_URL", API_BASE + "/twin/")
START_CMD = "cd backend && uvicorn main:app --port 8000"

st.set_page_config(
    page_title="AI Digital Twin - Smart Building",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(path, params=None, timeout=4):
    """Raw GET against the backend. Returns None on any failure — callers
    decide how to degrade gracefully instead of a raw traceback."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3, show_spinner=False)
def backend_online():
    return _get("/") is not None


def _p(room, **params):
    """Query parameters including the selected room (None = all rooms)."""
    if room:
        params["room_id"] = room
    return params


@st.cache_data(ttl=60, show_spinner=False)
def get_building():
    return _get("/api/building") or {}


@st.cache_data(ttl=2, show_spinner=False)
def get_rooms_live():
    return _get("/api/rooms/live") or {}


@st.cache_data(ttl=2, show_spinner=False)
def get_live(room=None):
    return _get("/api/live", _p(room)) or {}


@st.cache_data(ttl=2, show_spinner=False)
def get_history(limit=200, room=None):
    return _get("/api/history", _p(room, limit=limit)) or []


@st.cache_data(ttl=2, show_spinner=False)
def get_alerts(limit=50, room=None):
    return _get("/api/alerts", _p(room, limit=limit)) or []


@st.cache_data(ttl=2, show_spinner=False)
def get_actions(limit=50, room=None):
    return _get("/api/actions", _p(room, limit=limit)) or []


@st.cache_data(ttl=2, show_spinner=False)
def get_energy_pred(room=None):
    return _get("/api/predict/energy", _p(room)) or {}


@st.cache_data(ttl=2, show_spinner=False)
def get_occupancy_pred(room=None):
    return _get("/api/predict/occupancy", _p(room)) or {}


@st.cache_data(ttl=2, show_spinner=False)
def get_fault_pred(room=None):
    return _get("/api/predict/fault", _p(room)) or {}


@st.cache_data(ttl=5, show_spinner=False)
def get_energy_pred_history(limit=150, room=None):
    return _get("/api/predict/energy/history", _p(room, limit=limit)) or {"data": [], "mae_kw": None}


@st.cache_data(ttl=5, show_spinner=False)
def get_occupancy_pred_history(limit=150, room=None):
    return _get("/api/predict/occupancy/history", _p(room, limit=limit)) or {"data": [], "accuracy": None}


@st.cache_data(ttl=5, show_spinner=False)
def get_fault_pred_history(limit=150, room=None):
    return _get("/api/predict/fault/history", _p(room, limit=limit)) or {"data": [], "agreement": None}


@st.cache_data(ttl=10, show_spinner=False)
def get_daily_energy(room=None):
    return _get("/api/analytics/daily_energy", _p(room)) or []


# ---------------------------------------------------------------------------
# Small shared UI helpers
# ---------------------------------------------------------------------------


def backend_down_message():
    st.error("**Backend unavailable** — the dashboard can't reach the FastAPI service.")
    st.markdown(
        f"Start it in a terminal, then reload this page:\n```bash\n{START_CMD}\n```"
    )


def require_backend():
    """Call at the top of every page. Stops the page cleanly (no traceback)
    if the FastAPI backend isn't reachable."""
    if not backend_online():
        backend_down_message()
        st.stop()


def ai_badge():
    st.markdown(
        "<span style='background-color:#5B2A86;color:white;padding:2px 10px;"
        "border-radius:10px;font-size:0.8em;font-weight:600;'>AI PREDICTION</span>",
        unsafe_allow_html=True,
    )


def sensor_badge():
    st.markdown(
        "<span style='background-color:#1F6F5C;color:white;padding:2px 10px;"
        "border-radius:10px;font-size:0.8em;font-weight:600;'>SENSOR DATA — NO AI PREDICTION</span>",
        unsafe_allow_html=True,
    )


def selected_room():
    """Room chosen in the sidebar (room_id), or None before the backend is reachable."""
    return st.session_state.get("room_id")


def room_name(room_id):
    for r in get_building().get("rooms", []):
        if r["id"] == room_id:
            return r["name"]
    return room_id or "All rooms"


def room_caption():
    rid = selected_room()
    st.caption(f"Showing room **{room_name(rid)}** ({rid}) – change it in the sidebar.")


def history_df(limit=200, room=None):
    hist = get_history(limit, room)
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def severity_bucket(sev: str) -> str:
    """Maps the rule engine's real severity strings onto the three display
    buckets the dashboard groups alerts into. No new alert conditions are
    introduced here — only existing severities are relabelled for display."""
    sev = (sev or "").lower()
    if sev == "critical":
        return "critical"
    if sev in ("high", "warning"):
        return "warning"
    return "info"


RECOMMENDED_ACTION = {
    "high_power": "Reduce load or inspect equipment drawing excess power.",
    "predicted_fault": "Inspect the equipment associated with the flagged reading.",
    "sensor_anomaly": "Verify sensor connectivity and validate the reading.",
}


def rooms_table():
    """One row per room from /api/rooms/live - the same data the 3D view renders."""
    data = get_rooms_live()
    rows = []
    for e in data.get("rooms", []):
        rd = e.get("reading") or {}
        fp = e.get("fault_probability")
        rows.append({
            "Room": f"{e['name']} ({e['room_id']})",
            "Status": {"normal": "🟢 Normal", "warning": "🟠 Warning", "critical": "🔴 Critical"}.get(e["status"], "⚪ No data"),
            "Temperature (°C)": rd.get("temperature"),
            "Humidity (%)": rd.get("humidity"),
            "Occupancy": ("Occupied" if rd.get("occupancy") else "Empty") if rd else None,
            "Light Level (lux)": rd.get("light_level"),
            "Power (kW)": rd.get("power_consumption"),
            "Fault probability": f"{fp:.0%}" if fp is not None else "--",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        sim = data.get("simulation") or {}
        if sim:
            st.caption(f"Simulated time {sim.get('sim_time', '--').replace('T', ' ')} · "
                       f"{'running' if sim.get('running') else 'paused'} at {sim.get('speed')}× speed")
    else:
        st.info("Waiting for the first room readings…")


# ---------------------------------------------------------------------------
# PAGE: Overview / Digital Twin
# ---------------------------------------------------------------------------


def page_overview():
    require_backend()
    st.title("🏠 Digital Twin — Overview")
    st.caption("Landing page — current state of the simulated building at a glance.")
    room = selected_room()

    live = get_live(room)
    alerts = get_alerts(100)          # building-wide: the status banner covers every room
    energy_pred = get_energy_pred(room)
    occ_pred = get_occupancy_pred(room)
    fault_pred = get_fault_pred(room)

    if not live:
        st.info("Waiting for the first simulated reading… the backend just started.")
        st.stop()

    # --- overall status, from real alert severities only -------------------
    counts = {"critical": 0, "warning": 0, "info": 0}
    for a in alerts:
        counts[severity_bucket(a.get("severity"))] += 1

    if counts["critical"] > 0:
        status_label, status_color = "🔴 ALERT", "#B3261E"
    elif counts["warning"] > 0:
        status_label, status_color = "🟠 WARNING", "#B36B00"
    else:
        status_label, status_color = "🟢 NORMAL", "#1F6F5C"

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background-color:{status_color}20;"
        f"border:1px solid {status_color};font-size:1.3em;font-weight:700;color:{status_color};'>"
        f"Building Status: {status_label}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- all rooms ---------------------------------------------------------
    st.subheader("All Rooms")
    rooms_table()

    st.divider()

    # --- current building state --------------------------------------------
    st.subheader(f"Current State — {room_name(room)}")
    sensor_badge()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temperature", f"{live.get('temperature', '--')} °C")
    c2.metric("Humidity", f"{live.get('humidity', '--')} %")
    c3.metric("Occupancy", "Occupied" if live.get("occupancy") else "Empty")
    c4.metric("Light Level", f"{live.get('light_level', '--')} lux")
    c5.metric("Power Consumption", f"{live.get('power_consumption', '--')} kW")
    st.caption(f"Last simulated reading: {live.get('timestamp', '--')}")

    st.divider()

    # --- quick AI summary ----------------------------------------------------
    st.subheader("Quick AI Summary")
    ai_badge()
    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Predicted Energy Consumption",
        f"{energy_pred.get('predicted_power_kw', '--')} kW",
        help="Random Forest Regressor — see AI Analytics for detail.",
    )
    p2.metric(
        "Predicted Occupancy",
        "Likely Occupied" if occ_pred.get("predicted_occupied") else "Likely Empty",
        f"{occ_pred.get('probability', 0):.0%} confidence" if occ_pred.get("probability") is not None else None,
        help="Decision Tree Classifier — see AI Analytics for detail.",
    )
    p3.metric(
        "Fault / Anomaly Probability",
        f"{fault_pred.get('fault_probability', 0):.0%}" if fault_pred.get("fault_probability") is not None else "--",
        help="Isolation Forest — see AI Analytics for detail.",
    )

    st.divider()

    # --- alerts summary --------------------------------------------------
    st.subheader("Active Alerts Summary")
    a1, a2, a3 = st.columns(3)
    a1.metric("🔴 Critical", counts["critical"])
    a2.metric("🟠 Warning", counts["warning"])
    a3.metric("🔵 Information", counts["info"])
    st.page_link(pages["alerts"], label="Go to Alerts & Faults →", icon="🚨")

    st.divider()

    # --- digital twin status + workflow banner ------------------------------
    st.subheader("Digital Twin Status")
    st.info(
        "**Software-Simulated Digital Twin** — all sensor readings on this "
        "dashboard are measured from the simulated building in "
        "`backend/simulation.py` (one stream per room). No physical "
        "hardware is connected; the rest of the pipeline (storage, AI "
        "models, rule engine, dashboard, 3D view) runs exactly as it would "
        "against real IoT data."
    )
    st.page_link(pages["twin"], label="Open the 3D Digital Twin →", icon="🧊")

    flow_steps = [
        "Simulated Building / Sensors", "Data", "AI Analytics",
        "Digital Twin State", "Optimization / Feedback", "Alerts & Actions",
    ]
    flow_html = "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:6px;'>"
    for i, step in enumerate(flow_steps):
        flow_html += (
            "<div style='background-color:#2E2E38;color:white;padding:8px 14px;"
            "border-radius:8px;font-size:0.85em;font-weight:600;white-space:nowrap;'>"
            f"{step}</div>"
        )
        if i < len(flow_steps) - 1:
            flow_html += "<div style='font-size:1.2em;color:#888;'>→</div>"
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Sensor Monitoring (Module 1)
# ---------------------------------------------------------------------------


def page_sensors():
    require_backend()
    st.title("📡 Sensor Monitoring")
    st.caption("Module 1 — IoT + Data Simulation Layer (`backend/simulation.py`, `backend/simulator.py`, `backend/database.py`)")
    room_caption()
    sensor_badge()
    room = selected_room()

    live = get_live(room)
    if not live:
        st.info("Waiting for data…")
        st.stop()

    st.subheader("Current Sensor Values")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temperature", f"{live.get('temperature', '--')} °C")
    c2.metric("Humidity", f"{live.get('humidity', '--')} %")
    c3.metric("Occupancy", "Occupied" if live.get("occupancy") else "Empty")
    c4.metric("Light Level", f"{live.get('light_level', '--')} lux")
    c5.metric("Power", f"{live.get('power_consumption', '--')} kW")

    st.divider()
    st.subheader("Historical Sensor Graphs")
    sensor_badge()
    st.caption("Simulated IoT sensor data — plotted exactly as generated, with no AI processing.")

    df = history_df(300, room)
    if df.empty:
        st.info("Not enough history yet — check back in a few seconds.")
        return
    df_indexed = df.set_index("timestamp")

    st.markdown("**Temperature over time (°C)**")
    st.line_chart(df_indexed[["temperature"]])

    st.markdown("**Humidity over time (%)**")
    st.line_chart(df_indexed[["humidity"]])

    st.markdown("**Power Consumption over time (kW)**")
    st.line_chart(df_indexed[["power_consumption"]])

    st.markdown("**Light Level over time (lux)**")
    st.line_chart(df_indexed[["light_level"]])

    st.markdown("**Occupancy over time (0 = Empty, 1 = Occupied)**")
    st.line_chart(df_indexed[["occupancy"]])


# ---------------------------------------------------------------------------
# PAGE: AI Analytics (Module 3)
# ---------------------------------------------------------------------------


def page_ai_analytics():
    require_backend()
    st.title("🤖 AI Analytics")
    st.caption("Module 3 — AI & Analytics Engine (`backend/train_models.py`, `backend/main.py`)")
    room_caption()
    room = selected_room()
    st.write(
        "Three trained scikit-learn models run continuously in the backend. "
        "Every chart below is generated by applying those already-trained "
        "models to real simulated readings — the dashboard does not retrain "
        "or re-implement any model logic."
    )

    tab1, tab2, tab3 = st.tabs([
        "🌲 Energy — Random Forest", "🌳 Occupancy — Decision Tree", "🧩 Fault — Isolation Forest",
    ])

    # --- 3.1 Energy prediction -------------------------------------------
    with tab1:
        st.subheader("AI Model: Random Forest Regressor — Energy Prediction")
        ai_badge()
        result = get_energy_pred_history(150, room)
        data, mae = result.get("data", []), result.get("mae_kw")

        m1, m2 = st.columns(2)
        live_pred = get_energy_pred(room)
        m1.metric("Latest AI Predicted Power", f"{live_pred.get('predicted_power_kw', '--')} kW")
        m2.metric("Model MAE (this history window)", f"{mae} kW" if mae is not None else "--")

        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").rename(columns={
                "actual_power_kw": "Actual Sensor Power (kW)",
                "predicted_power_kw": "AI Predicted Power (kW)",
            })
            st.markdown("**AI Prediction — Actual vs Random Forest Predicted Energy**")
            st.line_chart(df[["Actual Sensor Power (kW)", "AI Predicted Power (kW)"]])
        else:
            st.info("Not enough data yet, or the energy model hasn't been trained.")

    # --- 3.2 Occupancy prediction -----------------------------------------
    with tab2:
        st.subheader("AI Model: Decision Tree Classifier — Occupancy Prediction")
        ai_badge()
        result = get_occupancy_pred_history(150, room)
        data, acc = result.get("data", []), result.get("accuracy")

        m1, m2 = st.columns(2)
        live_pred = get_occupancy_pred(room)
        m1.metric(
            "Latest AI Prediction",
            "Likely Occupied" if live_pred.get("predicted_occupied") else "Likely Empty",
            f"{live_pred.get('probability', 0):.0%} confidence" if live_pred.get("probability") is not None else None,
        )
        m2.metric("Model Accuracy (this history window)", f"{acc:.1%}" if acc is not None else "--")

        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").rename(columns={
                "actual_occupancy": "Actual Occupancy",
                "predicted_occupied": "AI Predicted Occupancy",
            })
            st.markdown("**AI Prediction — Actual vs Predicted Occupancy**")
            st.line_chart(df[["Actual Occupancy", "AI Predicted Occupancy"]])
            st.caption("0 = Empty, 1 = Occupied.")
        else:
            st.info("Not enough data yet, or the occupancy model hasn't been trained.")

    # --- 3.3 Fault / anomaly detection --------------------------------------
    with tab3:
        st.subheader("AI Model: Isolation Forest — Fault / Anomaly Detection")
        ai_badge()
        st.caption(
            "How it works: the Isolation Forest scores each reading with "
            "`decision_function`; lower (more negative) scores are more "
            "isolated / anomalous. The backend converts that score into a "
            "0–100% fault probability. Readings the model flags as outliers "
            "(equivalent to roughly >50% probability) are marked anomalous."
        )
        result = get_fault_pred_history(150, room)
        data, agreement = result.get("data", []), result.get("agreement")

        m1, m2 = st.columns(2)
        live_pred = get_fault_pred(room)
        m1.metric(
            "Latest Fault Probability",
            f"{live_pred.get('fault_probability', 0):.0%}" if live_pred.get("fault_probability") is not None else "--",
        )
        m2.metric(
            "Agreement with simulator's injected fault labels",
            f"{agreement:.1%}" if agreement is not None else "--",
        )

        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.markdown("**AI Anomaly Detection — Isolation Forest**")
            chart_df = df.set_index("timestamp").rename(
                columns={"fault_probability": "Fault Probability"}
            )
            st.line_chart(chart_df[["Fault Probability"]])

            anomalies = df[df["predicted_anomaly"] == 1]
            st.markdown(f"**Detected anomalies in this window: {len(anomalies)}**")
            if not anomalies.empty:
                show = anomalies[[
                    "timestamp", "fault_probability", "temperature",
                    "humidity", "power_consumption",
                ]].rename(columns={
                    "fault_probability": "Fault Probability",
                    "temperature": "Temperature (°C)",
                    "humidity": "Humidity (%)",
                    "power_consumption": "Power (kW)",
                })
                st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data yet, or the fault model hasn't been trained.")


# ---------------------------------------------------------------------------
# PAGE: Alerts & Faults
# ---------------------------------------------------------------------------


def alert_card(a):
    sev = severity_bucket(a.get("severity"))
    icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}[sev]
    label = {"critical": "Critical", "warning": "Warning", "info": "Information"}[sev]
    title = (a.get("alert_type") or "event").replace("_", " ").title()
    action = RECOMMENDED_ACTION.get(a.get("alert_type"), "Review the reading and confirm normal operation.")

    where = room_name(a.get("room_id")) if a.get("room_id") else "Building"
    with st.container(border=True):
        st.markdown(f"{icon} **{title}**  ·  {label}  ·  {where}  ·  `{a.get('timestamp', '')}`")
        st.write(f"**What happened / why:** {a.get('message', '')}")
        st.caption(f"Recommended action: {action}")


def page_alerts():
    require_backend()
    st.title("🚨 Alerts & Faults")
    st.caption("Raised by the rule engine (`backend/rule_engine.py`) from real sensor readings and AI predictions.")
    scope = st.radio("Show alerts for", ["All rooms", "Selected room"], horizontal=True, key="alert_scope")
    alerts = get_alerts(100, selected_room() if scope == "Selected room" else None)
    if not alerts:
        st.success("No alerts have been raised. The building is operating normally.")
        return

    buckets = {"critical": [], "warning": [], "info": []}
    for a in alerts:
        buckets[severity_bucket(a.get("severity"))].append(a)

    st.markdown(
        f"**{len(buckets['critical'])}** Critical &nbsp;·&nbsp; "
        f"**{len(buckets['warning'])}** Warning &nbsp;·&nbsp; "
        f"**{len(buckets['info'])}** Information"
    )
    st.divider()

    st.subheader("🔴 Critical")
    st.caption("Conditions requiring immediate attention.")
    if buckets["critical"]:
        for a in buckets["critical"]:
            alert_card(a)
    else:
        st.write("No critical alerts.")

    st.subheader("🟠 Warning")
    st.caption("Conditions that may require attention.")
    if buckets["warning"]:
        for a in buckets["warning"]:
            alert_card(a)
    else:
        st.write("No warnings.")

    st.subheader("🔵 Information")
    st.caption("Normal system events or informational messages.")
    if buckets["info"]:
        for a in buckets["info"]:
            alert_card(a)
    else:
        st.write("No informational events.")


# ---------------------------------------------------------------------------
# PAGE: Optimization & Feedback (Module 4)
# ---------------------------------------------------------------------------


def categorize_action(action: str) -> str:
    a = (action or "").upper()
    if "HVAC" in a:
        return "HVAC"
    if "LIGHT" in a:
        return "Lighting"
    if "PREDICTED FAULT" in a:
        return "Fault Response"
    if "POWER" in a:
        return "Energy"
    return "Other"


def page_optimization():
    require_backend()
    st.title("⚙️ Optimization & Feedback")
    st.caption("Module 4 — Optimization & Feedback Engine (`backend/rule_engine.py`)")
    st.write(
        "This module turns **Sensor Data + AI Predictions** into a "
        "**Decision**. Every entry below is a real action the rule engine "
        "logged — no new rules are shown here beyond what `rule_engine.py` "
        "actually implements."
    )

    scope = st.radio("Show actions for", ["All rooms", "Selected room"], horizontal=True, key="action_scope")
    actions = get_actions(100, selected_room() if scope == "Selected room" else None)
    if not actions:
        st.info("No optimization actions logged yet.")
        return

    categories = ["HVAC", "Lighting", "Energy", "Fault Response", "Other"]
    grouped = {c: [] for c in categories}
    for act in actions:
        grouped[categorize_action(act.get("action"))].append(act)

    active_cats = [c for c in categories if grouped[c]]
    tabs = st.tabs([f"{c} ({len(grouped[c])})" for c in active_cats] or ["No actions"])

    for tab, cat in zip(tabs, active_cats):
        with tab:
            for act in grouped[cat]:
                with st.container(border=True):
                    st.markdown(f"`{act.get('timestamp', '')}`  ·  {room_name(act.get('room_id')) if act.get('room_id') else 'Building'}")
                    st.markdown(f"**Decision:** {act.get('action', '')}")
                    st.write(f"**Reason:** {act.get('reason', '')}")


# ---------------------------------------------------------------------------
# PAGE: 3D Digital Twin
# ---------------------------------------------------------------------------


def page_twin():
    require_backend()
    # The 3D view polls the backend itself every 2 s, so the dashboard's
    # page-level auto-refresh is skipped on this page (it would reload the scene).
    st.session_state["_skip_autorefresh"] = True
    st.title("🧊 3D Digital Twin")
    st.caption(
        "Interactive model of the simulated building. Room colours, the room panel and the "
        "values on every other page come from the same backend endpoints. Readings are "
        "**simulated** — no physical sensors are connected."
    )
    room = selected_room()
    url = f"{TWIN_URL}?room={room}" if room else TWIN_URL
    st.link_button("Open the 3D view in its own browser tab ↗", url)
    components.iframe(url, height=820, scrolling=False)


# ---------------------------------------------------------------------------
# PAGE: Historical Analytics
# ---------------------------------------------------------------------------


def page_history():
    require_backend()
    st.title("📊 Historical Analytics")
    st.caption("Longer-run trends aggregated from real stored data.")

    tab1, tab2 = st.tabs(["Historical Sensor Analytics", "AI Prediction Results"])

    with tab1:
        sensor_badge()
        st.markdown("**Daily Energy Trend — whole building (kWh, simulated days)**")
        daily = get_daily_energy()
        if daily:
            daily_df = pd.DataFrame(daily).set_index("date")
            st.bar_chart(daily_df)
            st.caption("Power readings integrated over time (kW × hours between readings), summed over all rooms.")
        else:
            st.info("Not enough data yet.")

        st.markdown(f"**Occupancy Pattern by Hour of Day — {room_name(selected_room())}**")
        df = history_df(2000, selected_room())
        if not df.empty:
            df["hour"] = df["timestamp"].dt.hour
            by_hour = df.groupby("hour")["occupancy"].mean().reindex(range(24), fill_value=0)
            st.bar_chart(by_hour.rename("Avg. occupancy rate"))
        else:
            st.info("Not enough data yet.")

        st.markdown("**Alert Frequency by Type**")
        alerts = get_alerts(500)
        if alerts:
            adf = pd.DataFrame(alerts)
            counts = adf["alert_type"].value_counts()
            st.bar_chart(counts.rename("Count"))
        else:
            st.info("No alerts logged yet.")

        st.markdown("**System Activity — Actions Logged Over Time**")
        actions = get_actions(500)
        if actions:
            actdf = pd.DataFrame(actions)
            actdf["timestamp"] = pd.to_datetime(actdf["timestamp"])
            actdf["date"] = actdf["timestamp"].dt.date.astype(str)
            per_day = actdf.groupby("date").size()
            st.bar_chart(per_day.rename("Actions"))
        else:
            st.info("No actions logged yet.")

    with tab2:
        ai_badge()
        st.write(
            "For AI-generated actual-vs-predicted results (energy, "
            "occupancy, fault probability), see the **AI Analytics** page — "
            "kept separate here so historical sensor trends are never mixed "
            "with model output in the same chart."
        )
        st.page_link(pages["ai"], label="Go to AI Analytics →", icon="🤖")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "overview": st.Page(page_overview, title="Overview", icon="🏠", default=True),
    "twin": st.Page(page_twin, title="3D Digital Twin", icon="🧊"),
    "sensors": st.Page(page_sensors, title="Sensor Monitoring", icon="📡"),
    "ai": st.Page(page_ai_analytics, title="AI Analytics", icon="🤖"),
    "alerts": st.Page(page_alerts, title="Alerts & Faults", icon="🚨"),
    "optimization": st.Page(page_optimization, title="Optimization & Feedback", icon="⚙️"),
    "history": st.Page(page_history, title="Historical Analytics", icon="📊"),
}

st.sidebar.title("AI-Powered Digital Twin")
st.sidebar.caption("Smart Building — Software-Simulated")

if backend_online():
    st.sidebar.success("Backend connected")
    _rooms = get_building().get("rooms", [])
    if _rooms:
        _ids = [r["id"] for r in _rooms]
        if st.session_state.get("room_id") not in _ids:
            st.session_state["room_id"] = _ids[0]
        st.sidebar.selectbox(
            "Room / zone", _ids, key="room_id",
            format_func=lambda rid: f"{room_name(rid)} ({rid})",
            help="All pages show the readings of this room. The Overview also lists every room.",
        )
else:
    st.sidebar.error("Backend unavailable")
    st.sidebar.code(START_CMD, language="bash")

auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (s)", 2, 30, 5, disabled=not auto_refresh)

st.sidebar.divider()
st.sidebar.markdown("**Four-module architecture**")
st.sidebar.markdown(
    "1. IoT + Data Simulation *(3D building)*\n"
    "2. Digital Twin Dashboard *(this app + 3D view)*\n"
    "3. AI & Analytics Engine\n"
    "4. Optimization & Feedback"
)

st.session_state["_skip_autorefresh"] = False
nav = st.navigation(list(pages.values()))
nav.run()

st.sidebar.divider()
st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh and not st.session_state.get("_skip_autorefresh"):
    time.sleep(refresh_seconds)
    st.rerun()
