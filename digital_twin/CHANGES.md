# Changes in the 3D digital twin version

## New files
| File | Purpose |
|---|---|
| `backend/building.py` | Single definition of the building: 6 rooms on 2 floors, geometry, occupancy profiles, sensor list (the existing 5), fault scenarios |
| `backend/simulation.py` | Stateful, seeded simulation of the rooms; source of all readings; controls (pause, speed, reset, scenarios, HVAC/lights modes, closed-loop rule actions) |
| `frontend/twin/index.html`, `css/twin.css`, `js/app.js` | 3D digital twin page: room panel, controls, legend, labels, polling of the backend |
| `frontend/twin/js/gl.js` | Small dependency-free WebGL renderer (meshes, lighting, orbit camera, picking) |
| `frontend/twin/js/scene.js` | Programmatic building geometry (walls, glazing, doors, furniture, HVAC units, lights, sensor nodes, people) |
| `tests/*.py`, `tests/support/fastapi/` | 45 tests; FastAPI stand-in used only when FastAPI is not installed |
| `tools/fault_threshold_report.py` | Reproduces the evidence for the fault-alert threshold |
| `run_backend.bat`, `run_dashboard.bat`, `.env.example`, `.gitignore`, `requirements-dev.txt`, `models/README.md`, `docs/screenshots/*`, `CHANGES.md` | Startup helpers, settings, documentation |

## Modified files
| File | Change |
|---|---|
| `backend/main.py` | Background loop advances the building simulation and stores one reading per room; applies rule actions back to the rooms; first-start bootstrap (history + training); optional `room_id` on all existing endpoints; new `/api/building`, `/api/rooms/live`, `/api/simulation/*`; serves `/twin`; daily energy integrated to kWh; occupancy prediction uses the reading's timestamp; model-loading errors reported |
| `backend/database.py` | `room_id` column on all tables with in-place migration; per-room queries; bulk insert; configurable DB path |
| `backend/rule_engine.py` | Same rules; pure `decide()`; change-only logging per room; `room_status()`; fault threshold 0.7 → 0.52 (0.7 was unreachable); `THRESHOLDS` exported |
| `backend/simulator.py` | History generation and stand-alone live mode now run the building simulation (with the same rule logic) instead of independent random readings |
| `backend/train_models.py` | Trains on per-room readings (ignores pre-3D rows if both exist); writes `models/metrics.json`; configurable models dir. Algorithms, features and parameters unchanged |
| `dashboard/app.py` | Room selector; all pages per room; All-Rooms table; 3D Digital Twin page (iframe); alerts/actions labelled by room; daily energy in kWh; configurable API URL |
| `README.md` | Rewritten for the new version |

## Removed files
| File | Reason |
|---|---|
| `models/energy_model.pkl`, `models/fault_model.pkl`, `models/occupancy_model.pkl` | Trained on the old single-zone data with scikit-learn 1.8.0. The backend now trains matching models automatically on first start. The original files are in the original ZIP. |
