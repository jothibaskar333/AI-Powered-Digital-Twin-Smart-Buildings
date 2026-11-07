"""Shared test setup. Import this module before any backend module.

* Points the database and model directory at a fresh temporary folder, so tests
  never touch data/ or models/ of a real installation.
* Puts backend/ on sys.path.
* If the real `fastapi` package is not installed, puts tests/support on sys.path
  so the minimal FastAPI stand-in (tests/support/fastapi) is used instead.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
TMP = Path(tempfile.mkdtemp(prefix="dt_tests_"))
os.environ.setdefault("DT_DB_PATH", str(TMP / "test.db"))
os.environ.setdefault("DT_MODELS_DIR", str(TMP / "models"))
os.environ.setdefault("DT_HISTORY_DAYS", "3")

USING_FASTAPI_SHIM = importlib.util.find_spec("fastapi") is None
if USING_FASTAPI_SHIM:
    sys.path.insert(0, str(ROOT / "tests" / "support"))
sys.path.insert(0, str(BACKEND))


def fresh_db(name):
    """Switch the database module to a new empty database file."""
    import database
    database.DB_PATH = TMP / f"{name}.db"
    if database.DB_PATH.exists():
        database.DB_PATH.unlink()
    database.init_db()
    return database.DB_PATH
