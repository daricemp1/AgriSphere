# light sqlite wrapper for history, tasks, costs
import os
import sqlite3
from pathlib import Path

# -------- Resolve DB location (env overrides supported) --------
# AGRISPHERE_DB_PATH takes precedence; otherwise use AGRISPHERE_DB_DIR; else <project_root>/data/agrisphere.db
DB_FILE = os.getenv("AGRISPHERE_DB_PATH")
DB_DIR = os.getenv("AGRISPHERE_DB_DIR")

if DB_FILE:
    DB_PATH = Path(DB_FILE)
else:
    if DB_DIR:
        data_dir = Path(DB_DIR)
    else:
        here = Path(__file__).resolve()
        # If this file lives in a "utils"/"src" folder, parent[1] is likely the project root.
        project_root = (
            here.parents[1]
            if here.parent.name.lower() in {"utils", "src", "app"}
            else here.parent
        )
        data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = data_dir / "agrisphere.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Pragmas for performance + reliability
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")

    # Core tables
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            crop TEXT,
            stage TEXT,
            location TEXT,
            disease TEXT,
            precip REAL,
            tmax REAL,
            tmin REAL,
            wind REAL,
            soil REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            title TEXT NOT NULL,
            due TEXT,
            crop TEXT,
            disease TEXT,
            status TEXT DEFAULT 'open'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS costs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            item TEXT NOT NULL,
            qty REAL DEFAULT 0,
            unit_cost REAL DEFAULT 0,
            crop TEXT,
            disease TEXT,
            note TEXT
        )
        """
    )

    # Helpful indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_due ON tasks(status, due)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_costs_ts ON costs(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts)")


def get_conn() -> sqlite3.Connection:
    """One connection per process; Streamlit will cache this via st.cache_resource in app.py."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn
