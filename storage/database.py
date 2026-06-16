"""Minimal SQLite persistence for the screener.

Saves a ranked watchlist to a ``watchlist`` table, tagging every row with the
timestamp of the run that produced it so historical runs can be compared later.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Database file lives at the repo root and is git-ignored (*.db in .gitignore).
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "trading_agent.db"

_CREATE_WATCHLIST_TABLE = """
CREATE TABLE IF NOT EXISTS watchlist (
    run_timestamp  TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    sector         TEXT,
    momentum_score REAL,
    return_3m      REAL,
    return_6m      REAL
)
"""


def save_watchlist(
    watchlist: pd.DataFrame,
    db_path: Path = DEFAULT_DB_PATH,
    run_timestamp: str | None = None,
) -> str | None:
    """Append a watchlist DataFrame to the ``watchlist`` table.

    Each row is stamped with ``run_timestamp`` (UTC ISO-8601, generated if not
    supplied). Returns the timestamp used, or ``None`` if there was nothing to
    save.
    """
    if watchlist.empty:
        log.warning("Watchlist is empty; nothing to save.")
        return None

    run_timestamp = run_timestamp or datetime.now(timezone.utc).isoformat()

    # Prepend the run timestamp so column order matches the table schema.
    rows = watchlist.copy()
    rows.insert(0, "run_timestamp", run_timestamp)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_CREATE_WATCHLIST_TABLE)
        rows.to_sql("watchlist", connection, if_exists="append", index=False)
        connection.commit()
    finally:
        connection.close()

    log.info("Saved %d watchlist rows to %s (run %s).", len(rows), db_path, run_timestamp)
    return run_timestamp


def load_latest_watchlist(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load the watchlist rows from the most recent run.

    Returns the rows for the latest ``run_timestamp``, sorted best momentum first,
    or an empty DataFrame if the database/table is missing or has no rows.
    """
    if not Path(db_path).exists():
        log.warning("Database %s does not exist; no watchlist to load.", db_path)
        return pd.DataFrame()

    connection = sqlite3.connect(db_path)
    try:
        latest = connection.execute("SELECT MAX(run_timestamp) FROM watchlist").fetchone()[0]
        if latest is None:
            log.warning("Watchlist table is empty.")
            return pd.DataFrame()
        rows = pd.read_sql_query(
            "SELECT * FROM watchlist WHERE run_timestamp = ? ORDER BY momentum_score DESC",
            connection,
            params=(latest,),
        )
    except sqlite3.OperationalError:
        log.warning("No watchlist table found in %s.", db_path)
        return pd.DataFrame()
    finally:
        connection.close()

    log.info("Loaded %d watchlist rows from run %s.", len(rows), latest)
    return rows
