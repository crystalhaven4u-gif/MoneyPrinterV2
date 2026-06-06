"""
SQLite persistence for the data farmer: ``.mp/farm.db``.

One row per farmed video. Re-runs UPSERT (insert-or-update) keyed on
``video_id`` so repeated farms refresh metrics instead of duplicating rows.
"""

import os
import sqlite3
from typing import Optional

# Repo root = three levels up from this file (src/longform/storage.py).
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMNS = (
    "video_id",
    "niche",
    "title",
    "duration_s",
    "views",
    "published_at",
    "channel_id",
    "subs",
    "view_velocity",
    "outlier_score",
    "cc_feasibility",
    "score",
    "farmed_at",
)


def get_farm_db_path() -> str:
    """Absolute path to the farm database."""
    return os.path.join(_ROOT_DIR, ".mp", "farm.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_farm_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Optional[str] = None) -> None:
    """Creates the ``videos`` table if it does not exist."""
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id        TEXT PRIMARY KEY,
                niche           TEXT,
                title           TEXT,
                duration_s      INTEGER,
                views           INTEGER,
                published_at    TEXT,
                channel_id      TEXT,
                subs            INTEGER,
                view_velocity   REAL,
                outlier_score   REAL,
                cc_feasibility  INTEGER,
                score           REAL,
                farmed_at       TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def upsert_video(row: dict, db_path: Optional[str] = None) -> None:
    """
    Inserts or updates a farmed-video row keyed on ``video_id``.

    Args:
        row (dict): Must contain every column in ``COLUMNS``.
        db_path (str | None): Optional override for the database path.
    """
    missing = [column for column in COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Row is missing column(s): {', '.join(missing)}.")

    init_db(db_path)
    placeholders = ", ".join("?" for _ in COLUMNS)
    updates = ", ".join(f"{column}=excluded.{column}" for column in COLUMNS if column != "video_id")
    sql = (
        f"INSERT INTO videos ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(video_id) DO UPDATE SET {updates}"
    )
    values = [row[column] for column in COLUMNS]

    connection = _connect(db_path)
    try:
        connection.execute(sql, values)
        connection.commit()
    finally:
        connection.close()


def upsert_videos(rows, db_path: Optional[str] = None) -> int:
    """Upserts many rows; returns the count written."""
    count = 0
    for row in rows:
        upsert_video(row, db_path=db_path)
        count += 1
    return count


def get_all(db_path: Optional[str] = None) -> list:
    """Returns all farmed rows as dicts, highest score first."""
    init_db(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM videos ORDER BY score DESC, views DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_by_niche(niche: str, db_path: Optional[str] = None) -> list:
    """Returns farmed rows for one niche, highest score first."""
    init_db(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM videos WHERE niche = ? ORDER BY score DESC, views DESC",
            (niche,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
