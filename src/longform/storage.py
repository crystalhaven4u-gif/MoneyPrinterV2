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


# --------------------------------------------------------------------------- #
# Image generation ledger
# --------------------------------------------------------------------------- #
# One row per generated image. Records which provider served it and the cost
# (0 for the free backends), so spend stays auditable as backends are swapped.
IMAGE_LOG_COLUMNS = (
    "created_at",
    "provider",
    "prompt",
    "aspect_ratio",
    "width",
    "height",
    "output_path",
    "cost",
    "is_placeholder",
    "quality",
    "note",
)


def init_image_log(db_path: Optional[str] = None) -> None:
    """Creates the ``image_log`` table if it does not exist."""
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT,
                provider        TEXT,
                prompt          TEXT,
                aspect_ratio    TEXT,
                width           INTEGER,
                height          INTEGER,
                output_path     TEXT,
                cost            REAL,
                is_placeholder  INTEGER,
                quality         TEXT,
                note            TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def log_image(record: dict, db_path: Optional[str] = None) -> None:
    """
    Appends one image-generation event to the ledger.

    Args:
        record (dict): May contain any of ``IMAGE_LOG_COLUMNS``; missing keys
            are stored as NULL.
        db_path (str | None): Optional override for the database path.
    """
    init_image_log(db_path)
    placeholders = ", ".join("?" for _ in IMAGE_LOG_COLUMNS)
    values = [record.get(column) for column in IMAGE_LOG_COLUMNS]
    sql = (
        f"INSERT INTO image_log ({', '.join(IMAGE_LOG_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    connection = _connect(db_path)
    try:
        connection.execute(sql, values)
        connection.commit()
    finally:
        connection.close()


def get_image_log(db_path: Optional[str] = None) -> list:
    """Returns all image-generation events oldest-first."""
    init_image_log(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute("SELECT * FROM image_log ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Creative ledger (script / hook / packaging runs)
# --------------------------------------------------------------------------- #
# Three tables record what the creative layer produced, so every choice is
# auditable: creative_runs (one row per generated package), hook_variants (all
# cold-open candidates + scores), and title_variants (all titles + chosen flag).
CREATIVE_RUN_COLUMNS = (
    "run_id",
    "created_at",
    "niche",
    "topic",
    "prompt_version",
    "entry_count",
    "word_count",
    "chosen_title",
    "chosen_hook",
    "review_item_path",
)

HOOK_VARIANT_COLUMNS = (
    "run_id",
    "created_at",
    "hook_text",
    "curiosity",
    "depth_pull",
    "payoff_promise",
    "total",
    "is_best",
)

TITLE_VARIANT_COLUMNS = (
    "run_id",
    "created_at",
    "formula",
    "title",
    "is_chosen",
)


def init_creative_tables(db_path: Optional[str] = None) -> None:
    """Creates the creative-ledger tables if they do not exist."""
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS creative_runs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT,
                created_at        TEXT,
                niche             TEXT,
                topic             TEXT,
                prompt_version    TEXT,
                entry_count       INTEGER,
                word_count        INTEGER,
                chosen_title      TEXT,
                chosen_hook       TEXT,
                review_item_path  TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hook_variants (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT,
                created_at      TEXT,
                hook_text       TEXT,
                curiosity       REAL,
                depth_pull      REAL,
                payoff_promise  REAL,
                total           REAL,
                is_best         INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS title_variants (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT,
                created_at  TEXT,
                formula     TEXT,
                title       TEXT,
                is_chosen   INTEGER
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _insert(table: str, columns, record: dict, db_path: Optional[str]) -> None:
    placeholders = ", ".join("?" for _ in columns)
    values = [record.get(column) for column in columns]
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    connection = _connect(db_path)
    try:
        connection.execute(sql, values)
        connection.commit()
    finally:
        connection.close()


def log_creative_run(record: dict, db_path: Optional[str] = None) -> None:
    """Records one generated package (prompt_version, entry_count, choices)."""
    init_creative_tables(db_path)
    _insert("creative_runs", CREATIVE_RUN_COLUMNS, record, db_path)


def log_hook_variant(record: dict, db_path: Optional[str] = None) -> None:
    """Records one cold-open candidate and its scores."""
    init_creative_tables(db_path)
    _insert("hook_variants", HOOK_VARIANT_COLUMNS, record, db_path)


def log_title_variant(record: dict, db_path: Optional[str] = None) -> None:
    """Records one title candidate and whether it was chosen."""
    init_creative_tables(db_path)
    _insert("title_variants", TITLE_VARIANT_COLUMNS, record, db_path)


def update_creative_run_choices(
    run_id: str,
    chosen_title: Optional[str] = None,
    chosen_hook: Optional[str] = None,
    review_item_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Fills in the chosen title/hook + review path on an existing run row."""
    init_creative_tables(db_path)
    connection = _connect(db_path)
    try:
        connection.execute(
            "UPDATE creative_runs SET chosen_title = ?, chosen_hook = ?, "
            "review_item_path = ? WHERE run_id = ?",
            (chosen_title, chosen_hook, review_item_path, run_id),
        )
        connection.commit()
    finally:
        connection.close()


def get_creative_runs(db_path: Optional[str] = None) -> list:
    """Returns all creative runs oldest-first."""
    init_creative_tables(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute("SELECT * FROM creative_runs ORDER BY id").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_hook_variants(run_id: str, db_path: Optional[str] = None) -> list:
    """Returns hook candidates for a run, highest total first."""
    init_creative_tables(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM hook_variants WHERE run_id = ? ORDER BY total DESC",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_title_variants(run_id: str, db_path: Optional[str] = None) -> list:
    """Returns title candidates for a run."""
    init_creative_tables(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM title_variants WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
