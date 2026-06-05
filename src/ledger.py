"""
SQLite run ledger for the OffSzn content engine.

One row per generated video, tracked from generation through review and
publishing. Every pipeline stage updates its row. The ledger lives at
``.mp/ledger.db`` at the project root.

Schema (table ``videos``):
    id              TEXT  primary key (uuid hex)
    created_at      TEXT  ISO-8601 UTC timestamp
    lane            TEXT  content lane (e.g. "offszn", "faceless")
    topic           TEXT  generated topic/subject
    prompt_version  TEXT  version of the prompt pack used
    hook_id         TEXT  id of the hook used (see Phase 2)
    voice           TEXT  TTS voice id
    video_path      TEXT  absolute path to the rendered MP4
    status          TEXT  pending | approved | rejected | published
    platform        TEXT  platform(s) the video was published to
    platform_post_id TEXT id(s) returned by the publishing platform
    utm_campaign    TEXT  always "mpv2-{lane}-{video_id}"
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from config import ROOT_DIR

VALID_STATUSES = ("pending", "approved", "rejected", "published")

# Columns a caller is allowed to mutate after the row is created.
_UPDATABLE_COLUMNS = {
    "lane",
    "topic",
    "prompt_version",
    "hook_id",
    "voice",
    "video_path",
    "status",
    "platform",
    "platform_post_id",
    "utm_campaign",
}


def get_ledger_path() -> str:
    """
    Returns the absolute path to the ledger SQLite database.
    """
    return os.path.join(ROOT_DIR, ".mp", "ledger.db")


def build_utm_campaign(lane: str, video_id: str) -> str:
    """
    Builds the canonical utm_campaign value for a video.

    Args:
        lane (str): The content lane.
        video_id (str): The video id.

    Returns:
        utm_campaign (str): "mpv2-{lane}-{video_id}".
    """
    return f"mpv2-{lane}-{video_id}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_ledger_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Optional[str] = None) -> None:
    """
    Creates the ledger table if it does not already exist.

    Args:
        db_path (str | None): Optional override for the database path.

    Returns:
        None
    """
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                id               TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                lane             TEXT,
                topic            TEXT,
                prompt_version   TEXT,
                hook_id          TEXT,
                voice            TEXT,
                video_path       TEXT,
                status           TEXT NOT NULL DEFAULT 'pending',
                platform         TEXT,
                platform_post_id TEXT,
                utm_campaign     TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def create_entry(
    lane: str,
    topic: Optional[str] = None,
    prompt_version: Optional[str] = None,
    hook_id: Optional[str] = None,
    voice: Optional[str] = None,
    video_id: Optional[str] = None,
    status: str = "pending",
    db_path: Optional[str] = None,
) -> str:
    """
    Creates a new ledger row for a generated (or in-progress) video.

    Args:
        lane (str): Content lane.
        topic (str | None): Topic/subject (may be filled in later).
        prompt_version (str | None): Prompt pack version.
        hook_id (str | None): Hook id.
        voice (str | None): TTS voice id.
        video_id (str | None): Optional explicit id. A uuid hex is generated
            when omitted.
        status (str): Initial status. Defaults to "pending".
        db_path (str | None): Optional override for the database path.

    Returns:
        video_id (str): The id of the created row.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Expected one of {VALID_STATUSES}."
        )

    video_id = video_id or uuid4().hex
    utm_campaign = build_utm_campaign(lane, video_id)

    init_db(db_path)
    connection = _connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO videos (
                id, created_at, lane, topic, prompt_version, hook_id,
                voice, status, utm_campaign
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                _utc_now(),
                lane,
                topic,
                prompt_version,
                hook_id,
                voice,
                status,
                utm_campaign,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    return video_id


def update_entry(
    video_id: str,
    db_path: Optional[str] = None,
    **fields,
) -> None:
    """
    Updates one or more mutable columns on an existing ledger row.

    Args:
        video_id (str): The id of the row to update.
        db_path (str | None): Optional override for the database path.
        **fields: Column/value pairs to update.

    Raises:
        ValueError: If an unknown column or invalid status is supplied.
        KeyError: If no row matches ``video_id``.
    """
    if not fields:
        return

    invalid = set(fields) - _UPDATABLE_COLUMNS
    if invalid:
        raise ValueError(
            f"Cannot update unknown column(s): {', '.join(sorted(invalid))}."
        )

    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{fields['status']}'. Expected one of {VALID_STATUSES}."
        )

    assignments = ", ".join(f"{column} = ?" for column in fields)
    values = list(fields.values()) + [video_id]

    init_db(db_path)
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            f"UPDATE videos SET {assignments} WHERE id = ?",
            values,
        )
        connection.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"No ledger entry found for id '{video_id}'.")
    finally:
        connection.close()


def set_status(video_id: str, status: str, db_path: Optional[str] = None) -> None:
    """
    Convenience wrapper to update only a row's status.
    """
    update_entry(video_id, db_path=db_path, status=status)


def mark_published(
    video_id: str,
    platform: str,
    platform_post_id: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Marks a row as published and records the platform and post id.

    Args:
        video_id (str): The id of the row to update.
        platform (str): Platform name(s) the video was published to.
        platform_post_id (str): Id(s) returned by the platform.
        db_path (str | None): Optional override for the database path.
    """
    update_entry(
        video_id,
        db_path=db_path,
        status="published",
        platform=platform,
        platform_post_id=platform_post_id,
    )


def get_entry(video_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """
    Fetches a single ledger row as a dict, or None if it does not exist.
    """
    init_db(db_path)
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM videos WHERE id = ?",
            (video_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def list_entries(
    status: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list:
    """
    Lists ledger rows, optionally filtered by status, oldest first.

    Args:
        status (str | None): Optional status filter.
        db_path (str | None): Optional override for the database path.

    Returns:
        entries (list[dict]): Matching rows as dicts.
    """
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Expected one of {VALID_STATUSES}."
        )

    init_db(db_path)
    connection = _connect(db_path)
    try:
        if status is None:
            rows = connection.execute(
                "SELECT * FROM videos ORDER BY created_at"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM videos WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
