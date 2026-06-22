"""
Review-queue gate for the OffSzn content engine.

Distribution safety first: a finished video is NEVER published directly from a
pipeline run. Instead it lands in ``review_queue/pending/`` as an MP4 plus a
metadata JSON sidecar. A human approves or rejects it via ``src/review.py``;
only items that reach ``review_queue/approved/`` may be published, and the
publisher moves them to ``review_queue/published/`` once posted.

Layout (relative to the project root)::

    review_queue/
        pending/    <video_id>.mp4  <video_id>.json
        approved/   ...
        rejected/   ...
        published/  ...

Metadata JSON fields:
    video_id, lane, title, description, hashtags, target_platforms,
    hook_id, prompt_version, utm_campaign, status
"""

import json
import os
import shutil
from typing import Optional

from config import ROOT_DIR

import ledger

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
PUBLISHED = "published"

_STATES = (PENDING, APPROVED, REJECTED, PUBLISHED)

# Fields the metadata sidecar must contain when a video is submitted.
REQUIRED_METADATA_FIELDS = (
    "title",
    "description",
    "hashtags",
    "target_platforms",
    "hook_id",
    "prompt_version",
)


def get_queue_dir(base_dir: Optional[str] = None) -> str:
    """
    Returns the review-queue root directory.
    """
    return base_dir or os.path.join(ROOT_DIR, "review_queue")


def default_target_platforms() -> list:
    """
    Computes the default set of target platforms for a new video: YouTube plus
    any Post Bridge platforms (TikTok/Instagram) when Post Bridge is enabled.

    Returns:
        platforms (list[str]): Ordered, de-duplicated platform names.
    """
    from config import get_post_bridge_config

    platforms = ["youtube"]
    post_bridge = get_post_bridge_config()
    if post_bridge.get("enabled"):
        for platform in post_bridge.get("platforms", []):
            if platform not in platforms:
                platforms.append(platform)
    return platforms


def _state_dir(state: str, base_dir: Optional[str] = None) -> str:
    if state not in _STATES:
        raise ValueError(f"Unknown queue state '{state}'. Expected one of {_STATES}.")
    return os.path.join(get_queue_dir(base_dir), state)


def ensure_dirs(base_dir: Optional[str] = None) -> None:
    """
    Creates all review-queue subdirectories if they do not exist.
    """
    for state in _STATES:
        os.makedirs(_state_dir(state, base_dir), exist_ok=True)


def _video_path(state: str, video_id: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(_state_dir(state, base_dir), f"{video_id}.mp4")


def _meta_path(state: str, video_id: str, base_dir: Optional[str] = None) -> str:
    return os.path.join(_state_dir(state, base_dir), f"{video_id}.json")


def _try_ledger(call) -> None:
    """
    Best-effort ledger update. A missing row (e.g. in isolated tests) must not
    break filesystem bookkeeping.
    """
    try:
        call()
    except KeyError:
        pass


def submit(
    video_path: str,
    metadata: dict,
    video_id: str,
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    move: bool = False,
    update_ledger: bool = True,
) -> dict:
    """
    Submits a finished video to the pending review queue.

    Args:
        video_path (str): Path to the rendered MP4.
        metadata (dict): Metadata for the video. Must include the
            ``REQUIRED_METADATA_FIELDS``.
        video_id (str): The video id (ties the item to its ledger row).
        base_dir (str | None): Optional override for the queue root.
        db_path (str | None): Optional override for the ledger database.
        move (bool): Move the source MP4 instead of copying it.
        update_ledger (bool): Record the submission in the ledger.

    Returns:
        record (dict): The metadata sidecar that was written to disk.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ValueError: If required metadata fields are missing.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(
            f"Metadata is missing required field(s): {', '.join(missing)}."
        )

    ensure_dirs(base_dir)

    destination_video = _video_path(PENDING, video_id, base_dir)
    if move:
        shutil.move(video_path, destination_video)
    else:
        shutil.copy2(video_path, destination_video)

    record = dict(metadata)
    record["video_id"] = video_id
    record["status"] = PENDING

    with open(_meta_path(PENDING, video_id, base_dir), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    if update_ledger:
        _try_ledger(
            lambda: ledger.update_entry(
                video_id,
                db_path=db_path,
                status=PENDING,
                video_path=destination_video,
            )
        )

    return record


def _load_item(state: str, video_id: str, base_dir: Optional[str] = None) -> dict:
    with open(_meta_path(state, video_id, base_dir), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "video_id": video_id,
        "metadata": metadata,
        "video_path": _video_path(state, video_id, base_dir),
        "meta_path": _meta_path(state, video_id, base_dir),
        "state": state,
    }


def list_items(state: str, base_dir: Optional[str] = None) -> list:
    """
    Lists all items in the given queue state, oldest first.

    Args:
        state (str): One of pending/approved/rejected/published.
        base_dir (str | None): Optional override for the queue root.

    Returns:
        items (list[dict]): Items with video_id, metadata and paths.
    """
    directory = _state_dir(state, base_dir)
    if not os.path.isdir(directory):
        return []

    items = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        video_id = name[: -len(".json")]
        try:
            items.append(_load_item(state, video_id, base_dir))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def list_pending(base_dir: Optional[str] = None) -> list:
    """Lists items awaiting review."""
    return list_items(PENDING, base_dir)


def list_approved(base_dir: Optional[str] = None) -> list:
    """Lists items approved for publishing."""
    return list_items(APPROVED, base_dir)


def _move_item(
    video_id: str,
    from_state: str,
    to_state: str,
    base_dir: Optional[str] = None,
    new_status: Optional[str] = None,
) -> dict:
    source_meta = _meta_path(from_state, video_id, base_dir)
    if not os.path.exists(source_meta):
        raise FileNotFoundError(
            f"No '{from_state}' item found for id '{video_id}'."
        )

    ensure_dirs(base_dir)

    with open(source_meta, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if new_status is not None:
        metadata["status"] = new_status

    destination_meta = _meta_path(to_state, video_id, base_dir)
    with open(destination_meta, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.remove(source_meta)

    source_video = _video_path(from_state, video_id, base_dir)
    destination_video = _video_path(to_state, video_id, base_dir)
    if os.path.exists(source_video):
        shutil.move(source_video, destination_video)

    return {
        "video_id": video_id,
        "metadata": metadata,
        "video_path": destination_video,
        "meta_path": destination_meta,
        "state": to_state,
    }


def approve(
    video_id: str,
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Approves a pending item, moving it to ``approved/``.

    Args:
        video_id (str): The id of the pending item.
        base_dir (str | None): Optional override for the queue root.
        db_path (str | None): Optional override for the ledger database.

    Returns:
        item (dict): The moved item.
    """
    item = _move_item(video_id, PENDING, APPROVED, base_dir, new_status=APPROVED)
    _try_ledger(lambda: ledger.set_status(video_id, APPROVED, db_path=db_path))
    return item


def reject(
    video_id: str,
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Rejects a pending item, moving it to ``rejected/``.

    Args:
        video_id (str): The id of the pending item.
        base_dir (str | None): Optional override for the queue root.
        db_path (str | None): Optional override for the ledger database.

    Returns:
        item (dict): The moved item.
    """
    item = _move_item(video_id, PENDING, REJECTED, base_dir, new_status=REJECTED)
    _try_ledger(lambda: ledger.set_status(video_id, REJECTED, db_path=db_path))
    return item


def mark_published(
    video_id: str,
    platform: str,
    platform_post_id: str,
    base_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Moves an approved item to ``published/`` and records the result.

    Args:
        video_id (str): The id of the approved item.
        platform (str): Platform name(s) the video was published to.
        platform_post_id (str): Id(s) returned by the platform.
        base_dir (str | None): Optional override for the queue root.
        db_path (str | None): Optional override for the ledger database.

    Returns:
        item (dict): The moved item.
    """
    item = _move_item(video_id, APPROVED, PUBLISHED, base_dir, new_status=PUBLISHED)
    _try_ledger(
        lambda: ledger.mark_published(
            video_id, platform, platform_post_id, db_path=db_path
        )
    )
    return item
