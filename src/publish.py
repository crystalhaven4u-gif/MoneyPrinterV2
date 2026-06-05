#!/usr/bin/env python3
"""
Publisher: drains the approved review queue and posts each item.

This is the ONLY place that publishes, and it publishes ONLY from
``review_queue/approved/``. Items that are successfully posted are moved to
``review_queue/published/`` and their ledger row is marked ``published``.

Supported destinations:
    - YouTube (Data API v3) for items targeting "youtube".
    - TikTok / Instagram (Post Bridge) for items targeting those platforms,
      when Post Bridge is enabled in config.

Usage:
    python src/publish.py            # publish all approved items
    python src/publish.py --dry-run  # show what would be published
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ledger
import review_queue
from config import get_post_bridge_config
from status import error, info, success, warning

POST_BRIDGE_PLATFORMS = {"tiktok", "instagram"}


def _publish_youtube(item: dict):
    """
    Publishes a single item to YouTube via the Data API.

    Returns:
        (platform, post_id) on success, or (None, None) on failure.
    """
    from youtube_upload import upload_video, YouTubeUploadError

    metadata = item["metadata"]
    try:
        video_id = upload_video(
            video_path=item["video_path"],
            title=metadata.get("title", ""),
            description=metadata.get("description", ""),
            tags=metadata.get("hashtags") or None,
        )
        return "youtube", video_id
    except YouTubeUploadError as exc:
        warning(f"YouTube publish failed for {item['video_id']}: {exc}")
        return None, None


def _publish_post_bridge(item: dict, platforms: list):
    """
    Publishes a single item to TikTok/Instagram via Post Bridge.

    Returns:
        (platform_csv, post_id) on success, or (None, None) on failure/skip.
    """
    from classes.PostBridge import PostBridge, PostBridgeClientError
    from post_bridge_integration import (
        build_platform_configurations,
        resolve_social_account_ids,
    )

    config = get_post_bridge_config()
    if not config["enabled"]:
        info("Post Bridge is disabled; skipping TikTok/Instagram for this item.")
        return None, None
    if not config["api_key"]:
        warning("Post Bridge is enabled but no API key is configured.")
        return None, None

    metadata = item["metadata"]
    caption = metadata.get("title", "") or os.path.splitext(
        os.path.basename(item["video_path"])
    )[0]

    client = PostBridge(config["api_key"])
    try:
        account_ids = resolve_social_account_ids(
            client=client,
            configured_account_ids=config["account_ids"],
            platforms=platforms,
            interactive=False,
        )
        if not account_ids:
            warning(
                f"No Post Bridge accounts resolved for {item['video_id']}; skipping."
            )
            return None, None

        media_id = client.upload_media(item["video_path"])
        result = client.create_post(
            caption=caption,
            social_account_ids=account_ids,
            media_ids=[media_id],
            platform_configurations=build_platform_configurations(caption),
        )
        for warning_message in result.get("warnings", []):
            warning(f"Post Bridge warning: {warning_message}")
        return ",".join(platforms), str(result.get("id", ""))
    except PostBridgeClientError as exc:
        warning(f"Post Bridge publish failed for {item['video_id']}: {exc}")
        return None, None


def publish_item(item: dict, dry_run: bool = False) -> bool:
    """
    Publishes one approved item to all of its target platforms.

    Args:
        item (dict): An approved review-queue item.
        dry_run (bool): When True, only report what would happen.

    Returns:
        published (bool): True if at least one platform accepted the post.
    """
    metadata = item["metadata"]
    targets = metadata.get("target_platforms", []) or []
    post_bridge_targets = [p for p in targets if p in POST_BRIDGE_PLATFORMS]

    if dry_run:
        info(
            f"[dry-run] Would publish {item['video_id']} "
            f"to {', '.join(targets) or '(no targets)'}"
        )
        return False

    posted_platforms = []
    posted_ids = []

    if "youtube" in targets:
        platform, post_id = _publish_youtube(item)
        if platform:
            posted_platforms.append(platform)
            posted_ids.append(post_id)

    if post_bridge_targets:
        platform, post_id = _publish_post_bridge(item, post_bridge_targets)
        if platform:
            posted_platforms.append(platform)
            posted_ids.append(post_id)

    if not posted_platforms:
        warning(f"Nothing published for {item['video_id']}; leaving in approved/.")
        return False

    review_queue.mark_published(
        video_id=item["video_id"],
        platform=",".join(posted_platforms),
        platform_post_id=",".join(p for p in posted_ids if p),
    )
    success(
        f"Published {item['video_id']} to {', '.join(posted_platforms)} "
        f"(ids: {', '.join(p for p in posted_ids if p) or 'n/a'})."
    )
    return True


def publish_approved(dry_run: bool = False) -> int:
    """
    Publishes every item currently in the approved queue.

    Returns:
        published_count (int): Number of items successfully published.
    """
    items = review_queue.list_approved()
    if not items:
        info("No approved items to publish.")
        return 0

    info(f"Found {len(items)} approved item(s).")
    published_count = 0
    for item in items:
        if publish_item(item, dry_run=dry_run):
            published_count += 1

    if not dry_run:
        success(f"Published {published_count}/{len(items)} approved item(s).")
    return published_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish approved review-queue items.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be published without posting.",
    )
    args = parser.parse_args(argv)

    publish_approved(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
