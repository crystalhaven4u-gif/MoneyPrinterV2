"""
YouTube data farmer.

For each candidate niche probe in ``longform_discovery.json`` it:
  1. search.list (order=viewCount) and a second pass (order=date,
     publishedAfter=now-90d) over the probe's queries, videoDuration=long.
  2. videos.list on the hits -> duration, viewCount, publishedAt/title/channel.
     Keeps only duration > 30 min AND viewCount > 1,000,000.
  3. channels.list -> subscriberCount; outlier_score = views / subs.
  4. view_velocity = views / days_since_publish.
  5. a cheap creativeCommon search.list to estimate legal-footage feasibility.
  6. scores each survivor with the scout_scoring weights, applies the
     min_score_to_queue gate, and upserts everything into ``.mp/farm.db``.

Scoring (weights from config; signals normalized 0-1 across the run):
  trend_velocity  <- min-max(view_velocity)
  retention_proxy <- format_archetypes[probe.default_format]   (format prior)
  search_demand   <- min-max(views)
  competition_gap <- min-max(outlier_score)  (small channels beating big = gap)
  rpm_band        <- RPM_BAND_SCORE[probe.rpm_band or "mid"]

cc_feasibility is stored and reported but is NOT a scoring weight — in the
discovery config sourcing is a hard gate, not a weighted signal.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import storage
from .youtube_api import QuotaExceeded

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_DURATION_SECONDS = 30 * 60          # > 30 minutes
MIN_VIEWS = 1_000_000                    # > 1,000,000 views
DEFAULT_RECENCY_DAYS = 90
DEFAULT_MAX_RESULTS = 50

RPM_BAND_SCORE = {"low": 0.25, "mid": 0.5, "high": 0.75, "very_high": 1.0}

_DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?"
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def parse_duration_seconds(iso_duration: str) -> int:
    """Converts an ISO-8601 duration (e.g. PT1H2M3S) to whole seconds."""
    if not iso_duration:
        return 0
    match = _DURATION_RE.fullmatch(iso_duration.strip())
    if not match:
        return 0
    parts = {key: int(value) if value else 0 for key, value in match.groupdict().items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_since(published_at: str, now: datetime) -> float:
    """Days between publish and ``now``, floored at 1 to avoid div-by-zero."""
    published = _parse_dt(published_at)
    if published is None:
        return 1.0
    delta_days = (now - published).total_seconds() / 86400.0
    return max(delta_days, 1.0)


def compute_view_velocity(views: int, published_at: str, now: datetime) -> float:
    """views per day since publish."""
    return views / days_since(published_at, now)


def compute_outlier_score(views: int, subs: int) -> float:
    """
    views / subscribers. High when a small channel pulls big views (punching
    above its weight). Subs of 0 fall back to 1 to avoid div-by-zero while
    still rewarding the outlier.
    """
    return views / max(subs, 1)


def passes_filters(
    duration_s: int,
    views: int,
    min_duration_s: int = MIN_DURATION_SECONDS,
    min_views: int = MIN_VIEWS,
) -> bool:
    """True if a video clears the duration AND view thresholds."""
    return duration_s > min_duration_s and views > min_views


def _normalize(values) -> list:
    """Min-max normalize to 0-1. A zero range maps everything to 0.5 (neutral)."""
    values = list(values)
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.5 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]


def _chunks(sequence, size):
    sequence = list(sequence)
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


# --------------------------------------------------------------------------- #
# Discovery config
# --------------------------------------------------------------------------- #
def default_discovery_path() -> str:
    return os.path.join(_ROOT_DIR, "longform_discovery.json")


def load_discovery(path: Optional[str] = None) -> dict:
    with open(path or default_discovery_path(), "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Farming
# --------------------------------------------------------------------------- #
def _extract_video_ids(search_response: dict) -> list:
    ids = []
    for item in search_response.get("items", []):
        video_id = (item.get("id") or {}).get("videoId")
        if video_id:
            ids.append(video_id)
    return ids


def _farm_one_niche(client, probe, now, max_results, recency_days):
    """
    Farms a single probe. Raises QuotaExceeded if the budget is hit (the caller
    stops gracefully). Returns (survivors, cc_feasibility).
    """
    niche = probe["id"]
    queries = probe.get("queries", [])

    published_after = (now - timedelta(days=recency_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    video_ids = set()
    for query in queries:
        top = client.search_list(
            part="snippet", q=query, type="video",
            videoDuration="long", order="viewCount", maxResults=max_results,
        )
        video_ids.update(_extract_video_ids(top))

        recent = client.search_list(
            part="snippet", q=query, type="video",
            videoDuration="long", order="date",
            publishedAfter=published_after, maxResults=max_results,
        )
        video_ids.update(_extract_video_ids(recent))

    # Cheap CC feasibility probe (one per niche).
    cc_feasibility = 0
    if queries:
        cc = client.search_list(
            part="snippet", q=queries[0], type="video",
            videoLicense="creativeCommon", maxResults=1,
        )
        cc_feasibility = int((cc.get("pageInfo") or {}).get("totalResults", 0))

    # Details + filter.
    survivors = []
    channel_ids = set()
    for batch in _chunks(video_ids, 50):
        details = client.videos_list(batch)
        for item in details.get("items", []):
            duration_s = parse_duration_seconds(
                (item.get("contentDetails") or {}).get("duration", "")
            )
            views = int((item.get("statistics") or {}).get("viewCount", 0) or 0)
            if not passes_filters(duration_s, views):
                continue
            snippet = item.get("snippet") or {}
            channel_id = snippet.get("channelId", "")
            survivors.append(
                {
                    "video_id": item.get("id"),
                    "niche": niche,
                    "title": snippet.get("title", ""),
                    "duration_s": duration_s,
                    "views": views,
                    "published_at": snippet.get("publishedAt", ""),
                    "channel_id": channel_id,
                    "cc_feasibility": cc_feasibility,
                }
            )
            if channel_id:
                channel_ids.add(channel_id)

    # Subscriber counts -> outlier + velocity.
    subs_by_channel = {}
    for batch in _chunks(channel_ids, 50):
        channels = client.channels_list(batch)
        for item in channels.get("items", []):
            subs_by_channel[item.get("id")] = int(
                (item.get("statistics") or {}).get("subscriberCount", 0) or 0
            )

    for survivor in survivors:
        subs = subs_by_channel.get(survivor["channel_id"], 0)
        survivor["subs"] = subs
        survivor["view_velocity"] = compute_view_velocity(
            survivor["views"], survivor["published_at"], now
        )
        survivor["outlier_score"] = compute_outlier_score(survivor["views"], subs)

    return survivors, cc_feasibility


def score_videos(survivors, weights, archetypes, probe_by_id) -> list:
    """Scores survivors in place (adds 'score') and returns the list."""
    if not survivors:
        return survivors

    norm_velocity = _normalize(s["view_velocity"] for s in survivors)
    norm_views = _normalize(s["views"] for s in survivors)
    norm_outlier = _normalize(s["outlier_score"] for s in survivors)

    for index, survivor in enumerate(survivors):
        probe = probe_by_id.get(survivor["niche"], {})
        retention_proxy = float(archetypes.get(probe.get("default_format"), 0.5))
        rpm_band = RPM_BAND_SCORE.get(probe.get("rpm_band", "mid"), 0.5)

        score = (
            weights.get("trend_velocity", 0) * norm_velocity[index]
            + weights.get("retention_proxy", 0) * retention_proxy
            + weights.get("search_demand", 0) * norm_views[index]
            + weights.get("competition_gap", 0) * norm_outlier[index]
            + weights.get("rpm_band", 0) * rpm_band
        )
        survivor["score"] = round(score, 4)

    return survivors


def farm(
    client,
    discovery,
    db_path: Optional[str] = None,
    now: Optional[datetime] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    recency_days: int = DEFAULT_RECENCY_DAYS,
    persist: bool = True,
) -> dict:
    """
    Runs a full farm over every probe and returns the scored results.

    Returns:
        result (dict): videos (scored, persisted), cc_by_niche, stopped_early,
            min_score_to_queue, weights.
    """
    now = now or datetime.now(timezone.utc)
    scoring = discovery["scout_scoring"]
    weights = scoring["weights"]
    archetypes = discovery.get("format_archetypes", {})
    min_score = scoring.get("hard_gates", {}).get("min_score_to_queue", 0.6)
    probes = discovery["candidate_niche_probes"]["probes"]
    probe_by_id = {probe["id"]: probe for probe in probes}

    survivors = []
    cc_by_niche = {}
    stopped_early = False

    for probe in probes:
        try:
            niche_survivors, cc = _farm_one_niche(
                client, probe, now, max_results, recency_days
            )
        except QuotaExceeded:
            # Stop gracefully; score and persist whatever we already have.
            stopped_early = True
            break
        survivors.extend(niche_survivors)
        cc_by_niche[probe["id"]] = cc

    score_videos(survivors, weights, archetypes, probe_by_id)

    farmed_at = now.isoformat()
    if persist:
        for survivor in survivors:
            row = dict(survivor)
            row["farmed_at"] = farmed_at
            storage.upsert_video(row, db_path=db_path)

    return {
        "videos": survivors,
        "cc_by_niche": cc_by_niche,
        "stopped_early": stopped_early,
        "min_score_to_queue": min_score,
        "weights": weights,
        "quota_used": getattr(client, "quota_used", None),
    }


# --------------------------------------------------------------------------- #
# Topics slate output
# --------------------------------------------------------------------------- #
def default_topics_path() -> str:
    return os.path.join(_ROOT_DIR, "topics.longform.json")


def build_topics_slate(result, discovery, now: Optional[datetime] = None, examples_per_niche: int = 5) -> dict:
    """
    Builds the ranked niche slate (only videos clearing min_score_to_queue),
    each backed by its top example videos. Niches ranked by mean queued score.
    """
    now = now or datetime.now(timezone.utc)
    min_score = result["min_score_to_queue"]
    probe_by_id = {p["id"]: p for p in discovery["candidate_niche_probes"]["probes"]}

    queued = [v for v in result["videos"] if v.get("score", 0) >= min_score]

    by_niche = {}
    for video in queued:
        by_niche.setdefault(video["niche"], []).append(video)

    # Lazy import to avoid a hard dependency when only farming.
    from . import title_match

    niches = []
    for niche, videos in by_niche.items():
        videos_sorted = sorted(videos, key=lambda v: v["score"], reverse=True)
        scores = [v["score"] for v in videos_sorted]
        probe = probe_by_id.get(niche, {})
        ranking = title_match.tally_titles([v["title"] for v in videos_sorted])["ranking"]

        niches.append(
            {
                "niche": niche,
                "label": probe.get("label", niche),
                "default_format": probe.get("default_format"),
                "aggregate_score": round(sum(scores) / len(scores), 4),
                "video_count": len(videos_sorted),
                "cc_feasibility": result["cc_by_niche"].get(niche, 0),
                "dominant_title_formulas": ranking[:5],
                "examples": [
                    {
                        "video_id": v["video_id"],
                        "title": v["title"],
                        "url": f"https://www.youtube.com/watch?v={v['video_id']}",
                        "views": v["views"],
                        "view_velocity": round(v["view_velocity"], 1),
                        "outlier_score": round(v["outlier_score"], 2),
                        "duration_s": v["duration_s"],
                        "score": v["score"],
                    }
                    for v in videos_sorted[:examples_per_niche]
                ],
            }
        )

    niches.sort(key=lambda n: n["aggregate_score"], reverse=True)

    return {
        "generated_at": now.isoformat(),
        "min_score_to_queue": min_score,
        "weights": result["weights"],
        "stopped_early": result["stopped_early"],
        "quota_used": result.get("quota_used"),
        "niches": niches,
    }


def write_topics_slate(slate, path: Optional[str] = None) -> str:
    target = path or default_topics_path()
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(slate, handle, indent=2)
    return target


# --------------------------------------------------------------------------- #
# CLI entrypoint for a live farm
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:  # pragma: no cover - exercised live, not in tests
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Farm YouTube long-form data.")
    parser.add_argument("--budget", type=int, default=None, help="Daily quota budget override.")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--recency-days", type=int, default=DEFAULT_RECENCY_DAYS)
    args = parser.parse_args(argv)

    # Resolve the API key from config/env (lazy import to keep package light).
    sys.path.insert(0, os.path.join(_ROOT_DIR, "src"))
    try:
        from config import get_longform_youtube_api_key, get_longform_daily_quota_budget
        api_key = get_longform_youtube_api_key()
        budget = args.budget or get_longform_daily_quota_budget()
    except Exception:
        api_key = os.environ.get("YOUTUBE_API_KEY", "")
        budget = args.budget or 9000

    if not api_key:
        print("No YouTube Data API key. Set longform.youtube_api_key or YOUTUBE_API_KEY.")
        return 1

    from .youtube_api import YouTubeDataClient

    client = YouTubeDataClient(api_key, daily_budget=budget)
    discovery = load_discovery()

    print(f"Farming {len(discovery['candidate_niche_probes']['probes'])} niche probes "
          f"(budget {budget} units)...")
    result = farm(client, discovery, max_results=args.max_results, recency_days=args.recency_days)

    slate = build_topics_slate(result, discovery)
    path = write_topics_slate(slate)

    queued = sum(n["video_count"] for n in slate["niches"])
    print(f"Farmed {len(result['videos'])} videos; {queued} queued (score >= "
          f"{result['min_score_to_queue']}). Quota used: {client.quota_used}/{budget}"
          f"{' (stopped early)' if result['stopped_early'] else ''}.")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
