"""
Royalty-free music-bed fetcher for the production layer.

Fetches a mood-matched, commercial-safe track to ``assets/music/`` for the
compositor's ducked bed, caches it (so it isn't re-downloaded every run), and
returns the track's license/credit so it lands in the credits slate + the
description.

Note: Pixabay has no public music API endpoint (``/api/music/`` 404s,
``/api/audio/`` 403s), so the working free source here is Openverse audio
(keyless, CC-licensed). A Pixabay-music provider stub is kept in the chain in
case it is ever exposed. Fully non-blocking: any failure returns None and the
compositor falls back to a local track or silence.
"""

import json
import os
import re
from typing import Callable, Optional

import requests

from . import sourcer  # reuse is_commercial_safe + download


def _http(session):
    return session if session is not None else requests


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "music").lower()).strip("_") or "music"


def query_pixabay_music(mood: str, cfg: dict, session=None) -> list:
    """Pixabay has no public music API endpoint -> no candidates (kept for when
    it is exposed)."""
    return []


def query_openverse_audio(mood: str, cfg: dict, session=None) -> list:
    """Keyless Openverse audio, filtered to commercial-use licenses."""
    try:
        resp = _http(session).get(
            "https://api.openverse.org/v1/audio/",
            params={"q": mood, "license_type": "commercial", "page_size": 8},
            headers={"User-Agent": "MoneyPrinterV2-longform/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for item in resp.json().get("results", []):
            download = item.get("url") or ""
            if not download:
                continue
            license_str = item.get("license", "") or ""
            out.append({
                "source": "openverse", "source_id": str(item.get("id", "")),
                "url": item.get("foreign_landing_url", download), "download_url": download,
                "license": license_str,
                "attribution_required": "by" in license_str.lower(),
                "author": item.get("creator", ""), "title": item.get("title", "Untitled"),
            })
        return out
    except Exception:
        return []


MUSIC_PROVIDERS = (query_pixabay_music, query_openverse_audio)


def _credit(record: dict) -> str:
    if not record.get("attribution_required"):
        return ""
    title = record.get("title") or "Untitled"
    author = record.get("author") or "Unknown"
    return f"Music: \"{title}\" by {author} via {record['source']} ({record.get('license', '')})"


def fetch_music_bed(
    mood: str,
    dest_dir: str,
    cfg: Optional[dict] = None,
    session=None,
    providers=MUSIC_PROVIDERS,
    cache: bool = True,
) -> Optional[dict]:
    """
    Fetches one commercial-safe music track for ``mood`` into ``dest_dir``.

    Returns a record {path, source, source_id, url, license,
    attribution_required, attribution, title} or None (non-blocking). Caches on
    the mood slug so repeat runs reuse the file + its sidecar license.
    """
    os.makedirs(dest_dir, exist_ok=True)
    slug = _slug(mood)
    cache_audio = os.path.join(dest_dir, f"bed_{slug}.mp3")
    cache_meta = os.path.join(dest_dir, f"bed_{slug}.json")

    if cache and os.path.exists(cache_audio) and os.path.exists(cache_meta):
        try:
            with open(cache_meta, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["path"] = cache_audio
            return record
        except Exception:
            pass  # fall through and refetch

    for provider in providers:
        try:
            candidates = provider(mood, cfg or {}, session) or []
        except Exception:
            candidates = []
        for candidate in candidates:
            if not sourcer.is_commercial_safe(candidate.get("license", "")):
                continue
            path = sourcer.download(candidate.get("download_url", ""), cache_audio, session=session)
            if not path:
                continue
            record = {
                "path": path, "source": candidate["source"],
                "source_id": candidate.get("source_id", ""), "url": candidate.get("url", ""),
                "license": candidate.get("license", ""),
                "attribution_required": bool(candidate.get("attribution_required")),
                "title": candidate.get("title", "Untitled"), "author": candidate.get("author", ""),
            }
            record["attribution"] = _credit(record)
            try:
                with open(cache_meta, "w", encoding="utf-8") as handle:
                    json.dump({k: v for k, v in record.items() if k != "path"}, handle, indent=2)
            except Exception:
                pass
            return record
    return None
