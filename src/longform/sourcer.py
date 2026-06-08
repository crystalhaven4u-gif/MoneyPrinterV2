"""
Footage sourcer for the production layer.

For each shot in a script entry's shot list, this finds a commercial-safe visual
from the ``legal_sources`` in longform_discovery.json -- preferring real stock
video (Pexels/Pixabay with keys; Openverse / Internet Archive keyless) and
falling back to an AI image (Ken-Burns in compose) and finally a colored slate.

HARD RULE: every asset placed in a video is logged to the ledger with
{source, source_id, url, license, attribution_required}. A shot with no logged
license is a failure, not a silent skip -- so even the AI image and the slate
fallback carry an explicit (owned/generated) license. Perceptual hashing keeps a
clip/image from repeating within one video.
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from . import storage

# Licenses we accept for commercial use. NC/ND are rejected.
_COMMERCIAL_OK = ("cc0", "pdm", "publicdomain", "public domain", "by", "by-sa",
                  "pexels", "pixabay", "attribution")
_COMMERCIAL_BAD = ("nc", "nd", "noncommercial", "non-commercial", "noderiv")
_PHASH_THRESHOLD = 6  # max Hamming distance to treat two assets as duplicates


def _http(session):
    return session if session is not None else requests


def _slug_words(url: str) -> str:
    """Pulls human-readable words out of a stock URL slug (e.g. a Pexels page)."""
    try:
        path = re.sub(r"https?://[^/]+/", "", url or "")
        path = re.sub(r"[-/_]+", " ", path)
        path = re.sub(r"\b\d+\b", " ", path)  # drop the numeric id
        return " ".join(path.split()).strip()
    except Exception:
        return ""


def is_commercial_safe(license_str: str) -> bool:
    """True if a license string is usable commercially (rejects NC/ND)."""
    text = (license_str or "").strip().lower()
    if not text:
        return False
    if any(bad in text for bad in _COMMERCIAL_BAD):
        return False
    return any(ok in text for ok in _COMMERCIAL_OK)


# --------------------------------------------------------------------------- #
# Provider queries -> normalized candidates
#   candidate = {source, source_id, url, download_url, license,
#                attribution_required(bool), kind('video'|'image'), author}
# --------------------------------------------------------------------------- #
def query_pexels(shot: str, cfg: dict, session=None) -> list:
    key = (cfg or {}).get("pexels_api_key")
    if not key:
        return []
    try:
        resp = _http(session).get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": key},
            params={"query": shot, "per_page": 5, "size": "medium"},
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for video in resp.json().get("videos", []):
            files = sorted(
                video.get("video_files", []),
                key=lambda f: (f.get("width") or 0),
                reverse=True,
            )
            hd = next((f for f in files if (f.get("width") or 0) >= 1280), files[0] if files else None)
            if not hd:
                continue
            out.append({
                "source": "pexels", "source_id": str(video.get("id")),
                "url": video.get("url", ""), "download_url": hd.get("link", ""),
                "license": "Pexels License", "attribution_required": False,
                "kind": "video", "author": (video.get("user") or {}).get("name", ""),
                "description": _slug_words(video.get("url", "")),
            })
        return out
    except Exception:
        return []


def query_pixabay(shot: str, cfg: dict, session=None) -> list:
    key = (cfg or {}).get("pixabay_api_key")
    if not key:
        return []
    try:
        resp = _http(session).get(
            "https://pixabay.com/api/videos/",
            params={"key": key, "q": shot, "per_page": 5},
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for hit in resp.json().get("hits", []):
            streams = hit.get("videos", {})
            best = streams.get("large") or streams.get("medium") or streams.get("small")
            if not best or not best.get("url"):
                continue
            out.append({
                "source": "pixabay", "source_id": str(hit.get("id")),
                "url": hit.get("pageURL", ""), "download_url": best["url"],
                "license": "Pixabay License", "attribution_required": False,
                "kind": "video", "author": hit.get("user", ""),
                "description": str(hit.get("tags", "")),
            })
        return out
    except Exception:
        return []


def query_openverse(shot: str, cfg: dict, session=None) -> list:
    """Keyless Openverse images, filtered to commercial-use licenses."""
    try:
        resp = _http(session).get(
            "https://api.openverse.org/v1/images/",
            params={"q": shot, "license_type": "commercial", "page_size": 5},
            headers={"User-Agent": "MoneyPrinterV2-longform/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for item in resp.json().get("results", []):
            download = item.get("url") or ""
            if not download:
                continue
            out.append({
                "source": "openverse", "source_id": str(item.get("id", "")),
                "url": item.get("foreign_landing_url", download),
                "download_url": download,
                "license": item.get("license", ""),
                "attribution_required": "by" in (item.get("license", "") or "").lower(),
                "kind": "image", "author": item.get("creator", ""),
                "description": str(item.get("title", "")),
            })
        return out
    except Exception:
        return []


DEFAULT_PROVIDERS = (query_pexels, query_pixabay, query_openverse)


# --------------------------------------------------------------------------- #
# Relevance ranking -- fetch many candidates, keep the best visual match
# --------------------------------------------------------------------------- #
# A candidate scoring below this (0-10) is treated as a generic mismatch: the
# stock strategy declines it so the shot falls through to AI imagery instead.
MIN_RELEVANCE = 4.0


def make_relevance_ranker(llm: Callable[[str], str]) -> Callable:
    """Builds an LLM relevance ranker: ``ranker(intent, descriptions) -> [score]``.

    ``intent`` is the shot's visual intent (query/description/mood); each score
    is 0-10 for how well that candidate visually matches the intent. Fully
    non-blocking -- any failure returns an empty list so callers keep the
    providers' own order (and the first candidate wins as before).
    """
    if llm is None:
        return lambda intent, descriptions: []

    def ranker(intent: str, descriptions: list) -> list:
        listing = "\n".join(
            f"{index}. {desc or '(no description)'}" for index, desc in enumerate(descriptions)
        )
        prompt = (
            f"A video editor needs ONE clip/image for this shot:\n  INTENT: {intent}\n\n"
            f"Score how well each candidate below VISUALLY matches that intent, "
            f"0-10 (10 = perfect on-theme match; 0 = unrelated/generic). Judge the "
            f"described content only.\n\nCANDIDATES:\n{listing}\n\n"
            f'Return ONLY a JSON array, one object per candidate, in order: '
            f'[{{"index": <int>, "score": <0-10>}}]'
        )
        try:
            text = llm(prompt)
            fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if fence:
                text = fence.group(1)
            start, end = text.find("["), text.rfind("]")
            verdicts = json.loads(text[start:end + 1], strict=False)
        except Exception:
            return []
        scores = [0.0] * len(descriptions)
        for verdict in verdicts:
            try:
                index = int(verdict["index"])
                if 0 <= index < len(scores):
                    scores[index] = max(0.0, min(10.0, float(verdict.get("score", 0))))
            except (TypeError, ValueError, KeyError):
                continue
        return scores

    return ranker


def rank_candidates(intent: str, candidates: list, ranker: Optional[Callable]) -> tuple:
    """Orders ``candidates`` best-first by relevance to ``intent``.

    Returns (ordered_candidates, best_score). With no ranker (or on any failure)
    the original order is kept and best_score is None (meaning "unscored -- don't
    reject"). With a ranker, candidates sort by score descending and best_score
    is the top score so the caller can reject a generic-only match set.
    """
    if not candidates or ranker is None:
        return candidates, None
    descriptions = [c.get("description", "") for c in candidates]
    scores = ranker(intent, descriptions) or []
    if len(scores) != len(candidates):
        return candidates, None
    ordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    best_score = ordered[0][1] if ordered else None
    return [c for c, _ in ordered], best_score


# --------------------------------------------------------------------------- #
# Download + perceptual hashing
# --------------------------------------------------------------------------- #
def download(url: str, dest: str, session=None) -> Optional[str]:
    """Downloads ``url`` to ``dest``; returns the path or None on failure."""
    try:
        resp = _http(session).get(url, timeout=120, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)
        return dest if os.path.getsize(dest) > 0 else None
    except Exception:
        return None


def _frame_png(video_path: str, ffmpeg_path: str, out_png: str) -> Optional[str]:
    """Extracts the first frame of a video for hashing (needs ffmpeg)."""
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", video_path, "-frames:v", "1", out_png],
            capture_output=True, timeout=60,
        )
        return out_png if os.path.exists(out_png) else None
    except Exception:
        return None


def perceptual_hash(path: str, kind: str, ffmpeg_path: str = "ffmpeg"):
    """
    Perceptual hash of an asset. Images hash directly; videos hash their first
    frame. Falls back to a content md5 (as a string) if imagehash/ffmpeg fail, so
    dedupe still rejects byte-identical files.
    """
    try:
        import imagehash
        from PIL import Image

        target = path
        if kind == "video":
            frame = _frame_png(path, ffmpeg_path, path + ".frame.png")
            if not frame:
                raise RuntimeError("no frame")
            target = frame
        return imagehash.phash(Image.open(target))
    except Exception:
        try:
            with open(path, "rb") as handle:
                return hashlib.md5(handle.read()).hexdigest()
        except Exception:
            return None


def is_duplicate(new_hash, used_hashes) -> bool:
    """True if ``new_hash`` matches an already-used asset (perceptual or exact)."""
    if new_hash is None:
        return False
    for existing in used_hashes:
        if type(existing) is not type(new_hash):
            continue
        if isinstance(new_hash, str):
            if existing == new_hash:
                return True
        else:  # imagehash supports subtraction = Hamming distance
            try:
                if (new_hash - existing) <= _PHASH_THRESHOLD:
                    return True
            except Exception:
                continue
    return False


# --------------------------------------------------------------------------- #
# Slate fallback (no text -> no freetype; safe)
# --------------------------------------------------------------------------- #
def make_slate(path: str, size=(1920, 1080), rgb=(20, 24, 32)) -> str:
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.new("RGB", size, rgb).save(path, "PNG")
    return path


# --------------------------------------------------------------------------- #
# Sourcing one shot (with the full fallback chain) + logging
# --------------------------------------------------------------------------- #
def _attribution(candidate: dict) -> str:
    if not candidate.get("attribution_required"):
        return ""
    author = candidate.get("author") or "Unknown"
    return f"{author} via {candidate['source']} ({candidate.get('license', '')})".strip()


def _source_stock(
    search_query, intent, shot_index, dest_dir, providers, footage_cfg,
    used_hashes, session, ffmpeg_path, ranker, reject_generic=True,
) -> Optional[dict]:
    """Best commercial-safe stock match for a shot, or None.

    Gathers candidates across ALL providers, relevance-ranks them against the
    shot intent, and downloads best-first until one passes the perceptual-dedupe
    gate. When ``reject_generic`` is set and the best match only scores generic
    (< MIN_RELEVANCE), returns None so the shot can fall through to AI imagery --
    but the caller retries with ``reject_generic=False`` once AI is exhausted, so
    a real (if imperfect) clip always beats a colored slate.
    """
    candidates = []
    for provider in providers:
        try:
            found = provider(search_query, footage_cfg, session) or []
        except Exception:
            found = []
        candidates.extend(c for c in found if is_commercial_safe(c.get("license", "")))
    if not candidates:
        return None

    ordered, best_score = rank_candidates(intent, candidates, ranker)
    if reject_generic and best_score is not None and best_score < MIN_RELEVANCE:
        return None  # only generic mismatches -> prefer AI imagery instead

    for index, candidate in enumerate(ordered):
        ext = ".mp4" if candidate.get("kind") == "video" else ".jpg"
        dest = os.path.join(dest_dir, f"shot{shot_index}_{candidate['source']}_{index}{ext}")
        path = download(candidate.get("download_url", ""), dest, session=session)
        if not path:
            continue
        digest = perceptual_hash(path, candidate.get("kind", "image"), ffmpeg_path)
        if is_duplicate(digest, used_hashes):
            continue
        if digest is not None:
            used_hashes.add(digest)
        return {
            "kind": candidate.get("kind", "image"), "path": path,
            "source": candidate["source"], "source_id": candidate.get("source_id", ""),
            "url": candidate.get("url", ""), "license": candidate.get("license", ""),
            "attribution_required": bool(candidate.get("attribution_required")),
            "attribution": _attribution(candidate),
        }
    return None


def _source_ai(ai_prompt, shot_index, dest_dir, image_fallback_fn, used_hashes, ffmpeg_path) -> Optional[dict]:
    """AI image for a shot, or None. A flagged placeholder is treated as a
    FAILURE (returns None) so the shot falls through to stock/slate rather than
    silently shipping a gray placeholder as if it were real footage."""
    if image_fallback_fn is None:
        return None
    try:
        dest = os.path.join(dest_dir, f"shot{shot_index}_ai.png")
        result = image_fallback_fn(ai_prompt, dest)
        ai_path = getattr(result, "path", None) or (result.get("path") if isinstance(result, dict) else None)
        is_placeholder = getattr(result, "is_placeholder", None)
        if isinstance(result, dict):
            is_placeholder = result.get("is_placeholder")
        if is_placeholder:
            return None
        if not (ai_path and os.path.exists(ai_path)):
            return None
        digest = perceptual_hash(ai_path, "image", ffmpeg_path)
        if digest is not None:
            used_hashes.add(digest)
        return {
            "kind": "image", "path": ai_path, "source": "ai_generated",
            "source_id": "", "url": "", "license": "AI-generated (owned)",
            "attribution_required": False, "attribution": "",
        }
    except Exception:
        return None


def source_shot(
    shot_text: str,
    shot_index: int,
    entry_title: str,
    run_id: str,
    dest_dir: str,
    providers=DEFAULT_PROVIDERS,
    footage_cfg: Optional[dict] = None,
    image_fallback_fn: Optional[Callable] = None,
    used_hashes: Optional[set] = None,
    session=None,
    ffmpeg_path: str = "ffmpeg",
    db_path: Optional[str] = None,
    log: bool = True,
    search_query: Optional[str] = None,
    ai_prompt: Optional[str] = None,
    prefer_ai: bool = False,
    ranker: Optional[Callable] = None,
) -> dict:
    """
    Sources a single shot with an ordered fallback chain, always license-logged.

    For atmospheric/lore shots set ``prefer_ai=True`` so AI imagery is tried
    FIRST (eerie, on-theme), with stock as the fallback; for concrete real-world
    shots stock is tried first. Either way a flagged AI placeholder falls through
    rather than shipping, and a colored slate is the final guaranteed fallback.

    Args:
        search_query: the stock/AI library query (concrete nouns + mood);
            defaults to ``shot_text``.
        ai_prompt: the AI-image prompt; defaults to ``shot_text``.
        prefer_ai: try AI imagery before stock (atmosphere shots).
        ranker: optional relevance ranker (see ``make_relevance_ranker``).

    Returns the asset dict {kind, path, source, source_id, url, license,
    attribution_required, attribution, shot_text, entry_title, shot_index}.
    """
    used_hashes = used_hashes if used_hashes is not None else set()
    footage_cfg = footage_cfg or {}
    os.makedirs(dest_dir, exist_ok=True)
    search_query = (search_query or shot_text or "").strip()
    ai_prompt = (ai_prompt or shot_text or "").strip()
    intent = ai_prompt or search_query or (shot_text or "")

    def _stock(reject_generic=True):
        return _source_stock(
            search_query, intent, shot_index, dest_dir, providers, footage_cfg,
            used_hashes, session, ffmpeg_path, ranker, reject_generic=reject_generic,
        )

    def _ai():
        return _source_ai(ai_prompt, shot_index, dest_dir, image_fallback_fn, used_hashes, ffmpeg_path)

    # Atmosphere shots prefer AI first; concrete shots prefer (relevant) stock,
    # then AI. In BOTH chains the final step is lenient stock (reject_generic
    # off) so a real, if imperfect, clip always wins over a colored slate.
    if prefer_ai:
        order = [_ai, lambda: _stock(reject_generic=False)]
    else:
        order = [lambda: _stock(reject_generic=True), _ai, lambda: _stock(reject_generic=False)]
    for strategy in order:
        asset = strategy()
        if asset is not None:
            return _finalize(asset, shot_text, shot_index, entry_title, run_id, db_path, log)

    # Colored slate (last resort) -- still licensed + logged.
    slate = make_slate(os.path.join(dest_dir, f"shot{shot_index}_slate.png"))
    asset = {
        "kind": "slate", "path": slate, "source": "slate", "source_id": "",
        "url": "", "license": "Generated placeholder (owned)",
        "attribution_required": False, "attribution": "",
    }
    return _finalize(asset, shot_text, shot_index, entry_title, run_id, db_path, log)


def _finalize(asset, shot_text, shot_index, entry_title, run_id, db_path, log) -> dict:
    asset.update({"shot_text": shot_text, "shot_index": shot_index, "entry_title": entry_title})
    if log:
        storage.log_asset(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "shot_index": shot_index, "entry_title": entry_title,
                "shot_text": shot_text, "kind": asset["kind"],
                "source": asset["source"], "source_id": asset["source_id"],
                "url": asset["url"], "license": asset["license"],
                "attribution_required": int(asset["attribution_required"]),
                "attribution": asset["attribution"], "path": asset["path"],
            },
            db_path=db_path,
        )
    return asset


def build_default_providers() -> tuple:
    """The default provider order (Pexels, Pixabay, Openverse)."""
    return DEFAULT_PROVIDERS
