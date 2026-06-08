"""
Exemplar harvester: learn the genre's winning pattern from REAL top performers.

Instead of imitating a generic "be dramatic" prior, the creative layer learns
from the actual top-performing videos in a niche (farmed into ``.mp/farm.db``).
This module selects those exemplars and pulls the raw material a style spec is
distilled from (``style_spec.py``):

  * SELECTION -- blend views + view_velocity + outlier_score into one rank, run
    the existing relevance filter so off-topic bleed (e.g. a "Lost (TV series)"
    page mis-farmed into an iceberg niche) is excluded BEFORE harvesting.
  * ANTI-OVERFIT -- sample across MANY distinct channels, capped per channel
    (default 2), so we capture the genre's shared pattern, not one creator's
    clone.
  * MATERIAL -- per kept video, fetch the transcript (free youtube-transcript-
    api), title, opening hook (first ~45-60s), rough section/tier structure,
    chapter markers (from the description when available), and pacing signals.
    Transcripts are CLEANED of ASR noise with rough sentence boundaries.

Everything is non-blocking: a video with no transcript is skipped and the next
is pulled; any network failure degrades to fewer exemplars rather than raising.

LEGAL NOTE: harvested transcripts are transient raw material for pattern
distillation only. They are NEVER persisted into our review/output artifacts and
generation must never reproduce their wording or borrow their facts (see
``style_spec.py`` guards and CLAUDE.md).
"""

import re
from typing import Callable, Optional

from . import research, storage

# Blend weights for ranking exemplars (sum ~1.0). Views = raw reach,
# view_velocity = momentum, outlier_score = over-performance vs the channel.
BLEND_WEIGHTS = {"views": 0.40, "view_velocity": 0.35, "outlier_score": 0.25}

DEFAULT_MAX_KEEP = 8
DEFAULT_PER_CHANNEL_CAP = 2
HOOK_SECONDS = 55  # opening window treated as "the hook"

# ASR / caption noise to strip from transcripts.
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|&#39;|&quot;|&amp;")
_CHAPTER_LINE = re.compile(r"^\s*(?:\(?\d{1,2}:)?(\d{1,2}:\d{2})\)?\s+(.+?)\s*$")


# --------------------------------------------------------------------------- #
# Transcript fetch (injectable so tests never hit the network)
# --------------------------------------------------------------------------- #
def default_transcript_fn(video_id: str, languages=("en", "en-US", "en-GB")) -> Optional[list]:
    """Fetches a transcript as ``[{text, start, duration}]`` or None.

    Uses youtube-transcript-api (v1.x ``.fetch`` with a legacy fallback). Fully
    non-blocking: returns None on any error (disabled/missing transcript, network
    block, unavailable video) so the harvester just pulls the next candidate.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return None
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=list(languages))
        try:
            return fetched.to_raw_data()
        except AttributeError:
            return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    except Exception:
        pass
    try:  # legacy 0.x surface, just in case
        from youtube_transcript_api import YouTubeTranscriptApi as _A

        return _A.get_transcript(video_id, languages=list(languages))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Transcript cleaning + derivations
# --------------------------------------------------------------------------- #
def clean_transcript(snippets: list) -> str:
    """Joins ASR snippets into clean text with rough sentence boundaries.

    Strips bracketed noise ([Music], (inaudible)) and HTML entities, collapses
    whitespace, and inserts a sentence break where the gap between snippets is
    long enough to imply one (so the distiller sees rough sentences, not one run-
    on line).
    """
    parts = []
    prev_end = None
    for snip in snippets or []:
        text = _NOISE.sub(" ", str(snip.get("text", "")))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = snip.get("start")
        if prev_end is not None and isinstance(start, (int, float)) and start - prev_end > 1.2:
            if parts and parts[-1][-1] not in ".!?":
                parts[-1] += "."
        parts.append(text)
        try:
            prev_end = float(start) + float(snip.get("duration", 0) or 0)
        except (TypeError, ValueError):
            prev_end = None
    joined = " ".join(parts)
    joined = re.sub(r"\s+([.!?,])", r"\1", joined)
    return joined.strip()


def derive_hook(snippets: list, seconds: int = HOOK_SECONDS) -> str:
    """The opening ~``seconds`` of narration (the cold-open hook), cleaned."""
    window = []
    for snip in snippets or []:
        start = snip.get("start", 0) or 0
        try:
            if float(start) > seconds:
                break
        except (TypeError, ValueError):
            pass
        window.append(snip)
    return clean_transcript(window)


def parse_chapters(description: str) -> list:
    """Extracts ``[{time, title}]`` chapter markers from a video description."""
    chapters = []
    for line in (description or "").splitlines():
        match = _CHAPTER_LINE.match(line)
        if match:
            title = match.group(2).strip(" -–—:·•").strip()
            if title and len(title) < 120:
                chapters.append({"time": match.group(1), "title": title})
    return chapters


def pacing_signals(clean_text: str, duration_s) -> dict:
    """Words-per-minute + sentence-length stats from cleaned narration."""
    words = re.findall(r"\b\w+\b", clean_text or "")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", clean_text or "") if s.strip()]
    wpm = None
    try:
        if duration_s and float(duration_s) > 0:
            wpm = round(len(words) / (float(duration_s) / 60.0), 1)
    except (TypeError, ValueError):
        wpm = None
    avg_sentence_words = round(len(words) / len(sentences), 1) if sentences else 0.0
    return {"word_count": len(words), "wpm": wpm,
            "sentences": len(sentences), "avg_sentence_words": avg_sentence_words}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _blended_rank(rows: list) -> list:
    """Returns rows sorted by a min-max-normalized blend of the three metrics."""
    def col(name):
        return [float(r.get(name) or 0.0) for r in rows]

    norms = {}
    for metric in BLEND_WEIGHTS:
        values = col(metric)
        lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
        span = hi - lo
        norms[metric] = [(v - lo) / span if span else 0.0 for v in values]

    scored = []
    for index, row in enumerate(rows):
        blended = sum(BLEND_WEIGHTS[m] * norms[m][index] for m in BLEND_WEIGHTS)
        scored.append({**row, "blended": round(blended, 4)})
    scored.sort(key=lambda r: r["blended"], reverse=True)
    return scored


def _relevance_filter(niche: str, rows: list, llm) -> list:
    """Drops off-topic rows using the existing LLM relevance filter (keeps order;
    non-blocking -- keeps all rows if the LLM is unavailable or errors)."""
    if not rows or llm is None:
        return rows
    subject = niche.replace("_", " ").replace("deepdive", "deep-dive").strip()
    entries = [{"title": r.get("title", ""), "fact": r.get("title", ""), "url": r.get("video_id", "")}
               for r in rows]
    try:
        kept, _rejected = research.filter_relevant(
            f"{subject} YouTube videos (the format/genre, any subject)", entries, llm
        )
    except Exception:
        return rows
    kept_ids = {e.get("url") for e in kept}
    return [r for r in rows if r.get("video_id") in kept_ids]


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #
def harvest_exemplars(
    niche: str,
    llm: Optional[Callable] = None,
    max_keep: int = DEFAULT_MAX_KEEP,
    per_channel_cap: int = DEFAULT_PER_CHANNEL_CAP,
    transcript_fn: Optional[Callable] = None,
    description_fn: Optional[Callable] = None,
    db_path: Optional[str] = None,
    rows: Optional[list] = None,
) -> list:
    """
    Harvests up to ``max_keep`` exemplars for ``niche`` from farm.db.

    Pipeline: blended rank -> relevance filter -> per-channel-capped selection,
    fetching a transcript for each candidate (skipping those without one) until
    ``max_keep`` are collected.

    Args:
        llm: relevance-filter LLM (optional; skipped if None).
        transcript_fn(video_id) -> [{text,start,duration}] | None (injectable).
        description_fn(video_id) -> str | None for chapter parsing (optional).
        rows: pre-supplied niche rows (tests); defaults to storage.get_by_niche.

    Returns:
        list of exemplar dicts: {video_id, channel_id, title, views, blended,
        hook, transcript (cleaned, TRANSIENT), structure {chapters, pacing}}.
    """
    transcript_fn = transcript_fn or default_transcript_fn
    source_rows = rows if rows is not None else storage.get_by_niche(niche, db_path=db_path)
    if not source_rows:
        return []

    ranked = _blended_rank(source_rows)
    ranked = _relevance_filter(niche, ranked, llm)

    kept, per_channel = [], {}
    for row in ranked:
        if len(kept) >= max_keep:
            break
        channel = row.get("channel_id") or "unknown"
        if per_channel.get(channel, 0) >= per_channel_cap:
            continue
        snippets = transcript_fn(row.get("video_id"))
        if not snippets:
            continue  # no transcript -> skip, pull the next candidate
        clean = clean_transcript(snippets)
        if len(clean.split()) < 50:
            continue  # too thin to learn from
        description = None
        if description_fn is not None:
            try:
                description = description_fn(row.get("video_id"))
            except Exception:
                description = None
        kept.append({
            "video_id": row.get("video_id"),
            "channel_id": channel,
            "channel_title": row.get("channel_title", ""),
            "title": row.get("title", ""),
            "views": int(row.get("views") or 0),
            "blended": row.get("blended"),
            "hook": derive_hook(snippets),
            "transcript": clean,
            "structure": {
                "chapters": parse_chapters(description) if description else [],
                "pacing": pacing_signals(clean, row.get("duration_s")),
            },
        })
        per_channel[channel] = per_channel.get(channel, 0) + 1

    return kept


# --------------------------------------------------------------------------- #
# Optional metadata enrichment via the Data API (cheap: 1 unit / batch)
# --------------------------------------------------------------------------- #
def make_description_fn(client) -> Callable:
    """Builds a ``description_fn(video_id)`` backed by a YouTubeDataClient.

    Caches per call only; the caller decides whether to use it (it spends 1
    quota unit per video). Non-blocking -- returns None on any error.
    """
    def description_fn(video_id: str) -> Optional[str]:
        try:
            data = client.videos_list([video_id], part="snippet")
            items = data.get("items") or []
            if items:
                return (items[0].get("snippet") or {}).get("description", "")
        except Exception:
            return None
        return None

    return description_fn
