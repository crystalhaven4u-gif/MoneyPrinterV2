"""
Free, keyless factual grounding for the iceberg script architect.

For a chosen iceberg topic this gathers a pool of REAL candidate entries with
short facts (so narration is grounded rather than invented "AI slop"), using the
Wikipedia REST/Action APIs -- no key, no cost. The whole step is non-blocking: if
search returns nothing or errors, callers get an empty pool and fall back to
model-only generation.

The HTTP layer is injectable (``searcher`` / ``session``) so tests never touch
the network.
"""

import os
import re
import time
from typing import Callable, Optional
from urllib.parse import quote

import requests

# Politeness delay between Wikipedia requests (reduces 429s on shared IPs).
# Patchable so tests never actually sleep.
_sleep = time.sleep
_REQUEST_DELAY = 0.2

WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
_USER_AGENT = "MoneyPrinterV2-longform/1.0 (research grounding)"


def _http(session):
    return session if session is not None else requests


def _fulltext_search(query: str, limit: int, session=None) -> list:
    """Returns candidate page titles via Wikipedia full-text search (relevance
    ranked -- far more topical than prefix opensearch)."""
    response = _http(session).get(
        WIKI_ACTION_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srnamespace": 0,
            "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return [hit.get("title", "") for hit in (data.get("query", {}).get("search", [])) if hit.get("title")]


def _is_relevant(topic: str, title: str, fact: str) -> bool:
    """Keep entries that share a meaningful token with the topic (drop noise)."""
    topic_tokens = {t for t in re.findall(r"\w+", topic.lower()) if len(t) > 3}
    if not topic_tokens:
        return True
    haystack = f"{title} {fact}".lower()
    return any(token in haystack for token in topic_tokens)


def _summary(title: str, session=None) -> dict:
    """Returns {title, fact, url} from the Wikipedia REST summary endpoint."""
    response = _http(session).get(
        f"{WIKI_REST_SUMMARY}/{quote(title.replace(' ', '_'))}",
        headers={"User-Agent": _USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    url = ((data.get("content_urls") or {}).get("desktop") or {}).get("page", "")
    return {
        "title": data.get("title", title),
        "fact": (data.get("extract", "") or "").strip(),
        "url": url,
    }


def _wikipedia_searcher(session=None) -> Callable[[str, int], list]:
    """Default searcher: opensearch across query variants + REST summaries."""

    def search(topic: str, max_entries: int) -> list:
        # Quote the topic as a phrase first so an ambiguous term (e.g. "lost
        # media") matches the concept, not every page containing one word.
        queries = [
            f'"{topic}"',
            topic,
            f"{topic} controversies unexplained",
        ]
        seen_titles = set()
        candidates = []
        for query in queries:
            try:
                for title in _fulltext_search(query, max_entries, session=session):
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        candidates.append(title)
            except Exception:
                continue
            if len(candidates) >= max_entries * 2:
                break

        entries = []
        for title in candidates:
            if len(entries) >= max_entries:
                break
            summary = None
            for attempt in range(2):  # one retry to ride out a transient 429
                try:
                    summary = _summary(title, session=session)
                    break
                except Exception:
                    _sleep(_REQUEST_DELAY * (attempt + 1))
            if summary and summary.get("fact") and _is_relevant(
                topic, summary["title"], summary["fact"]
            ):
                entries.append(summary)
            _sleep(_REQUEST_DELAY)
        return entries

    return search


def gather_entries(
    topic: str,
    max_entries: int = 12,
    searcher: Optional[Callable[[str, int], list]] = None,
    session=None,
) -> dict:
    """
    Gathers grounded candidate entries for ``topic``.

    Args:
        topic (str): The iceberg topic.
        max_entries (int): Cap on candidate entries.
        searcher: ``searcher(topic, max_entries) -> list[{title, fact, url}]``.
            Defaults to the Wikipedia searcher; injected in tests.
        session: requests-like HTTP object for the default searcher.

    Returns:
        result (dict): {"entries": [{title, fact, url}], "sources": [url, ...]}.
        Always returns cleanly (empty pool) on any failure -- never raises.
    """
    fn = searcher or _wikipedia_searcher(session)
    try:
        raw = fn(topic, max_entries) or []
    except Exception:
        raw = []

    entries = []
    sources = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fact = (item.get("fact") or "").strip()
        if not fact:
            continue
        entry = {
            "title": (item.get("title") or "").strip(),
            "fact": fact,
            "url": (item.get("url") or "").strip(),
        }
        if not entry["title"]:
            continue
        entries.append(entry)
        if entry["url"] and entry["url"] not in sources:
            sources.append(entry["url"])
        if len(entries) >= max_entries:
            break

    return {"entries": entries, "sources": sources}


def format_entries_for_prompt(entries, limit: Optional[int] = None) -> str:
    """Renders candidate entries as a compact, factual bullet list for a prompt."""
    chosen = entries[: limit or len(entries)]
    if not chosen:
        return "(no grounded candidates found; use only widely-established facts)"
    lines = []
    for entry in chosen:
        fact = entry["fact"]
        if len(fact) > 280:
            fact = fact[:277].rstrip() + "..."
        lines.append(f"  - {entry['title']}: {fact}")
    return "\n".join(lines)
