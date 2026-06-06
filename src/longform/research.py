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
from typing import Callable, Optional
from urllib.parse import quote

import requests

WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
_USER_AGENT = "MoneyPrinterV2-longform/1.0 (research grounding)"


def _http(session):
    return session if session is not None else requests


def _opensearch(query: str, limit: int, session=None) -> list:
    """Returns [(title, url)] candidates for a query via Wikipedia opensearch."""
    response = _http(session).get(
        WIKI_ACTION_API,
        params={
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,
            "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    titles = data[1] if len(data) > 1 else []
    urls = data[3] if len(data) > 3 else []
    return list(zip(titles, urls + [""] * (len(titles) - len(urls))))


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
        queries = [
            topic,
            f"{topic} list",
            f"{topic} controversies",
            f"{topic} unexplained",
        ]
        seen_titles = set()
        candidates = []
        for query in queries:
            try:
                for title, url in _opensearch(query, max_entries, session=session):
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        candidates.append((title, url))
            except Exception:
                continue
            if len(candidates) >= max_entries * 2:
                break

        entries = []
        for title, _url in candidates:
            if len(entries) >= max_entries:
                break
            try:
                summary = _summary(title, session=session)
            except Exception:
                continue
            if summary.get("fact"):
                entries.append(summary)
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
