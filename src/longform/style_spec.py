"""
Style distillation: turn harvested exemplars into a reusable, per-niche STYLE
SPEC the creative layer learns from.

An LLM reads the cleaned exemplars (``exemplars.py``) and abstracts the genre's
shared winning pattern into a STRUCTURED JSON spec -- hook formula, narrator
tone, pacing/sentence-rhythm, how dread is built, how tiers open/close loops and
transition, typical words-per-entry, the escalation pattern, title phrasing
patterns, and chapter style. The spec is cached per niche and refreshable.

LEGAL GUARDS (also documented in CLAUDE.md):
  * The spec stores ABSTRACTED PATTERNS, plus at most very short (<=15 word)
    illustrative snippets -- never full transcripts.
  * Generation must never reproduce exemplar wording or borrow their facts; our
    facts always come from our own Wikipedia grounding. ``find_leak`` detects any
    long verbatim run from an exemplar leaking into generated text, and the
    sanitizer scrubs the spec itself before it is cached/used.

Non-blocking: if no exemplars are available (no transcripts / offline), a clearly
marked low-confidence default spec is returned so the creative layer still runs.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from . import exemplars as exemplars_module

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_ROOT_DIR, ".mp")

SNIPPET_MAX_WORDS = 15      # legal cap on any illustrative snippet in the spec
LEAK_MIN_WORDS = 8          # a verbatim run >= this many words counts as a leak

# The fields a distilled spec must expose. Missing fields are filled with the
# default-spec value so downstream injection never KeyErrors.
SPEC_FIELDS = (
    "hook_formula",
    "narrator_tone",
    "pacing_notes",
    "dread_build",
    "tier_transitions",
    "words_per_entry",
    "escalation_pattern",
    "title_patterns",
    "chapter_style",
    "illustrative_snippets",
)


# --------------------------------------------------------------------------- #
# Default / low-confidence spec (used when no exemplars are available)
# --------------------------------------------------------------------------- #
def default_spec(niche: str) -> dict:
    """A safe, generic spec so the creative layer runs even with no exemplars."""
    return {
        "niche": niche,
        "low_confidence": True,
        "learned_from": [],
        "hook_formula": "Open on a visceral question or stake; never 'Welcome to'.",
        "narrator_tone": "calm, ominous documentary narrator; second person sparingly",
        "pacing_notes": "short dread-beats mixed with longer descriptive lines",
        "dread_build": "escalate unease tier by tier; imply more than you state",
        "tier_transitions": "end each tier on a stakes open loop pulling deeper",
        "words_per_entry": 180,
        "escalation_pattern": "each tier darker and more unsettling than the last",
        "title_patterns": ["The {Topic} Iceberg Explained", "The DISTURBING {Topic} Iceberg"],
        "chapter_style": "Tier label + entry name (e.g. 'Surface: ...')",
        "illustrative_snippets": [],
    }


# --------------------------------------------------------------------------- #
# Anti-plagiarism leak detection
# --------------------------------------------------------------------------- #
def _norm_words(text: str) -> list:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _ngrams(words: list, n: int) -> set:
    return {" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1))}


def find_leak(generated_text: str, exemplar_texts, min_words: int = LEAK_MIN_WORDS) -> Optional[str]:
    """Returns the first verbatim run of >= ``min_words`` shared between
    ``generated_text`` and any exemplar text, or None if there is no leak."""
    gen_words = _norm_words(generated_text)
    if len(gen_words) < min_words:
        return None
    gen_grams = _ngrams(gen_words, min_words)
    if not gen_grams:
        return None
    for exemplar in exemplar_texts or []:
        ex_grams = _ngrams(_norm_words(exemplar), min_words)
        hit = gen_grams & ex_grams
        if hit:
            return next(iter(hit))
    return None


def assert_no_leak(generated_text: str, exemplar_texts, min_words: int = LEAK_MIN_WORDS) -> None:
    """Raises ValueError if a long verbatim exemplar run leaks into the text."""
    leak = find_leak(generated_text, exemplar_texts, min_words=min_words)
    if leak:
        raise ValueError(f"exemplar wording leaked into generated text: '{leak}'")


# --------------------------------------------------------------------------- #
# Spec sanitization (enforce the legal guards in code)
# --------------------------------------------------------------------------- #
def _truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def sanitize_spec(spec: dict, exemplar_texts) -> dict:
    """Enforces the legal guards on a distilled spec: snippets <= 15 words, and
    no field may contain a long verbatim run copied from an exemplar."""
    snippets = spec.get("illustrative_snippets") or []
    clean_snippets = []
    for snippet in snippets if isinstance(snippets, list) else [snippets]:
        text = _truncate_words(str(snippet), SNIPPET_MAX_WORDS)
        if text.strip() and not find_leak(text, exemplar_texts, min_words=LEAK_MIN_WORDS):
            clean_snippets.append(text.strip())
    spec["illustrative_snippets"] = clean_snippets[:6]

    # Scrub any long verbatim leak from the free-text pattern fields.
    for field in ("hook_formula", "narrator_tone", "pacing_notes", "dread_build",
                  "tier_transitions", "escalation_pattern", "chapter_style"):
        value = spec.get(field)
        if isinstance(value, str) and find_leak(value, exemplar_texts, min_words=LEAK_MIN_WORDS):
            spec[field] = default_spec(spec.get("niche", "")).get(field, "")
    return spec


# --------------------------------------------------------------------------- #
# Distillation (LLM)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in distillation output")
    return json.loads(cleaned[start:end + 1], strict=False)


def _build_distill_prompt(niche: str, exemplars: list) -> str:
    blocks = []
    for index, ex in enumerate(exemplars, start=1):
        pacing = ex.get("structure", {}).get("pacing", {})
        sample = (ex.get("transcript", "") or "")[:1100]
        blocks.append(
            f"--- EXEMPLAR {index} ---\n"
            f"TITLE: {ex.get('title','')}\n"
            f"WPM: {pacing.get('wpm')} | avg sentence words: {pacing.get('avg_sentence_words')}\n"
            f"OPENING HOOK: {ex.get('hook','')[:600]}\n"
            f"TRANSCRIPT SAMPLE: {sample}\n"
        )
    subject = niche.replace("_", " ").replace("deepdive", "deep-dive")
    return (
        f"You are a senior YouTube content strategist. Study these REAL "
        f"top-performing '{subject}' videos and ABSTRACT the shared pattern that "
        f"makes the format work. Output the pattern ONLY -- do NOT copy any "
        f"wording, sentences, or facts from them.\n\n"
        + "\n".join(blocks)
        + "\n\nReturn ONLY a JSON object describing the reusable STYLE SPEC:\n"
        f"{{\n"
        f'  "hook_formula": "how the best cold opens grab the viewer (the move, not the words)",\n'
        f'  "narrator_tone": "voice/tone of the narrator",\n'
        f'  "pacing_notes": "sentence rhythm + pacing pattern",\n'
        f'  "dread_build": "how tension/curiosity/dread is built",\n'
        f'  "tier_transitions": "how sections open and close loops + transition phrasing patterns",\n'
        f'  "words_per_entry": <typical words per section, integer>,\n'
        f'  "escalation_pattern": "how intensity escalates across the video",\n'
        f'  "title_patterns": ["pattern with {{Topic}} placeholder", "..."],\n'
        f'  "chapter_style": "how chapters/sections are named",\n'
        f'  "illustrative_snippets": ["<=15-word paraphrased illustration", "..."]\n'
        f"}}\n"
        f"Rules: patterns must be ABSTRACT and reusable for ANY subject in this "
        f"genre. Each illustrative_snippet must be <= 15 words and must NOT be a "
        f"verbatim quote. No commentary outside the JSON."
    )


def distill_style_spec(niche: str, exemplars: list, llm: Callable) -> dict:
    """LLM-distills exemplars into a structured, sanitized style spec."""
    if not exemplars or llm is None:
        return default_spec(niche)
    try:
        raw = _extract_json(llm(_build_distill_prompt(niche, exemplars)))
    except Exception:
        return default_spec(niche)

    base = default_spec(niche)
    spec = {field: raw.get(field, base.get(field)) for field in SPEC_FIELDS}
    try:
        spec["words_per_entry"] = int(spec.get("words_per_entry") or base["words_per_entry"])
    except (TypeError, ValueError):
        spec["words_per_entry"] = base["words_per_entry"]
    if not isinstance(spec.get("title_patterns"), list):
        spec["title_patterns"] = base["title_patterns"]
    spec["niche"] = niche
    spec["low_confidence"] = False
    spec["learned_from"] = [
        {"video_id": ex.get("video_id"), "channel_id": ex.get("channel_id"),
         "channel_title": ex.get("channel_title", ""), "title": ex.get("title"),
         "views": ex.get("views"), "blended": ex.get("blended")}
        for ex in exemplars
    ]
    spec["distinct_channels"] = len({ex.get("channel_id") for ex in exemplars})
    spec["created_at"] = datetime.now(timezone.utc).isoformat()

    exemplar_texts = [ex.get("transcript", "") for ex in exemplars]
    return sanitize_spec(spec, exemplar_texts)


# --------------------------------------------------------------------------- #
# Retention re-rank hook (self-improving -- wired once the retention loop exists)
# --------------------------------------------------------------------------- #
def rerank_by_retention(exemplars: list, retention_data: Optional[dict] = None) -> list:
    """FUTURE HOOK: once we have our OWN retention data, re-rank exemplars by what
    actually held viewers (not just YouTube view counts) so the style spec
    becomes self-improving. Until then this is a no-op pass-through.
    """
    if not retention_data:
        return exemplars
    return sorted(
        exemplars,
        key=lambda ex: retention_data.get(ex.get("video_id"), ex.get("blended") or 0.0),
        reverse=True,
    )


# --------------------------------------------------------------------------- #
# Cache + build
# --------------------------------------------------------------------------- #
def cache_path(niche: str, cache_dir: Optional[str] = None) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", (niche or "niche").lower()).strip("_") or "niche"
    return os.path.join(cache_dir or _CACHE_DIR, f"style_spec_{safe}.json")


def load_cached(niche: str, cache_dir: Optional[str] = None) -> Optional[dict]:
    path = cache_path(niche, cache_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None
    return None


def build_style_spec(
    niche: str,
    llm: Optional[Callable] = None,
    refresh: bool = False,
    max_keep: int = exemplars_module.DEFAULT_MAX_KEEP,
    transcript_fn: Optional[Callable] = None,
    description_fn: Optional[Callable] = None,
    db_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    rows: Optional[list] = None,
    write_cache: bool = True,
) -> dict:
    """
    Returns the per-niche style spec, building (and caching) it if needed.

    With ``refresh=False`` a cached spec is reused; ``refresh=True`` re-harvests
    and re-distills. The cache NEVER contains transcripts -- only the abstracted
    spec + the learned-from metadata. Non-blocking throughout.
    """
    if not refresh:
        cached = load_cached(niche, cache_dir)
        if cached:
            return cached

    if llm is None:
        try:
            from .llm import default_llm

            llm = default_llm()
        except Exception:
            llm = None

    exemplars = exemplars_module.harvest_exemplars(
        niche, llm=llm, max_keep=max_keep, transcript_fn=transcript_fn,
        description_fn=description_fn, db_path=db_path, rows=rows,
    )
    spec = distill_style_spec(niche, exemplars, llm)

    if write_cache:
        try:
            path = cache_path(niche, cache_dir)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(spec, handle, indent=2)  # transcripts already excluded
        except Exception:
            pass
    return spec


# --------------------------------------------------------------------------- #
# Prompt formatting (injection into the creative layer)
# --------------------------------------------------------------------------- #
def format_spec_for_prompt(spec: Optional[dict]) -> str:
    """Renders the spec as a compact LEARNED-STYLE block for a generation prompt.

    Returns "" when there is no usable spec so callers fall back to their own
    built-in guidance.
    """
    if not spec:
        return ""
    lines = [
        f"- Hook formula: {spec.get('hook_formula','')}",
        f"- Narrator tone: {spec.get('narrator_tone','')}",
        f"- Pacing: {spec.get('pacing_notes','')}",
        f"- Building dread: {spec.get('dread_build','')}",
        f"- Tier transitions: {spec.get('tier_transitions','')}",
        f"- Escalation: {spec.get('escalation_pattern','')}",
    ]
    snippets = spec.get("illustrative_snippets") or []
    if snippets:
        lines.append("- Illustrative cadence (paraphrase, never copy): "
                     + " | ".join(snippets[:4]))
    return "\n".join(lines)
