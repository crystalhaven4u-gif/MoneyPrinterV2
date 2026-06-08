"""
Iceberg deep-dive script architect (staged + length-enforced + grounded).

Single-shot generation produced short, hallucination-prone scripts. This module
generates in stages, which both small and large models handle far better:

  a) OUTLINE  -- the iceberg skeleton: ordered tiers (surface -> deepest), each
                 with a title + one-line premise + planned open-loop, plus the
                 cold_hook and final_payoff.
  b) PER-ENTRY -- each tier's narration generated on its own, to its own word
                 budget (target_minutes*140 / entry_count), with a short running
                 context (previous closing beat + the loop it must resolve) so
                 coherence holds across chunks.
  c) LENGTH    -- if an entry lands under ~85% of budget, re-prompt to EXPAND
                 with concrete specifics (not filler); capped at 3 retries; the
                 per-entry word count is logged.
  d) REASSEMBLE into the existing schema (cold_hook -> tiers -> final_payoff) so
                 nothing downstream changes.

Optional factual GROUNDING (config script.grounding, default on) feeds REAL
candidate entries + facts (via research.py) into the outline and per-entry
prompts so narration is grounded, not invented. Falls back cleanly to model-only.
Few-shot tone still comes from the real farm.db winners.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import yaml

from . import research, storage

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_PROMPT_PATH = os.path.join(_ROOT_DIR, "prompts", "longform_iceberg.yaml")
DEFAULT_TARGET_MINUTES = 10
DEFAULT_FEW_SHOT_N = 6
DEFAULT_WPM = 140
MIN_LENGTH_RATIO = 0.85           # expand an entry below 85% of its word budget
MAX_EXPAND_RETRIES = 3
MAX_PARSE_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Prompt config + few-shot
# --------------------------------------------------------------------------- #
def load_prompt_config(path: Optional[str] = None) -> dict:
    """Loads the versioned iceberg prompt YAML."""
    with open(path or DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_few_shot_winners(
    niche: str = "iceberg_deepdive",
    db_path: Optional[str] = None,
    limit: int = DEFAULT_FEW_SHOT_N,
) -> list:
    """Pulls the top real winners for ``niche`` from farm.db as few-shot tone."""
    rows = storage.get_by_niche(niche, db_path=db_path)
    examples = []
    for row in rows[:limit]:
        duration_min = int((row.get("duration_s") or 0) // 60)
        examples.append({"title": row.get("title", ""), "duration_min": duration_min})
    return examples


def _format_few_shot(examples) -> str:
    if not examples:
        return "(no examples available)"
    return "\n".join(f"  - {ex['title']} — {ex['duration_min']} min" for ex in examples)


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> str:
    """Pulls the first balanced JSON object out of an LLM reply."""
    if not text:
        raise ValueError("empty model output")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for index in range(start, len(cleaned)):
        if cleaned[index] == "{":
            depth += 1
        elif cleaned[index] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ValueError("unbalanced JSON object in model output")


def _word_count(*parts) -> int:
    return sum(len(re.findall(r"\b\w+\b", part or "")) for part in parts)


def _coerce_shot_list(value) -> list:
    if isinstance(value, str):
        value = [value]
    return [str(shot).strip() for shot in (value or []) if str(shot).strip()]


# Visual shot specs (v3): each shot carries a description (what it shows), a
# dedicated stock/AI SEARCH QUERY (concrete filmable nouns + mood, separate from
# narration), a mood word, and a type: "atmosphere" (eerie/lore -> prefer AI
# imagery) or "concrete" (a real-world thing -> prefer stock footage).
_VALID_SHOT_TYPES = ("atmosphere", "concrete")


def _coerce_shots(value) -> list:
    """Normalizes a model shot_list into rich shot specs.

    Accepts a list of strings (legacy) or a list of objects with any of
    {description, query, search_query, mood, type}. Always returns a list of
    {description, query, mood, type} dicts; strings become concrete shots whose
    query equals the description.
    """
    if isinstance(value, (str, dict)):
        value = [value]
    shots = []
    for raw in value or []:
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            shots.append({"description": text, "query": text, "mood": "", "type": "concrete"})
            continue
        if not isinstance(raw, dict):
            continue
        description = str(raw.get("description") or raw.get("shot") or "").strip()
        query = str(raw.get("query") or raw.get("search_query") or description).strip()
        if not (description or query):
            continue
        description = description or query
        query = query or description
        shot_type = str(raw.get("type") or "").strip().lower()
        if shot_type not in _VALID_SHOT_TYPES:
            shot_type = "concrete"
        shots.append({
            "description": description,
            "query": query,
            "mood": str(raw.get("mood") or "").strip(),
            "type": shot_type,
        })
    return shots


def parse_outline(text: str) -> dict:
    """
    Parses an OUTLINE reply into {cold_hook, final_payoff, tiers:[{tier, label,
    entry_title, premise, open_loop}]}. Raises ValueError without usable tiers.
    """
    data = json.loads(_extract_json(text), strict=False)
    if not isinstance(data, dict):
        raise ValueError("outline JSON is not an object")
    raw_tiers = data.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("outline has no tiers")

    tiers = []
    for index, raw in enumerate(raw_tiers, start=1):
        if not isinstance(raw, dict):
            continue
        entry_title = str(raw.get("entry_title", "")).strip()
        if not entry_title:
            continue
        tiers.append(
            {
                "tier": int(raw.get("tier", index) or index),
                "label": str(raw.get("label", f"Tier {index}")).strip() or f"Tier {index}",
                "entry_title": entry_title,
                "premise": str(raw.get("premise", "")).strip(),
                "open_loop": str(raw.get("open_loop", "")).strip(),
            }
        )
    if not tiers:
        raise ValueError("outline has no usable tier entries")

    return {
        "cold_hook": str(data.get("cold_hook", "")).strip(),
        "final_payoff": str(data.get("final_payoff", "")).strip(),
        "tiers": tiers,
    }


def parse_entry(text: str) -> dict:
    """Parses a PER-ENTRY reply into {narration, shots, shot_list, open_loop}."""
    data = json.loads(_extract_json(text), strict=False)
    if not isinstance(data, dict):
        raise ValueError("entry JSON is not an object")
    narration = str(data.get("narration", "")).strip()
    if not narration:
        raise ValueError("entry has no narration")
    open_loop = str(data.get("open_loop", "")).strip()
    if open_loop.lower() == "open_loop":  # model echoed the schema key, not a value
        open_loop = ""
    shots = _coerce_shots(data.get("shot_list"))
    return {
        "narration": narration,
        "shots": shots,
        "shot_list": [shot["description"] for shot in shots],
        "open_loop": open_loop,
    }


def parse_rewrite(text: str) -> str:
    """Parses a NARRATIVE-REWRITE reply into the rewritten narration string.

    Accepts a JSON object with a ``narration`` key, or (fallback) treats the
    whole reply as prose after stripping any code fence/preamble. Raises
    ValueError only when nothing usable remains.
    """
    if not text or not text.strip():
        raise ValueError("empty rewrite output")
    try:
        data = json.loads(_extract_json(text), strict=False)
        if isinstance(data, dict):
            narration = str(data.get("narration", "")).strip()
            if narration:
                return narration
    except ValueError:
        pass
    # Fallback: strip a leading ```fence and any "Here is..." preamble line.
    cleaned = text.strip()
    fence = re.search(r"```(?:\w+)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        raise ValueError("no rewrite text found")
    return cleaned


# Legacy single-shot parser, retained for compatibility/tests.
def parse_iceberg_script(text: str) -> dict:
    """Parses a full single-shot iceberg script (legacy contract)."""
    data = json.loads(_extract_json(text), strict=False)
    if not isinstance(data, dict):
        raise ValueError("parsed JSON is not an object")
    raw_tiers = data.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("script has no tiers")
    tiers = []
    for index, raw in enumerate(raw_tiers, start=1):
        if not isinstance(raw, dict):
            continue
        tiers.append(
            {
                "tier": int(raw.get("tier", index) or index),
                "label": str(raw.get("label", f"Tier {index}")),
                "entry_title": str(raw.get("entry_title", "")).strip(),
                "narration": str(raw.get("narration", "")).strip(),
                "shot_list": _coerce_shot_list(raw.get("shot_list")),
                "open_loop": str(raw.get("open_loop", "")).strip(),
            }
        )
    if not tiers:
        raise ValueError("script has no usable tier entries")
    cold_hook = str(data.get("cold_hook", "")).strip()
    final_payoff = str(data.get("final_payoff", "")).strip()
    return {
        "iceberg_topic": str(data.get("iceberg_topic", "")).strip(),
        "cold_hook": cold_hook,
        "tiers": tiers,
        "final_payoff": final_payoff,
        "entry_count": len(tiers),
        "word_count": _word_count(cold_hook, final_payoff, *[t["narration"] for t in tiers]),
    }


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _style_block(prompt_cfg: dict) -> str:
    return yaml.safe_dump(prompt_cfg.get("style", {}), sort_keys=False)


def build_outline_prompt(topic, few_shot, grounded_block, n_tiers, prompt_cfg) -> str:
    return (
        f"You are an expert YouTube iceberg-video script architect.\n"
        f"Build the OUTLINE (skeleton) for an iceberg deep-dive on: {topic}.\n\n"
        f"Order {n_tiers} tiers from SURFACE (well-known) to the DEEPEST / most "
        f"unsettling layer. For each tier give: a short label (Surface, Shallows, "
        f"Deep, Abyss, Bedrock...), an entry_title (a real thing on the iceberg "
        f"chart), a one-line premise, and a one-line open_loop teasing the NEXT "
        f"deeper tier. Also write a 0-15s cold_hook that names the iceberg and "
        f"teases (without revealing) the deepest layer, and a final_payoff line "
        f"that resolves it.\n\n"
        f"PREFER REAL, VERIFIABLE entries. Real candidate entries with facts:\n"
        f"{grounded_block}\n\n"
        f"Top real iceberg videos (for tone/scope only — do not copy topics):\n"
        f"{_format_few_shot(few_shot)}\n\n"
        f"STYLE:\n{_style_block(prompt_cfg)}\n"
        f"Return ONLY JSON: {{\"cold_hook\": str, \"tiers\": [{{\"tier\": int, "
        f"\"label\": str, \"entry_title\": str, \"premise\": str, "
        f"\"open_loop\": str}}], \"final_payoff\": str}}"
    )


def build_entry_prompt(topic, tier, word_budget, running_context, fact, prompt_cfg) -> str:
    fact_block = fact.strip() if fact else "(no specific source facts — use only widely-established knowledge; do NOT invent specifics)"
    context_block = running_context.strip() if running_context else "(this is the opening tier, right after the cold open)"
    return (
        f"Write the NARRATION for ONE tier of the '{topic}' iceberg video.\n\n"
        f"TIER: {tier['label']} — {tier['entry_title']}\n"
        f"PREMISE: {tier.get('premise', '')}\n"
        f"GROUND-TRUTH FACTS (rely on these; do not contradict them; do not "
        f"fabricate names/dates/numbers beyond established knowledge):\n{fact_block}\n\n"
        f"CONTEXT SO FAR (continue smoothly; pick up the thread it teases):\n"
        f"{context_block}\n\n"
        f"Write about {word_budget} words of vivid, FACTUAL documentary narration "
        f"(calm authoritative voice, short sentences, escalating unease). End on a "
        f"one-sentence open_loop teasing the next deeper tier"
        + (f" (planned: {tier['open_loop']})" if tier.get("open_loop") else "")
        + ".\n\n"
        f"Then give a shot_list of 2-4 visuals. For EACH shot provide:\n"
        f"  - description: what is on screen\n"
        f"  - query: a concrete VISUAL SEARCH QUERY of filmable nouns + mood for a "
        f"stock/AI library (NOT a sentence from the narration); e.g. 'ominous red "
        f"door dark liminal hallway', 'flickering fluorescent office at night'\n"
        f"  - mood: one or two mood words (e.g. 'dread', 'eerie calm')\n"
        f"  - type: 'atmosphere' for eerie/lore/abstract shots that suit AI imagery, "
        f"or 'concrete' for real-world things that suit stock footage\n\n"
        f"Return ONLY JSON: {{\"narration\": str, \"shot_list\": [{{\"description\": "
        f"str, \"query\": str, \"mood\": str, \"type\": str}}], \"open_loop\": str}}"
    )


def build_expand_prompt(topic, tier, narration, deficit_words, fact) -> str:
    fact_block = fact.strip() if fact else "(use only established knowledge; do NOT invent specifics)"
    return (
        f"This tier of the '{topic}' iceberg is TOO SHORT. Expand it by about "
        f"{deficit_words} more words of CONCRETE, specific detail — mechanisms, "
        f"consequences, vivid imagery, established names/dates — NOT filler, "
        f"repetition, or padding. Keep it factual and on-topic, preserve the "
        f"ending open_loop.\n\n"
        f"TIER: {tier['label']} — {tier['entry_title']}\n"
        f"FACTS:\n{fact_block}\n\n"
        f"CURRENT NARRATION:\n{narration}\n\n"
        f"Return ONLY JSON (the FULL expanded version): {{\"narration\": str, "
        f"\"shot_list\": [str], \"open_loop\": str}}"
    )


# --------------------------------------------------------------------------- #
# Pass 2 -- narrative rewrite (cinematic, escalating dread; facts preserved)
# --------------------------------------------------------------------------- #
def _narrative_cfg(prompt_cfg: dict) -> dict:
    cfg = (prompt_cfg or {}).get("narrative") or {}
    return cfg if isinstance(cfg, dict) else {}


def _narrative_rules(prompt_cfg: dict) -> str:
    rules = _narrative_cfg(prompt_cfg).get("rules") or []
    return "\n".join(f"  - {rule}" for rule in rules) if rules else "  - (use cinematic, escalating dread)"


def _narrative_few_shot(prompt_cfg: dict) -> str:
    examples = _narrative_cfg(prompt_cfg).get("few_shot") or []
    if not examples:
        return "(no tone examples)"
    return "\n".join(f'  - "{ex}"' for ex in examples)


_PRESERVE = (
    "HARD RULE: preserve every proper noun, name, date, number, and concrete "
    "fact from the SOURCE exactly. Do NOT invent any new specifics, events, or "
    "details that are not already in the source text. You are re-voicing it, not "
    "adding to it."
)


def _learned_section(style_block: str) -> str:
    if not style_block:
        return ""
    return (
        "LEARNED STYLE PATTERNS (distilled from real top-performing videos in "
        "this genre -- ADOPT these patterns; never copy any exemplar's words or "
        "facts):\n" + style_block + "\n\n"
    )


def build_cold_open_rewrite_prompt(topic, factual_cold_hook, first_tier_title, prompt_cfg, style_block="") -> str:
    return (
        f"You are an ominous documentary narrator opening a YouTube iceberg "
        f"deep-dive on '{topic}'.\n\n"
        f"{_learned_section(style_block)}"
        f"Rewrite the COLD OPEN below into 3-5 spoken sentences that GRIP a viewer "
        f"in the first 10 seconds. The first 2-3 sentences must land a visceral "
        f"hook -- a disturbing question, a stake, or an unsettling image -- so the "
        f"viewer NEEDS the answer. Speak directly to them ('you', 'imagine'). "
        f"Tease that we descend tier by tier to something at the very bottom, "
        f"WITHOUT revealing it. Absolutely never begin with 'Welcome to' or "
        f"'In this video'.\n\n"
        f"TONE EXEMPLARS (cadence + menace only -- never reuse their words):\n"
        f"{_narrative_few_shot(prompt_cfg)}\n\n"
        f"RULES:\n{_narrative_rules(prompt_cfg)}\n\n"
        f"{_PRESERVE}\n\n"
        f"SOURCE COLD OPEN:\n{factual_cold_hook}\n\n"
        f"Return ONLY JSON: {{\"narration\": str}}"
    )


def build_tier_rewrite_prompt(topic, label, entry_title, position, total, factual, open_loop, prompt_cfg, style_block="") -> str:
    depth = (
        "This is the FIRST tier -- unsettling but still recognizable."
        if position == 1 else
        f"This is tier {position} of {total} -- it must feel DARKER and more "
        f"disturbing than every tier before it."
    )
    return (
        f"You are an ominous documentary narrator descending the '{topic}' "
        f"iceberg. Rewrite ONE tier's narration into cinematic, escalating-dread "
        f"narration.\n\n"
        f"{_learned_section(style_block)}"
        f"TIER: {label} -- {entry_title}\n{depth}\n\n"
        f"Requirements: vary the pacing (mix short, punchy dread-beats with longer "
        f"descriptive lines -- no monotone paragraphs); address the viewer "
        f"directly where it bites; strip all encyclopedic phrasing ('is a "
        f"fictional', 'refers to', 'is a concept'), hedging and filler; keep "
        f"roughly the same length. End on a one-sentence open loop with STAKES "
        f"that pulls the viewer to the next, deeper tier"
        + (f" (the next descent: {open_loop})" if open_loop else "")
        + " -- never 'next, we'll look at...'.\n\n"
        f"TONE EXEMPLARS (cadence + menace only -- never reuse their words):\n"
        f"{_narrative_few_shot(prompt_cfg)}\n\n"
        f"RULES:\n{_narrative_rules(prompt_cfg)}\n\n"
        f"{_PRESERVE}\n\n"
        f"SOURCE NARRATION:\n{factual}\n\n"
        f"Return ONLY JSON: {{\"narration\": str}}"
    )


def build_payoff_rewrite_prompt(topic, factual_payoff, deepest_title, prompt_cfg, style_block="") -> str:
    return (
        f"You are an ominous documentary narrator closing the '{topic}' iceberg "
        f"deep-dive at its deepest layer ('{deepest_title}').\n\n"
        f"{_learned_section(style_block)}"
        f"Rewrite the FINAL PAYOFF below into 2-4 spoken sentences that resolve the "
        f"descent and land the dread -- the moment the title is paid off. Speak to "
        f"the viewer. No 'thanks for watching', no 'subscribe'.\n\n"
        f"RULES:\n{_narrative_rules(prompt_cfg)}\n\n"
        f"{_PRESERVE}\n\n"
        f"SOURCE FINAL PAYOFF:\n{factual_payoff}\n\n"
        f"Return ONLY JSON: {{\"narration\": str}}"
    )


def _rewrite(llm, prompt, fallback) -> str:
    """Runs one rewrite; returns the rewritten text or the factual fallback on
    any failure (non-blocking -- Pass 2 must never lose Pass 1's content)."""
    try:
        out = parse_rewrite(llm(prompt))
    except Exception:
        return fallback
    return out or fallback


def narrative_rewrite(script: dict, llm, prompt_cfg: dict, style_spec: Optional[dict] = None) -> dict:
    """Re-voices a factual script (Pass 1) into cinematic narration (Pass 2).

    When a ``style_spec`` is given (distilled from real top-performing videos via
    style_spec.py), its learned hook formula / pacing / transitions / escalation
    are injected as guidance instead of only the generic built-in instructions --
    adopting the genre's proven pattern. Mutates and returns ``script``:
    cold_hook, every tier narration, and final_payoff are rewritten; the
    originals are kept under ``factual_*`` for traceability. Non-blocking -- any
    section that fails keeps its factual text.
    """
    style_block = ""
    if style_spec:
        try:
            from . import style_spec as style_spec_module

            style_block = style_spec_module.format_spec_for_prompt(style_spec)
        except Exception:
            style_block = ""

    topic = script.get("iceberg_topic", "the topic")
    tiers = script.get("tiers", [])
    total = len(tiers)
    deepest_title = tiers[-1]["entry_title"] if tiers else topic
    first_title = tiers[0]["entry_title"] if tiers else ""

    factual_cold = script.get("cold_hook", "")
    script["factual_cold_hook"] = factual_cold
    script["cold_hook"] = _rewrite(
        llm, build_cold_open_rewrite_prompt(topic, factual_cold, first_title, prompt_cfg, style_block), factual_cold
    )

    for position, tier in enumerate(tiers, start=1):
        factual = tier.get("narration", "")
        tier["factual_narration"] = factual
        tier["narration"] = _rewrite(
            llm,
            build_tier_rewrite_prompt(
                topic, tier.get("label", ""), tier.get("entry_title", ""),
                position, total, factual, tier.get("open_loop", ""), prompt_cfg, style_block,
            ),
            factual,
        )

    factual_payoff = script.get("final_payoff", "")
    script["factual_payoff"] = factual_payoff
    script["final_payoff"] = _rewrite(
        llm, build_payoff_rewrite_prompt(topic, factual_payoff, deepest_title, prompt_cfg, style_block), factual_payoff
    )

    script["word_count"] = _word_count(
        script["cold_hook"], script["final_payoff"], *[t["narration"] for t in tiers]
    )
    script["narrative_pass"] = True
    return script


# --------------------------------------------------------------------------- #
# Generation primitives
# --------------------------------------------------------------------------- #
def _call_parse(llm, prompt, parser, attempts=MAX_PARSE_ATTEMPTS):
    last = None
    for _ in range(max(1, attempts)):
        try:
            return parser(llm(prompt))
        except ValueError as exc:
            last = exc
    raise ValueError(f"unparseable model output: {last}")


def _closing_beat(narration: str) -> str:
    """Last sentence or two of a narration, for the running context."""
    sentences = re.split(r"(?<=[.!?])\s+", narration.strip())
    return " ".join(sentences[-2:]).strip() if sentences else narration[-240:]


def _match_fact(entry_title: str, pool: list) -> str:
    """Best-matching grounded fact for an entry title (substring overlap)."""
    title = (entry_title or "").lower()
    if not title:
        return ""
    for entry in pool:
        name = (entry.get("title") or "").lower()
        if name and (name in title or title in name):
            return entry.get("fact", "")
    # token-overlap fallback
    title_tokens = set(re.findall(r"\w+", title))
    best, best_overlap = "", 0
    for entry in pool:
        tokens = set(re.findall(r"\w+", (entry.get("title") or "").lower()))
        overlap = len(title_tokens & tokens)
        if overlap > best_overlap:
            best, best_overlap = entry.get("fact", ""), overlap
    return best if best_overlap >= 2 else ""


def generate_entry(
    topic, tier, word_budget, running_context, fact, prompt_cfg, llm,
    min_ratio=MIN_LENGTH_RATIO, max_expand=MAX_EXPAND_RETRIES,
) -> dict:
    """
    Generates one tier's narration to its word budget, expanding if too short.

    Returns the entry dict plus 'word_count' and 'expansions' (retries used).
    """
    entry = _call_parse(
        llm, build_entry_prompt(topic, tier, word_budget, running_context, fact, prompt_cfg),
        parse_entry,
    )
    floor = int(word_budget * min_ratio)
    expansions = 0
    while _word_count(entry["narration"]) < floor and expansions < max_expand:
        deficit = word_budget - _word_count(entry["narration"])
        try:
            expanded = _call_parse(
                llm, build_expand_prompt(topic, tier, entry["narration"], deficit, fact),
                parse_entry,
            )
        except ValueError:
            break
        expansions += 1
        # Keep the longer narration; merge in any shots/open_loop it produced.
        if _word_count(expanded["narration"]) > _word_count(entry["narration"]):
            entry["narration"] = expanded["narration"]
            entry["shots"] = expanded.get("shots") or entry.get("shots", [])
            entry["shot_list"] = expanded["shot_list"] or entry["shot_list"]
            entry["open_loop"] = expanded["open_loop"] or entry["open_loop"]

    entry["word_count"] = _word_count(entry["narration"])
    entry["expansions"] = expansions
    return entry


def generate_outline(topic, few_shot, pool, n_tiers, prompt_cfg, llm) -> dict:
    """Generates and parses the iceberg outline (skeleton)."""
    grounded_block = research.format_entries_for_prompt(pool)
    return _call_parse(
        llm, build_outline_prompt(topic, few_shot, grounded_block, n_tiers, prompt_cfg),
        parse_outline,
    )


# --------------------------------------------------------------------------- #
# Top-level staged generation
# --------------------------------------------------------------------------- #
def generate_script(
    topic: str,
    llm: Optional[Callable[[str], str]] = None,
    target_minutes: int = DEFAULT_TARGET_MINUTES,
    prompt_path: Optional[str] = None,
    db_path: Optional[str] = None,
    few_shot_n: int = DEFAULT_FEW_SHOT_N,
    run_id: Optional[str] = None,
    grounding: Optional[bool] = None,
    searcher: Optional[Callable] = None,
    narrative: Optional[bool] = None,
    style_spec: Optional[dict] = None,
    niche: str = "iceberg_deepdive",
    log: bool = True,
) -> dict:
    """
    Generates a tiered iceberg script via staged outline + per-entry passes.

    Args:
        grounding: force grounding on/off; defaults to config script.grounding.
        searcher: optional research searcher (injected in tests).

    Returns:
        script (dict): iceberg_topic, cold_hook, tiers (tier/label/entry_title/
        narration/shot_list/open_loop), final_payoff, entry_count, word_count,
        per_entry_word_counts, prompt_version, run_id, sources, grounded.
    """
    if llm is None:
        from .llm import default_llm

        llm = default_llm()

    prompt_cfg = load_prompt_config(prompt_path)
    prompt_version = str(prompt_cfg.get("version", "unknown"))
    wpm = int(prompt_cfg.get("target_wpm", DEFAULT_WPM))
    total_budget = wpm * target_minutes

    # Grounding (non-blocking).
    if grounding is None:
        try:
            import sys

            src_dir = os.path.join(_ROOT_DIR, "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            from config import get_script_grounding

            grounding = get_script_grounding()
        except Exception:
            grounding = True
    pool, sources, rejected_sources = [], [], []
    if grounding:
        result = research.gather_entries(topic, searcher=searcher, llm=llm)
        pool, sources = result["entries"], result["sources"]
        rejected_sources = result.get("rejected", [])

    few_shot = get_few_shot_winners(db_path=db_path, limit=few_shot_n)
    n_tiers = max(5, min(9, round(target_minutes * 0.8)))

    # a) OUTLINE
    outline = generate_outline(topic, few_shot, pool, n_tiers, prompt_cfg, llm)
    tiers_plan = outline["tiers"]
    entry_count = len(tiers_plan)
    per_entry_budget = max(80, int(total_budget / entry_count))

    # b) + c) PER-ENTRY with running context + length enforcement
    tiers = []
    per_entry_word_counts = []
    running_context = outline["cold_hook"]
    for plan in tiers_plan:
        fact = _match_fact(plan["entry_title"], pool)
        try:
            entry = generate_entry(
                topic, plan, per_entry_budget, running_context, fact, prompt_cfg, llm
            )
        except ValueError:
            # Non-blocking: a single unparseable tier must not abort the run.
            # Fall back to the outline premise as minimal narration.
            premise = plan.get("premise") or plan["entry_title"]
            entry = {
                "narration": premise,
                "shots": [],
                "shot_list": [],
                "open_loop": plan.get("open_loop", ""),
                "word_count": _word_count(premise),
                "expansions": 0,
            }
        tiers.append(
            {
                "tier": plan["tier"],
                "label": plan["label"],
                "entry_title": plan["entry_title"],
                "narration": entry["narration"],
                "shots": entry.get("shots", []),
                "shot_list": entry["shot_list"],
                "open_loop": entry["open_loop"] or plan["open_loop"],
            }
        )
        per_entry_word_counts.append(
            {"entry_title": plan["entry_title"], "word_count": entry["word_count"],
             "expansions": entry["expansions"]}
        )
        running_context = (
            f"Previous tier ({plan['label']} — {plan['entry_title']}) ended: "
            f"{_closing_beat(entry['narration'])} It teased: "
            f"{tiers[-1]['open_loop']}"
        )

    # d) REASSEMBLE into the existing schema
    word_count = _word_count(
        outline["cold_hook"], outline["final_payoff"], *[t["narration"] for t in tiers]
    )
    script = {
        "iceberg_topic": topic,
        "cold_hook": outline["cold_hook"],
        "tiers": tiers,
        "final_payoff": outline["final_payoff"],
        "entry_count": entry_count,
        "word_count": word_count,
        "per_entry_word_counts": per_entry_word_counts,
        "prompt_version": prompt_version,
        "run_id": run_id,
        "grounded": bool(pool),
        "sources": sources,
        "rejected_sources": rejected_sources,
        "narrative_pass": False,
    }

    # e) PASS 2 -- narrative rewrite (cinematic, escalating dread; facts kept).
    if narrative is None:
        try:
            from config import get_script_narrative

            narrative = get_script_narrative()
        except Exception:
            narrative = True
    if narrative:
        # Use the learned per-niche style spec when one is available. Only the
        # CACHED spec is loaded here (no live harvest during script gen); build
        # it separately via style_spec.build_style_spec / the CLI.
        if style_spec is None:
            try:
                from . import style_spec as style_spec_module

                style_spec = style_spec_module.load_cached(niche)
            except Exception:
                style_spec = None
        try:
            narrative_rewrite(script, llm, prompt_cfg, style_spec=style_spec)
        except Exception:
            # Non-blocking: if Pass 2 wholesale fails, keep the factual script.
            script["narrative_pass"] = False
        script["style_spec_used"] = bool(style_spec and not style_spec.get("low_confidence"))
        if style_spec and style_spec.get("chapter_style"):
            script["chapter_style"] = style_spec["chapter_style"]
    word_count = script["word_count"]

    if log:
        storage.log_creative_run(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "niche": "iceberg_deepdive",
                "topic": topic,
                "prompt_version": prompt_version,
                "entry_count": entry_count,
                "word_count": word_count,
                "chosen_title": None,
                "chosen_hook": None,
                "review_item_path": None,
                "sources": json.dumps(sources),
            },
            db_path=db_path,
        )

    return script
