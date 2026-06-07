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
    """Parses a PER-ENTRY reply into {narration, shot_list, open_loop}."""
    data = json.loads(_extract_json(text), strict=False)
    if not isinstance(data, dict):
        raise ValueError("entry JSON is not an object")
    narration = str(data.get("narration", "")).strip()
    if not narration:
        raise ValueError("entry has no narration")
    return {
        "narration": narration,
        "shot_list": _coerce_shot_list(data.get("shot_list")),
        "open_loop": str(data.get("open_loop", "")).strip(),
    }


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
        + ". Give a shot_list of 2-4 concrete visuals.\n\n"
        f"Return ONLY JSON: {{\"narration\": str, \"shot_list\": [str], "
        f"\"open_loop\": str}}"
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
    }

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
