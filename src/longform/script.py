"""
Iceberg deep-dive script architect.

Generates a tiered iceberg narration script for a topic:
  - a 0-15s cold_hook that names the iceberg and teases the deepest layer,
  - tiered entries ordered surface -> obscure -> deepest, EACH ending on a mini
    open-loop teasing the next tier (each entry carries a shot list),
  - a final_payoff that resolves the deepest layer.

The generator is few-shotted on the REAL top ``iceberg_deepdive`` winners from
``.mp/farm.db`` (their titles + durations), not generic priors. The structural
prompt lives in ``prompts/longform_iceberg.yaml`` and is versioned; the version
and the produced entry count are recorded in the ledger.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import yaml

from . import storage

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_PROMPT_PATH = os.path.join(_ROOT_DIR, "prompts", "longform_iceberg.yaml")
DEFAULT_TARGET_MINUTES = 10
DEFAULT_FEW_SHOT_N = 6


# --------------------------------------------------------------------------- #
# Prompt + few-shot assembly
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
    """
    Pulls the top real winners for ``niche`` from farm.db as few-shot examples.

    Returns a list of {"title", "duration_min"} dicts, highest score first.
    """
    rows = storage.get_by_niche(niche, db_path=db_path)
    examples = []
    for row in rows[:limit]:
        duration_min = int((row.get("duration_s") or 0) // 60)
        examples.append({"title": row.get("title", ""), "duration_min": duration_min})
    return examples


def _format_few_shot(examples) -> str:
    if not examples:
        return "(no examples available)"
    return "\n".join(
        f"  - {ex['title']} — {ex['duration_min']} min" for ex in examples
    )


def build_script_prompt(
    topic: str,
    prompt_cfg: dict,
    few_shot,
    target_minutes: int = DEFAULT_TARGET_MINUTES,
) -> str:
    """Renders the final prompt string from the YAML config + live few-shot."""
    target_wpm = int(prompt_cfg.get("target_wpm", 140))
    word_budget = target_wpm * target_minutes
    # A 10-min iceberg comfortably holds ~7-9 tiers; scale gently with length.
    tier_hint = max(5, min(9, round(target_minutes * 0.8)))

    instructions = prompt_cfg.get("instructions", "")
    rendered = instructions.format(
        word_budget=word_budget,
        target_wpm=target_wpm,
        target_minutes=target_minutes,
        tier_hint=tier_hint,
        few_shot=_format_few_shot(few_shot),
    )

    schema = prompt_cfg.get("output_schema", "")
    structure = yaml.safe_dump(prompt_cfg.get("structure", {}), sort_keys=False)
    style = yaml.safe_dump(prompt_cfg.get("style", {}), sort_keys=False)

    return (
        f"TOPIC: {topic}\n\n"
        f"{rendered}\n\n"
        f"STRUCTURE:\n{structure}\n"
        f"STYLE:\n{style}\n"
        f"OUTPUT JSON SCHEMA:\n{schema}\n"
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> str:
    """Pulls the first balanced JSON object out of an LLM reply."""
    if not text:
        raise ValueError("empty model output")
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ValueError("unbalanced JSON object in model output")


def _word_count(*parts) -> int:
    return sum(len(re.findall(r"\b\w+\b", part or "")) for part in parts)


def parse_iceberg_script(text: str) -> dict:
    """
    Parses an LLM reply into a validated iceberg script.

    Returns a dict with: iceberg_topic, cold_hook, tiers (each with tier, label,
    entry_title, narration, shot_list, open_loop), final_payoff, entry_count,
    word_count.

    Raises ValueError if no usable JSON or no tiers are present.
    """
    data = json.loads(_extract_json(text))
    if not isinstance(data, dict):
        raise ValueError("parsed JSON is not an object")

    raw_tiers = data.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("script has no tiers")

    tiers = []
    for index, raw in enumerate(raw_tiers, start=1):
        if not isinstance(raw, dict):
            continue
        shot_list = raw.get("shot_list") or []
        if isinstance(shot_list, str):
            shot_list = [shot_list]
        tiers.append(
            {
                "tier": int(raw.get("tier", index) or index),
                "label": str(raw.get("label", f"Tier {index}")),
                "entry_title": str(raw.get("entry_title", "")).strip(),
                "narration": str(raw.get("narration", "")).strip(),
                "shot_list": [str(shot).strip() for shot in shot_list if str(shot).strip()],
                "open_loop": str(raw.get("open_loop", "")).strip(),
            }
        )

    if not tiers:
        raise ValueError("script has no usable tier entries")

    cold_hook = str(data.get("cold_hook", "")).strip()
    final_payoff = str(data.get("final_payoff", "")).strip()
    narration_words = _word_count(
        cold_hook, final_payoff, *[tier["narration"] for tier in tiers]
    )

    return {
        "iceberg_topic": str(data.get("iceberg_topic", "")).strip(),
        "cold_hook": cold_hook,
        "tiers": tiers,
        "final_payoff": final_payoff,
        "entry_count": len(tiers),
        "word_count": narration_words,
    }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_script(
    topic: str,
    llm: Optional[Callable[[str], str]] = None,
    target_minutes: int = DEFAULT_TARGET_MINUTES,
    prompt_path: Optional[str] = None,
    db_path: Optional[str] = None,
    few_shot_n: int = DEFAULT_FEW_SHOT_N,
    run_id: Optional[str] = None,
    max_attempts: int = 2,
    log: bool = True,
) -> dict:
    """
    Generates and parses a tiered iceberg script for ``topic``.

    Args:
        topic (str): The iceberg topic.
        llm: ``llm(prompt) -> str``; defaults to the configured Ollama client.
        target_minutes (int): Drives the word budget at the prompt's wpm.
        db_path (str | None): farm.db path (few-shot source + ledger target).
        run_id (str | None): Correlation id stored on the ledger row.
        max_attempts (int): Retries if the model returns unparseable output.
        log (bool): Whether to record prompt_version + entry_count in the ledger.

    Returns:
        script (dict): parsed script plus prompt_version and run_id.
    """
    if llm is None:
        from .llm import default_llm

        llm = default_llm()

    prompt_cfg = load_prompt_config(prompt_path)
    prompt_version = str(prompt_cfg.get("version", "unknown"))
    few_shot = get_few_shot_winners(db_path=db_path, limit=few_shot_n)
    prompt = build_script_prompt(topic, prompt_cfg, few_shot, target_minutes)

    last_error = None
    script = None
    for _ in range(max(1, max_attempts)):
        raw = llm(prompt)
        try:
            script = parse_iceberg_script(raw)
            break
        except ValueError as exc:
            last_error = exc
            continue
    if script is None:
        raise ValueError(f"could not parse a valid iceberg script: {last_error}")

    script["prompt_version"] = prompt_version
    script["run_id"] = run_id
    if not script.get("iceberg_topic"):
        script["iceberg_topic"] = topic

    if log:
        storage.log_creative_run(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "niche": "iceberg_deepdive",
                "topic": topic,
                "prompt_version": prompt_version,
                "entry_count": script["entry_count"],
                "word_count": script["word_count"],
                "chosen_title": None,
                "chosen_hook": None,
                "review_item_path": None,
            },
            db_path=db_path,
        )

    return script
