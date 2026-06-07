"""
Cold-open hook engine for iceberg deep-dives.

Generates 6-10 cold-open variants for a topic, has the LLM self-score each on
three axes -- curiosity, "how deep does it go" pull, and payoff-promise -- then
keeps the best. All variants and their scores are recorded in the ledger so the
retention loop can later learn which hooks actually held viewers.
"""

import json
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from . import storage

# How the three self-scored axes combine into one rankable number. Curiosity and
# depth-pull drive the click; payoff-promise guards against empty clickbait.
SCORE_WEIGHTS = {"curiosity": 0.4, "depth_pull": 0.35, "payoff_promise": 0.25}

MIN_HOOKS = 6
MAX_HOOKS = 10


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _extract_json(text: str):
    """Returns the first JSON array or object found in an LLM reply."""
    if not text:
        raise ValueError("empty model output")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
    raise ValueError("no JSON found in model output")


def _hooks_from_lines(text: str) -> list:
    """Fallback parser: small models often return a numbered/bulleted list."""
    hooks = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip leading list markers ("1.", "2)", "-", "*", "•") and quotes.
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        # Skip obvious headers/labels ("Here are 8 hooks:", "Hooks:").
        if line.endswith(":") or len(line) < 12:
            continue
        hooks.append(line)
    return hooks


def parse_hook_list(text: str) -> list:
    """
    Parses an LLM reply into a clean list of hook strings.

    Accepts a JSON array of strings, a JSON array of {"hook": ...} objects, or
    (fallback) a plain numbered/bulleted list. Raises ValueError if nothing
    usable is found.
    """
    hooks = []
    try:
        data = _extract_json(text)
        if isinstance(data, dict):
            data = data.get("hooks", [])
        for item in data or []:
            if isinstance(item, str):
                value = item.strip()
            elif isinstance(item, dict):
                value = str(item.get("hook", "")).strip()
            else:
                value = ""
            if value:
                hooks.append(value)
    except ValueError:
        hooks = []

    if not hooks:
        hooks = _hooks_from_lines(text)
    if not hooks:
        raise ValueError("no hooks parsed from model output")
    return hooks


def _clamp_unit10(value) -> float:
    """Coerces a score to a 0-10 float (accepts 0-1 inputs too)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= number <= 1.0:
        number *= 10.0
    return max(0.0, min(10.0, number))


def parse_scores(text: str) -> dict:
    """
    Parses an LLM self-score reply into {curiosity, depth_pull, payoff_promise},
    each a 0-10 float. Missing axes default to 0.
    """
    data = _extract_json(text)
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    return {
        "curiosity": _clamp_unit10(data.get("curiosity")),
        "depth_pull": _clamp_unit10(data.get("depth_pull")),
        "payoff_promise": _clamp_unit10(data.get("payoff_promise")),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def aggregate_score(scores: dict) -> float:
    """Weighted total of the three axes, rounded to 3 dp."""
    total = sum(
        SCORE_WEIGHTS[axis] * float(scores.get(axis, 0.0)) for axis in SCORE_WEIGHTS
    )
    return round(total, 3)


def select_best(scored: list) -> dict:
    """
    Returns the highest-scoring hook. Ties break on curiosity, then on the hook
    text (stable + deterministic).
    """
    if not scored:
        raise ValueError("no scored hooks to choose from")
    return max(
        scored,
        key=lambda item: (
            item["total"],
            item["scores"]["curiosity"],
            item["hook"],
        ),
    )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _bare_topic(topic: str) -> str:
    """Strips a leading article so "the {topic} iceberg" reads naturally."""
    return re.sub(r"^(the|a|an)\s+", "", (topic or "").strip(), flags=re.IGNORECASE)


def _hooks_prompt(topic: str, n: int) -> str:
    topic = _bare_topic(topic)
    return (
        f"You write cold opens for long-form 'iceberg' YouTube deep-dives.\n"
        f"Topic: the {topic} iceberg.\n\n"
        f"Write {n} DISTINCT cold-open hooks (1-2 sentences each, spoken aloud "
        f"in the first 15 seconds). Each must name the iceberg, hint at how deep "
        f"it goes, and promise a payoff at the bottom -- without revealing it.\n\n"
        f'Return ONLY a JSON array of {n} strings.'
    )


def _score_prompt(topic: str, hook: str) -> str:
    topic = _bare_topic(topic)
    return (
        f"Rate this cold-open hook for a '{topic} iceberg' deep-dive on three "
        f"axes, each 0-10:\n"
        f"  curiosity: how strongly it makes a viewer need to know more\n"
        f"  depth_pull: how much it sells 'how deep does this go?'\n"
        f"  payoff_promise: how credibly it promises a real payoff (not empty "
        f"clickbait)\n\n"
        f"HOOK: {hook}\n\n"
        f'Return ONLY JSON: {{"curiosity": <n>, "depth_pull": <n>, '
        f'"payoff_promise": <n>}}'
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def run_hook_engine(
    topic: str,
    llm: Optional[Callable[[str], str]] = None,
    n: int = 8,
    db_path: Optional[str] = None,
    run_id: Optional[str] = None,
    log: bool = True,
) -> dict:
    """
    Generates, self-scores, and ranks cold-open hooks for ``topic``.

    Returns:
        result (dict): best (the chosen variant) and variants (all, scored,
        sorted best-first). Each variant is {hook, scores, total, is_best}.
    """
    if llm is None:
        from .llm import default_llm

        llm = default_llm()

    n = max(MIN_HOOKS, min(MAX_HOOKS, n))
    hooks = parse_hook_list(llm(_hooks_prompt(topic, n)))

    scored = []
    for hook in hooks:
        try:
            scores = parse_scores(llm(_score_prompt(topic, hook)))
        except ValueError:
            scores = {"curiosity": 0.0, "depth_pull": 0.0, "payoff_promise": 0.0}
        scored.append({"hook": hook, "scores": scores, "total": aggregate_score(scores)})

    best = select_best(scored)
    for variant in scored:
        variant["is_best"] = variant is best
    scored.sort(key=lambda item: item["total"], reverse=True)

    if log:
        created_at = datetime.now(timezone.utc).isoformat()
        for variant in scored:
            storage.log_hook_variant(
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "hook_text": variant["hook"],
                    "curiosity": variant["scores"]["curiosity"],
                    "depth_pull": variant["scores"]["depth_pull"],
                    "payoff_promise": variant["scores"]["payoff_promise"],
                    "total": variant["total"],
                    "is_best": int(variant["is_best"]),
                },
                db_path=db_path,
            )

    return {"best": best, "variants": scored}
