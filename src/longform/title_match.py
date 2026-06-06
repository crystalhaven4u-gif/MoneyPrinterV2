"""
Title-formula matcher.

Matches real high-performer titles against the formula bank in
``title_formulas.json`` and tallies which formulas dominate per niche. Each
formula id has a heuristic regex (or set of regexes); a title can match more
than one formula.

The patterns are keyed by the formula ids defined in title_formulas.json. If
the bank gains a formula with no pattern here, it is reported via
``unmatched_formula_ids`` rather than silently ignored.
"""

import json
import os
import re
from collections import Counter
from typing import Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_formulas_path() -> str:
    return os.path.join(_ROOT_DIR, "title_formulas.json")


# Heuristic detectors per formula id. Order does not matter; all are tested.
FORMULA_PATTERNS = {
    "compression": [
        r"\bin\s+\d+\s*(hrs?|hours?|mins?|minutes?|days?|years?)\b",
        r"\b\d+\s*(years?|hours?)\b.*\bin\b",
    ],
    "entire_history": [
        r"\bentire history of\b",
        r"\bhistory of\b.*\bexplained\b",
    ],
    "curiosity_gap": [
        r"^the\s+\w+\s+that\b",
        r"\bthe\s+\w+\s+that\s+(started|changed|ended|saved|destroyed|created)\b",
    ],
    "the_man_who": [
        r"\bthe\s+(man|woman|boy|girl|men|people)\s+who\b",
        r"\bthe day\b",
    ],
    "iceberg": [r"\biceberg\b"],
    "specific_number": [
        r"^\d+\s+\w+",
        r"^\W*\d+\s+\w+\s+(that|nobody|you|which|to)\b",
    ],
    "hidden_truth": [
        r"\bwhat they(?:'re| are| never| don't| do not| aren't)\b",
        r"\bthe real reason\b",
        r"\bwhat they never told you\b",
        r"\bnobody talks about\b",
    ],
    "authority_lead": [
        r"\b(former|ex|retired|veteran|professional|expert|detective|lawyer|doctor|scientist|nasa|engineer)\b.*\bexplains?\b",
        r"\bexplains?:\b",
    ],
    "identity_challenge": [
        r"\byou'?ve\b",
        r"\bit'?s time to\b",
        r"\byou (need|have) to\b",
    ],
    "blueprint": [
        r"\bblueprint\b",
        r"\bmy (exact )?(framework|system|method|blueprint|process)\b",
    ],
    "warning": [
        r"\bstop (doing|using)\b",
        r"\bwhy your\b.*\b(is|are) (failing|killing|dying)\b",
        r"\bbefore it'?s too late\b",
    ],
    "versus": [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bwho (really |actually )?won\b",
    ],
    "provocative_question": [
        r"^(what|why|how|who|where|when|did|is|are|was|were|can|could|should|would|will)\b.*\?\s*$",
        r"\?\s*$",
    ],
    "contrarian": [
        r"\bwhy everything you know\b",
        r"\bis wrong\b",
        r"\bmyth\b",
        r"\bisn'?t what (you think|it seems)\b",
    ],
    "extreme_stakes": [
        r"\b\d+\s*(seconds?|minutes?|hours?|days?)\s+to\b",
        r"\bto (stop|save|prevent|survive|escape)\b",
        r"\b(deadliest|deadly|most dangerous|nuclear war|end of the world)\b",
    ],
}


def load_formula_ids(formulas_path: Optional[str] = None) -> list:
    """Returns the ordered list of formula ids defined in the bank."""
    path = formulas_path or default_formulas_path()
    with open(path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    return [formula["id"] for formula in bank.get("formulas", [])]


def match_title(title: str) -> list:
    """
    Returns the list of formula ids whose heuristic matches ``title``.

    Matching is case-insensitive and a title may match multiple formulas.
    """
    if not title:
        return []
    text = title.strip().lower()
    matched = []
    for formula_id, patterns in FORMULA_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append(formula_id)
                break
    return matched


def tally_titles(titles, formulas_path: Optional[str] = None) -> dict:
    """
    Tallies formula matches across a list of titles.

    Returns:
        result (dict):
            counts (Counter): formula_id -> number of titles matching it
            ranking (list[tuple]): [(formula_id, count), ...] desc
            matched_titles (int): titles that matched at least one formula
            total_titles (int): titles considered
            unmatched_formula_ids (list): formula ids in the bank with no
                heuristic pattern defined here
    """
    counts = Counter()
    matched_titles = 0
    for title in titles:
        hits = match_title(title)
        if hits:
            matched_titles += 1
        counts.update(hits)

    ranking = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    unmatched = []
    bank_ids = load_formula_ids(formulas_path)
    unmatched = [fid for fid in bank_ids if fid not in FORMULA_PATTERNS]

    return {
        "counts": counts,
        "ranking": ranking,
        "matched_titles": matched_titles,
        "total_titles": len(list(titles)) if not isinstance(titles, list) else len(titles),
        "unmatched_formula_ids": unmatched,
    }


def analyze_by_niche(rows, formulas_path: Optional[str] = None) -> dict:
    """
    Groups farmed rows by niche and tallies dominant title formulas per niche.

    Args:
        rows (list[dict]): farmed rows; each needs "niche" and "title".

    Returns:
        per_niche (dict): niche -> ranking list [(formula_id, count), ...]
    """
    titles_by_niche = {}
    for row in rows:
        titles_by_niche.setdefault(row.get("niche", "unknown"), []).append(
            row.get("title", "")
        )

    return {
        niche: tally_titles(titles, formulas_path)["ranking"]
        for niche, titles in titles_by_niche.items()
    }
