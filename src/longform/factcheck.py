"""
Specific-claim extraction + verification for the two-pass script architect.

This is the accuracy backbone:

  * PASS 2 ENTITY LOCK -- Pass 2 may only RESTYLE Pass 1. ``new_specifics`` diffs
    the named entities / dates / numbers in a re-voiced section against the set
    allowed by Pass 1; anything new (an invented person like "Mark", a fake date,
    a number) is a violation to be re-prompted away, then stripped/flagged.
  * PASS 1 VERIFICATION -- ``verify_claims`` extracts every date / number /
    proper-noun claim from generated narration and checks whether the entry's
    SOURCE text supports it, so unsupported specifics can be surfaced to the
    human reviewer instead of silently shipping.

Deliberately dependency-free (regex + stdlib): no spaCy/NLTK. Heuristic, tuned to
favour FEW false strips in the lock (so good styling survives) while still
catching invented people/dates/numbers, and to over-surface (not under-surface)
in the advisory confidence report.
"""

import re

_MONTHS = ("January February March April May June July August September October "
           "November December").split()
_MONTH_RE = "(?:%s)" % "|".join(_MONTHS)

# Date forms, richest first. Years are the catch-all.
_DATE_PATTERNS = [
    re.compile(r"\b%s\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b" % _MONTH_RE, re.I),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+%s\s+\d{4}\b" % _MONTH_RE, re.I),
    re.compile(r"\b%s\s+\d{4}\b" % _MONTH_RE, re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(?:1\d{3}|20\d{2})\b"),
]
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_PROPER_RUN = re.compile(r"[A-Z][A-Za-z'’]+(?:\s+[A-Z][A-Za-z'’]+)*")
_TITLED = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sir|Saint|St|Lord|Lady|Detective|Officer|Agent|"
    r"Captain|Sergeant|President|King|Queen)\.?\s+[A-Z][A-Za-z'’]+"
    r"(?:\s+[A-Z][A-Za-z'’]+)*"
)

# Common words that are capitalized for grammar/style, not because they name a
# specific thing. Kept lowercase. Sentence-initial capitals are skipped
# structurally; this catches the mid-sentence stylistic ones.
_STOP = {
    "the", "a", "an", "and", "or", "but", "so", "yet", "for", "nor", "to", "of",
    "in", "on", "at", "by", "as", "if", "it", "is", "be", "this", "that", "these",
    "those", "there", "here", "then", "than", "now", "once", "over", "under",
    "into", "out", "up", "down", "with", "without", "you", "your", "youre",
    "youve", "yours", "i", "im", "ive", "we", "weve", "were", "our", "ours",
    "they", "them", "their", "theirs", "he", "his", "him", "she", "her", "hers",
    "imagine", "consider", "pause", "stay", "picture", "think", "look", "listen",
    "remember", "welcome", "today", "most", "many", "some", "every", "each",
    "what", "when", "where", "why", "how", "who", "which", "whose", "maybe",
    "perhaps", "because", "although", "while", "after", "before", "until",
    "deeper", "darker", "beneath", "below", "above", "next", "first", "last",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
_STOP.update(m.lower() for m in _MONTHS)


_ABBREV = ("Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "vs", "etc",
           "Inc", "Ltd", "Lt", "Sgt", "Capt", "Gen", "No")


def _sentences(text: str) -> list:
    """Splits into sentences, protecting common abbreviation periods (so
    "Dr. Emma Taylor" is not split after "Dr.")."""
    protected = text or ""
    for abbrev in _ABBREV:
        protected = re.sub(r"\b%s\." % abbrev, abbrev + "\x00", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return [p.replace("\x00", ".") for p in parts if p]


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"['’]s\b", "", text)        # possessive
    text = text.strip(".,;:!?\"'()[]’‘").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _norm_number(text: str) -> str:
    return (text or "").replace(",", "").strip()


def extract_dates(text: str) -> set:
    found, scratch = set(), text or ""
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(scratch):
            found.add(_norm(match.group(0)))
    return found


def extract_numbers(text: str) -> set:
    # Remove date spans first so years/days aren't double-counted as numbers.
    scratch = text or ""
    for pattern in _DATE_PATTERNS:
        scratch = pattern.sub(" ", scratch)
    return {_norm_number(m.group(0)) for m in _NUMBER_RE.finditer(scratch)}


def extract_proper_nouns(text: str, skip_initial: bool = False, explode: bool = False) -> set:
    """Proper-noun phrases in ``text``.

    skip_initial: ignore a run that begins a sentence (grammatical capital, not a
        name) -- used when scanning a re-voiced candidate so styling like
        "Imagine ..." is not mistaken for an entity.
    explode: also add the individual component tokens of multi-word names -- used
        when building the ALLOWED set so "Backrooms" matches "The Backrooms".
    """
    out = set()
    for sentence in _sentences(text):
        for match in _TITLED.finditer(sentence):  # titled names anywhere
            out.add(_norm(match.group(0)))
        for match in _PROPER_RUN.finditer(sentence):
            phrase = match.group(0).strip()
            tokens = phrase.split()
            if skip_initial and match.start() == 0:
                tokens = tokens[1:]  # drop the grammatical sentence-initial capital
            if not tokens:
                continue
            if len(tokens) >= 2:
                out.add(_norm(" ".join(tokens)))
                if explode:
                    for token in tokens:
                        normed = _norm(token)
                        if len(normed) >= 3 and normed not in _STOP:
                            out.add(normed)
            else:
                normed = _norm(tokens[0])
                if len(normed) >= 3 and normed not in _STOP:
                    out.add(normed)
    return out


def extract_specifics(text: str, skip_initial: bool = False, explode: bool = False) -> dict:
    """All specifics in ``text``: {proper_nouns, dates, numbers} (normalized)."""
    return {
        "proper_nouns": extract_proper_nouns(text, skip_initial=skip_initial, explode=explode),
        "dates": extract_dates(text),
        "numbers": extract_numbers(text),
    }


def allowed_specifics(*texts) -> dict:
    """The specifics Pass 1 permits -- proper nouns exploded into component
    tokens so a restyled mention of part of a name is not a false violation."""
    blob = "\n".join(t for t in texts if t)
    return extract_specifics(blob, skip_initial=False, explode=True)


def new_specifics(allowed: dict, candidate_text: str) -> dict:
    """Specifics present in ``candidate_text`` (a Pass-2 section) but NOT allowed
    by Pass 1. Returns {proper_nouns, dates, numbers, all} -- ``all`` is a flat,
    sorted list of the offending strings for re-prompting / stripping / flagging.
    """
    cand = extract_specifics(candidate_text, skip_initial=True, explode=False)
    allowed_proper = allowed.get("proper_nouns", set())

    bad_proper = set()
    for phrase in cand["proper_nouns"]:
        if phrase in allowed_proper:
            continue
        tokens = phrase.split()
        if len(tokens) > 1 and all(tok in allowed_proper for tok in tokens):
            continue  # every component is known -> not an invention
        bad_proper.add(phrase)

    bad_dates = cand["dates"] - allowed.get("dates", set())
    bad_numbers = cand["numbers"] - allowed.get("numbers", set())
    flat = sorted(bad_proper) + sorted(bad_dates) + sorted(bad_numbers)
    return {"proper_nouns": bad_proper, "dates": bad_dates,
            "numbers": bad_numbers, "all": flat}


def strip_sentences_with(text: str, terms) -> str:
    """Drops sentences that still contain any of ``terms`` (last-resort hard
    strip after re-prompts fail). Case-insensitive substring match."""
    if not terms:
        return text
    lowered = [t for t in terms if t]
    kept = []
    for sentence in _sentences(text):
        low = sentence.lower()
        if any(term in low for term in lowered):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


# --------------------------------------------------------------------------- #
# Pass 1 verification (claim support against the source extract)
# --------------------------------------------------------------------------- #
# A handful of genre-common, broadly-known references that should not be flagged
# as "unsupported" just because a short source extract omits them.
_COMMON_KNOWLEDGE = {
    "4chan", "reddit", "youtube", "internet", "wikipedia", "google", "discord",
    "twitter", "facebook", "creepypasta",
}


def _word_in(needle: str, haystack: str) -> bool:
    """Whole-token containment (so '9' does not match inside '2019')."""
    return re.search(r"(?<!\w)%s(?!\w)" % re.escape(needle), haystack) is not None


def _supported(claim: str, source_norm: str) -> bool:
    claim = (claim or "").strip()
    if not claim:
        return True
    low = claim.lower()
    if low in _COMMON_KNOWLEDGE:
        return True
    if _word_in(low, source_norm):
        return True
    tokens = [t for t in re.split(r"\s+", low) if len(t) >= 3]
    if tokens and all(_word_in(tok, source_norm) for tok in tokens):
        return True  # every meaningful token of the name appears in the source
    return False


def verify_claims(narration: str, source_text: str) -> dict:
    """Checks each specific in ``narration`` against ``source_text``.

    Returns {checked, supported, flagged:[{type, claim}], score} where score is
    supported/checked (1.0 when nothing to check). Proper nouns, dates and
    numbers are all verified; unsupported ones are flagged for the reviewer.
    """
    source_norm = re.sub(r"\s+", " ", (source_text or "").lower())
    # skip_initial: don't flag a sentence-initial capital (grammar, not a name);
    # titled names ("Dr. Emma Taylor") are still caught regardless of position.
    spec = extract_specifics(narration, skip_initial=True, explode=False)
    flagged, checked, supported = [], 0, 0
    for kind, key in (("name", "proper_nouns"), ("date", "dates"), ("number", "numbers")):
        for claim in sorted(spec[key]):
            checked += 1
            if _supported(claim, source_norm):
                supported += 1
            else:
                flagged.append({"type": kind, "claim": claim})
    score = round(supported / checked, 3) if checked else 1.0
    return {"checked": checked, "supported": supported, "flagged": flagged, "score": score}
