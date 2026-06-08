"""
Packaging for iceberg deep-dives: titles, thumbnails, and the review-queue item.

- Titles: 5 variants built from the proven iceberg formula plus 'specific_number'
  and 'curiosity_gap' variants (see title_formulas.json), styled after the real
  winners in farm.db (intensifiers, closing brackets, depth challenges).
- Thumbnails: 3 tiered-iceberg concepts, each rendered at 1280x720 by calling
  generate_image() from image_providers.py (the free Pollinations path) and then
  overlaying big, readable PIL text.
- Everything (script, all hook variants + scores, all title variants, thumbnail
  concepts) is written into a single review-queue item under
  review_queue/pending/<run_id>/; the chosen title/hook are logged in the ledger.

Nothing here publishes -- it only stages a human review item.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

from . import image_providers, storage

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REVIEW_BASE = os.path.join(_ROOT_DIR, "review_queue", "pending")
THUMB_SIZE = (1280, 720)


# --------------------------------------------------------------------------- #
# Titles
# --------------------------------------------------------------------------- #
def _display_topic(topic: str) -> str:
    """
    Normalizes a topic for use inside templates that supply their own article.

    Strips a leading "The/A/An" so "The Backrooms" -> "Backrooms" (avoiding
    "The The Backrooms Iceberg"), and title-cases an all-lowercase topic.
    """
    topic = (topic or "").strip()
    topic = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.IGNORECASE)
    if topic and topic == topic.lower():
        return topic.title()
    return topic


def _odd_near(value: Optional[int]) -> int:
    """Nearest 'specific, odd' count from the formula rules (7/9/13 feel)."""
    candidates = [7, 9, 13]
    if not value or value < 5:
        return 9
    # round value up to an odd number, capped at 13.
    odd = value if value % 2 == 1 else value + 1
    return min(odd, 13) if odd >= 7 else min(candidates, key=lambda c: abs(c - odd))


def build_title_variants(topic: str, entry_count: Optional[int] = None) -> list:
    """
    Builds 5 title variants for the iceberg on ``topic``.

    Returns a list of {"formula", "title"} dicts: three using the 'iceberg'
    formula (base, intensified, depth-challenge), one 'specific_number', and one
    'curiosity_gap'.
    """
    name = _display_topic(topic)
    number = _odd_near(entry_count)
    return [
        {"formula": "iceberg", "title": f"The {name} Iceberg Explained"},
        {
            "formula": "iceberg",
            "title": f"The DISTURBING {name} Iceberg (It Gets Worse)",
        },
        {
            "formula": "specific_number",
            "title": f"{number} Layers of the {name} Iceberg That Go Too Deep",
        },
        {
            "formula": "curiosity_gap",
            "title": f"The {name} Secret Hiding at the Bottom of the Iceberg",
        },
        {
            "formula": "iceberg",
            "title": f"The {name} Iceberg Nobody Has Reached the Bottom Of",
        },
    ]


def build_titles_with_spec(topic: str, entry_count: Optional[int] = None,
                           style_spec: Optional[dict] = None) -> list:
    """Title variants augmented with the learned style spec's title patterns.

    Falls back to the proven built-in variants when no spec is supplied, so the
    default behaviour (and its tests) are unchanged. Spec patterns may use a
    ``{Topic}`` / ``{topic}`` placeholder and are filled with the display topic;
    they are de-duplicated against the built-ins and capped so the list stays
    review-friendly.
    """
    base = build_title_variants(topic, entry_count=entry_count)
    patterns = (style_spec or {}).get("title_patterns") or []
    if not patterns:
        return base

    name = _display_topic(topic)
    learned, seen = [], {t["title"].lower() for t in base}
    for pattern in patterns:
        try:
            title = str(pattern).replace("{Topic}", name).replace("{topic}", name).strip()
        except Exception:
            continue
        if not title or "{" in title or title.lower() in seen:
            continue
        seen.add(title.lower())
        learned.append({"formula": "learned_pattern", "title": title})
        if len(learned) >= 2:
            break
    # Learned patterns first (they reflect what actually performs), then proven.
    return (learned + base)[:6]


# --------------------------------------------------------------------------- #
# Thumbnail concepts
# --------------------------------------------------------------------------- #
def build_thumbnail_concepts(topic: str) -> list:
    """Builds 3 distinct tiered-iceberg thumbnail concepts for ``topic``."""
    name = _display_topic(topic)
    upper = name.upper()
    return [
        {
            "concept_id": "split_waterline",
            "visual_prompt": (
                f"dramatic photoreal iceberg split at the waterline, small bright "
                f"tip above calm water and a vast dark mass plunging into black "
                f"depths below, themed around {name}, cinematic, high contrast, "
                f"ominous, 16:9 YouTube thumbnail"
            ),
            "overlay_text": f"THE {upper}\nICEBERG",
            "palette": "icy cyan over near-black deep",
        },
        {
            "concept_id": "tiered_glow",
            "visual_prompt": (
                f"a towering iceberg cross-section divided into glowing horizontal "
                f"tiers descending into darkness, each tier dimmer and more "
                f"unsettling, {name} motifs, volumetric light, eerie, cinematic, "
                f"16:9 YouTube thumbnail"
            ),
            "overlay_text": f"{upper}\nHOW DEEP?",
            "palette": "teal glow fading to abyssal black",
        },
        {
            "concept_id": "abyss_focus",
            "visual_prompt": (
                f"extreme low-angle looking down the bottom tier of an iceberg into "
                f"a pitch-black abyss, a single faint unsettling detail about {name} "
                f"barely visible at the very bottom, dread, fog, cinematic, 16:9 "
                f"YouTube thumbnail"
            ),
            "overlay_text": f"THE BOTTOM OF\nTHE {upper} ICEBERG",
            "palette": "deep blue-black, single cold highlight",
        },
    ]


# --------------------------------------------------------------------------- #
# Rendering (generate_image + PIL text overlay)
# --------------------------------------------------------------------------- #
# Heavy, readable fonts for thumbnail text, best-first. The bundled
# fonts/bold_font.ttf is a LAST resort: it has broken metrics (zero-height
# bbox) that segfault freetype on multi-word strings in Pillow 9.x, so prefer a
# real system font wherever one exists.
# Thumbnail text is rendered with ImageMagick (a project dependency, configured
# at imagemagick_path) rather than Pillow/freetype: the local Pillow build
# segfaults nondeterministically on repeated freetype text rendering. ImageMagick
# rasterizes text in its own process, so it is both robust and higher quality.
# If ImageMagick is unavailable we fall back to a text-less cover-fit so a render
# never crashes.
_MIN_POINTSIZE = 44
_MAX_POINTSIZE = 120
_CHAR_WIDTH_RATIO = 0.55      # avg glyph width / pointsize, for size heuristic
_LINE_RATIO = 1.18            # line spacing as a multiple of pointsize


def _imagemagick_path() -> Optional[str]:
    """Resolves the ImageMagick `magick` binary from config, then PATH."""
    try:
        import sys

        src_dir = os.path.join(_ROOT_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from config import get_imagemagick_path

        path = (get_imagemagick_path() or "").strip()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    import shutil

    return shutil.which("magick") or shutil.which("convert")


def _coverfit_no_text(base_image_path: str, out_path: str) -> str:
    """Fallback: cover-fit the base to 1280x720 with no text (never crashes)."""
    from PIL import Image

    image = Image.open(base_image_path).convert("RGB")
    target_w, target_h = THUMB_SIZE
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)))
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    resized.crop((left, top, left + target_w, top + target_h)).save(out_path, "PNG")
    return out_path


def _pointsize_for(lines: list) -> int:
    """Heuristic pointsize that fits the widest line into 1280x720."""
    target_w, target_h = THUMB_SIZE
    max_chars = max((len(line) for line in lines), default=1) or 1
    by_width = (target_w * 0.90) / (max_chars * _CHAR_WIDTH_RATIO)
    by_height = (target_h * 0.52) / (len(lines) * _LINE_RATIO)
    return int(max(_MIN_POINTSIZE, min(_MAX_POINTSIZE, by_width, by_height)))


def overlay_text(
    base_image_path: str, text: str, out_path: str, magick_path: Optional[str] = None
) -> str:
    """
    Renders large, high-contrast title text onto a 1280x720 thumbnail using
    ImageMagick (white fill + black outline, bottom-centered, one annotate pass
    per line). Falls back to a text-less cover-fit if ImageMagick is missing or
    fails, so a render never crashes.
    """
    import subprocess

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [line.strip() for line in (text or "").split("\n") if line.strip()]
    magick = magick_path or _imagemagick_path()

    if not lines or not magick:
        return _coverfit_no_text(base_image_path, out_path)

    pointsize = _pointsize_for(lines)
    line_step = int(pointsize * _LINE_RATIO)

    command = [
        magick, base_image_path,
        "-resize", "1280x720^", "-gravity", "center", "-extent", "1280x720",
        "-gravity", "south", "-font", "Arial-Bold", "-pointsize", str(pointsize),
    ]
    # Bottom-up: last line sits lowest. Each line: black outline then white fill.
    for index, line in enumerate(reversed(lines)):
        offset = 44 + index * line_step
        command += [
            "-strokewidth", str(max(6, pointsize // 9)), "-stroke", "black",
            "-fill", "none", "-annotate", f"+0+{offset}", line,
            "-stroke", "none", "-fill", "white", "-annotate", f"+0+{offset}", line,
        ]
    command.append(out_path)

    try:
        result = subprocess.run(command, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path):
            return out_path
        # Retry once without an explicit font (some installs lack Arial-Bold).
        no_font = [arg for arg in command if arg not in ("-font", "Arial-Bold")]
        result = subprocess.run(no_font, capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(out_path):
            return out_path
    except Exception:
        pass

    return _coverfit_no_text(base_image_path, out_path)


def render_thumbnail(
    concept: dict,
    output_path: str,
    generate_image_fn: Optional[Callable] = None,
    do_overlay: bool = True,
    quality: str = "standard",
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Renders one thumbnail: generate a 1280x720 base via generate_image(), then
    overlay the concept's text.

    Args:
        generate_image_fn: defaults to image_providers.generate_image. Injected
            (mocked) in tests so no real network call happens.

    Returns:
        dict: concept_id, path, provider, is_placeholder, overlay_text,
        visual_prompt.
    """
    gen = generate_image_fn or image_providers.generate_image
    base_path = output_path
    if do_overlay:
        root, ext = os.path.splitext(output_path)
        base_path = f"{root}_base{ext or '.png'}"

    result = gen(
        prompt=concept["visual_prompt"],
        aspect_ratio="16:9",
        output_path=base_path,
        quality=quality,
        config=config,
        session=session,
        db_path=db_path,
    )

    final_path = output_path
    if do_overlay:
        overlay_text(result.path, concept.get("overlay_text", ""), output_path)
    else:
        final_path = result.path

    return {
        "concept_id": concept.get("concept_id"),
        "path": final_path,
        "provider": getattr(result, "provider", None),
        "is_placeholder": getattr(result, "is_placeholder", None),
        "overlay_text": concept.get("overlay_text", ""),
        "visual_prompt": concept["visual_prompt"],
    }


# --------------------------------------------------------------------------- #
# Review-queue item
# --------------------------------------------------------------------------- #
def submit_to_review(
    run_id: str,
    topic: str,
    script: dict,
    hooks_result: dict,
    titles: list,
    concepts: list,
    base_dir: Optional[str] = None,
    render: bool = True,
    generate_image_fn: Optional[Callable] = None,
    do_overlay: bool = True,
    config: Optional[dict] = None,
    session=None,
    db_path: Optional[str] = None,
) -> dict:
    """
    Writes a single review item with the script, all hook + title variants, and
    rendered thumbnails, then logs the chosen title/hook in the ledger.

    Returns a summary dict including the item directory and chosen fields.
    """
    item_dir = os.path.join(base_dir or DEFAULT_REVIEW_BASE, run_id)
    os.makedirs(item_dir, exist_ok=True)

    thumbnails = []
    if render:
        for index, concept in enumerate(concepts, start=1):
            out = os.path.join(item_dir, f"thumb_{index}_{concept['concept_id']}.png")
            thumbnails.append(
                render_thumbnail(
                    concept,
                    out,
                    generate_image_fn=generate_image_fn,
                    do_overlay=do_overlay,
                    config=config,
                    session=session,
                    db_path=db_path,
                )
            )
    else:
        thumbnails = [dict(concept, path=None) for concept in concepts]

    best_hook = (hooks_result or {}).get("best", {})
    chosen_title = titles[0]["title"] if titles else None
    chosen_hook = best_hook.get("hook")

    package = {
        "run_id": run_id,
        "niche": "iceberg_deepdive",
        "topic": topic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "prompt_version": script.get("prompt_version"),
        "chosen": {"title": chosen_title, "hook": chosen_hook},
        "script": script,
        "hook_variants": (hooks_result or {}).get("variants", []),
        "title_variants": titles,
        "thumbnails": thumbnails,
    }

    with open(os.path.join(item_dir, "package.json"), "w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2)

    # Ledger: finalize the run row + record every title variant.
    storage.update_creative_run_choices(
        run_id=run_id,
        chosen_title=chosen_title,
        chosen_hook=chosen_hook,
        review_item_path=item_dir,
        db_path=db_path,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    for index, variant in enumerate(titles):
        storage.log_title_variant(
            {
                "run_id": run_id,
                "created_at": created_at,
                "formula": variant.get("formula"),
                "title": variant.get("title"),
                "is_chosen": int(index == 0),
            },
            db_path=db_path,
        )

    return {
        "run_id": run_id,
        "item_dir": item_dir,
        "chosen_title": chosen_title,
        "chosen_hook": chosen_hook,
        "thumbnails": thumbnails,
        "package_path": os.path.join(item_dir, "package.json"),
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def generate_iceberg_package(
    topic: str,
    run_id: str,
    llm: Optional[Callable] = None,
    target_minutes: int = 10,
    db_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    generate_image_fn: Optional[Callable] = None,
    n_hooks: int = 8,
) -> dict:
    """
    End-to-end: script -> hooks -> titles -> thumbnails -> review item.

    Returns the submit_to_review summary plus the script and hooks results.
    """
    from . import hooks as hooks_module
    from . import script as script_module
    from . import style_spec as style_spec_module

    # Learned per-niche style spec (cached; built separately). Non-blocking.
    try:
        spec = style_spec_module.load_cached("iceberg_deepdive")
    except Exception:
        spec = None

    script = script_module.generate_script(
        topic, llm=llm, target_minutes=target_minutes, db_path=db_path, run_id=run_id,
        style_spec=spec,
    )
    hooks_result = hooks_module.run_hook_engine(
        topic, llm=llm, n=n_hooks, db_path=db_path, run_id=run_id, style_spec=spec,
    )
    titles = build_titles_with_spec(topic, entry_count=script["entry_count"], style_spec=spec)
    concepts = build_thumbnail_concepts(topic)

    summary = submit_to_review(
        run_id=run_id,
        topic=topic,
        script=script,
        hooks_result=hooks_result,
        titles=titles,
        concepts=concepts,
        base_dir=base_dir,
        generate_image_fn=generate_image_fn,
        db_path=db_path,
    )
    summary["script"] = script
    summary["hooks_result"] = hooks_result
    return summary
